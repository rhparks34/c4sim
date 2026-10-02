"""Train the model on one fold: python -m src.train --fold K [--smoke] [--out DIR].

Writes per-epoch EMA and live checkpoints, history.csv (per-epoch metrics, including each output's masked RMSE) and run.json.
--smoke is a quick CPU check on real data: 16 training plays, 4 validation plays, 2 epochs of 4 updates with batches of 2.
"""
import argparse
import copy
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import paths
from src.recipe import RECIPE
from src.model import evaluate, inputs, network
from src.model.batches import PlayDataset, SideStore, TrainingItems, plan, seeded_batches
from src.model.losses import objective

OUTPUTS = ('standard', 'route_break', 'midpoint')
QB_FEATURE_NAMES = ('qb_dx', 'qb_dy')
read_play_ids = inputs.read_play_ids


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class EMA:
    """Exponential moving average of the weights; buffers are copied."""

    def __init__(self, model):
        self.module = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for target, source in zip(self.module.parameters(), model.parameters()):
            target.lerp_(source, 1. - RECIPE.ema_decay)
        for target, source in zip(self.module.buffers(), model.buffers()):
            target.copy_(source)


def build_model(device):
    """The saved random initialization (the same draw on every machine), then every random generator reseeded."""
    manifest = inputs.load_json(paths.PREPARED / 'initial_state_manifest.json')
    path = paths.PREPARED / 'initial_state.pt'
    assert inputs.sha256(path.read_bytes()) == manifest['file_sha256'], 'initial_state.pt does not match its manifest'
    state = torch.load(path, map_location='cpu', weights_only=False)
    seed_everything(RECIPE.seed)
    model = network.C4Sim()
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    seed_everything(RECIPE.seed)
    return model


def set_qb_statistics(model, stats):
    """The model rebuilds the QB's position from the normalized qb_dx / qb_dy features, so its buffers must hold this fold's
    statistics (the same values the dataset normalizes with)."""
    mean = np.array([stats['feature_mean'].get(n, 0.0) for n in QB_FEATURE_NAMES], dtype=np.float32)
    std = np.array([stats['feature_std'].get(n, 1.0) for n in QB_FEATURE_NAMES], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    with torch.no_grad():
        model.qb_mean.copy_(torch.from_numpy(mean)); model.qb_std.copy_(torch.from_numpy(std))
    assert torch.equal(model.qb_mean.cpu(), torch.from_numpy(mean)) and torch.equal(model.qb_std.cpu(), torch.from_numpy(std))


def side_inputs(fold):
    return SideStore(inputs.load_side(fold, 'retrieval'), inputs.load_side(fold, 'orientation'), inputs.load_side(fold, 'position_blocker'))


def masked_rmse(d, mask):
    """RMSE of the reported prediction after the route-breakdown scoring mask (rows past a masked play's score window leave)."""
    w = d.game_play_id.map(dict(zip(mask.game_play_id, mask.score_window)))
    keep = w.isna() | (d.frame_index <= w)
    return evaluate.rmse(d[keep.to_numpy()], 'candidate')


def validate(model, ids, data, stats, side, device, mask):
    """-> {output: (metrics, predictions)}: one forward pass per reflection; the other outputs replay it."""
    view = network.OutputView(model); out = {}
    for name in OUTPUTS:
        view.reset(name)
        metrics, d = evaluate.validation(view, ids, data, PlayDataset(ids, data, stats, side, is_train=False), device)
        metrics['masked_rmse'] = masked_rmse(d, mask)
        out[name] = (metrics, d)
    view.cache = []
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=range(4))
    ap.add_argument('--smoke', action='store_true', help='CPU check: 16 training plays, 4 validation plays, 2 epochs of 4 updates')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)
    out_dir = args.out or paths.OUT_ROOT / 'train' / f'fold{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    assert not (out_dir / 'history.csv').exists(), f'{out_dir} already holds a run'
    torch.set_num_threads(2)
    lambdas = dict(segment=RECIPE.rule_path_weight, priv=RECIPE.teacher_weight)
    population = inputs.load_json(paths.fold_dir(args.fold) / 'population.json')
    stats = inputs.load_json(paths.fold_dir(args.fold) / 'stats.json')
    side = side_inputs(args.fold)
    model = build_model(args.device)
    set_qb_statistics(model, stats)
    data, _ = inputs.load_plays(args.fold)
    ids = read_play_ids(args.fold)
    train_ids = ids[ids.split.eq('train')].reset_index(drop=True)
    valid_ids = ids[ids.split.eq('validation')].reset_index(drop=True)
    assert not set(train_ids.game_id) & set(valid_ids.game_id)
    if args.smoke:
        train_ids = train_ids.head(16).reset_index(drop=True); valid_ids = valid_ids.head(4).reset_index(drop=True)
    else:
        assert len(train_ids) == population['training_plays'] and len(valid_ids) == population['validation_plays']
    mask = pd.read_csv(paths.SCORING_MASK, dtype={'game_play_id': str})
    mean_rows = population['training_mean_native_rows']       # mean valid defender-frames per training play
    items = TrainingItems(PlayDataset(train_ids, data, stats, side, is_train=True))
    batch_size, epochs, steps = (2, 2, 4) if args.smoke else (RECIPE.batch_size, RECIPE.epochs, len(train_ids) // RECIPE.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE.lr, weight_decay=RECIPE.weight_decay)
    # The route-break readout is clipped apart from the rest of the model.
    groups = {'shared': [p for n, p in model.named_parameters() if not n.startswith('route_break_')],
              'route_break': [p for n, p in model.named_parameters() if n.startswith('route_break_')]}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RECIPE.schedule_epochs)
    ema = EMA(model)
    history = []; updates = 0
    for epoch in range(epochs):
        model.train(); stage_sse = np.zeros(3); row_count = 0; sums = {}; grad_sum = clip_sum = rb_grad_sum = 0.
        workers = 0 if args.smoke else RECIPE.loader_workers
        loader = DataLoader(items, batch_sampler=plan(len(train_ids), epoch, batch_size, steps), num_workers=workers,
                            pin_memory=args.device == 'cuda', generator=torch.Generator().manual_seed(840000 + epoch),
                            **({'multiprocessing_context': 'fork'} if workers else {}))
        for step, batch in enumerate(seeded_batches(loader, epoch)):
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            network.stash(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, parts, sse, rule_parts = objective(model, batch, mean_rows, lambdas)
            assert torch.isfinite(loss), 'non-finite loss'
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(groups['shared'], RECIPE.clip_norm, error_if_nonfinite=True)
            rb_grad = torch.nn.utils.clip_grad_norm_(groups['route_break'], RECIPE.clip_norm, error_if_nonfinite=True)
            optimizer.step(); ema.update(model); updates += 1
            stage_sse += sse.cpu().numpy(); row_count += int(batch['def_frame_mask'].sum())
            grad_sum += float(grad); clip_sum += min(float(grad), 1.); rb_grad_sum += float(rb_grad)
            for name, value in dict(total=loss, **parts, **rule_parts).items():
                sums[name] = sums.get(name, 0.) + float(value)
        assert step + 1 == steps
        scheduler.step()
        outputs = validate(ema.module, valid_ids, data, stats, side, args.device, mask)
        torch.save({k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}, out_dir / f'ep{epoch}_ema.pt')
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, out_dir / f'ep{epoch}_live.pt')
        row = dict(epoch=epoch, optimizer_updates_cumulative=updates,
                   **{f'stage{i + 1}_train_rmse': float(np.sqrt(stage_sse[i] / (2 * row_count))) for i in range(3)},
                   grad_norm_unclipped_mean=grad_sum / steps, grad_norm_clipped_mean=clip_sum / steps,
                   route_break_grad_norm_unclipped_mean=rb_grad_sum / steps,
                   **{k + '_mean': v / steps for k, v in sums.items()},
                   **{o + '_' + k: v for o, (om, _) in outputs.items() for k, v in om.items()})
        history.append(row); pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)
        print('EPOCH ' + json.dumps(row), flush=True)
    best = {o: min(history, key=lambda r: r[o + '_masked_rmse'])['epoch'] for o in OUTPUTS}
    run = dict(fold=args.fold, smoke=args.smoke, device=args.device, epochs=epochs, schedule_epochs=RECIPE.schedule_epochs,
               steps_per_epoch=steps, batch_size=batch_size, training_plays=len(train_ids), validation_plays=len(valid_ids),
               mean_native_rows=mean_rows, parameters=sum(p.numel() for p in model.parameters()),
               best_epoch_by_masked_rmse=best, python=platform.python_version(), torch=torch.__version__,
               numpy=np.__version__, pandas=pd.__version__)
    (out_dir / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    print('DONE ' + json.dumps(run), flush=True)


if __name__ == '__main__':
    main()
