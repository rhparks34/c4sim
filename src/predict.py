"""Predict every validation play of one fold: python -m src.predict --fold K [--output standard|route_break|midpoint] [--checkpoint PATH].

Writes OUT_ROOT/predictions/fold{K}/{output}.parquet, one row per valid defender-frame, in the trainer's validation order.
candidate_x/y is the reported prediction (reflection averaging, then smoothing); tta_x/y is before smoothing; actual_x/y is real.
"""
import os

for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')   # one BLAS thread per process (the verified setting)

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import paths  # noqa: E402
from src.model import evaluate, inputs, network  # noqa: E402
from src.model.batches import PlayDataset  # noqa: E402
from src.train import OUTPUTS, masked_rmse, side_inputs  # noqa: E402

COLUMNS = evaluate.COLUMNS + ['game_id']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=range(4))
    ap.add_argument('--output', choices=OUTPUTS, help='one output (default: all three)')
    ap.add_argument('--checkpoint', type=Path, help='checkpoint for --output (default: CKPT_ROOT/fold{K}/{output}.pt)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)
    assert args.checkpoint is None or args.output, '--checkpoint needs --output'
    torch.set_num_threads(args.threads)
    out_dir = paths.OUT_ROOT / 'predictions' / f'fold{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = inputs.load_json(paths.fold_dir(args.fold) / 'stats.json')
    side = side_inputs(args.fold)
    data, ids = inputs.load_plays(args.fold)
    valid_ids = ids[ids.split.eq('validation')].reset_index(drop=True)
    mask = pd.read_csv(paths.SCORING_MASK, dtype={'game_play_id': str})
    for name in ([args.output] if args.output else OUTPUTS):
        path = args.checkpoint or paths.CKPT_ROOT / f'fold{args.fold}' / f'{name}.pt'
        model = network.C4Sim()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False), strict=True)
        model.eval(); model.to(args.device)
        view = network.OutputView(model)
        view.reset('standard')
        if name != 'standard':                       # the other outputs replay the standard pass
            evaluate.validation(view, valid_ids, data, PlayDataset(valid_ids, data, stats, side, is_train=False), args.device)
            view.reset(name)
        metrics, d = evaluate.validation(view, valid_ids, data, PlayDataset(valid_ids, data, stats, side, is_train=False), args.device)
        d = d[COLUMNS]
        d.to_parquet(out_dir / f'{name}.parquet', index=False)
        print(json.dumps(dict(fold=args.fold, output=name, checkpoint=str(path), rows=len(d), smoothed_rmse=metrics['smoothed_rmse'],
                              masked_rmse=masked_rmse(d, mask))), flush=True)


if __name__ == '__main__':
    main()
