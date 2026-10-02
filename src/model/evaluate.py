"""Validation predictions: every validation play is predicted as recorded and mirrored (y-flip), the two are averaged
(reflection averaging), each defender track is smoothed, and one row per valid defender-frame is returned."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.model import network
from src.model.smoothing import smooth_track

COLUMNS = ['game_play_id', 'nfl_id', 'frame_id', 'actual_x', 'actual_y', 'tta_x', 'tta_y', 'candidate_x', 'candidate_y',
           'track_rank', 'frame_index']


def predict_batches(model, loader, device):
    """-> predictions, targets [N,d,t,2] (displacements) and the valid defender-frame mask [N,d,t]."""
    model.eval()
    model.to(device)
    preds, targets, valid = [], [], []
    with torch.inference_mode():
        for batch in loader:
            network.stash(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            pred, _ = model(batch['features'], batch['route_context'], route_recv_mask=batch['route_recv_mask'],
                            route_frame_mask=batch['route_frame_mask'], def_mask=batch['def_mask'], frame_mask=batch['frame_mask'],
                            def_frame_mask=batch['def_frame_mask'], snap_xy=batch['snap_xy'], los_x=batch['los_x'],
                            zone_landmarks=batch['zone_landmarks'], rule_tokens=batch['rule_tokens'], rule_previews=batch['rule_previews'],
                            rule_option_valid=batch['rule_option_valid'], rule_boundary=batch['rule_boundary'], rule_boundary_valid=batch['rule_boundary_valid'])
            preds.append(pred.cpu().numpy())
            targets.append(batch.get('target_abs', batch['target']).cpu().numpy())
            valid.append(batch['def_frame_mask'].cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), np.concatenate(valid, axis=0) > 0


def rmse(d, stem):
    return float(np.sqrt(np.mean((d[[stem + '_x', stem + '_y']].to_numpy() - d[['actual_x', 'actual_y']].to_numpy()) ** 2)))


def validation(model, ids, data, dataset, device):
    """dataset: the evaluation dataset of `ids` (its force_flip switches the reflection). -> (metrics, prediction table)."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    dataset.force_flip = False
    pred, target, valid = predict_batches(model, loader, device)
    dataset.force_flip = True
    flip, flip_target, flip_valid = predict_batches(model, loader, device)
    assert np.array_equal(valid, flip_valid)
    flip[:, :, :, 1] *= -1
    np.testing.assert_allclose(target[valid], (flip_target * np.array([1, -1]))[valid], atol=1e-5)
    avg = (pred + flip) * .5
    rows = []
    for i, gp in enumerate(ids.game_play_id):
        p = data[gp]
        for j, nfl in enumerate(p['defender_ids']):
            ix = np.flatnonzero(valid[i, j])
            xy = avg[i, j, ix] + p['defense_snap_xy'][j]
            actual = target[i, j, ix] + p['defense_snap_xy'][j]
            sx, sy = smooth_track(xy[:, 0], xy[:, 1])
            for k, t in enumerate(ix):
                rows.append((gp, str(nfl), int(p['frame_ids'][t]), actual[k, 0], actual[k, 1],
                             xy[k, 0], xy[k, 1], sx[k], sy[k], k + 1, int(t) + 1))
    d = pd.DataFrame(rows, columns=COLUMNS)
    d['game_id'] = d.game_play_id.str.split('_').str[0]
    raw = float(np.sqrt(np.mean((pred[valid] - target[valid]) ** 2)))
    return {'raw_rmse': raw, 'tta_rmse': rmse(d, 'tta'), 'smoothed_rmse': rmse(d, 'candidate')}, d
