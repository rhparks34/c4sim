"""Evaluation baselines of one fold: python -m src.eval.baselines --fold K (after src.predict).

Rows: every defender-frame the fold's standard prediction file holds. For each, the real position and four baselines that learn
nothing from the scored games:
  static    the defender's snap position, held
  cv        snap position + snap velocity x elapsed time; the velocity is the tracking speed and direction at the snap frame
            (zero when the defender has no snap row)
  rules     the coded Cover-4 rule simulator's own path (evaluation-only file eval/plays/rule_sim_paths.csv, at most 40 frames,
            then held at its last position; NaN for defenders it never simulated)
  analogs   the weighted mean path of the route-matched analog defenders, from the fold's retrieval side input (snap position
            when no analog is valid)
The scorers use the rows and play sets of this file; the baselines themselves are not in the table.
-> DATA_ROOT/eval/baselines/fold{K}.parquet
"""
import argparse

import numpy as np
import pandas as pd

from src.eval.common import prediction_path
from src.model import inputs
from src.paths import EVAL
from src.prep.frames import oriented, read_frames
from src.prep.plays import read_plays

KEY = ['game_play_id', 'nfl_id', 'frame_id']


def frame_table(predictions):
    d = predictions[['game_play_id', 'game_id', 'nfl_id', 'frame_id', 'frame_index', 'actual_x', 'actual_y']].copy()
    d['nfl_id'] = d.nfl_id.astype(str)
    return d.sort_values(['game_play_id', 'nfl_id', 'frame_id']).reset_index(drop=True)


def add_static_cv(d):
    g = d.groupby(['game_play_id', 'nfl_id'])
    d['static_x'] = g.actual_x.transform('first'); d['static_y'] = g.actual_y.transform('first')
    plays = read_plays().set_index('game_play_id')
    f = oriented(read_frames(2018, ['game_play_id', 'nfl_id', 'frame_id', 'x', 'y', 's', 'dir', 'going_left'], set(d.game_play_id)))
    f = f[f.frame_id.eq(f.game_play_id.map(plays.snap_frame))].drop_duplicates(['game_play_id', 'nfl_id'])
    r = np.deg2rad(f['dir'].to_numpy()); s = np.nan_to_num(f.s.to_numpy())
    v = pd.DataFrame(dict(game_play_id=f.game_play_id.to_numpy(), nfl_id=f.nfl_id.astype(str).to_numpy(),
                          vx=np.nan_to_num(s * np.sin(r)), vy=np.nan_to_num(s * np.cos(r))))
    d = d.merge(v, on=['game_play_id', 'nfl_id'], how='left')
    d['cv_has_velocity'] = d.vx.notna().astype(np.int8)
    d[['vx', 'vy']] = d[['vx', 'vy']].fillna(0.)
    elapsed = (d.frame_index - 1) * 0.1
    d['cv_x'] = d.static_x + d.vx * elapsed; d['cv_y'] = d.static_y + d.vy * elapsed
    return d.drop(columns=['vx', 'vy'])


def add_rules(d):
    s = pd.read_csv(EVAL / 'plays' / 'rule_sim_paths.csv', usecols=KEY + ['pred_x', 'pred_y'])
    s['nfl_id'] = s.nfl_id.astype(np.int64).astype(str)
    s = s[s.game_play_id.isin(set(d.game_play_id))].rename(columns={'pred_x': 'rules_x', 'pred_y': 'rules_y'})
    d = d.merge(s, on=KEY, how='left')
    d['rules_defined'] = d.rules_x.notna().astype(np.int8)
    has = d.groupby(['game_play_id', 'nfl_id']).rules_defined.transform('max') > 0
    d[['rules_x', 'rules_y']] = d.groupby(['game_play_id', 'nfl_id'])[['rules_x', 'rules_y']].ffill()
    d.loc[~has, ['rules_x', 'rules_y']] = np.nan
    return d


def analog_rows(records, side, plays):
    """Each defender-frame of `plays`: the analog mean position (channels 0-1 x 10 yd from the snap spot) and the valid share."""
    rows = []
    for gp in plays:
        r = records[gp]; ch = side[gp].float().numpy(); snap = r['defense_snap_xy']; dfm = r['precomputed_base']['def_frame_mask']
        for j, nid in enumerate(r['defender_ids']):
            for t in np.flatnonzero(dfm[j]):
                ok = ch[j, t, 4] > 0
                rows.append((gp, str(nid), int(r['frame_ids'][t]), snap[j, 0] + (ch[j, t, 0] * 10. if ok else 0.),
                             snap[j, 1] + (ch[j, t, 1] * 10. if ok else 0.), float(ch[j, t, 4])))
    return pd.DataFrame(rows, columns=KEY + ['analogs_x', 'analogs_y', 'analogs_frac'])


def build(fold, predictions, records, side):
    d = frame_table(predictions)
    d['fold'] = fold
    d = add_rules(add_static_cv(d))
    return d.merge(analog_rows(records, side, sorted(set(d.game_play_id))), on=KEY, how='left')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=range(4))
    args = ap.parse_args(argv)
    records, _ = inputs.load_plays(args.fold)
    d = build(args.fold, pd.read_parquet(prediction_path(args.fold, 'standard')), records, inputs.load_side(args.fold, 'retrieval'))
    out = EVAL / 'baselines'; out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / f'fold{args.fold}.parquet', index=False)
    print(f'baselines fold {args.fold}: {len(d):,} rows, {d.game_play_id.nunique():,} plays', flush=True)


if __name__ == '__main__':
    main()
