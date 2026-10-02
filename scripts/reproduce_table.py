"""Rebuild the four-fold table of the abstract from prediction files.

usage: python scripts/reproduce_table.py [--predictions DIR] [--out DIR] [--only accuracy,openness,...]
  --predictions  folder with fold{K}/{standard,route_break,midpoint}.parquet (default OUT_ROOT/predictions)
  --out          default OUT_ROOT/table; writes table.md, table.json, pooled_intervals.txt, numbers.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import accuracy, breaks, classifier, completion, openness, reads, table  # noqa: E402
from src.eval.common import PREDICTIONS  # noqa: E402
from src.paths import OUT_ROOT  # noqa: E402

FAMILIES = ['accuracy', 'openness', 'completion', 'breaks', 'reads', 'classifier']
COL = {0: 'fold 0', 1: 'fold 1', 2: 'fold 2', 3: 'fold 3', 'pooled': 'pooled'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, default=PREDICTIONS); ap.add_argument('--out', type=Path, default=OUT_ROOT / 'table')
    ap.add_argument('--only', default=','.join(FAMILIES))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); only = a.only.split(',')
    T, I, extra = {}, {}, []
    if 'accuracy' in only:
        t0 = time.time(); r = accuracy.score(a.predictions)
        for k, v in r.items():
            for label, m in (('Mean error (yd)', 'mean_rmse'), ('Median play (yd)', 'median_play_rmse')):
                T.setdefault(label, {})[COL[k]] = {'real': None, **{s: v['defined'][s][m] for s in v['defined']}}
        I['accuracy'] = r['pooled']
        print(f'accuracy {time.time() - t0:.0f} s', flush=True)
    if 'openness' in only:
        t0 = time.time(); r = openness.score(a.predictions)
        for k, v in r.items():
            for label, part in (('Openness at throw (yd)', 'runners'), ('Tightest receiver (yd)', 'tightest')):
                T.setdefault(label, {})[COL[k]] = {s: v[part][s][0] for s in openness.SRC}
        I['openness'] = {s: r['pooled']['runners'][f'{s}_minus_real'] for s in openness.SRC[1:]}
        I['tightest'] = {s: r['pooled']['tightest'][f'{s}_minus_real'] for s in openness.SRC[1:]}
        print(f'openness {time.time() - t0:.0f} s', flush=True)
    if 'completion' in only:
        t0 = time.time(); r = completion.score(a.predictions)
        for k, v in r.items():
            T.setdefault('Completion probability (%)', {})[COL[k]] = {s: v[s]['mean'][0] for s in table.SRC}
            I[f'completion_{k}'] = {s: v[s]['bias_pp'] for s in table.SRC[1:]}
        print(f'completion {time.time() - t0:.0f} s', flush=True)
    if 'breaks' in only:
        t0 = time.time(); r = breaks.score(a.predictions)
        for k, v in r.items():
            T.setdefault('1 s after a cut (yd)', {})[COL[k]] = {'real': v['real'], **v['sources']}
            T.setdefault('  breaks / plays / games', {})[COL[k]] = (v['n'], v['plays'], v['games'])
        I['cut'] = r['pooled']['minus_real']
        print(f'breaks {time.time() - t0:.0f} s', flush=True)
    if 'reads' in only:
        t0 = time.time(); r = reads.score(a.predictions)
        rows = {'read_2': ('#2 read: safety 5+ yd, vert / flat (%)', '  #2 read plays vert/flat'),
                'read_solo': ('Solo: backside safety 2+ yd, vert / short (%)', '  Solo plays vert/short')}
        for k, v in r.items():
            for rid, (label, count) in rows.items():
                x = v[rid]; ca, cb = x['classes']
                T.setdefault(label, {})[COL[k]] = {s: (x[s]['a'][0], x[s]['b'][0]) for s in table.SRC}
                T.setdefault(count, {})[COL[k]] = (x['n_plays'][ca], x['n_plays'][cb], x['n_games'])
        for rid in reads.TABLE_READS:
            I[rid] = {s: [100 * e for e in r['pooled'][rid][s]['diff']] for s in table.SRC}
        print(f'reads {time.time() - t0:.0f} s', flush=True)
    if 'classifier' in only:
        t0 = time.time(); r = classifier.score(a.predictions)
        cls = {'real': r['real'], 'standard': r['standard'], 'midpoint': r['midpoint'], 'route-break': r['route_break'],
               'frozen at snap': r['static'], 'plays': r['plays'], 'games': r['games']}
        (a.out / 'classifier.json').write_text(json.dumps(cls, indent=1))
        ids = [('C001', 'real'), ('C002', 'standard'), ('C003', 'midpoint'), ('C004', 'route-break'), ('C008', 'frozen at snap'), ('C009', 'real')]
        extra += [(i, f'{100 * cls[k]:.1f}%') for i, k in ids]
        print(f'classifier (fold 0, {r["plays"]} plays, {r["games"]} games): ' + ', '.join(f'{k} {100 * cls[k]:.1f}%' for k in list(cls)[:5]))
        print(f'classifier {time.time() - t0:.0f} s', flush=True)
    md, nums = table.table(T)
    txt, inums = table.intervals(I)
    (a.out / 'table.md').write_text(md); (a.out / 'table.json').write_text(json.dumps(T, indent=1, default=list))
    (a.out / 'pooled_intervals.txt').write_text(txt)
    with open(a.out / 'numbers.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['id', 'value']); w.writerows(nums + inums + extra)
    print(md + '\n' + txt)


if __name__ == '__main__':
    main()
