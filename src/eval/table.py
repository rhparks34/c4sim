"""Assemble the four-fold table, the pooled intervals and numbers.csv from the scorers' results.

Each cell lists the real defense first, then the standard output, the midpoint and the route-break output.
numbers.csv gives every printed number an id: T001... in table order (row, column, then position in the cell), I001... in the
order of the pooled-interval lines, C001... for the coverage classifier.
"""
import json

COLS = ['fold 0', 'fold 1', 'fold 2', 'fold 3', 'pooled']
SRC = ['real', 'standard', 'midpoint', 'route_break']
SHORT = {'standard': 'standard', 'midpoint': 'midpoint', 'route_break': 'route-break', 'real': 'real'}
ROWS = [  # label, kind, columns that hold a value
    ('Mean error (yd)', 'acc', COLS), ('Median play (yd)', 'acc', COLS),
    ('Openness at throw (yd)', 'f2', COLS), ('Tightest receiver (yd)', 'f2', COLS),
    ('Completion probability (%)', 'pct', COLS[:4]),
    ('1 s after a cut (yd)', 'f2', COLS), ('  breaks / plays / games', 'count', COLS),
    ('#2 read: safety 5+ yd, vert / flat (%)', 'read', COLS), ('  #2 read plays vert/flat', 'count', COLS),
    ('Solo: backside safety 2+ yd, vert / short (%)', 'read', COLS), ('  Solo plays vert/short', 'count', COLS),
]
FMT = {'acc': lambda x: f'{x:.2f}', 'f2': lambda x: f'{x:.2f}', 'pct': lambda x: f'{100 * x:.1f}',
       'read': lambda x: f'{round(100 * x[0])}/{round(100 * x[1])}'}


def cell_items(kind, v):
    """Printed pieces of one cell, in order."""
    if kind == 'count':
        return [str(x) for x in v]
    return [FMT[kind](v[s]) for s in SRC if not (kind == 'acc' and s == 'real')]


def cell_text(kind, v):
    if v is None:
        return '—'
    if kind == 'count':
        return ' / '.join(cell_items(kind, v))
    items = cell_items(kind, v)
    return ', '.join((['—'] if kind == 'acc' else []) + items)


def table(T):
    """T: {row label: {column: cell value}} -> (markdown, numbers [(id, value)]); a missing cell prints '—' and gets no number."""
    lines = ['| Measure (real, standard, midpoint, route-break) | ' + ' | '.join(COLS) + ' |', '|---|' + '---|' * len(COLS)]
    numbers, n = [], 0
    for label, kind, cols in ROWS:
        vals = T.get(label, {})
        lines.append(f'| {label} | ' + ' | '.join(cell_text(kind, vals.get(c)) for c in COLS) + ' |')
        for c in cols:
            width = 3 if kind in ('acc', 'count') else 4
            v = vals.get(c)
            items = cell_items(kind, v) if v is not None else [None] * width
            for x in items:
                n += 1
                if x is not None:
                    numbers.append((f'T{n:03d}', x))
    return '\n'.join(lines) + '\n', numbers


def interval(e, k=2, sign=True):
    f = (lambda x: f'{x:+.{k}f}') if sign else (lambda x: f'{x:.{k}f}')
    return f'{f(e[0])} [{f(e[1])}, {f(e[2])}]'


def intervals(I):
    """I: {line key: content}. -> (text, numbers). Lines and ids follow the published pooled_intervals.txt."""
    out, numbers = [], []
    trio = ['standard', 'midpoint', 'route_break']
    heads = [('openness', 'Openness at throw, source - real (yd)'), ('tightest', 'Tightest receiver, source - real (yd)'),
             ('cut', '1 s after a cut, source - real (yd)')]
    by_src = {s: [] for s in trio}
    for key, head in heads:
        if key in I:
            d = {SHORT[s]: interval(I[key][s]) for s in trio}
            out.append(f'{head}: {d}')
            for s in trio:
                by_src[s].append(d[SHORT[s]])
        else:
            for s in trio:
                by_src[s].append(None)
    if 'accuracy' in I:
        a = I['accuracy']
        d = {SHORT[s]: (round(a['defined'][s]['mean_rmse'], 4), round(a['defined'][s]['median_play_rmse'], 4)) for s in trio}
        out.append(f'accuracy pooled (mean error, median play): {d}')
        delta = {SHORT[s]: a['delta'][s] for s in ('route_break', 'midpoint')}
        out.append(f'accuracy, output - standard: {json.dumps(delta)}')
        for s in trio:
            by_src[s].append(str(d[SHORT[s]]))
    else:
        for s in trio:
            by_src[s].append(None)
    n = 0
    for s in trio:
        for v in by_src[s]:
            n += 1
            if v is not None:
                numbers.append((f'I{n:03d}', v))
    for s in ('route_break', 'midpoint'):
        for m in ('mean_rmse', 'median_play_rmse', 'median_euclid'):
            n += 1
            if 'accuracy' in I:
                e = I['accuracy']['delta'][s][m]
                numbers.append((f'I{n:03d}', f'{e["est"]:.6f} [{e["ci"][0]:.6f}, {e["ci"][1]:.6f}]'))
    for key, label in (('read_2', '#2 read, safety 5+ yd: vert - flat'), ('read_solo', 'Solo read, backside safety 2+ yd: vert - short')):
        d = {SHORT[s]: interval(I[key][s], 0) + ' pts' for s in SRC} if key in I else None
        if d:
            out.append(f'{label}: {d}')
        for s in SRC:
            n += 1
            if d:
                numbers.append((f'I{n:03d}', d[SHORT[s]]))
    for k in range(4):
        key = f'completion_{k}'
        d = {SHORT[s]: interval(I[key][s]) for s in trio} if key in I else None
        if d:
            out.append(f'completion fold {k}, source - real (pts): {d}')
        for s in trio:
            n += 1
            if d:
                numbers.append((f'I{n:03d}', d[SHORT[s]]))
    return '\n'.join(out) + '\n', numbers
