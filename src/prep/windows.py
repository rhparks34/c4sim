"""Stage 4: each play's window, the frames after the snap that are inputs and targets (the same in training and validation).

python -m src.prep.windows

The window runs from the snap to the throw, capped at 80 frames (8 s). On scramble plays (charted SCRAMBLE or SCRAMBLE_ROLLOUT_*) and
plays with no charted dropback type it stops before the quarterback breaks down: the departure is the first frame where the QB is
more than 3 yd to the side of his snap spot or past the line of scrimmage; the breakdown onset walks back from there while he keeps
moving away at 2 yd/s or more; the window keeps frames 1 .. onset - 1. Plays whose window is shorter than 6 frames are left out.
-> $DATA_ROOT/prepared/windows.parquet (game_play_id, n_frames, window, truncated_by, departure, onset)
"""
import numpy as np
import pandas as pd

from src.paths import PREPARED
from src.prep.plays import read_plays
from src.prep.tracks import read_tracks

CAP, LATERAL, AWAY_SPEED = 80, 3.0, 0.2     # frames; yd; yd per frame (2 yd/s)
MIN_WINDOW = 6


def windows_path():
    return PREPARED / 'windows.parquet'


def read_windows():
    return pd.read_parquet(windows_path())


def breakdown(qb_x, qb_y, los_x):
    """-> (departure frame, onset frame), 1-based, or (None, None) when the QB never leaves the pocket."""
    dy = qb_y - qb_y[0]
    lateral = np.abs(dy) > LATERAL
    past = qb_x > los_x
    hit = np.flatnonzero(lateral | past)
    if not len(hit):
        return None, None
    i = int(hit[0])
    sideways = bool(lateral[i])
    step = np.diff(qb_y) * np.sign(dy[i]) if sideways else np.diff(qb_x)   # step[k]: move from frame k to k + 1 (0-based)
    j = i
    while j > 0 and step[j - 1] >= AWAY_SPEED:
        j -= 1
    return i + 1, j + 1


def play_window(entry, dropback_family):
    n = len(entry['frame_ids'])
    departure = onset = None
    if dropback_family in ('SCR', 'OTHER'):
        qb = entry['qb_data'][0]
        departure, onset = breakdown(qb[:, 0], qb[:, 1], entry['los_x'])
    window = min(n, CAP, onset - 1 if onset is not None else n)
    if onset is not None and onset - 1 < min(n, CAP):
        cut = 'scramble' if dropback_family == 'SCR' else 'unlabeled_departure'
    else:
        cut = 'cap80' if n > CAP else 'none'
    return dict(n_frames=n, window=window, truncated_by=cut, departure=departure, onset=onset)


def main():
    plays = read_plays().set_index('game_play_id')
    rows = []
    for season in (2018, 2021, 2022, 2023):
        for gp, entry in read_tracks(season).items():
            rows.append(dict(game_play_id=gp, **play_window(entry, plays.at[gp, 'dropback_family'])))
    w = pd.DataFrame(rows)
    w['kept'] = w.window >= MIN_WINDOW
    w.to_parquet(windows_path(), index=False)
    print(f'windows: {len(w):,} plays, {int((~w.kept).sum())} shorter than {MIN_WINDOW} frames', w.truncated_by.value_counts().to_dict(),
          flush=True)


if __name__ == '__main__':
    main()
