"""Stage 2: the play table.

python -m src.prep.plays

One row per play of the population (metadata/plays.csv: every play the four training runs used, plus 62 Cover-4 passes that are
training-only):
  season             2018, 2021, 2022 or 2023 (2023 includes the week-18 games played in January 2024)
  coverage_family    quarters, cover3, cover1, cover2, cover6, cover0, man2 or other. 2018: the NGS charting published by nflverse;
                     2021-2022: PFF's pff_passCoverage; 2023: team_coverage_type
  play_action        as charted (2021-2023); 2018 from the tracking event or the detector (play_action.py). pa_known = 1 everywhere
  dropback           the charted dropback type, upper case (MISSING when none); dropback_family TRAD / DROLL (designed rollout) /
                     SCR (scramble) / OTHER
  snap_frame, release_frame  the play's first ball_snap / snap_direct and first pass_forward / pass_shovel event; the automatic
                     event only when the play has no manual one. The 2023 files run from the snap to the throw.
  fold               the fold of a scored 2018 game (metadata/folds.csv), else -1
  kept_out_of_validation     from metadata/plays.csv
-> $DATA_ROOT/prepared/plays.parquet
"""
import numpy as np
import pandas as pd

from src.paths import PREPARED, REPO
from src.prep import raw
from src.prep.frames import read_frames
from src.prep.play_action import flags_2018

COVERAGE_FAMILIES = ('quarters', 'cover3', 'cover1', 'cover2', 'cover6', 'cover0', 'man2', 'other')   # the model's one-hot order
FAMILY_NGS = {'COVER_4': 'quarters', 'COVER_3': 'cover3', 'COVER_1': 'cover1', 'COVER_2': 'cover2', 'COVER_6': 'cover6',
              'COVER_0': 'cover0', '2_MAN': 'man2', 'PREVENT': 'other'}
FAMILY_PFF = {'Quarters': 'quarters', 'Cover-3': 'cover3', 'Cover-3 Seam': 'cover3', 'Cover-3 Cloud Left': 'cover3',
              'Cover-3 Cloud Right': 'cover3', 'Cover-3 Double Cloud': 'cover3', 'Cover-1': 'cover1', 'Cover-1 Double': 'cover1',
              'Cover-2': 'cover2', 'Cover-6': 'cover6', 'Cover 6-Left': 'cover6', 'Cover-6 Right': 'cover6', 'Cover-0': 'cover0',
              '2-Man': 'man2'}                                       # anything else PFF charts (Red Zone, Bracket, ...) is 'other'
FAMILY_2023 = {'COVER_4_ZONE': 'quarters', 'COVER_3_ZONE': 'cover3', 'COVER_1_MAN': 'cover1', 'COVER_2_ZONE': 'cover2',
               'COVER_6_ZONE': 'cover6', 'COVER_0_MAN': 'cover0', 'COVER_2_MAN': 'man2', 'PREVENT': 'other'}
DROPBACK_FAMILY = {'TRADITIONAL': 'TRAD', 'DESIGNED_ROLLOUT_LEFT': 'DROLL', 'DESIGNED_ROLLOUT_RIGHT': 'DROLL', 'SCRAMBLE': 'SCR',
                   'SCRAMBLE_ROLLOUT_LEFT': 'SCR', 'SCRAMBLE_ROLLOUT_RIGHT': 'SCR'}
SNAP_MANUAL, SNAP_AUTO = ('ball_snap', 'snap_direct'), ('autoevent_ballsnap',)
RELEASE_MANUAL, RELEASE_AUTO = ('pass_forward', 'pass_shovel'), ('autoevent_passforward',)


def plays_path():
    return PREPARED / 'plays.parquet'


def read_plays():
    return pd.read_parquet(plays_path())


def population():
    return pd.read_csv(REPO / 'metadata' / 'plays.csv', dtype={'game_play_id': str})


def first_event(events, manual, automatic):
    """Per play, the first frame of a manual event, else of an automatic one."""
    first = lambda names: events[events.event.isin(names)].groupby('game_play_id').frame_id.min()
    return first(manual).combine_first(first(automatic))


def bounds(season, wanted):
    frames = read_frames(season, ['game_play_id', 'frame_id', 'event'], wanted)
    if season == 2023:
        g = frames.groupby('game_play_id').frame_id
        return pd.DataFrame({'snap_frame': g.min(), 'release_frame': g.max()})
    events = frames[frames.event.ne('')].drop_duplicates()
    b = pd.DataFrame({'snap_frame': first_event(events, SNAP_MANUAL, SNAP_AUTO),
                      'release_frame': first_event(events, RELEASE_MANUAL, RELEASE_AUTO)}).dropna()
    return b[b.release_frame >= b.snap_frame].astype(int)


def season_labels(season):
    """Coverage call, play action and dropback type of every play of a season, as charted."""
    p = raw.read_plays(season)
    if season == 2023:
        p['game_play_id'] = p.game_id + '_' + p.play_id
        return pd.DataFrame({'game_play_id': p.game_play_id, 'coverage_family': p.team_coverage_type.map(FAMILY_2023),
                             'play_action': p.play_action.str.upper().eq('TRUE').astype(int), 'dropback': p.dropback_type})
    p['game_play_id'] = p.gameId + '_' + p.playId
    if season == 2018:
        ngs = raw.read_nflverse_coverage_2018()
        ngs = ngs[ngs.defense_coverage_type.fillna('').str.strip().ne('')]
        call = dict(zip(ngs.old_game_id + '_' + ngs.play_id, ngs.defense_coverage_type))
        flags = flags_2018()
        return pd.DataFrame({'game_play_id': p.game_play_id, 'coverage_family': p.game_play_id.map(call).map(FAMILY_NGS),
                             'play_action': p.game_play_id.map(flags), 'dropback': p.typeDropback})
    family = p.pff_passCoverage.map(lambda c: FAMILY_PFF.get(c, 'other') if isinstance(c, str) else None)
    pa = p.pff_playAction.astype(float) if season == 2021 else p.playAction.str.upper().eq('TRUE').astype(float)
    return pd.DataFrame({'game_play_id': p.game_play_id, 'coverage_family': family, 'play_action': pa,
                         'dropback': p.dropBackType if season == 2021 else p.dropbackType})


def main():
    pop = population()
    folds = pd.read_csv(REPO / 'metadata' / 'folds.csv', dtype={'game_id': str})
    fold_of = dict(zip(folds.game_id, folds.fold.astype(int)))
    parts = []
    for season in (2018, 2021, 2022, 2023):
        wanted = set(pop.game_play_id[pop.season.eq(season)])
        lab = season_labels(season).drop_duplicates('game_play_id').set_index('game_play_id').reindex(sorted(wanted))
        t = lab.join(bounds(season, wanted))
        t.insert(0, 'season', season)
        parts.append(t.rename_axis('game_play_id').reset_index())
    t = pd.concat(parts, ignore_index=True)
    missing = t[t.coverage_family.isna() | t.snap_frame.isna()]
    assert missing.empty, ('plays without a coverage call or snap/release events', missing.game_play_id.head().tolist())
    t['game_id'] = t.game_play_id.str.split('_').str[0]
    t['coverage_family_idx'] = t.coverage_family.map({f: i for i, f in enumerate(COVERAGE_FAMILIES)}).astype(int)
    t['play_action'] = t.play_action.astype(int)
    t['pa_known'] = 1
    t['dropback'] = t.dropback.fillna('MISSING').str.upper()
    t['dropback_family'] = t.dropback.map(DROPBACK_FAMILY).fillna('OTHER')
    t[['snap_frame', 'release_frame']] = t[['snap_frame', 'release_frame']].astype(int)
    t['n_frames'] = t.release_frame - t.snap_frame + 1
    t['fold'] = t.game_id.map(fold_of).fillna(-1).astype(int)
    t['kept_out_of_validation'] = t.game_play_id.map(dict(zip(pop.game_play_id, pop.kept_out_of_validation))).astype(int)
    t.to_parquet(plays_path(), index=False)
    print(f'plays: {len(t):,}', t.groupby('season').size().to_dict(), 'families', t.coverage_family.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    main()
