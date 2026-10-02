"""Every read of a raw input file, one function per file family. Nothing else in the code opens a raw file.

Expected layout under $DATA_ROOT/raw/ (README, "Running it"; each folder is one Kaggle download, unpacked as published):
  nfl-big-data-bowl-2021/             2018 season   games.csv, plays.csv, players.csv, week1-17.csv
  nfl-big-data-bowl-2023/             2021 season   games.csv, plays.csv, players.csv, pffScoutingData.csv, week1-8.csv
  nfl-big-data-bowl-2025/             2022 season   games.csv, plays.csv, players.csv, player_play.csv, tracking_week_1-9.csv
  nfl-big-data-bowl-2026-analytics/   2023 season   supplementary_data.csv, train/input_2023_w01-18.csv
  nflverse/pbp_participation_2018.csv 2018 coverage labels (not on Kaggle)
"""
import pandas as pd

from src.paths import RAW

DIRS = {2018: RAW / 'nfl-big-data-bowl-2021', 2021: RAW / 'nfl-big-data-bowl-2023', 2022: RAW / 'nfl-big-data-bowl-2025',
        2023: RAW / 'nfl-big-data-bowl-2026-analytics'}
SOURCES = {
    2018: 'https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data',
    2021: 'https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/data',
    # The competition page no longer serves its files; the same files are in the Kaggle dataset
    # alexandermeau/nfl-big-data-bowl-archived-data-2025 (folder big-data-bowl-data/).
    2022: 'https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data',
    2023: 'https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data',
}
NFLVERSE_2018 = RAW / 'nflverse' / 'pbp_participation_2018.csv'


def _csv(path, columns=None, text=False):
    """text=True: every field as its published text (missing = NaN); else the fast reader with typed columns (tracking)."""
    if columns is not None:
        header = pd.read_csv(path, nrows=0).columns
        columns = [c for c in columns if c in header]
    if text:
        return pd.read_csv(path, usecols=columns, dtype=str, low_memory=False)
    return pd.read_csv(path, usecols=columns, engine='pyarrow')


def tracking_files(season):
    """One season's tracking files in week order."""
    d = DIRS[season]
    if season == 2018:     # Kaggle nfl-big-data-bowl-2021: week{1..17}.csv
        files = [d / f'week{w}.csv' for w in range(1, 18)]
    elif season == 2021:   # Kaggle nfl-big-data-bowl-2023: week{1..8}.csv
        files = [d / f'week{w}.csv' for w in range(1, 9)]
    elif season == 2022:   # Kaggle nfl-big-data-bowl-2025: tracking_week_{1..9}.csv
        files = [d / f'tracking_week_{w}.csv' for w in range(1, 10)]
    else:                  # Kaggle nfl-big-data-bowl-2026-analytics: train/input_2023_w{01..18}.csv (snap to throw)
        files = [d / 'train' / f'input_2023_w{w:02d}.csv' for w in range(1, 19)]
    missing = [f for f in files if not f.exists()]
    assert not missing, f'missing raw tracking files: {missing[:3]}'
    return files


def read_tracking(path, columns=None):
    """One week of player tracking (10 Hz) as published (the columns asked for that the file has)."""
    return _csv(path, columns)


def read_games(season):
    # Kaggle: games.csv of the season's competition (2023: supplementary_data.csv carries the game fields)
    return _csv(DIRS[season] / 'games.csv', text=True) if season != 2023 else read_supplementary_2023()


def read_plays(season):
    """Play-level fields as published (2023: supplementary_data.csv)."""
    # Kaggle: plays.csv (2018, 2021, 2022); supplementary_data.csv (2023)
    return _csv(DIRS[season] / 'plays.csv', text=True) if season != 2023 else read_supplementary_2023()


def read_players(season):
    # Kaggle: players.csv (2018, 2021, 2022); the 2023 input files carry each player's position
    return _csv(DIRS[season] / 'players.csv', text=True)


def read_pff_scouting_2021(columns=None):
    # Kaggle nfl-big-data-bowl-2023: pffScoutingData.csv (pff_role per player and play)
    return _csv(DIRS[2021] / 'pffScoutingData.csv', columns, text=True)


def read_player_play_2022(columns=None):
    # Kaggle nfl-big-data-bowl-2025: player_play.csv (wasInitialPassRusher, wasRunningRoute, coverage assignments)
    return _csv(DIRS[2022] / 'player_play.csv', columns, text=True)


def read_supplementary_2023():
    # Kaggle nfl-big-data-bowl-2026-analytics: supplementary_data.csv (team_coverage_type, dropback_type, teams, play action)
    return _csv(DIRS[2023] / 'supplementary_data.csv', text=True)


def read_nflverse_coverage_2018():
    """2018 coverage calls charted by NGS, as published by nflverse (matched to the tracking on old_game_id and play_id)."""
    # nflverse: https://github.com/nflverse/nflverse-data/releases/tag/pbp_participation, pbp_participation_2018.csv
    return _csv(NFLVERSE_2018, ['old_game_id', 'play_id', 'defense_coverage_type', 'defense_man_zone_type'], text=True)
