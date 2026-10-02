"""Every file location the code uses, from three roots set by environment variables.

DATA_ROOT  Big Data Bowl downloads (raw/), the prepared model inputs (prepared/) and evaluation assets (eval/).
CKPT_ROOT  trained checkpoints (one per fold and output).
OUT_ROOT   everything the code writes: training runs, predictions, tables.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get('DATA_ROOT', REPO / 'data'))
CKPT_ROOT = Path(os.environ.get('CKPT_ROOT', REPO / 'checkpoints'))
OUT_ROOT = Path(os.environ.get('OUT_ROOT', REPO / 'outputs'))

RAW = DATA_ROOT / 'raw'            # Kaggle downloads, one folder per competition (README, "Running it")
PREPARED = DATA_ROOT / 'prepared'  # model inputs, written by src/prep/
EVAL = DATA_ROOT / 'eval'          # evaluation-only assets (coverage classifier, completion-probability engines)
SCORING_MASK = REPO / 'metadata' / 'scoring_mask.csv'   # the 54 route-breakdown plays and their last scored frame


def fold_dir(fold):
    """Prepared inputs that differ by fold: ids, statistics, side inputs and (folds 1-3) the fold overlay."""
    return PREPARED / f'fold{fold}'
