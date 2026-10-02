"""Read the prepared inputs of one fold : play records, side inputs, play ids and statistics.

Play records: one store for every play (prepared/records/), each shard's SHA-256 checked against its manifest; a fold keeps its listed
plays, and its validation plays never carry the training-only rule teacher.
"""
import gzip
import hashlib
import io
import json

import pandas as pd
import torch

from src import paths

TEACHER = 'rule_teacher'


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_play_ids(fold):
    """game_play_id, game_id, n_def, split (train / validation) of the fold."""
    return pd.read_csv(paths.fold_dir(fold) / 'play_ids.csv', dtype=str)


def load_plays(fold):
    """-> ({game_play_id: record} for the fold's listed plays, play ids)."""
    ids = read_play_ids(fold)
    split = dict(zip(ids.game_play_id, ids.split))
    store = paths.PREPARED / 'records'
    data = {}
    for entry in json.loads((store / 'manifest.json').read_text())['shards']:
        raw = (store / entry['filename']).read_bytes()
        assert sha256(raw) == entry['sha256'], ('record shard hash', entry['filename'])
        for gp, rec in torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False).items():
            if gp not in split:
                continue
            if split[gp] == 'validation':
                rec.pop(TEACHER, None)            # the teacher is a training target only
            data[gp] = rec
    assert len(data) == len(ids), ('listed plays missing from the record store', len(ids) - len(data))
    return data, ids


def load_side(fold, name):
    """One side input of the fold ('retrieval', 'orientation', 'position_blocker'): {game_play_id: tensors}."""
    folder = paths.fold_dir(fold)
    manifest = json.loads((folder / f'{name}_manifest.json').read_text())
    pt, gz = folder / f'{name}.pt', folder / f'{name}.pt.gz'
    raw = pt.read_bytes() if pt.exists() else gzip.decompress(gz.read_bytes())
    assert sha256(raw) == manifest['side_sha256'], ('side input hash', name)
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)


def load_json(path):
    return json.loads(path.read_text())
