"""Per-play tensors and the training batch plan.

PlayDataset turns one prepared play record into model inputs and targets. Training items are mirrored across the field's long axis
with probability 0.5 and get entity dropout (p 0.1); evaluation items are mirrored only when `force_flip` is set.
The plan, the per-item seeds and the per-update dropout seeds make every epoch reproducible.
"""
import random

import numpy as np
import torch

from src.recipe import RECIPE

MAX_DEFENDERS = 8
MAX_ROUTE_RUNNERS = 5
MAX_RECEIVERS = 5                    # nearest receivers described in each defender's features
N_PER_RECEIVER_FEATS = 6
N_PLAY_STATE_FEATS = 7
N_AGGREGATE_FEATS = 4
FIELD_HEIGHT = 53.3
ROUTE_NORM_X, ROUTE_NORM_Y = 30.0, FIELD_HEIGHT / 2
N_COVERAGE_FAMILIES = 8              # quarters, cover3, cover1, cover2, cover6, cover0, man2, other
N_OPTIONS = 12                       # coded rule options per defender
N_TOKEN_FEATS, N_BOUNDARY_FEATS = 68, 28
N_PREVIEW_STEPS = 11

FEATURE_NAMES = []
for _i in range(MAX_RECEIVERS):
    FEATURE_NAMES += [f'recv_dx_{_i}', f'recv_dy_{_i}', f'recv_dist_{_i}', f'recv_vx_{_i}', f'recv_vy_{_i}', f'recv_closing_speed_{_i}']
FEATURE_NAMES += ['time_frac', 'snap_x', 'snap_y', 'qb_dx', 'qb_dy', 'qb_pocket_depth', 'los_depth',
                  'n_receivers_nearby', 'nearest_recv_dist', 'mean_recv_depth', 'any_route_break']

# Mirroring (y -> 53.3 - y) permutes left/right rule options and zone landmarks and negates lateral channels.
OPTION_FLIP = np.array([0, 1, 2, 3, 4, 8, 7, 6, 5, 11, 10, 9])
ZONE_FLIP = np.array([3, 2, 1, 0, 6, 5, 4]); ZONE_START = 5
TOKEN_SIGN = np.array([1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1,
                       1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
                       1, 1], dtype=np.float32)
PREVIEW_SIGN = np.array([1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1], dtype=np.float32)
PREVIEW_SCALE = np.array([8, 8, 10, 10, 7, 7, 1, 1, 1, 8, 8], dtype=np.float32)   # dx dy vx vy ax ay f_frac lam power_frac c1_dx c1_dy
BOUNDARY_SIGN = np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
                         dtype=np.float32)
ANALOG_LATERAL = (1, 6)              # analog channels that change sign under the mirror
SIDE_KEYS = ('analogs', 'orientation_qb', 'orientation_receiver', 'orientation_defender', 'position', 'blocker', 'receiver_slots')


def receiver_slots(play):
    """[80, 5]: receiver block r of every defender's features holds a real route runner at frame t iff r < the number of route
    runners tracked at t (blocks are filled nearest first). The same for every defender and unchanged by the mirror."""
    tracked = np.asarray(play['offense_mask']) > 0
    n = int(play['n_frames'])
    count = tracked[:, :n].sum(0) if tracked.size else np.zeros(n, int)
    out = np.zeros((RECIPE.max_frames, MAX_RECEIVERS), dtype=np.float32)
    t = min(n, RECIPE.max_frames)
    out[:t] = (np.arange(MAX_RECEIVERS)[None, :] < count[:t, None]).astype(np.float32)
    return out


def zone_landmarks(arr, flip):
    """[d,13,7] zone-landmark features, as recorded or mirrored (only the zone block is permuted)."""
    if arr is None:
        return np.zeros((MAX_DEFENDERS, 13, 7), dtype=np.float32)
    if not flip:
        return arr.copy()
    out = arr.copy()
    out[:, ZONE_START:ZONE_START + len(ZONE_FLIP)] = arr[:, ZONE_START + ZONE_FLIP]
    return out


def rule_tokens(tok, opt_valid, flip):
    if tok is None:
        return (np.zeros((MAX_DEFENDERS, N_OPTIONS, N_TOKEN_FEATS), dtype=np.float32),
                np.zeros((MAX_DEFENDERS, N_OPTIONS), dtype=np.float32))
    if not flip:
        return tok.copy(), opt_valid.copy()
    out = tok[:, OPTION_FLIP, :] * TOKEN_SIGN[None, None, :]
    return out.astype(np.float32), opt_valid[:, OPTION_FLIP].copy()


def rule_previews(prev, flip):
    """[d,12,11,11] ten-step previews of each option (stored as float16) -> scaled float32, clipped to +-4."""
    if prev is None:
        return np.zeros((MAX_DEFENDERS, N_OPTIONS, N_PREVIEW_STEPS, len(PREVIEW_SCALE)), dtype=np.float32)
    out = prev.astype(np.float32)
    if flip:
        out = out[:, OPTION_FLIP] * PREVIEW_SIGN[None, None, None, :]
    return np.clip(out / PREVIEW_SCALE[None, None, None, :], -4.0, 4.0)


def rule_boundary(bh, vpre, flip):
    if bh is None:
        return (np.zeros((MAX_DEFENDERS, N_BOUNDARY_FEATS), dtype=np.float32),
                np.zeros((MAX_DEFENDERS, 2), dtype=np.float32))
    if not flip:
        return bh.copy(), vpre.copy()
    out_vp = vpre.copy()
    out_vp[:, 1] = -out_vp[:, 1]
    return (bh * BOUNDARY_SIGN[None, :]).astype(np.float32), out_vp


def rule_teacher(priv, flip):
    """Privileged per-option acceleration targets (+ unused second block), mask, initial velocity and its validity."""
    da0, ddp, mask, dv0, dv0_valid = priv
    if not flip:
        return da0.copy(), mask.copy(), dv0.copy(), dv0_valid.copy()
    da0f = da0[:, OPTION_FLIP].copy()
    da0f[..., 1] *= -1.0
    dv0f = dv0.copy()
    dv0f[:, 1] *= -1.0
    return da0f, mask[:, OPTION_FLIP].copy(), dv0f, dv0_valid.copy()


class SideStore:
    """Per-play side inputs: route-matched analog plays, body orientation, position class and blocker flag."""

    def __init__(self, analogs, orientation, position_blocker):
        self.analogs, self.orientation, self.position_blocker = analogs, orientation, position_blocker

    def tensors(self, gp, flipped):
        r = self.analogs[gp].clone()
        assert r.shape == (MAX_DEFENDERS, RECIPE.max_frames, 8), r.shape
        o = self.orientation[gp]
        qb, recv, dfn = o['qb'].clone(), o['recv'].clone(), o['dfn'].clone()
        if flipped:
            for c in ANALOG_LATERAL:
                r[..., c] = -r[..., c]
            qb[..., 1] = -qb[..., 1]; recv[..., 1] = -recv[..., 1]; dfn[..., 1] = -dfn[..., 1]
        pb = self.position_blocker[gp]
        return {'analogs': r, 'orientation_qb': qb, 'orientation_receiver': recv, 'orientation_defender': dfn,
                'position': pb['pos'].clone().long(), 'blocker': pb['blk'].clone().float()}


class PlayDataset(torch.utils.data.Dataset):
    """One item = one play: every defender's inputs and targets plus the route runners and the side inputs."""

    def __init__(self, play_ids, data, stats, side, is_train, max_seq_len=RECIPE.max_frames):
        self.play_ids_df = play_ids.reset_index(drop=True)
        self.data_dict = data
        self.side = side
        self.is_train = is_train
        self.force_flip = None
        self.max_seq_len = max_seq_len
        self.feat_mean = np.array([stats['feature_mean'].get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32)
        feat_std = np.array([stats['feature_std'].get(n, 1.0) for n in FEATURE_NAMES], dtype=np.float32)
        self.feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    def __len__(self):
        return len(self.play_ids_df)

    def __getitem__(self, idx):
        sample = self._play(idx)
        gp = self.play_ids_df.iloc[idx % len(self.play_ids_df)]['game_play_id']
        # Recover which view was drawn from the snap positions, then attach the matching side inputs.
        base_y = torch.as_tensor(self.data_dict[gp]['precomputed_base']['snap_xy'][:, 1], dtype=torch.float32)
        got_y = sample['snap_xy'][:, 1]; valid = sample['def_mask'] > 0
        if torch.allclose(got_y[valid], base_y[valid], atol=1e-4):
            flipped = False
        else:
            assert torch.allclose(got_y[valid], 53.3 - base_y[valid], atol=1e-3), gp
            flipped = True
        sample.update(self.side.tensors(gp, flipped))
        sample['receiver_slots'] = torch.from_numpy(receiver_slots(self.data_dict[gp]))
        return sample

    def _play(self, idx):
        idx = idx % len(self.play_ids_df)
        gp_id = self.play_ids_df.iloc[idx]['game_play_id']
        play = self.data_dict[gp_id]
        if self.force_flip is not None:
            use_flip = bool(self.force_flip)
        else:
            use_flip = self.is_train and random.random() < 0.5
        cached = play['precomputed_flip'] if use_flip else play['precomputed_base']
        feat = cached['features'].copy()
        target = cached['target'].copy()
        target_abs = cached.get('target_abs', target).copy()
        def_mask = cached['def_mask'].copy()
        frame_mask = cached['frame_mask'].copy()
        snap_xy = cached['snap_xy'].copy()
        def_frame_mask = cached['def_frame_mask'].copy()
        route_context = cached['route_context'].copy()
        route_recv_mask = cached['route_recv_mask'].copy()
        route_frame_mask = cached['route_frame_mask'].copy()

        feat = (feat - self.feat_mean) / self.feat_std
        feat = np.nan_to_num(feat, nan=0.0)
        feat = np.concatenate([feat, self._coverage_onehot(play, def_frame_mask)], axis=-1)
        feat = np.concatenate([feat, self._play_action(play, def_frame_mask)], axis=-1)
        dropped = self._entity_dropout(feat, def_mask, def_frame_mask, route_context, route_recv_mask, route_frame_mask,
                                       snap_xy, play['los_x'])
        target = np.nan_to_num(target, nan=0.0)
        target_abs = np.nan_to_num(target_abs, nan=0.0)
        landmarks = zone_landmarks(play.get('zone_landmarks'), use_flip)
        for s in dropped:
            if 0 <= int(s) < MAX_ROUTE_RUNNERS:
                landmarks[:, int(s), :] = 0.0
        out = {
            'features': torch.tensor(feat, dtype=torch.float32),
            'zone_landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'target_abs': torch.tensor(target_abs, dtype=torch.float32),
            'def_mask': torch.tensor(def_mask, dtype=torch.float32),
            'frame_mask': torch.tensor(frame_mask, dtype=torch.float32),
            'def_frame_mask': torch.tensor(def_frame_mask, dtype=torch.float32),
            'snap_xy': torch.tensor(snap_xy, dtype=torch.float32),
            'los_x': torch.tensor(float(play['los_x']), dtype=torch.float32),
            'route_context': torch.tensor(route_context, dtype=torch.float32),
            'route_recv_mask': torch.tensor(route_recv_mask, dtype=torch.float32),
            'route_frame_mask': torch.tensor(route_frame_mask, dtype=torch.float32),
        }
        out.update(self._rule_tensors(play, use_flip, dropped))
        return out

    def _coverage_onehot(self, play, def_frame_mask):
        oh = np.zeros((MAX_DEFENDERS, self.max_seq_len, N_COVERAGE_FAMILIES), dtype=np.float32)
        oh[:, :, int(play.get('coverage_family_idx', 0))] = 1.0
        return oh * def_frame_mask[:, :, None]

    def _play_action(self, play, def_frame_mask):
        pa = np.zeros((MAX_DEFENDERS, self.max_seq_len, 2), dtype=np.float32)
        pa[:, :, 0] = float(play.get('play_action', 0))
        pa[:, :, 1] = float(play.get('pa_known', 0))
        return pa * def_frame_mask[:, :, None]

    def _entity_dropout(self, feat, def_mask, def_frame_mask, route_context, route_recv_mask, route_frame_mask, snap_xy, los_x):
        """Training only, probability 0.1: drop 1-2 route runners (never below 3) and, with probability 0.5, one defender
        (never below 3). In place. -> the dropped route slots."""
        dropped = []
        if not (self.is_train and RECIPE.entity_dropout > 0 and random.random() < RECIPE.entity_dropout):
            return dropped
        recv_valid = np.where(route_recv_mask > 0)[0]
        if len(recv_valid) > 3:
            k = random.randint(1, min(2, len(recv_valid) - 3))
            dropped = [int(r) for r in random.sample(list(recv_valid), k)]
            # The per-defender receiver blocks are ranked by distance at every frame: re-rank them without the dropped runners.
            raw_routes = route_context.copy()
            n_block = MAX_RECEIVERS * N_PER_RECEIVER_FEATS
            for d in np.where(def_mask > 0)[0]:
                sx, sy = snap_xy[d]
                for t in range(self.max_seq_len):
                    valid_t = [int(r) for r in recv_valid if route_frame_mask[r, t] > 0]
                    if not valid_t:
                        continue
                    rx = np.array([raw_routes[r, t, 0] * ROUTE_NORM_X + float(los_x) for r in valid_t])
                    ry = np.array([raw_routes[r, t, 1] * ROUTE_NORM_Y + FIELD_HEIGHT / 2 for r in valid_t])
                    order = [valid_t[i] for i in np.argsort((rx - sx) ** 2 + (ry - sy) ** 2)]
                    old = feat[d, t, :n_block].reshape(MAX_RECEIVERS, N_PER_RECEIVER_FEATS).copy()
                    compact = np.zeros_like(old)
                    kept = [r for r in order if r not in dropped]
                    old_rank = {r: i for i, r in enumerate(order)}
                    for new_rank, r in enumerate(kept[:MAX_RECEIVERS]):
                        compact[new_rank] = old[old_rank[r]]
                    feat[d, t, :n_block] = compact.reshape(-1)
            # Aggregate receiver statistics would still count the dropped runners: neutralise them.
            agg0 = n_block + N_PLAY_STATE_FEATS
            feat[..., agg0:agg0 + N_AGGREGATE_FEATS] = 0.0
            for r in dropped:
                route_context[r] = 0.0
                route_recv_mask[r] = 0.0
                route_frame_mask[r] = 0.0
        def_valid = np.where(def_mask > 0)[0]
        if len(def_valid) > 3 and random.random() < 0.5:
            d = random.choice(list(def_valid))
            feat[d] = 0.0
            def_mask[d] = 0.0
            def_frame_mask[d] = 0.0
        return dropped

    def _rule_tensors(self, play, flip, dropped):
        """Coded-rule inputs (every item) and the privileged teacher targets (training items only)."""
        tok, opt_valid = rule_tokens(play.get('rule_tokens'), play.get('rule_option_valid'), flip)
        prev = rule_previews(play.get('rule_previews'), flip)
        bh, vpre = rule_boundary(play.get('rule_boundary'), play.get('rule_velocity_before'), flip)
        bh_valid = play.get('rule_boundary_valid')
        bh_valid = bh_valid.copy() if bh_valid is not None else np.zeros(MAX_DEFENDERS, dtype=np.float32)
        for s in dropped:
            if 0 <= int(s) < MAX_ROUTE_RUNNERS:
                tok[:, int(s)] = 0.0
                prev[:, int(s)] = 0.0
                opt_valid[:, int(s)] = 0.0
        out = {
            'rule_tokens': torch.tensor(tok, dtype=torch.float32),
            'rule_previews': torch.tensor(prev, dtype=torch.float32),
            'rule_option_valid': torch.tensor(opt_valid, dtype=torch.float32),
            'rule_boundary': torch.tensor(bh, dtype=torch.float32),
            'rule_velocity_before': torch.tensor(vpre, dtype=torch.float32),
            'rule_boundary_valid': torch.tensor(bh_valid, dtype=torch.float32),
        }
        if self.is_train:
            priv = play.get('rule_teacher')
            if priv is None:
                da0 = np.zeros((MAX_DEFENDERS, N_OPTIONS, 2), np.float32)
                pmask = np.zeros((MAX_DEFENDERS, N_OPTIONS), np.float32)
                dv0 = np.zeros((MAX_DEFENDERS, 2), np.float32)
                dv0_valid = np.zeros(MAX_DEFENDERS, np.float32)
            else:
                da0, pmask, dv0, dv0_valid = rule_teacher(priv, flip)
            for s in dropped:
                if 0 <= int(s) < MAX_ROUTE_RUNNERS:
                    pmask[:, int(s)] = 0.0
            out.update({
                'teacher_acceleration': torch.tensor(da0, dtype=torch.float32),
                'teacher_mask': torch.tensor(pmask, dtype=torch.float32),
                'teacher_velocity': torch.tensor(dv0, dtype=torch.float32),
                'teacher_velocity_valid': torch.tensor(dv0_valid, dtype=torch.float32),
            })
        return out


class TrainingItems(torch.utils.data.Dataset):
    """Items are (index, seed, aux): the play is built under its own python-random seed; on every 4th ("aux") batch the teacher
    masks are zeroed, so the teacher loss trains on three batches of four."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        index, seed, aux = item
        state = random.getstate(); random.seed(seed)
        try:
            sample = self.dataset[index]
        finally:
            random.setstate(state)
        if aux:
            for key in ('teacher_mask', 'teacher_velocity_valid'):
                sample[key].zero_()
        return sample


def plan(n_plays, epoch, batch, steps):
    """The epoch's batches: a seeded permutation of the training plays cut into `steps` batches, each item with its own seed."""
    order = np.random.default_rng(800000 + epoch).permutation(n_plays)
    result = []
    for step in range(steps):
        aux = step % 4 == 3
        ix = order[step * batch:(step + 1) * batch]
        assert len(ix) == batch
        result.append([(int(j), 820000 + epoch * steps * batch + step * batch + k, int(aux)) for k, j in enumerate(ix)])
    return result


def seeded_batches(loader, epoch):
    """Reseed torch before each update so dropout draws depend only on (epoch, step)."""
    for i, b in enumerate(loader):
        torch.manual_seed(830000 + epoch * 100000 + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(830000 + epoch * 100000 + i)
        yield b
