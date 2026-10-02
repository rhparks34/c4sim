"""The training recipe shared by all four folds, in one place."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    epochs: int = 28                      # epochs trained ...
    schedule_epochs: int = 36             # ... on a cosine learning-rate schedule laid out for 36 epochs
    batch_size: int = 16                  # plays per update
    lr: float = 3e-4                      # AdamW, one optimizer over every parameter
    weight_decay: float = 1e-4
    clip_norm: float = 1.0                # gradient norm clip, applied separately to the route-break readout and the rest
    ema_decay: float = .9999              # reported checkpoints are the exponential moving average of the weights
    seed: int = 42
    loader_workers: int = 2
    max_frames: int = 80                  # frames after the snap
    entity_dropout: float = 0.1           # training-time route-runner / defender dropout probability per play
    late_beta: float = math.log(2.0) / 2.0             # late-weighted loss: the last frame of a track weighs 2x the first
    position_weight: float = 0.22423153368992357       # weight of the late-weighted squared error next to the likelihood
    round_weight: float = .25             # loss on each of the first two refinement rounds
    openness_weight: float = 2.0          # openness penalty on the route-break output
    # The two coded-rule losses: weights set once, at the random initialization, so each loss's gradient norm matched the trunk's
    # (64 training plays); fixed ever since.
    rule_path_weight: float = 2.480482651657872     # reaction paths toward the coded-rule options
    teacher_weight: float = 2.32177344533824        # the rule teacher (training plays only)
    smoothing_lambda: float = 128.        # Whittaker smoothing of each predicted track ...
    smoothing_ramp: int = 12              # ... blended in over the first 12 frames


RECIPE = Recipe()
