"""Post-hoc smoothing of each predicted track: a Whittaker smoother (second differences, lambda 128) anchored at the snap, blended
in over the first 12 frames so the track leaves the snap position unchanged."""
import numpy as np
from scipy.linalg import solveh_banded

from src.recipe import RECIPE


def whittaker_anchored(y, lam):
    """Minimise sum w (z - y)^2 + lam * sum (second difference of z)^2 with the first point pinned (weight 1e6)."""
    n = len(y)
    if n < 4 or lam <= 0:
        return y.copy()
    weights = np.ones(n)
    weights[0] = 1e6
    main = np.zeros(n)
    main[0] = main[-1] = 1.0
    main[1] = main[-2] = 5.0
    main[2:-2] = 6.0
    off1 = np.zeros(n - 1)
    off1[0] = off1[-1] = -2.0
    off1[1:-1] = -4.0
    off2 = np.ones(n - 2)
    ab = np.zeros((3, n))
    ab[0, 2:] = lam * off2
    ab[1, 1:] = lam * off1
    ab[2, :] = weights + lam * main
    return solveh_banded(ab, weights * y, lower=False)


def smooth_track(px, py, lam=RECIPE.smoothing_lambda, ramp=RECIPE.smoothing_ramp):
    """Smoothed x and y of one track; frame k uses weight min(k / ramp, 1) on the smoothed value."""
    sx, sy = whittaker_anchored(px, lam), whittaker_anchored(py, lam)
    if ramp > 0:
        n = len(px)
        w = np.minimum(np.arange(n) / float(ramp), 1.0)
        sx = (1 - w) * px + w * sx
        sy = (1 - w) * py + w * sy
    return sx, sy
