"""Track smoothing used by the evaluation engines: a Whittaker smoother anchored at the first frame, blended in over a ramp.

lam 512, ramp 12: the first frames keep the raw position (weight frame / ramp on the smoothed one) so a track starts exactly where
the player stood.
"""
import numpy as np
from scipy.linalg import solveh_banded

LAM, RAMP = 512, 12


def whittaker_anchored(y, lam):
    """Second-difference Whittaker smoother; the first point has weight 1e6 so it stays put."""
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


def smooth_track(px, py, lam=LAM, ramp=RAMP):
    sx, sy = whittaker_anchored(px, lam), whittaker_anchored(py, lam)
    if ramp > 0:
        n = len(px)
        w = np.minimum(np.arange(n) / float(ramp), 1.0)
        sx = (1 - w) * px + w * sx
        sy = (1 - w) * py + w * sy
    return sx, sy
