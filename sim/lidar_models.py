"""Lidar range error models, for the localisation bench and for episodes (perturbation 'lidar_model').

gauss(s): independent Gaussian noise. ld19(...): a realistic LDROBOT LD19 model, see test_localization.py.
A model is f(ranges, hit geom ids, rng) -> ranges with errors; inf stays inf.
"""
import numpy as np


def gauss(s):
    return lambda r, g, rng: r + rng.normal(0, s, len(r)) * np.isfinite(r)


def ld19(seed_bias=0, bias=0.045, sig=0.010, q=0.015, mixed=0.5, outl=0.01):
    B = np.random.default_rng(seed_bias).uniform(-bias, bias, 4096)

    def f(r, g, rng):
        out = r.copy(); fin = np.isfinite(r)
        out[fin] = r[fin] + B[g[fin] % 4096] + rng.normal(0, sig, fin.sum())
        if q: out[fin] = np.round(out[fin] / q) * q
        if mixed:
            n = len(r); rr = np.where(fin, r, 99.0)
            for k in np.nonzero(fin)[0]:
                for j in ((k - 1) % n, (k + 1) % n):
                    if abs(rr[j] - rr[k]) > 0.1 and rr[j] < 50 and rng.random() < mixed / 2:
                        out[k] = rr[k] + rng.random() * (rr[j] - rr[k]); break
        if outl:
            o = fin & (rng.random(len(r)) < outl); out[o] = rng.uniform(0.15, 3.0, o.sum())
        return out
    return f


MODELS = {'gauss10': gauss(0.01), 'gauss30': gauss(0.03), 'ld19': ld19(0), 'ld19b': ld19(1), 'ld19c': ld19(2),
          'ld19x2': ld19(0, sig=0.02, outl=0.03),
          'ld19cal': ld19(0, bias=0.02)}      # after a per-unit range calibration: residual bias within +-20 mm
