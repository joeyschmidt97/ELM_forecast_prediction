"""Four-model reservoir ablation: ELM-rate forecasting with geometry seed context.

Runs the *same* echo-state-network forecast of the ELM-rate stream under four
static-context conditions, to isolate the value of geometry as a seed:

  1. **time-series only**       — context = none
  2. **+ scalar shape**         — LCFS shape scalars
  3. **+ flux-surface (PCA)**   — PCA of the full 10-surface r(theta) arrays
  4. **+ per-surface profile**  — PCA of the per-surface {kappa, delta, R0, a}

Each shot becomes one sequence: input = [elm_rate(t), <static geometry context>],
target = elm_rate(t + horizon). The geometry context is constant within a shot
(the seed the forecaster coasts on). Evaluation is **leave-one-shot-out**: the
readout trains on the other shots and forecasts the held-out one, so the context
must generalize across discharges. Reported metric is forecast NRMSE (lower =
better); model 1 is the floor that geometry must beat.

Honesty: 6 usable shots -> the cross-shot context effect is estimated from 6
points. This is the harness; the comparison sharpens as the geometry∩ELM set grows.

Run (TPED env, after ``pip install reservoirpy``)::

    conda activate TPED
    PYTHONPATH=C:/Users/joesc/git python src/reservoir_harness.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import reservoirpy
from reservoirpy.nodes import Reservoir
from reservoirpy.observables import nrmse
from sklearn.linear_model import Ridge as SKRidge

reservoirpy.set_seed(42)

SCALARS = ["kappa", "delta_lower", "delta_upper", "A", "q95"]


# --------------------------------------------------------------------------- #
# Per-shot streams + geometry context vectors for each model
# --------------------------------------------------------------------------- #
def load_streams(labels_dir, downsample=20, horizon=5, n_pca=3):
    geo = pd.read_csv(os.path.join(labels_dir, "geometry_elm_joined.csv"))
    summ = pd.read_csv(os.path.join(labels_dir, "elm_shot_summary.csv"))
    fs_dir = os.path.join(labels_dir, "fluxsurfaces")
    use = geo[geo.signal_ok == True].drop_duplicates("shot").set_index("shot")

    streams, scal, flux_raw, ps_raw = {}, {}, {}, {}
    for shot, g in use.iterrows():
        s = summ[summ.shot == shot]
        t_lo, t_hi = float(s.t_lo.iloc[0]), float(s.t_hi.iloc[0])
        d = np.load(os.path.join(labels_dir, "targets", f"{shot}.npz"))
        t, rate = d["time"], d["elm_rate"]
        m = (t >= t_lo) & (t <= t_hi)
        rate = rate[m][::downsample].astype(float)
        if len(rate) <= horizon + 20:
            continue
        streams[shot] = rate
        scal[shot] = g[SCALARS].to_numpy(float)
        fd = np.load(os.path.join(fs_dir, g["fs_file"]))
        flux_raw[shot] = fd["r_theta"].ravel()
        ps_raw[shot] = np.concatenate([fd[f"ps_{k}"] for k in
                                       ["kappa", "delta_upper", "delta_lower", "R0", "a"]])

    shots = list(streams)
    # standardized context blocks (fit across shots; slight LOSO leak, acknowledged)
    def reduce(raw, k):
        X = np.vstack([raw[sh] for sh in shots])
        mu = np.nanmean(X, 0); X = np.where(np.isnan(X), mu, X)[:, ~np.isnan(mu)]
        Z = PCA(n_components=min(k, X.shape[0] - 1)).fit_transform(
            StandardScaler().fit_transform(X))
        return {sh: Z[i] for i, sh in enumerate(shots)}

    ctx = {
        "1 time-only": {sh: np.zeros(0) for sh in shots},
        "2 +scalar": dict(zip(shots, StandardScaler().fit_transform(
            np.vstack([scal[sh] for sh in shots])))),
        "3 +flux-PCA": reduce(flux_raw, n_pca),
        "4 +persurf-PCA": reduce(ps_raw, n_pca),
    }
    return streams, ctx, shots, horizon


def make_io(rate, horizon):
    """Reservoir input = standardized rate(t); target = rate(t+horizon).

    Geometry is *not* an input here — it enters at the readout (below), so the
    reservoir dynamics are identical across all four models and the comparison is
    fair (only the linear readout sees the static context).
    """
    r = (rate - rate.mean()) / (rate.std() + 1e-9)
    T = len(r) - horizon
    return r[:T, None], r[horizon:horizon + T, None]


def _states(U, units, sr, lr):
    """Reservoir states for input U. Fresh seeded reservoir => identical weights
    and a zero initial state for every sequence (reservoirpy 0.4 has no reset arg)."""
    return Reservoir(units, sr=sr, lr=lr, seed=42).run(U)


# --------------------------------------------------------------------------- #
# Leave-one-shot-out: shared reservoir states, geometry injected at the readout
# --------------------------------------------------------------------------- #
def loso_all_models(streams, ctx, shots, horizon, units=300, sr=0.9, lr=0.3,
                    alpha=100.0):
    """Return {model_name: (mean_nrmse, std)} under leave-one-shot-out CV.

    Reservoir states are geometry-free and identical across models; each model
    fits a ridge readout on [states | static_context].
    """
    # states depend only on the rate stream -> compute once for all shots/folds
    S = {sh: _states(make_io(streams[sh], horizon)[0], units, sr, lr) for sh in shots}
    Y = {sh: make_io(streams[sh], horizon)[1] for sh in shots}
    results = {name: [] for name in ctx}
    for held in shots:
        for name, cmap in ctx.items():
            def feat(sh):
                c = cmap[sh]
                return (np.hstack([S[sh], np.tile(c, (len(S[sh]), 1))])
                        if c.size else S[sh])
            Xtr = np.vstack([feat(sh) for sh in shots if sh != held])
            Ytr = np.vstack([Y[sh] for sh in shots if sh != held])
            ro = SKRidge(alpha=alpha).fit(Xtr, Ytr)
            yt = Y[held].ravel()
            pred = np.asarray(ro.predict(feat(held))).ravel()
            results[name].append(float(np.sqrt(np.mean((yt - pred) ** 2))
                                       / (yt.std() + 1e-9)))
    return {n: (float(np.mean(e)), float(np.std(e))) for n, e in results.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="4-model reservoir ablation for ELM-rate forecasting.")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--downsample", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--units", type=int, default=300)
    ap.add_argument("--n-pca", type=int, default=3)
    args = ap.parse_args(argv)

    streams, ctx, shots, horizon = load_streams(
        args.labels_dir, args.downsample, args.horizon, args.n_pca)
    print(f"shots={len(shots)}  downsample={args.downsample} "
          f"(dt~{args.downsample*0.2:.0f} ms)  horizon={horizon} steps "
          f"(~{horizon*args.downsample*0.2:.0f} ms)  units={args.units}\n")
    res = loso_all_models(streams, ctx, shots, horizon, args.units)
    print(f"{'model':18s}{'ctx_dim':>8}{'NRMSE(LOSO)':>14}{'std':>8}")
    base = res["1 time-only"][0]
    for name, cmap in ctx.items():
        mean, sd = res[name]
        cdim = len(next(iter(cmap.values())))
        tag = "" if name == "1 time-only" else f"  ({(mean-base)/base*100:+.1f}% vs time-only)"
        print(f"{name:18s}{cdim:>8}{mean:>14.3f}{sd:>8.3f}{tag}")
    print("\nLower NRMSE = better. Model 1 is the geometry-free floor; geometry is "
          "injected only at the readout so the reservoir is identical across models. "
          "N=6 shots -> noisy; harness + preliminary.")


if __name__ == "__main__":
    main()
