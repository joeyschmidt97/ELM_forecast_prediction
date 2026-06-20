"""ELM predictor (reservoir) with geometry as static reservoir input — 4-model ablation.

Matches the construction in ``examples/reservoir_comp.ipynb``: an echo-state
network **forecasts the normalized lower-divertor D-alpha signal ``horizon``
steps ahead** (``to_forecasting`` style). The forecast trace *is* the ELM
prediction — its spikes are ELMs. Quality is scored by NRMSE of the predicted
vs. true future D-alpha, and shown as prediction overlays.

The static geometry "seed" is injected as **extra constant input channels** to
the reservoir, so it biases the reservoir *dynamics* (operating point), not just
a readout offset. (A readout-only constant is absorbed by a zero-mean signal —
which is why injecting at the readout did nothing.) Input scaling for the
geometry channels is set below the signal channel so geometry biases gently.

Four models share identical reservoir hyper-parameters; only the input differs:

  1. **time-only**       input = [dalpha(t)]
  2. **+scalar**         input = [dalpha(t), LCFS scalars]
  3. **+flux-PCA**       input = [dalpha(t), PCA(full r(theta))]
  4. **+persurf-PCA**    input = [dalpha(t), PCA(per-surface kappa/delta/...)]

Evaluation is **leave-one-shot-out**: the readout trains on the other shots and
forecasts the held-out shot, so geometry must transfer across discharges (a
within-shot split would let geometry merely memorize shot identity). The metric
is NRMSE; model 1 is the geometry-free baseline the others must beat.

Run (TPED env)::

    conda activate TPED
    PYTHONPATH=C:/Users/joesc/git python src/reservoir_harness.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge as SKRidge
from sklearn.preprocessing import StandardScaler

import reservoirpy
from reservoirpy.nodes import Reservoir

reservoirpy.set_seed(42)

SCALARS = ["kappa", "delta_lower", "delta_upper", "A", "q95"]


# --------------------------------------------------------------------------- #
# Per-shot D-alpha streams + geometry context vectors
# --------------------------------------------------------------------------- #
def load_data(labels_dir, db_path, downsample=10, n_pca=3):
    import pickle
    geo = pd.read_csv(os.path.join(labels_dir, "geometry_elm_joined.csv"))
    summ = pd.read_csv(os.path.join(labels_dir, "elm_shot_summary.csv"))
    fs_dir = os.path.join(labels_dir, "fluxsurfaces")
    use = geo[geo.signal_ok == True].drop_duplicates("shot").set_index("shot")
    db = {int(d["shot"]): d for d in pickle.load(open(db_path, "rb"))}

    streams, scal, flux_raw, ps_raw = {}, {}, {}, {}
    for shot, g in use.iterrows():
        s = summ[summ.shot == shot]
        t_lo, t_hi = float(s.t_lo.iloc[0]), float(s.t_hi.iloc[0])
        rec = db[shot]
        t = np.asarray(rec["time lower dalpha"], float)
        y = np.abs(np.asarray(rec["data lower dalpha"], float))
        m = (t >= t_lo) & (t <= t_hi)
        y = y[m]
        # block-mean downsample (mild denoise), then normalize to [0,1]
        n = (len(y) // downsample) * downsample
        y = y[:n].reshape(-1, downsample).mean(1)
        y = (y - y.min()) / (y.max() - y.min() + 1e-9)
        streams[shot] = y
        scal[shot] = g[SCALARS].to_numpy(float)
        fd = np.load(os.path.join(fs_dir, g["fs_file"]))
        flux_raw[shot] = fd["r_theta"].ravel()
        ps_raw[shot] = np.concatenate([fd[f"ps_{k}"] for k in
                                       ["kappa", "delta_upper", "delta_lower", "R0", "a"]])

    shots = list(streams)

    def reduce(raw, k):
        X = np.vstack([raw[sh] for sh in shots])
        mu = np.nanmean(X, 0); X = np.where(np.isnan(X), mu, X)[:, ~np.isnan(mu)]
        Z = PCA(n_components=min(k, X.shape[0] - 1)).fit_transform(
            StandardScaler().fit_transform(X))
        return {sh: Z[i] for i, sh in enumerate(shots)}

    rng = np.random.default_rng(0)
    ctx = {
        "1 time-only": {sh: np.zeros(0) for sh in shots},
        "2 +scalar": dict(zip(shots, StandardScaler().fit_transform(
            np.vstack([scal[sh] for sh in shots])))),
        "3 +flux-PCA": reduce(flux_raw, n_pca),
        "4 +persurf-PCA": reduce(ps_raw, n_pca),
        # control: random constant per shot — isolates "is it geometry, or just
        # a per-shot identifier the model can exploit?"
        "5 +random(ctrl)": {sh: rng.standard_normal(n_pca) for sh in shots},
    }
    return streams, ctx, shots


def make_io(y, ctx, horizon):
    """Input [dalpha(t), context...] and target dalpha(t+horizon)."""
    T = len(y) - horizon
    U = y[:T, None]
    if ctx.size:
        U = np.hstack([U, np.tile(ctx, (T, 1))])
    return U, y[horizon:horizon + T, None]


# --------------------------------------------------------------------------- #
# Reservoir states with geometry as input (biases dynamics)
# --------------------------------------------------------------------------- #
def _reservoir(input_dim, units, sr, lr, geo_scale):
    iscale = np.ones(input_dim)
    iscale[1:] = geo_scale            # gentler drive for the static geometry channels
    return Reservoir(units, sr=sr, lr=lr, input_scaling=iscale, seed=42)


def pooled_timesplit(streams, ctx_map, shots, horizon, train_frac, units, sr, lr,
                     geo_scale, ridge, warmup=20):
    """Pooled time-split: one readout trained on the early part of ALL shots
    (each with its own geometry input), tested on the later part of each shot.

    This is the realistic seed scenario — the shot's geometry is known at forecast
    time — and avoids the leave-one-shot-out extrapolation pathology while still
    asking whether geometry adds information beyond the D-alpha signal itself.
    Returns mean/std NRMSE over shots, per-shot NRMSE, and held-out predictions.
    """
    Xtr, Ytr, test = [], [], {}
    for sh in shots:
        U, Y = make_io(streams[sh], ctx_map[sh], horizon)
        res = _reservoir(U.shape[1], units, sr, lr, geo_scale)
        S = res.run(U)
        ntr = int(train_frac * len(S))
        Xtr.append(S[warmup:ntr]); Ytr.append(Y[warmup:ntr])
        test[sh] = (S[ntr:], Y[ntr:])
    ro = SKRidge(alpha=ridge).fit(np.vstack(Xtr), np.vstack(Ytr))
    errs, preds = {}, {}
    for sh, (S, Y) in test.items():
        p = np.asarray(ro.predict(S)).ravel(); yt = Y.ravel()
        errs[sh] = float(np.sqrt(np.mean((yt - p) ** 2)) / (yt.std() + 1e-9))
        preds[sh] = (yt, p)
    vals = list(errs.values())
    return float(np.mean(vals)), float(np.std(vals)), errs, preds


def run_all(streams, ctx, shots, horizon=10, train_frac=0.5, units=400, sr=0.95,
            lr=0.3, geo_scale=0.3, ridge=10.0):
    out = {}
    for name, cmap in ctx.items():
        mean, sd, per, preds = pooled_timesplit(streams, cmap, shots, horizon,
                                                train_frac, units, sr, lr, geo_scale, ridge)
        out[name] = dict(mean=mean, std=sd, per_shot=per, preds=preds)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reservoir ELM predictor, geometry-as-input ablation.")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--db", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dalpha_db.pkl"))
    ap.add_argument("--downsample", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--units", type=int, default=400)
    ap.add_argument("--geo-scale", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=10.0)
    args = ap.parse_args(argv)

    streams, ctx, shots = load_data(args.labels_dir, args.db, args.downsample)
    dt_ms = args.downsample * 0.2
    print(f"shots={len(shots)}  dt~{dt_ms:.1f} ms  horizon={args.horizon} "
          f"(~{args.horizon*dt_ms:.0f} ms ahead)  train_frac={args.train_frac}  "
          f"units={args.units}\n")
    res = run_all(streams, ctx, shots, args.horizon, args.train_frac, args.units,
                  geo_scale=args.geo_scale, ridge=args.ridge)
    base = res["1 time-only"]["mean"]
    print(f"{'model':18s}{'ctx_dim':>8}{'NRMSE(LOSO)':>13}{'std':>7}{'d vs base':>11}")
    for name, r in res.items():
        cd = len(next(iter(ctx[name].values())))
        tag = "" if name == "1 time-only" else f"{(r['mean']-base)/base*100:+.1f}%"
        print(f"{name:18s}{cd:>8}{r['mean']:>13.3f}{r['std']:>7.3f}{tag:>11}")
    print("\nForecast of the D-alpha trace `horizon` ahead; lower NRMSE = better. "
          "Geometry enters as reservoir input channels. N=6 -> noisy.")
    return res


if __name__ == "__main__":
    main()
