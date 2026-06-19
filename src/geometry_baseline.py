"""Geometry -> ELM baseline models: scalar shape vs full flux-surface arrays.

Two regressors predict an ELM target (default: forward ELM frequency over each
seed's coast window) from a discharge's equilibrium geometry, to test whether the
**full flux-surface shape** carries predictive information that the scalar shape
parameters wash out:

  * **scalar model** — last-closed-flux-surface scalars (A, eps, kappa,
    delta_u/l, q95, ...).
  * **flux-surface model** — the resampled r(theta) of N nested flux surfaces
    (edge -> core), flattened and PCA-reduced.

Both are evaluated with **leave-one-shot-out** CV (the only honest scheme at this
N) against a predict-the-mean baseline. With ~6 usable seeds the numbers are
illustrative, not conclusive — this is the comparison *framework*; it sharpens as
the geometry∩ELM set grows.

Run (TPED env has scikit-learn)::

    conda activate TPED
    PYTHONPATH=C:/Users/joesc/git python src/geometry_baseline.py --target fwd_freq_hz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCALAR_FEATURES = ["A", "eps", "kappa", "delta_upper", "delta_lower", "delta",
                   "q95", "q0", "qmin", "betap0"]


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #
def load_scalar_X(df: pd.DataFrame):
    cols = [c for c in SCALAR_FEATURES if c in df.columns and df[c].notna().any()]
    X = df[cols].to_numpy(float)
    # impute any stray NaN with column means
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu, X)
    return X, cols


def load_flux_X(df: pd.DataFrame, fs_dir: str):
    """Flatten r(theta) for every seed; impute NaN surfaces by column mean."""
    mats = []
    for _, r in df.iterrows():
        d = np.load(os.path.join(fs_dir, r["fs_file"]))
        mats.append(d["r_theta"].ravel())
    X = np.vstack(mats)
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu, X)
    keep = ~np.isnan(mu)                      # drop all-NaN columns
    return X[:, keep], int(keep.sum())


def load_persurface_X(df: pd.DataFrame, fs_dir: str):
    """Per-surface shape profile (model 4): {kappa, delta_u/l, R0, a} x N surfaces."""
    keys = ["ps_kappa", "ps_delta_upper", "ps_delta_lower", "ps_R0", "ps_a"]
    mats = []
    for _, r in df.iterrows():
        d = np.load(os.path.join(fs_dir, r["fs_file"]))
        if keys[0] not in d:
            raise KeyError("per-surface scalars not in npz — re-run geometry_features.py")
        mats.append(np.concatenate([d[k] for k in keys]))
    X = np.vstack(mats)
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu, X)
    keep = ~np.isnan(mu)
    return X[:, keep], int(keep.sum())


# --------------------------------------------------------------------------- #
# Leave-one-shot-out evaluation
# --------------------------------------------------------------------------- #
def loso_predict(X, y, groups, pipe):
    """Out-of-fold predictions under leave-one-group(shot)-out CV."""
    pred = np.full(len(y), np.nan)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
    return pred


def metrics(y, pred):
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-30
    rho, p = spearmanr(y, pred)
    return dict(R2_loso=1 - ss_res / ss_tot,
                RMSE=float(np.sqrt(np.mean((y - pred) ** 2))),
                MAE=float(np.mean(np.abs(y - pred))),
                spearman=float(rho), spearman_p=float(p))


def make_pipe(kind: str, alpha: float, n_pca: int, n_train: int):
    steps = [("scale", StandardScaler())]
    if kind == "flux":
        steps.append(("pca", PCA(n_components=min(n_pca, max(1, n_train - 1)))))
    steps.append(("ridge", Ridge(alpha=alpha)))
    return Pipeline(steps)


def run(target: str, labels_dir: str, alpha: float, n_pca: int, min_elms: int):
    df = pd.read_csv(os.path.join(labels_dir, "geometry_elm_joined.csv"))
    use = df[(df.signal_ok == True) & (df.fwd_n_elms >= min_elms)
             & df[target].notna()].reset_index(drop=True)
    groups = use["shot"].to_numpy()
    y = use[target].to_numpy(float)
    n_train = len(use) - 1
    print(f"target={target}   usable seeds={len(use)} "
          f"(shots={use['shot'].nunique()})   y: mean={y.mean():.2f} std={y.std():.2f}\n")

    fs_dir = os.path.join(labels_dir, "fluxsurfaces")
    out = []
    Xs, scols = load_scalar_X(use)
    out.append(("scalar(LCFS)", Xs.shape[1],
                metrics(y, loso_predict(Xs, y, groups,
                                        make_pipe("scalar", alpha, n_pca, n_train)))))
    Xp, npf = load_persurface_X(use, fs_dir)
    out.append(("per-surface profile", npf,
                metrics(y, loso_predict(Xp, y, groups,
                                        make_pipe("scalar", alpha, n_pca, n_train)))))
    Xf, nfeat = load_flux_X(use, fs_dir)
    out.append((f"flux-surface(PCA{min(n_pca, n_train-1)})", nfeat,
                metrics(y, loso_predict(Xf, y, groups,
                                        make_pipe("flux", alpha, n_pca, n_train)))))

    print(f"{'model':24s}{'nfeat':>7}{'R2_loso':>10}{'RMSE':>9}{'spearman':>10}{'p':>7}")
    for name, nf, m in out:
        print(f"{name:24s}{nf:>7}{m['R2_loso']:>10.3f}{m['RMSE']:>9.2f}"
              f"{m['spearman']:>10.2f}{m['spearman_p']:>7.2f}")
    print(f"\n(scalar features: {scols})")
    print("R2_loso<0 means worse than predicting the mean — expected at this N; "
          "treat as framework, not result.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scalar vs flux-surface geometry->ELM baseline.")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--target", default="fwd_freq_hz",
                    help="ELM target column (e.g. fwd_freq_hz, seed_time_to_next, fwd_median_dt)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-pca", type=int, default=3)
    ap.add_argument("--min-elms", type=int, default=2)
    args = ap.parse_args(argv)
    return run(args.target, args.labels_dir, args.alpha, args.n_pca, args.min_elms)


if __name__ == "__main__":
    main()
