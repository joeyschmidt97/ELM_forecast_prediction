"""Cox proportional-hazards scoring of inter-ELM intervals (failure analysis).

Treats each **inter-ELM interval as a survival time** (the "time to failure",
the failure being the next ELM crash) and fits a Cox PH model with discharge
geometry as covariates. This is the survival-analysis view of the project: it
yields per-interval hazard scores, interpretable hazard ratios (e.g. does higher
elongation lengthen the inter-ELM period?), and a concordance (C-index) that
quantifies how well geometry ranks ELM risk.

Unit of analysis: one row per inter-ELM interval. ``duration`` = onset_k →
onset_{k+1}; ``event`` = 1 (an ELM ended it). The final interval per shot
(last onset → end of the analyzed window) is right-censored (``event`` = 0).
Covariates come from the geometry **seed** whose coast window contains the
interval start (multi-time geometry handled).

Covariate sets compared:
  * **scalar** — LCFS shape scalars (interpretable hazard ratios).
  * **flux-PCA** — PCA of the full 10-surface r(theta) arrays (does the full
    shape rank risk better than scalars?).

Honesty: geometry is ~constant within a shot, so the *effective* N for the
covariate effect is the number of shots (6), even though there are hundreds of
intervals. In-sample C-index is optimistic; the **leave-one-shot-out C-index** is
the real generalization metric. Treat as framework + preliminary at this N.

Run (TPED env, after ``pip install lifelines``)::

    conda activate TPED
    PYTHONPATH=C:/Users/joesc/git python src/elm_hazard_cox.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCALAR_COVARIATES = ["kappa", "delta_lower", "delta_upper", "A", "q95"]


# --------------------------------------------------------------------------- #
# Build the inter-ELM interval survival dataset
# --------------------------------------------------------------------------- #
def _seed_for_time(gsub: pd.DataFrame, t: float):
    """The geometry seed (row) whose coast window contains time ``t``."""
    hit = gsub[(gsub.t_geo <= t) & (t < gsub.coast_until)]
    if len(hit):
        return hit.iloc[0]
    return gsub.iloc[0] if t < gsub.t_geo.min() else gsub.iloc[-1]


def build_intervals(labels_dir: str, primary="lower", min_elms=3) -> pd.DataFrame:
    events = pd.read_csv(os.path.join(labels_dir, "elm_events.csv"))
    summ = pd.read_csv(os.path.join(labels_dir, "elm_shot_summary.csv"))
    geo = pd.read_csv(os.path.join(labels_dir, "geometry_elm_joined.csv"))
    rows = []
    for shot, gsub in geo.groupby("shot"):
        s = summ[summ.shot == shot]
        if not len(s) or not bool(s["signal_ok"].iloc[0]):
            continue
        t_hi = float(s["t_hi"].iloc[0])
        ons = np.sort(events[(events.shot == shot) &
                             (events.divertor == primary)]["onset_time"].to_numpy())
        if ons.size < min_elms:
            continue
        gsub = gsub.sort_values("t_geo")
        for k in range(ons.size - 1):
            rows.append(_make_row(shot, ons[k], ons[k + 1] - ons[k], 1, gsub))
        if t_hi > ons[-1]:                                   # censored tail
            rows.append(_make_row(shot, ons[-1], t_hi - ons[-1], 0, gsub))
    return pd.DataFrame(rows)


def _make_row(shot, start, duration, event, gsub):
    seed = _seed_for_time(gsub, start)
    row = dict(shot=int(shot), start=float(start), duration=float(duration),
               event=int(event), fs_file=seed.get("fs_file", None))
    for c in SCALAR_COVARIATES:
        row[c] = float(seed[c]) if c in seed and pd.notna(seed[c]) else np.nan
    return row


def attach_flux_pca(df: pd.DataFrame, fs_dir: str, n_pca=3) -> list:
    """Add flux-surface PCA components as covariates (fit on the seed vectors)."""
    files = df["fs_file"].dropna().unique()
    mats = {f: np.load(os.path.join(fs_dir, f))["r_theta"].ravel() for f in files}
    X = np.vstack([mats[f] for f in files])
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu, X)[:, ~np.isnan(mu)]
    Z = PCA(n_components=n_pca).fit_transform(StandardScaler().fit_transform(X))
    pc = {f: Z[i] for i, f in enumerate(files)}
    cols = [f"fluxPC{i+1}" for i in range(n_pca)]
    for j, c in enumerate(cols):
        df[c] = [pc[f][j] if pd.notna(f) and f in pc else np.nan
                 for f in df["fs_file"]]
    return cols


# --------------------------------------------------------------------------- #
# Fit + score
# --------------------------------------------------------------------------- #
def fit_and_score(df: pd.DataFrame, covariates: list, penalizer=0.1):
    """Fit Cox on all data (in-sample) and report leave-one-shot-out C-index."""
    d = df.dropna(subset=covariates + ["duration", "event"]).copy()
    d = d[d["duration"] > 0]
    # standardize covariates (hazard ratios become per-SD; aids convergence)
    d[covariates] = StandardScaler().fit_transform(d[covariates])

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d[covariates + ["duration", "event"]], "duration", "event")
    in_c = cph.concordance_index_

    # leave-one-shot-out C-index (pooled over held-out intervals)
    risk = np.full(len(d), np.nan)
    for shot in d["shot"].unique():
        tr, te = d[d.shot != shot], d[d.shot == shot]
        if tr["event"].sum() < 2 or len(te) == 0:
            continue
        m = CoxPHFitter(penalizer=penalizer)
        m.fit(tr[covariates + ["duration", "event"]], "duration", "event")
        risk[d.shot.to_numpy() == shot] = m.predict_partial_hazard(te).to_numpy()
    ok = ~np.isnan(risk)
    loso_c = (concordance_index(d["duration"].to_numpy()[ok], -risk[ok],
                                d["event"].to_numpy()[ok])
              if ok.sum() > 2 else np.nan)
    return cph, in_c, loso_c, len(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cox PH hazard scoring of inter-ELM intervals.")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--penalizer", type=float, default=0.1)
    ap.add_argument("--n-pca", type=int, default=3)
    args = ap.parse_args(argv)

    df = build_intervals(args.labels_dir)
    n_shots = df["shot"].nunique()
    print(f"intervals={len(df)}  shots={n_shots}  events={int(df.event.sum())}  "
          f"censored={int((df.event == 0).sum())}")
    print(f"median inter-ELM duration={df[df.event==1].duration.median()*1e3:.1f} ms\n")

    flux_cols = attach_flux_pca(df, os.path.join(args.labels_dir, "fluxsurfaces"),
                                args.n_pca)
    sets = {"scalar(LCFS)": SCALAR_COVARIATES, "flux-PCA": flux_cols}

    print(f"{'covariates':18s}{'n_int':>7}{'C(in-sample)':>14}{'C(LOSO)':>10}")
    fits = {}
    for name, cov in sets.items():
        cph, in_c, loso_c, n = fit_and_score(df, cov, args.penalizer)
        fits[name] = cph
        print(f"{name:18s}{n:>7}{in_c:>14.3f}{loso_c:>10.3f}")

    print("\nScalar hazard ratios (per +1 SD; HR>1 = shorter interval / higher ELM risk):")
    hr = fits["scalar(LCFS)"].summary[["coef", "exp(coef)", "p"]]
    hr.columns = ["coef", "HR", "p"]
    print(hr.round(3).to_string())
    print("\nC=0.5 is random ranking; LOSO C is the honest metric. N=6 shots -> "
          "wide uncertainty; framework + preliminary.")
    return df, fits


if __name__ == "__main__":
    main()
