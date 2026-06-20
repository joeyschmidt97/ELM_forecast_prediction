"""Run every ELM/geometry analysis and write inspectable figures + tables.

Outputs land under ``analysis/`` (next to this file):

    analysis/
      SUMMARY.md                      key numbers, links every figure
      geometry/  flux_surfaces.png  persurface_profiles.png
                 geometry_vs_elm.png shape_table.csv
      labels/    elm_frequency.png   interval_hist.png  shot_summary.csv
      baseline/  pred_vs_actual.png  metrics.csv
      cox/       hazard_ratios.png   survival_curves.png  cindex.csv
      reservoir/ nrmse_bars.png      example_forecast.png  nrmse.csv

Run (TPED env)::

    conda activate TPED
    PYTHONPATH=C:/Users/joesc/git python analysis/run_all.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
LABELS = os.path.join(REPO, "data", "labels")
FS_DIR = os.path.join(LABELS, "fluxsurfaces")
OUT = os.path.join(REPO, "analysis")

from src import geometry_baseline as gb
from src import elm_hazard_cox as cox
from src import reservoir_harness as rh

OVERLAP = [129015, 129038, 132543, 132588, 139057, 141300, 141309]


def _dir(name):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Geometry: flux surfaces, per-surface profiles, geometry-vs-ELM scatter
# --------------------------------------------------------------------------- #
def geometry_figs(joined):
    d = _dir("geometry")
    use = joined[joined.signal_ok == True].drop_duplicates("shot")

    # (1) nested flux surfaces in (R,Z) for each shot
    n = len(use)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.0), sharey=True)
    for ax, (_, r) in zip(np.atleast_1d(axes), use.iterrows()):
        fs = np.load(os.path.join(FS_DIR, r["fs_file"]))
        th, rmag, zmag = fs["theta"], fs["rmag"], fs["zmag"]
        for i, lv in enumerate(fs["levels"]):
            rr = fs["r_theta"][i]
            if np.isnan(rr).all():
                continue
            R = rmag + rr * np.cos(th)
            Z = zmag + rr * np.sin(th)
            R = np.append(R, R[0]); Z = np.append(Z, Z[0])
            ax.plot(R, Z, color=plt.cm.viridis(i / len(fs["levels"])), lw=0.8)
        ax.plot(rmag, zmag, "r+", ms=8)
        ax.set_aspect("equal"); ax.set_title(f"{int(r['shot'])}\n@{r['t_geo']:.3f}s", fontsize=8)
        ax.set_xlabel("R (m)")
    np.atleast_1d(axes)[0].set_ylabel("Z (m)")
    fig.suptitle("Nested flux surfaces (edge=yellow → core=purple), magnetic axis = +", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "flux_surfaces.png"), dpi=110); plt.close(fig)

    # (2) per-surface shape profiles kappa(psiN), delta_lower(psiN)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for _, r in use.iterrows():
        fs = np.load(os.path.join(FS_DIR, r["fs_file"]))
        lv = fs["levels"]
        a1.plot(lv, fs["ps_kappa"], "-o", ms=3, label=int(r["shot"]))
        a2.plot(lv, fs["ps_delta_lower"], "-o", ms=3, label=int(r["shot"]))
    for a, t in ((a1, "elongation κ"), (a2, "lower triangularity δ")):
        a.set_xlabel("ψ_N (1=edge → 0.1=core)"); a.set_title(f"{t} vs radius"); a.invert_xaxis()
    a1.set_ylabel("κ"); a2.set_ylabel("δ_lower"); a1.legend(fontsize=7, ncol=2)
    fig.suptitle("Per-surface shape profile (model 4) — the radial info LCFS scalars wash out", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "persurface_profiles.png"), dpi=110); plt.close(fig)

    # (3) geometry-vs-ELM scatter with Spearman
    from scipy.stats import spearmanr
    pairs = [("kappa", "fwd_freq_hz"), ("delta", "seed_time_to_next"),
             ("delta_upper", "fwd_freq_hz"), ("A", "fwd_freq_hz")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (xf, yf) in zip(axes, pairs):
        g = use.dropna(subset=[xf, yf])
        ax.scatter(g[xf], g[yf], c="tab:blue")
        for _, r in g.iterrows():
            ax.annotate(int(r["shot"]), (r[xf], r[yf]), fontsize=6)
        rho, p = spearmanr(g[xf], g[yf])
        ax.set_xlabel(xf); ax.set_ylabel(yf); ax.set_title(f"ρ={rho:+.2f} (p={p:.2f})", fontsize=9)
    fig.suptitle("Geometry vs ELM behaviour (N=6 usable shots — exploratory)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "geometry_vs_elm.png"), dpi=110); plt.close(fig)

    cols = ["shot", "t_geo", "A", "kappa", "delta_lower", "delta_upper", "q95",
            "fwd_freq_hz", "seed_time_to_next", "signal_ok"]
    joined[[c for c in cols if c in joined]].to_csv(os.path.join(d, "shape_table.csv"), index=False)


# --------------------------------------------------------------------------- #
# Labels: ELM frequency per shot, inter-ELM interval histograms
# --------------------------------------------------------------------------- #
def label_figs():
    d = _dir("labels")
    summ = pd.read_csv(os.path.join(LABELS, "elm_shot_summary.csv"))
    ev = pd.read_csv(os.path.join(LABELS, "elm_events.csv"))
    ov = summ[summ.shot.isin(OVERLAP)].copy()

    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["tab:green" if ok else "tab:red" for ok in ov.signal_ok]
    ax.bar(ov.shot.astype(str), ov.elm_freq_hz.fillna(0), color=colors)
    ax.set_ylabel("ELM frequency (Hz)"); ax.set_title("ELM frequency per overlap shot (red = signal_ok=False)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "elm_frequency.png"), dpi=110); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, shot in zip(axes.ravel(), [s for s in OVERLAP if s != 129038]):
        on = np.sort(ev[(ev.shot == shot) & (ev.divertor == "lower")]["onset_time"].to_numpy())
        if on.size >= 2:
            ax.hist(np.diff(on) * 1e3, bins=15, color="tab:blue")
        ax.set_title(f"{shot}  (n={on.size})", fontsize=9); ax.set_xlabel("inter-ELM Δt (ms)")
    fig.suptitle("Inter-ELM interval distributions (lower divertor)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "interval_hist.png"), dpi=110); plt.close(fig)
    ov.to_csv(os.path.join(d, "shot_summary.csv"), index=False)


# --------------------------------------------------------------------------- #
# Baseline: predicted-vs-actual under LOSO for each representation
# --------------------------------------------------------------------------- #
def baseline_figs():
    d = _dir("baseline")
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    targets = ["fwd_freq_hz", "seed_time_to_next"]
    for ti, target in enumerate(targets):
        df = pd.read_csv(os.path.join(LABELS, "geometry_elm_joined.csv"))
        use = df[(df.signal_ok == True) & (df.fwd_n_elms >= 2) & df[target].notna()].reset_index(drop=True)
        groups = use["shot"].to_numpy(); y = use[target].to_numpy(float); nt = len(use) - 1
        Xs, _ = gb.load_scalar_X(use)
        Xp, _ = gb.load_persurface_X(use, FS_DIR)
        Xf, _ = gb.load_flux_X(use, FS_DIR)
        feats = [("scalar", Xs, "scalar"), ("per-surface", Xp, "scalar"), ("flux-surface", Xf, "flux")]
        for ax, (name, X, kind) in zip(axes[ti], feats):
            pred = gb.loso_predict(X, y, groups, gb.make_pipe(kind, 1.0, 3, nt))
            m = gb.metrics(y, pred)
            ax.scatter(y, pred, c="tab:purple")
            lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel(f"actual {target}"); ax.set_ylabel("LOSO predicted")
            ax.set_title(f"{name}\nR²={m['R2_loso']:.2f}  ρ={m['spearman']:+.2f}", fontsize=9)
            rows.append(dict(target=target, model=name, **m))
    fig.suptitle("Geometry → ELM: leave-one-shot-out predicted vs actual (N=6)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(d, "pred_vs_actual.png"), dpi=110); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(d, "metrics.csv"), index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cox: hazard ratios, survival curves, C-index table
# --------------------------------------------------------------------------- #
def cox_figs():
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from scipy.stats import spearmanr
    d = _dir("cox")
    df = cox.build_intervals(LABELS)
    dd = df[df.duration > 0].reset_index(drop=True)

    # leak-free LOSO risk for each representation (+ no-geometry = 0.5 baseline,
    # + random per-shot constant control, exactly as in the reservoir ablation)
    Xsc = dd[cox.SCALAR_COVARIATES].to_numpy(float)
    Xps = cox._persurf_raw(dd, FS_DIR)
    Xfl = cox._flux_raw(dd, FS_DIR)
    rng = np.random.default_rng(0)
    rmap = {sh: rng.standard_normal(3) for sh in dd.shot.unique()}
    Xrnd = np.vstack([rmap[s] for s in dd.shot])
    reps = [("scalar(LCFS)", Xsc, None), ("per-surface", Xps, 3),
            ("flux-surface", Xfl, 3), ("random(ctrl)", Xrnd, None)]
    risks, rows = {}, []
    for name, X, npca in reps:
        r = cox.loso_risk(dd, X, npca)
        risks[name] = r
        rows.append(dict(model=name, C_loso=cox.cindex_from_risk(dd, r)))
    cdf = pd.DataFrame(rows)
    cdf.to_csv(os.path.join(d, "cindex.csv"), index=False)

    # (1) THE comparison: C-index by representation vs the no-geometry baseline
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = ["crimson" if "random" in m else "teal" for m in cdf.model]
    ax.bar(cdf.model, cdf.C_loso, color=colors)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="no geometry (random ranking)")
    ax.set_ylabel("LOSO C-index"); ax.set_ylim(0.3, 0.7)
    ax.set_title("Cox ELM-hazard discrimination: full geometry vs scalar vs none\n"
                 "(>0.5 beats no-geometry; teal=geometry, red=random-ID control)")
    ax.legend(fontsize=8); plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(d, "cindex_comparison.png"), dpi=110); plt.close(fig)

    # (2) visual discrimination: split intervals by predicted risk -> KM curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    kmf = KaplanMeierFitter()
    for ax, name in zip(axes, ["scalar(LCFS)", "flux-surface"]):
        r = risks[name]; ok = ~np.isnan(r); med = np.median(r[ok])
        for grp, lab, c in [(r > med, "high predicted risk", "tab:red"),
                            (r <= med, "low predicted risk", "tab:green")]:
            m = ok & grp
            kmf.fit(dd.duration[m] * 1e3, dd.event[m], label=lab)
            kmf.plot_survival_function(ax=ax, ci_show=False, color=c)
        ax.set_title(f"{name}  (C={cox.cindex_from_risk(dd, r):.3f})", fontsize=9)
        ax.set_xlabel("time since last ELM (ms)")
    axes[0].set_ylabel("P(no ELM yet)")
    fig.suptitle("Do high- vs low-risk intervals separate? (wider gap = better discrimination)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "risk_stratified_survival.png"), dpi=110); plt.close(fig)

    # (3) per-shot: predicted risk vs actual median inter-ELM duration
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ["scalar(LCFS)", "flux-surface"]):
        r = risks[name]
        g = dd.assign(risk=r).groupby("shot").agg(
            risk=("risk", "mean"), med_dt=("duration", "median")).dropna()
        ax.scatter(g.risk, g.med_dt * 1e3, c="tab:purple")
        for sh, row in g.iterrows():
            ax.annotate(int(sh), (row.risk, row.med_dt * 1e3), fontsize=7)
        rho, p = spearmanr(g.risk, g.med_dt)
        ax.set_xlabel("mean predicted hazard (LOSO)"); ax.set_ylabel("actual median Δt (ms)")
        ax.set_title(f"{name}  ρ={rho:+.2f} (p={p:.2f})", fontsize=9)
    fig.suptitle("Per-shot: does predicted hazard track actual ELM rate? (want negative slope)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "per_shot_risk.png"), dpi=110); plt.close(fig)

    # supporting: scalar hazard-ratio forest (in-sample fit)
    sdf = dd[cox.SCALAR_COVARIATES + ["duration", "event"]].copy()
    from sklearn.preprocessing import StandardScaler
    sdf[cox.SCALAR_COVARIATES] = StandardScaler().fit_transform(sdf[cox.SCALAR_COVARIATES])
    cph = CoxPHFitter(penalizer=0.1).fit(sdf, "duration", "event")
    s = cph.summary
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    yv = np.arange(len(s))
    ax.errorbar(s["exp(coef)"], yv,
                xerr=[s["exp(coef)"] - s["exp(coef) lower 95%"],
                      s["exp(coef) upper 95%"] - s["exp(coef)"]],
                fmt="o", color="tab:blue", capsize=3)
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_yticks(yv); ax.set_yticklabels(s.index)
    ax.set_xlabel("hazard ratio (per +1 SD; >1 = higher ELM risk)")
    ax.set_title("Scalar geometry hazard ratios (in-sample)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "hazard_ratios.png"), dpi=110); plt.close(fig)
    return cdf, s


# --------------------------------------------------------------------------- #
# Reservoir: NRMSE bars + example held-out forecast trace
# --------------------------------------------------------------------------- #
def reservoir_figs():
    d = _dir("reservoir")
    db = os.path.join(REPO, "data", "dalpha_db.pkl")
    streams, ctx, shots = rh.load_data(LABELS, db, downsample=10, n_pca=3)
    res = rh.run_all(streams, ctx, shots, horizon=10, train_frac=0.5, units=400)
    df = pd.DataFrame([dict(model=k, nrmse=v["mean"], std=v["std"]) for k, v in res.items()])
    df.to_csv(os.path.join(d, "nrmse.csv"), index=False)

    # (1) difference-score bars; random control in red, time-only grey
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = ["grey" if "time-only" in m else ("crimson" if "random" in m else "teal")
              for m in df.model]
    ax.bar(df.model, df.nrmse, yerr=df["std"], color=colors, capsize=3)
    ax.axhline(df.nrmse.iloc[0], color="grey", ls="--", lw=0.8)
    ax.set_ylabel("NRMSE (pooled time-split)")
    ax.set_ylim(min(df.nrmse) * 0.9, max(df.nrmse) * 1.05)
    ax.set_title("D-alpha forecast skill by static context\n(teal=geometry, red=random-ID control, grey=no context)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(d, "nrmse_bars.png"), dpi=110); plt.close(fig)

    # (2) ELM-prediction overlays: true vs predicted D-alpha, baseline + best geometry
    best = min([m for m in res if m not in ("1 time-only", "5 +random(ctrl)")],
               key=lambda m: res[m]["mean"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 7))
    for ax, shot in zip(axes, shots[:3]):
        yt, p1 = res["1 time-only"]["preds"][shot]
        _, pg = res[best]["preds"][shot]
        ax.plot(yt, color="k", lw=0.8, label="true Dα (future)")
        ax.plot(p1, color="tab:blue", lw=0.9, label="pred: time-only")
        ax.plot(pg, color="tab:green", lw=0.9, label=f"pred: {best}")
        ax.set_title(f"shot {int(shot)}  (test half)", fontsize=9); ax.set_ylabel("norm Dα")
    axes[0].legend(fontsize=8, ncol=3); axes[-1].set_xlabel("time step (~2 ms each)")
    fig.suptitle("ELM prediction: forecast D-alpha 20 ms ahead (baseline test) — "
                 "geometry ≈ no change", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "elm_prediction.png"), dpi=110); plt.close(fig)
    return df


# --------------------------------------------------------------------------- #
def write_summary(base_m, cox_c, cox_hr, res_df):
    lines = ["# ELM forecast — analysis results\n",
             "All figures generated by `analysis/run_all.py`. N=6 usable shots "
             "(129038 dropped: signal_ok=False). Treat as framework + preliminary.\n",
             "## Geometry",
             "- ![flux surfaces](geometry/flux_surfaces.png)",
             "- ![per-surface profiles](geometry/persurface_profiles.png)",
             "- ![geometry vs ELM](geometry/geometry_vs_elm.png)  ([table](geometry/shape_table.csv))\n",
             "## ELM labels",
             "- ![ELM frequency](labels/elm_frequency.png)  ![interval hist](labels/interval_hist.png)\n",
             "## Geometry → ELM baseline (leave-one-shot-out)",
             "- ![pred vs actual](baseline/pred_vs_actual.png)  ([metrics](baseline/metrics.csv))\n```",
             base_m.round(3).to_string(index=False), "```\n",
             "## Cox PH hazard (inter-ELM survival) — the geometry comparison",
             "Baseline = no geometry (C=0.5). Leave-one-shot-out C-index, per-fold PCA (no leak); "
             "`random(ctrl)` = random per-shot constant.",
             "- ![C-index comparison](cox/cindex_comparison.png)",
             "- ![risk-stratified survival](cox/risk_stratified_survival.png)",
             "- ![per-shot risk](cox/per_shot_risk.png)  ![hazard ratios](cox/hazard_ratios.png)\n```",
             cox_c.round(3).to_string(index=False), "\n", cox_hr.round(3).to_string(), "```\n",
             "## Reservoir ELM predictor (forecast D-alpha 20 ms ahead) + geometry-as-input ablation",
             "Geometry injected as reservoir input channels; pooled time-split (train on early "
             "half of all shots, forecast later half). `+random(ctrl)` = random per-shot constant.",
             "- ![nrmse](reservoir/nrmse_bars.png)  ![elm prediction](reservoir/elm_prediction.png)\n```",
             res_df.round(3).to_string(index=False), "```",
             "_Geometry improves ~5% but the random-ID control improves the same — the gain is "
             "per-shot specialization, not geometry physics._\n"]
    with open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)
    joined = pd.read_csv(os.path.join(LABELS, "geometry_elm_joined.csv"))
    print("geometry figures..."); geometry_figs(joined)
    print("label figures...");    label_figs()
    print("baseline figures..."); base_m = baseline_figs()
    print("cox figures...");      cox_c, cox_hr = cox_figs()
    print("reservoir figures..."); res_df = reservoir_figs()
    write_summary(base_m, cox_c, cox_hr, res_df)
    print(f"\nDONE -> {OUT}\\SUMMARY.md  (+ geometry/ labels/ baseline/ cox/ reservoir/)")


if __name__ == "__main__":
    main()
