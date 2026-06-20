"""Geometry (shape) feature extraction for NSTX discharges + join to ELM labels.

Pulls EFIT equilibrium shape scalars from the g-files in ``ST_research`` via the
TPED ``GFileData`` parser, then joins them to the ELM labels produced by
``elm_labeling.py`` — matched to the D-alpha time base at the g-file snapshot
time ``t_geo``.

These static shape features are the *seed / initial condition* for the
decaying-memory forecaster: geometry (+ future mode-ID) is injected once at
``t_geo`` and the model predicts ELM activity forward in time. The join therefore
also computes **forward-from-seed** ELM targets (the quantities the seed predicts):
the time-matched ``time_to_next`` / ``time_since_last`` / ``elm_rate`` at ``t_geo``
and the ELM statistics over the window ``[t_geo, t_hi]``.

Requires the TPED conda env (xarray) and the TPED package on the path::

    conda activate TPED
    PYTHONPATH=C:/Users/joesc/git python src/geometry_features.py

Outputs (to ``data/labels/``):
  * ``geometry_features.csv``      — one row per shot: shape scalars + t_geo.
  * ``geometry_elm_joined.csv``    — geometry + ELM summary + time-matched seed
                                     targets + forward-from-seed ELM stats.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# --- repo / data locations (overridable via CLI) --------------------------- #
GIT_ROOT = r"C:\Users\joesc\git"
# directories searched recursively for EFIT g-files
SEARCH_ROOTS = [os.path.join(GIT_ROOT, "ST_research"),
                os.path.join(GIT_ROOT, "TPED", "data")]
ELM_DB = os.path.join(GIT_ROOT, "ELM_forecast_prediction", "data", "dalpha_db.pkl")

_GFILE_RE = re.compile(r"^g(\d{6})[._]")


def _import_gfiledata(git_root: str):
    """Import TPED's GFileData, adding the dir above TPED to sys.path."""
    if git_root not in sys.path:
        sys.path.insert(0, git_root)
    from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData
    return GFileData


# --------------------------------------------------------------------------- #
# g-file discovery and snapshot time
# --------------------------------------------------------------------------- #
def _canonical_score(path: str) -> tuple:
    """Rank candidate g-files for one shot; lower sorts first (more canonical).

    Prefers the clean time-stamped EFIT form ``g<shot>.<time>`` over
    reconstruction/sensitivity variants (``_ave``, ``p1j0.9``/``p1j1.1`` pressure
    scans, ``alpha`` scaling, ``_efit*``/``_e8099`` identifiers).
    """
    name = os.path.basename(path)
    variant = any(tag in name for tag in
                  ("p1j0.9", "p1j1.1", "alpha", "_efit", "_e8099", "_m8099"))
    dotted = 0 if re.match(r"^g\d{6}\.\d", name) else 1  # g123456.00650 form
    return (variant, dotted, name)


def discover_gfiles(roots=SEARCH_ROOTS) -> dict:
    """Map shot -> list of g-file paths found recursively under ``roots``."""
    out: dict[int, list[str]] = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "g*"), recursive=True):
            if os.path.isfile(p) and _GFILE_RE.match(os.path.basename(p)):
                shot = int(_GFILE_RE.match(os.path.basename(p)).group(1))
                out.setdefault(shot, []).append(p)
    return out


def find_gfile(shot: int, roots=SEARCH_ROOTS) -> str | None:
    """Most canonical g-file for ``shot`` across all search roots."""
    cands = discover_gfiles(roots).get(shot, [])
    return sorted(cands, key=_canonical_score)[0] if cands else None


def elm_db_shots(db_path: str = ELM_DB) -> set:
    """Set of shot numbers present in the ELM D-alpha database."""
    import pickle
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    return {int(d["shot"]) for d in db}


def discover_overlap(roots=SEARCH_ROOTS, db_path: str = ELM_DB) -> list:
    """Shots that have BOTH a g-file and ELM data — the geometry∩ELM set."""
    return sorted(set(discover_gfiles(roots)) & elm_db_shots(db_path))


def gfile_time(path: str) -> float:
    """Snapshot time in seconds from the g-file name (zero-padded milliseconds).

    Handles both the dotted form ``g132588.00650`` -> 0.650 s and the underscore
    form ``g129038_525_ave.p1j1`` -> 0.525 s by taking the first time token after
    the 6-digit shot number. (The EFIT ``time_ms`` header attr is unreliable on
    these NSTX reconstructions — often a constant ~110 — so the filename wins.)
    """
    m = re.match(r"g\d{6}[._]0*(\d+)", os.path.basename(path))
    return int(m.group(1)) * 1e-3 if m else np.nan


# --------------------------------------------------------------------------- #
# Shape scalars from the EFIT boundary + q-profile
# --------------------------------------------------------------------------- #
def shape_from_boundary(R, Z) -> dict:
    """Standard plasma shape parameters from the last-closed-flux-surface."""
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    Rmax, Rmin, Zmax, Zmin = R.max(), R.min(), Z.max(), Z.min()
    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    R_top = R[np.argmax(Z)]
    R_bot = R[np.argmin(Z)]
    return dict(
        R0=R0, a=a, A=R0 / a, eps=a / R0,
        kappa=(Zmax - Zmin) / (2 * a),
        delta_upper=(R0 - R_top) / a,
        delta_lower=(R0 - R_bot) / a,
        delta=0.5 * ((R0 - R_top) + (R0 - R_bot)) / a,
    )


def q_features(q) -> dict:
    """q-profile scalars; q assumed on a uniform normalized-flux grid 0..1."""
    q = np.abs(np.asarray(q, float))
    psin = np.linspace(0.0, 1.0, q.size)
    return dict(q0=float(q[0]), qmin=float(q.min()),
                q95=float(np.interp(0.95, psin, q)), q_edge=float(q[-1]))


def extract_shape(gfile_path: str, GFileData) -> dict:
    """Parse one g-file into a flat dict of shape + scalar equilibrium features."""
    ds = GFileData(gfile_path).gfile_to_xarray()
    rkey = "RBDRY" if "RBDRY" in ds else ("RBBBS" if "RBBBS" in ds else None)
    zkey = "ZBDRY" if "ZBDRY" in ds else ("ZBBBS" if "ZBBBS" in ds else None)
    if rkey is None:
        raise ValueError(f"no boundary array in {os.path.basename(gfile_path)}")
    feats = shape_from_boundary(ds[rkey].values, ds[zkey].values)
    qkey = "q" if "q" in ds else ("qpsi" if "qpsi" in ds else None)
    if qkey is not None:
        feats.update(q_features(ds[qkey].values))
    # a few directly-available footer scalars, when present
    for src, dst in [("BETAP0", "betap0"), ("PLASMA", "Ip"), ("RCENTR", "Rcentr")]:
        if src in ds:
            feats[dst] = float(np.asarray(ds[src].values).ravel()[0])
    return feats


def seed_time_s(ds, path: str) -> float:
    """Snapshot time in seconds — prefer the filename, fall back to EFIT attr.

    The filename encodes the intended time reliably; the EFIT ``time_ms`` header
    is often a constant placeholder (~110 ms) on these NSTX reconstructions.
    """
    t = gfile_time(path)
    if np.isfinite(t):
        return t
    tm = ds.attrs.get("time_ms", None) if ds is not None else None
    return float(tm) * 1e-3 if tm else np.nan


# --------------------------------------------------------------------------- #
# Full flux-surface shape arrays (information scalars wash out)
# --------------------------------------------------------------------------- #
def _point_in_poly(x, y, poly) -> bool:
    """Ray-casting test: is (x, y) inside the closed polygon ``poly`` (N,2)?"""
    R, Z = poly[:, 0], poly[:, 1]
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        if ((Z[i] > y) != (Z[j] > y)) and \
           (x < (R[j] - R[i]) * (y - Z[i]) / (Z[j] - Z[i] + 1e-30) + R[i]):
            inside = not inside
        j = i
    return inside


def _resample_theta(seg, rmag, zmag, n_theta):
    """Radius r(θ) about the magnetic axis, on a uniform poloidal-angle grid."""
    R, Z = seg[:, 0], seg[:, 1]
    th = np.arctan2(Z - zmag, R - rmag)
    r = np.hypot(R - rmag, Z - zmag)
    o = np.argsort(th)
    th, r = th[o], r[o]
    th, idx = np.unique(th, return_index=True)
    r = r[idx]
    grid = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    return grid, np.interp(grid, th, r, period=2 * np.pi)


def flux_surface_profile(ds, n_surfaces=10, n_theta=64, level_lo=0.1, level_hi=0.95):
    """Resampled r(θ) for ``n_surfaces`` flux surfaces, edge → core.

    Returns dict: ``levels`` (ψ_N, edge→core), ``theta`` (n_theta,), ``r_theta``
    ((n_surfaces, n_theta) radius about the magnetic axis), ``rmag``/``zmag``, and
    ``per_surface`` (R0/a/κ/δ per surface — a shape-vs-radius profile). NaN rows
    mark surfaces whose closed contour could not be located.
    """
    import contourpy
    R = ds["R"].values
    Z = ds["Z"].values
    psi = ds["psi_RZ"].values
    if ds["psi_RZ"].dims[0] == "R":
        psi = psi.T
    a = ds.attrs
    psin = (psi - a["psiax"]) / (a["psisep"] - a["psiax"])
    rmag, zmag = a["rmag"], a["zmag"]
    levels = np.linspace(level_hi, level_lo, n_surfaces)  # edge -> core

    cg = contourpy.contour_generator(R, Z, psin)
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    r_theta = np.full((n_surfaces, n_theta), np.nan)
    per_surface = []
    for i, lv in enumerate(levels):
        segs = [s for s in cg.lines(float(lv)) if len(s) >= 8
                and _point_in_poly(rmag, zmag, s)]
        if not segs:
            per_surface.append({}); continue
        # the true flux surface is the TIGHTEST closed loop around the axis; a
        # larger spurious contour can also enclose the axis because psi_N is
        # non-monotonic in the SOL/private-flux region, so pick min radial extent.
        seg = min(segs, key=lambda s: np.max(np.hypot(s[:, 0] - rmag, s[:, 1] - zmag)))
        _, r_theta[i] = _resample_theta(seg, rmag, zmag, n_theta)
        per_surface.append(shape_from_boundary(seg[:, 0], seg[:, 1]))
    return dict(levels=levels, theta=theta, r_theta=r_theta,
                rmag=float(rmag), zmag=float(zmag), per_surface=per_surface)


# --------------------------------------------------------------------------- #
# Seed table (one row per geometry time point; multi-time "coast" seeds)
# --------------------------------------------------------------------------- #
def build_seed_table(shots=None, git_root=GIT_ROOT, roots=SEARCH_ROOTS,
                     db_path=ELM_DB, all_times=True, time_tol_ms=15.0,
                     n_surfaces=10, n_theta=64, fs_dir=None):
    """One row per (shot, geometry time). Saves flux-surface arrays to ``fs_dir``.

    Multiple time points per shot are kept as separate **seeds**: the forecaster
    coasts on each geometry until the next time point overrides it. EFIT
    reconstruction variants at the same time (``p1j0.9`` etc.) collapse to one
    canonical g-file. With ``all_times=False`` only the earliest seed per shot is kept.
    """
    if shots is None:
        shots = discover_overlap(roots, db_path)
        print(f"auto-discovered {len(shots)} shots with g-file AND ELM data: {shots}")
    GFileData = _import_gfiledata(git_root)
    gmap = discover_gfiles(roots)
    if fs_dir:
        os.makedirs(fs_dir, exist_ok=True)
    rows = []
    for shot in shots:
        # parse every candidate, group by time (dedup EFIT variants/near-identical)
        parsed = []
        for p in sorted(gmap.get(shot, []), key=_canonical_score):
            try:
                ds = GFileData(p).gfile_to_xarray()
            except Exception as e:  # pragma: no cover
                print(f"  [warn] {shot} {os.path.basename(p)}: {e}"); continue
            if ds is None:                          # invalid/foreign g-file
                continue
            parsed.append((seed_time_s(ds, p), p, ds))
        if not parsed:
            print(f"  [skip] {shot}: no readable g-file"); continue
        # group by rounded time; canonical candidate already first (sorted)
        groups = {}
        for t, p, ds in parsed:
            key = round(t / (time_tol_ms * 1e-3))
            groups.setdefault(key, (t, p, ds))   # keep first (most canonical)
        seeds = sorted(groups.values(), key=lambda x: x[0])
        if not all_times:
            seeds = seeds[:1]
        for t_geo, p, ds in seeds:
            row = dict(shot=shot, gfile=os.path.basename(p), t_geo=t_geo)
            try:
                row.update(extract_shape(p, GFileData))
                if fs_dir:
                    fs = flux_surface_profile(ds, n_surfaces, n_theta)
                    # per-surface scalar profile (model 4): shape vs radius
                    ps = fs.pop("per_surface")
                    ps_keys = ["kappa", "delta_upper", "delta_lower", "R0", "a"]
                    ps_arr = {f"ps_{k}": np.array([d.get(k, np.nan) for d in ps])
                              for k in ps_keys}
                    fn = os.path.join(fs_dir, f"{shot}_{round(t_geo*1000)}.npz")
                    np.savez_compressed(fn, **fs, **ps_arr)
                    row["fs_file"] = os.path.basename(fn)
            except Exception as e:  # pragma: no cover
                print(f"  [warn] {shot}@{t_geo:.3f}s: {e}")
            rows.append(row)
    return pd.DataFrame(rows)


# backward-compatible alias (single seed per shot)
def build_geometry_table(shots=None, git_root=GIT_ROOT, roots=SEARCH_ROOTS,
                         db_path: str = ELM_DB):
    return build_seed_table(shots, git_root, roots, db_path, all_times=False,
                            fs_dir=None)


# --------------------------------------------------------------------------- #
# Join to ELM labels at the seed time (time-step matched)
# --------------------------------------------------------------------------- #
def _seed_targets(shot: int, t_geo: float, labels_dir: str):
    """Time-matched seed values from the per-shot target arrays at ``t_geo``."""
    npz = os.path.join(labels_dir, "targets", f"{shot}.npz")
    if not os.path.exists(npz) or not np.isfinite(t_geo):
        return {}
    d = np.load(npz)
    t = d["time"]
    idx = int(np.argmin(np.abs(t - t_geo)))
    return dict(seed_idx=idx, seed_time=float(t[idx]),
                seed_dt_match=float(t[idx] - t_geo),
                seed_time_to_next=float(d["time_to_next"][idx]),
                seed_time_since_last=float(d["time_since_last"][idx]),
                seed_elm_rate=float(d["elm_rate"][idx]))


def _forward_stats(shot: int, t_start: float, t_end: float, events: pd.DataFrame,
                   primary="lower"):
    """ELM statistics over the coast window [t_start, t_end) (onset-based)."""
    ev = events[(events.shot == shot) & (events.divertor == primary)]
    on = ev["onset_time"].to_numpy()
    fwd = np.sort(on[(on >= t_start) & (on < t_end)])
    out = dict(coast_until=t_end, coast_s=t_end - t_start,
               fwd_n_elms=int(fwd.size), fwd_freq_hz=np.nan, fwd_median_dt=np.nan)
    if t_end > t_start:
        out["fwd_freq_hz"] = fwd.size / (t_end - t_start)
    if fwd.size >= 2:
        out["fwd_median_dt"] = float(np.median(np.diff(fwd)))
    return out


def join_with_labels(geom: pd.DataFrame, labels_dir: str, primary="lower"):
    """Join seeds to ELM labels with multi-time **coast** windows.

    Each seed's forecast window runs from its geometry time to the next seed's
    time within the same shot (or to ``t_hi`` for the last) — i.e. the forecaster
    coasts on each geometry until a newer one overrides it.
    """
    summary = pd.read_csv(os.path.join(labels_dir, "elm_shot_summary.csv"))
    events = pd.read_csv(os.path.join(labels_dir, "elm_events.csv"))
    geom = geom.sort_values(["shot", "t_geo"]).reset_index(drop=True)
    rows = []
    for shot, grp in geom.groupby("shot"):
        s = summary[summary.shot == shot]
        t_hi = float(s["t_hi"].iloc[0]) if len(s) else np.nan
        t_lo = float(s["t_lo"].iloc[0]) if len(s) else np.nan
        times = grp["t_geo"].to_numpy()
        for i, (_, g) in enumerate(grp.iterrows()):
            t_geo = float(g["t_geo"])
            coast_until = float(times[i + 1]) if i + 1 < len(times) else t_hi
            row = g.to_dict()
            row["seed_in_window"] = bool(np.isfinite(t_hi) and t_lo <= t_geo <= t_hi)
            if len(s):
                for c in ["signal_ok", "n_elms", "elm_freq_hz", "median_dt", "cv_dt",
                          "weibull_k", "both_live", "coincidence_frac", "t_lo", "t_hi"]:
                    row[c] = s[c].iloc[0]
            row.update(_seed_targets(shot, t_geo, labels_dir))
            row.update(_forward_stats(shot, t_geo, coast_until, events, primary))
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract NSTX shape features and join to ELM labels.")
    ap.add_argument("--git-root", default=GIT_ROOT)
    ap.add_argument("--db", default=ELM_DB, help="ELM D-alpha database (for overlap discovery)")
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--shots", type=int, nargs="*", default=None,
                    help="explicit shots; default = auto-discover g-file ∩ ELM-data set")
    ap.add_argument("--primary", default="lower")
    ap.add_argument("--single-time", action="store_true",
                    help="one seed per shot (default: all geometry time points)")
    ap.add_argument("--n-surfaces", type=int, default=10)
    ap.add_argument("--n-theta", type=int, default=64)
    args = ap.parse_args(argv)

    os.makedirs(args.labels_dir, exist_ok=True)
    fs_dir = os.path.join(args.labels_dir, "fluxsurfaces")
    geom = build_seed_table(args.shots, args.git_root, SEARCH_ROOTS, args.db,
                            all_times=not args.single_time, n_surfaces=args.n_surfaces,
                            n_theta=args.n_theta, fs_dir=fs_dir)
    geom.to_csv(os.path.join(args.labels_dir, "geometry_features.csv"), index=False)

    joined = join_with_labels(geom, args.labels_dir, args.primary)
    joined.to_csv(os.path.join(args.labels_dir, "geometry_elm_joined.csv"), index=False)

    show = ["shot", "t_geo", "coast_until", "A", "kappa", "delta_lower", "q95",
            "seed_dt_match", "fwd_n_elms", "fwd_freq_hz", "signal_ok"]
    show = [c for c in show if c in joined.columns]
    print(joined[show].to_string(index=False))
    print(f"\n{len(geom)} seeds across {geom['shot'].nunique()} shots; "
          f"flux surfaces -> {fs_dir}/")
    print(f"written: {args.labels_dir}/geometry_features.csv, geometry_elm_joined.csv")
    return geom, joined


if __name__ == "__main__":
    main()
