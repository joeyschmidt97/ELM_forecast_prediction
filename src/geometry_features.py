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
ST_ROOT = os.path.join(GIT_ROOT, "ST_research", "NSTXU_discharges")
OVERLAP_SHOTS = [129015, 129038, 132543, 132588, 139057, 141300, 141309]


def _import_gfiledata(git_root: str):
    """Import TPED's GFileData, adding the dir above TPED to sys.path."""
    if git_root not in sys.path:
        sys.path.insert(0, git_root)
    from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData
    return GFileData


# --------------------------------------------------------------------------- #
# g-file discovery and snapshot time
# --------------------------------------------------------------------------- #
def find_gfile(shot: int, st_root: str = ST_ROOT) -> str | None:
    hits = sorted(glob.glob(os.path.join(st_root, str(shot), f"g{shot}.*")))
    return hits[0] if hits else None


def gfile_time(path: str) -> float:
    """Snapshot time in seconds parsed from a g-file name.

    ``g132588.00650`` -> 0.650 s; ``g129015.00409_450`` (averaged window) ->
    midpoint 0.4295 s. Times are zero-padded milliseconds.
    """
    stem = os.path.basename(path).split(".", 1)[1]  # e.g. '00650' or '00409_450'
    nums = [int(x) for x in re.findall(r"\d+", stem)]
    if not nums:
        return np.nan
    if len(nums) >= 2:                      # averaged window: midpoint
        return 0.5 * (nums[0] + nums[1]) * 1e-3
    return nums[0] * 1e-3


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


def build_geometry_table(shots=OVERLAP_SHOTS, git_root=GIT_ROOT, st_root=ST_ROOT):
    GFileData = _import_gfiledata(git_root)
    rows = []
    for shot in shots:
        g = find_gfile(shot, st_root)
        if g is None:
            print(f"  [skip] {shot}: no g-file"); continue
        row = dict(shot=shot, gfile=os.path.basename(g), t_geo=gfile_time(g))
        try:
            row.update(extract_shape(g, GFileData))
        except Exception as e:  # pragma: no cover
            print(f"  [warn] {shot}: {e}")
        rows.append(row)
    return pd.DataFrame(rows)


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


def _forward_stats(shot: int, t_geo: float, t_hi: float, events: pd.DataFrame,
                   primary="lower"):
    """ELM statistics over the forecast window [t_geo, t_hi] (onset-based)."""
    ev = events[(events.shot == shot) & (events.divertor == primary)]
    fwd = np.sort(ev[ev.onset_time >= t_geo]["onset_time"].to_numpy())
    out = dict(fwd_n_elms=int(fwd.size), fwd_freq_hz=np.nan, fwd_median_dt=np.nan)
    if t_hi > t_geo:
        out["fwd_freq_hz"] = fwd.size / (t_hi - t_geo)
    if fwd.size >= 2:
        out["fwd_median_dt"] = float(np.median(np.diff(fwd)))
    return out


def join_with_labels(geom: pd.DataFrame, labels_dir: str, primary="lower"):
    summary = pd.read_csv(os.path.join(labels_dir, "elm_shot_summary.csv"))
    events = pd.read_csv(os.path.join(labels_dir, "elm_events.csv"))
    rows = []
    for _, g in geom.iterrows():
        shot, t_geo = int(g["shot"]), float(g["t_geo"])
        s = summary[summary.shot == shot]
        t_hi = float(s["t_hi"].iloc[0]) if len(s) else np.nan
        row = g.to_dict()
        # seed falls inside the analyzed window?
        t_lo = float(s["t_lo"].iloc[0]) if len(s) else np.nan
        row["seed_in_window"] = bool(np.isfinite(t_hi) and t_lo <= t_geo <= t_hi)
        if len(s):
            for c in ["signal_ok", "n_elms", "elm_freq_hz", "median_dt", "cv_dt",
                      "weibull_k", "both_live", "coincidence_frac", "t_lo", "t_hi"]:
                row[c] = s[c].iloc[0]
        row.update(_seed_targets(shot, t_geo, labels_dir))
        row.update(_forward_stats(shot, t_geo, t_hi, events, primary))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract NSTX shape features and join to ELM labels.")
    ap.add_argument("--git-root", default=GIT_ROOT)
    ap.add_argument("--st-root", default=ST_ROOT)
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--shots", type=int, nargs="*", default=OVERLAP_SHOTS)
    ap.add_argument("--primary", default="lower")
    args = ap.parse_args(argv)

    geom = build_geometry_table(args.shots, args.git_root, args.st_root)
    os.makedirs(args.labels_dir, exist_ok=True)
    geom.to_csv(os.path.join(args.labels_dir, "geometry_features.csv"), index=False)

    joined = join_with_labels(geom, args.labels_dir, args.primary)
    joined.to_csv(os.path.join(args.labels_dir, "geometry_elm_joined.csv"), index=False)

    show = ["shot", "t_geo", "A", "kappa", "delta_lower", "q95", "seed_dt_match",
            "seed_time_to_next", "fwd_n_elms", "fwd_freq_hz", "signal_ok"]
    show = [c for c in show if c in joined.columns]
    print(joined[show].to_string(index=False))
    print(f"\nwritten: {args.labels_dir}/geometry_features.csv, geometry_elm_joined.csv")
    return geom, joined


if __name__ == "__main__":
    main()
