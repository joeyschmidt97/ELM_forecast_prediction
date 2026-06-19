"""ELM labeling pipeline for NSTX divertor D-alpha signals.

Detects ELM bursts in the raw upper/lower divertor D-alpha time series stored in
``dalpha_db.pkl`` (a list of per-shot dicts), extracts ELM onset times and
inter-ELM intervals, computes per-shot ELM statistics, and emits forecaster-ready
target series (time-to-next-ELM, time-since-last-ELM, smoothed ELM rate, and a
binary event train) on the native uniform time base.

The forecasting model (reservoir / decaying-memory LSTM) is *seeded* by static
geometry + mode-ID features and predicts forward in time; the targets produced
here are the ground truth it forecasts toward.

Database record schema (one dict per shot)::

    {'shot': float,
     'data lower dalpha': float32[N], 'time lower dalpha': float32[N],
     'data upper dalpha': float32[N], 'time upper dalpha': float32[N]}

Signal characteristics (confirmed on this DB):
  * 5 kHz sampling (dt = 0.2 ms), N = 31744.
  * D-alpha sits on a slowly varying baseline (~3 a.u.); ELMs are sharp spikes.
  * The upper-divertor trace is sign-inverted (negative) -> rectified here.
  * The back portion of each record is a saturation rail (|y| = max, e.g. 10.235)
    that is auto-detected and masked before detection.

CLI::

    python -m src.elm_labeling --db data/dalpha_db.pkl --out data/labels

Runs under the ``fmc`` (numpy/scipy/pandas) or ``TPED`` conda env.
"""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

try:  # Weibull shape is optional; guard if scipy.stats unavailable
    from scipy.stats import weibull_min
    _HAVE_WEIBULL = True
except Exception:  # pragma: no cover
    _HAVE_WEIBULL = False

DIVERTORS = ("lower", "upper")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class LabelConfig:
    """Detection and target-generation parameters."""

    t_min: float = 0.0           # ignore pre-shot samples (s)
    t_max: float | None = None   # hard upper window (s); None -> auto rail trim
    rail_frac: float = 0.98      # |y| >= rail_frac * max  treated as saturated
    rail_run: int = 25           # consecutive rail samples that mark the tail
    baseline_ms: float = 40.0    # rolling-median baseline window (ms)
    active_win_ms: float = 25.0  # long window for the plasma-on envelope (ms)
    active_floor_frac: float = 0.12  # plasma-on where envelope > frac * p95(envelope)
    min_signal_amp: float = 0.2  # dead-channel gate: windowed p95(|y|) must exceed this (a.u.)
    height_mad: float = 6.0      # peak height threshold in MAD units of residual
    prominence_mad: float = 4.0  # peak prominence threshold in MAD units
    min_interval_ms: float = 1.5  # refractory: minimum inter-ELM spacing (ms)
    onset_frac: float = 0.2      # onset = foot of rise where resid falls below frac*peak
    onset_max_back_ms: float = 8.0  # max look-back from peak when locating onset (ms)
    coincidence_tol_ms: float = 3.0  # lower/upper onset match tolerance (ms)
    rate_kernel_ms: float = 10.0  # Gaussian sigma for smoothed ELM rate (ms)
    primary: str = "lower"       # divertor used for per-shot statistics
    min_elms_stats: int = 3      # below this, distribution stats are NaN


# --------------------------------------------------------------------------- #
# Signal preconditioning
# --------------------------------------------------------------------------- #
def _valid_window(t, y, cfg: LabelConfig):
    """Return a boolean mask of physically valid (non-rail, in-window) samples."""
    y = np.abs(np.asarray(y, dtype=float))
    t = np.asarray(t, dtype=float)
    ymax = y.max()
    rail = y >= cfg.rail_frac * ymax

    # auto-detect the start of the trailing saturation rail (first long run)
    t_rail = np.inf
    if cfg.t_max is None and rail.any():
        run = 0
        for i, r in enumerate(rail):
            run = run + 1 if r else 0
            if run >= cfg.rail_run:
                t_rail = t[i - run + 1]
                break
    t_hi = cfg.t_max if cfg.t_max is not None else t_rail

    mask = (t >= cfg.t_min) & (t <= t_hi) & (~rail)
    if mask.sum() < 10:
        return mask

    # Plasma-on envelope: a long rolling median of |y| within the windowed region.
    # Detection is restricted to where the plasma is actually emitting, which
    # removes post-shot baseline noise that a purely MAD-relative threshold would
    # otherwise fire on (the dominant false-positive source on the upper divertor).
    dt = float(np.median(np.diff(t)))
    win = max(3, int(round((cfg.active_win_ms * 1e-3) / dt)))
    env = pd.Series(np.where(mask, y, np.nan)).rolling(
        win, center=True, min_periods=max(3, win // 4)).median().to_numpy()
    floor = cfg.active_floor_frac * np.nanpercentile(env[mask], 95)
    mask &= np.nan_to_num(env, nan=-np.inf) > floor
    return mask


def _baseline(y, dt, cfg: LabelConfig):
    """Slowly varying baseline via centered rolling median."""
    win = max(3, int(round((cfg.baseline_ms * 1e-3) / dt)))
    if win % 2 == 0:
        win += 1
    s = pd.Series(y)
    base = s.rolling(win, center=True, min_periods=1).median().to_numpy()
    return base


def _find_onsets(resid, peaks, dt, cfg: LabelConfig):
    """Locate ELM onset (foot of the rising edge) for each detected peak.

    Walks back from each peak to the last sample where the baseline-subtracted
    residual was still below ``onset_frac`` of the peak height. This marks the
    start of the sharp ELM rise — the physically meaningful event time for
    inter-ELM intervals (cf. the conventional dDα/dt "difference method";
    Eldon et al., as benchmarked by Song 2023). Falls back to the peak itself
    if no clear foot is found within the look-back window.
    """
    max_back = max(1, int(round((cfg.onset_max_back_ms * 1e-3) / dt)))
    onsets = np.empty(peaks.size, dtype=int)
    for j, p in enumerate(peaks):
        thr = cfg.onset_frac * resid[p]
        lo = max(0, p - max_back)
        i = p
        while i > lo and resid[i] > thr:
            i -= 1
        onsets[j] = i
    return onsets


def _match_onsets(a, b, tol):
    """Boolean mask over ``a``: True where some element of ``b`` is within ``tol``.

    Greedy nearest-time matching between two sorted onset-time arrays — used for
    lower/upper divertor coincidence confirmation.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.zeros(a.size, dtype=bool)
    idx = np.searchsorted(b, a)
    matched = np.zeros(a.size, dtype=bool)
    for k, (ai, j) in enumerate(zip(a, idx)):
        cand = [b[m] for m in (j - 1, j) if 0 <= m < b.size]
        if cand and min(abs(ai - c) for c in cand) <= tol:
            matched[k] = True
    return matched


# --------------------------------------------------------------------------- #
# Core detection
# --------------------------------------------------------------------------- #
def _empty_result(dt, n_valid, t_lo, t_hi, sig_p95):
    z = np.array([])
    zi = np.array([], dtype=int)
    return dict(onset_times=z, onset_idx=zi, peak_times=z, peak_idx=zi,
                amplitude=z, dt=dt, mad=np.nan, n_valid=int(n_valid),
                t_lo=t_lo, t_hi=t_hi, sig_p95=sig_p95, signal_ok=False)


def detect_elms(t, y, cfg: LabelConfig):
    """Detect ELMs in one rectified D-alpha trace, returning onset and peak.

    Returns
    -------
    dict with keys: ``onset_times`` / ``onset_idx`` (foot of the rise, into the
    original array), ``peak_times`` / ``peak_idx`` (burst maximum), ``amplitude``
    (peak residual height in MAD units), ``dt``, ``mad``, ``n_valid``, ``t_lo``,
    ``t_hi``, ``sig_p95``, ``signal_ok``.
    """
    t = np.asarray(t, dtype=float)
    y = np.abs(np.asarray(y, dtype=float))
    dt = float(np.median(np.diff(t)))

    mask = _valid_window(t, y, cfg)
    idx_valid = np.flatnonzero(mask)
    if idx_valid.size < 10:
        return _empty_result(dt, idx_valid.size, np.nan, np.nan, np.nan)

    tv, yv = t[mask], y[mask]

    # Dead-channel gate: a divertor whose in-window signal never rises to a
    # physical D-alpha level (e.g. an unplugged/biased-off channel sitting at the
    # noise floor) carries no ELMs. Without this, the relative MAD threshold fires
    # on pure noise (observed on the upper divertor of several shots, e.g. 132588).
    sig_p95 = float(np.percentile(yv, 95))
    if sig_p95 < cfg.min_signal_amp:
        return _empty_result(dt, idx_valid.size, float(tv[0]), float(tv[-1]), sig_p95)

    resid = yv - _baseline(yv, dt, cfg)
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-12

    distance = max(1, int(round((cfg.min_interval_ms * 1e-3) / dt)))
    peaks, props = find_peaks(
        resid,
        height=cfg.height_mad * mad,
        prominence=cfg.prominence_mad * mad,
        distance=distance,
    )
    onsets = _find_onsets(resid, peaks, dt, cfg)  # foot of each rise (event time)

    return dict(
        onset_times=tv[onsets],
        onset_idx=idx_valid[onsets],
        peak_times=tv[peaks],
        peak_idx=idx_valid[peaks],
        amplitude=props["peak_heights"] / mad,
        dt=dt,
        mad=float(mad),
        n_valid=int(idx_valid.size),
        t_lo=float(tv[0]),
        t_hi=float(tv[-1]),
        sig_p95=sig_p95,
        signal_ok=True,
    )


# --------------------------------------------------------------------------- #
# Per-shot statistics
# --------------------------------------------------------------------------- #
def shot_statistics(onset_times, t_lo, t_hi, cfg: LabelConfig) -> dict:
    """Inter-ELM interval statistics from a sorted onset-time array."""
    n = int(onset_times.size)
    out = dict(n_elms=n, t_first=np.nan, t_last=np.nan, window_s=t_hi - t_lo,
               elm_freq_hz=np.nan, mean_dt=np.nan, median_dt=np.nan, std_dt=np.nan,
               cv_dt=np.nan, min_dt=np.nan, max_dt=np.nan, weibull_k=np.nan)
    if n == 0:
        return out
    out["t_first"], out["t_last"] = float(onset_times[0]), float(onset_times[-1])
    out["elm_freq_hz"] = n / (t_hi - t_lo) if t_hi > t_lo else np.nan
    if n >= 2:
        dts = np.diff(onset_times)
        out.update(mean_dt=float(dts.mean()), median_dt=float(np.median(dts)),
                   std_dt=float(dts.std()), min_dt=float(dts.min()),
                   max_dt=float(dts.max()))
        out["cv_dt"] = out["std_dt"] / out["mean_dt"] if out["mean_dt"] else np.nan
        if n >= max(5, cfg.min_elms_stats) and _HAVE_WEIBULL:
            try:
                k, _, _ = weibull_min.fit(dts, floc=0)
                out["weibull_k"] = float(k)
            except Exception:
                pass
    return out


# --------------------------------------------------------------------------- #
# Forecaster targets (native uniform time base)
# --------------------------------------------------------------------------- #
def build_targets(t, onset_times, dt, cfg: LabelConfig) -> dict:
    """Build forward/backward time-to-ELM signals and a smoothed ELM rate.

    All arrays are on the full original time grid ``t``. Samples outside the
    valid window or beyond the last ELM have NaN where the quantity is undefined.
    """
    t = np.asarray(t, dtype=float)
    train = np.zeros_like(t)
    time_to_next = np.full_like(t, np.nan)
    time_since_last = np.full_like(t, np.nan)

    if onset_times.size:
        ot = np.sort(onset_times)
        # nearest-sample event train
        ev_idx = np.searchsorted(t, ot)
        ev_idx = np.clip(ev_idx, 0, len(t) - 1)
        train[ev_idx] = 1.0
        # forward: time until the next onset (defined up to the last ELM)
        nxt = np.searchsorted(ot, t, side="left")
        valid = nxt < ot.size
        time_to_next[valid] = ot[nxt[valid]] - t[valid]
        # backward: time since the previous onset (defined after the first ELM)
        prv = np.searchsorted(ot, t, side="right") - 1
        valid = prv >= 0
        time_since_last[valid] = t[valid] - ot[prv[valid]]

    # smoothed ELM rate via Gaussian kernel on the impulse train
    sigma = max(1, int(round((cfg.rate_kernel_ms * 1e-3) / dt)))
    half = int(3 * sigma)
    g = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    g /= g.sum() * dt  # normalize to events per second
    rate = np.convolve(train, g, mode="same")

    return dict(time=t, elm_train=train, time_to_next=time_to_next,
                time_since_last=time_since_last, elm_rate=rate)


# --------------------------------------------------------------------------- #
# Database driver
# --------------------------------------------------------------------------- #
def label_database(db, cfg: LabelConfig):
    """Label every shot; return (events_df, summary_df, targets_by_shot)."""
    events, summaries, targets = [], [], {}

    tol = cfg.coincidence_tol_ms * 1e-3
    other = {"lower": "upper", "upper": "lower"}

    for rec in db:
        shot = int(rec["shot"])
        per_div = {div: detect_elms(rec[f"time {div} dalpha"],
                                    rec[f"data {div} dalpha"], cfg)
                   for div in DIVERTORS}

        # lower/upper coincidence: only meaningful when both channels are live
        both_live = per_div["lower"]["signal_ok"] and per_div["upper"]["signal_ok"]
        matched = {}
        for div in DIVERTORS:
            if both_live:
                matched[div] = _match_onsets(
                    per_div[div]["onset_times"], per_div[other[div]]["onset_times"], tol)
            else:
                matched[div] = np.full(per_div[div]["onset_times"].size, np.nan)

        for div in DIVERTORS:
            res = per_div[div]
            for k in range(res["onset_times"].size):
                m = matched[div][k]
                events.append(dict(
                    shot=shot, divertor=div, elm_index=k,
                    onset_time=float(res["onset_times"][k]),
                    peak_time=float(res["peak_times"][k]),
                    amplitude=float(res["amplitude"][k]),
                    matched=(bool(m) if np.isfinite(m) else np.nan)))

        # per-shot stats from the primary (onset-based)
        prim = per_div[cfg.primary]
        stats = shot_statistics(prim["onset_times"], prim["t_lo"], prim["t_hi"], cfg)
        coinc_frac = (float(np.nanmean(matched[cfg.primary]))
                      if both_live and prim["onset_times"].size else np.nan)
        stats.update(shot=shot,
                     n_elms_lower=per_div["lower"]["onset_times"].size,
                     n_elms_upper=per_div["upper"]["onset_times"].size,
                     primary_p95=prim["sig_p95"], signal_ok=prim["signal_ok"],
                     both_live=both_live, coincidence_frac=coinc_frac,
                     dt_s=prim["dt"], t_lo=prim["t_lo"], t_hi=prim["t_hi"])
        summaries.append(stats)

        tgt = build_targets(np.asarray(rec[f"time {cfg.primary} dalpha"], dtype=float),
                            prim["onset_times"], prim["dt"], cfg)
        targets[shot] = tgt

    events_df = pd.DataFrame(events).sort_values(["shot", "divertor", "onset_time"])
    # interval-to-next within each (shot, divertor)
    events_df["interval_to_next"] = (
        events_df.groupby(["shot", "divertor"])["onset_time"].diff().shift(-1)
    )
    cols = ["shot", "signal_ok", "primary_p95", "n_elms", "n_elms_lower",
            "n_elms_upper", "both_live", "coincidence_frac", "elm_freq_hz",
            "median_dt", "mean_dt", "cv_dt", "min_dt", "max_dt", "weibull_k",
            "t_first", "t_last", "window_s", "t_lo", "t_hi", "dt_s"]
    summary_df = pd.DataFrame(summaries)[cols].sort_values("shot")
    return events_df, summary_df, targets


def main(argv=None):
    ap = argparse.ArgumentParser(description="Label ELMs in NSTX D-alpha database.")
    ap.add_argument("--db", default="data/dalpha_db.pkl")
    ap.add_argument("--out", default="data/labels")
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--height-mad", type=float, default=6.0)
    ap.add_argument("--min-interval-ms", type=float, default=1.5)
    ap.add_argument("--primary", choices=DIVERTORS, default="lower")
    ap.add_argument("--save-targets", action="store_true",
                    help="write per-shot target arrays as .npz")
    args = ap.parse_args(argv)

    cfg = LabelConfig(t_min=args.t_min, t_max=args.t_max,
                      height_mad=args.height_mad,
                      min_interval_ms=args.min_interval_ms, primary=args.primary)

    with open(args.db, "rb") as f:
        db = pickle.load(f)

    events_df, summary_df, targets = label_database(db, cfg)

    os.makedirs(args.out, exist_ok=True)
    events_df.to_csv(os.path.join(args.out, "elm_events.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out, "elm_shot_summary.csv"), index=False)
    if args.save_targets:
        tdir = os.path.join(args.out, "targets")
        os.makedirs(tdir, exist_ok=True)
        for shot, tgt in targets.items():
            np.savez_compressed(os.path.join(tdir, f"{shot}.npz"), **tgt)

    tot = int(summary_df["n_elms"].sum())
    print(f"shots: {len(summary_df)}   total ELMs (primary={cfg.primary}): {tot}")
    print(f"median ELMs/shot: {summary_df['n_elms'].median():.0f}   "
          f"median freq: {summary_df['elm_freq_hz'].median():.1f} Hz")
    print(f"written: {args.out}/elm_events.csv, elm_shot_summary.csv"
          + ("  + targets/*.npz" if args.save_targets else ""))
    return events_df, summary_df, targets


if __name__ == "__main__":
    main()
