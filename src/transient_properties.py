"""
Discovery-notable light curve properties.

These are the quantities an astronomer (or a broker filter) would read off a
light curve to decide whether a source is worth following up: how bright it
got, when it peaked, how fast it rose and faded, how long it stayed bright,
and what colour it was. The evaluation pipeline computes them *identically*
from the original light curve and from the light curve decoded out of the
latent, so the difference is a direct measure of what the compression lost.

All property functions operate on physical-unit arrays:
    t        (L,) days since first detection
    mag      (L,) PSF magnitude
    mag_err  (L,) magnitude uncertainty
    band     (L,) integer band index (0=g, 1=r, 2=i)

Conversion helpers translate to/from the model's native feature layout
    [log1p(dt), log1p(dt_prev), logflux, logflux_err, band_g, band_r, band_i, ...]
where logflux = -0.4 * mag (see preprocess_alerts.py). Extra channels beyond
the first seven (alert metadata) are carried through untouched.
"""

import numpy as np

BAND_G, BAND_R, BAND_I = 0, 1, 2
BAND_NAMES = {BAND_G: "g", BAND_R: "r", BAND_I: "i"}

# 2.5 * log10(2): magnitude difference corresponding to a factor 2 in flux.
HALF_FLUX_MAG = 2.5 * np.log10(2.0)

PROPERTY_NAMES = [
    "duration",
    "mag_first",
    "peak_mag",
    "t_peak",
    "rise_rate",
    "decline_rate",
    "amplitude",
    "t_half_width",
    "color_peak",
    "mean_color",
]

PROPERTY_UNITS = {
    "duration": "days",
    "mag_first": "mag",
    "peak_mag": "mag",
    "t_peak": "days",
    "rise_rate": "mag/day",
    "decline_rate": "mag/day",
    "amplitude": "mag",
    "t_half_width": "days",
    "color_peak": "mag",
    "mean_color": "mag",
}

PROPERTY_DESCRIPTIONS = {
    "duration": "time span between first and last detection",
    "mag_first": "discovery magnitude (first detection)",
    "peak_mag": "brightest magnitude in any band",
    "t_peak": "time of peak brightness since first detection",
    "rise_rate": "pre-peak brightening rate in the peak band (+ = brightening)",
    "decline_rate": "post-peak fading rate in the peak band (+ = fading)",
    "amplitude": "max - min magnitude in the peak band",
    "t_half_width": "time the peak band stays within a factor 2 of peak flux",
    "color_peak": "g - r within +/- color_window days of peak",
    "mean_color": "mean g - mean r over the whole light curve",
}

# Tolerances used for the "fraction recovered within tolerance" metric.
# Roughly: photometric-noise-level for magnitudes, ~one cadence gap for times.
PROPERTY_TOLERANCES = {
    "duration": 2.0,
    "mag_first": 0.2,
    "peak_mag": 0.2,
    "t_peak": 2.0,
    "rise_rate": 0.05,
    "decline_rate": 0.02,
    "amplitude": 0.2,
    "t_half_width": 3.0,
    "color_peak": 0.1,
    "mean_color": 0.1,
}


# ---------------------------------------------------------------------------
# Native feature layout <-> physical units
# ---------------------------------------------------------------------------


def _apply_norm(cont, norm, inverse):
    """Apply (or undo) an affine normalization on the 4 continuous channels."""
    if norm is None:
        return cont
    mean, std = norm
    mean = np.asarray(mean, dtype=np.float64).flatten()[:4]
    std = np.asarray(std, dtype=np.float64).flatten()[:4]
    if inverse:
        return cont * (std + 1e-8) + mean
    return (cont - mean) / (std + 1e-8)


def x_to_physical(x, norm=None):
    """Convert native features (L, C>=7) to (t, mag, mag_err, band).

    Args:
        x: array-like (L, C). Only the first 7 channels are read.
        norm: optional (mean, std) applied to the continuous channels during
            preprocessing (PhotoNPZDataset with stats). None = identity.

    Returns:
        t (days since first detection), mag, mag_err, band — all (L,) float64
        except band which is int64. Reconstructed inputs may carry slightly
        negative log-times; they are clipped to zero.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 7:
        raise ValueError(f"expected (L, C>=7) features, got shape {x.shape}")
    cont = _apply_norm(x[:, :4], norm, inverse=True)
    t = np.expm1(np.clip(cont[:, 0], 0.0, None))
    mag = -2.5 * cont[:, 2]
    mag_err = 2.5 * np.abs(cont[:, 3])
    band = x[:, 4:7].argmax(axis=1).astype(np.int64)
    return t, mag, mag_err, band


def physical_to_x(t, mag, mag_err, band, extra=None, norm=None):
    """Build native features from physical arrays.

    Args:
        t: (L,) observation times in days; must be sorted ascending. Times are
            re-referenced to t[0] so the first row has dt=0.
        mag, mag_err: (L,) magnitudes and uncertainties
        band: (L,) integer band index in {0, 1, 2}
        extra: optional (L, K) extra channels (alert metadata) appended as-is
        norm: optional (mean, std) to apply to the continuous channels

    Returns:
        x: (L, 7 + K) float32
    """
    t = np.asarray(t, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    mag_err = np.asarray(mag_err, dtype=np.float64)
    band = np.asarray(band, dtype=np.int64)
    L = len(t)
    if L == 0:
        raise ValueError("cannot build features from an empty light curve")
    if np.any(np.diff(t) < 0):
        raise ValueError("times must be sorted ascending")

    dt = t - t[0]
    dt_prev = np.zeros(L)
    dt_prev[1:] = np.diff(t)
    cont = np.column_stack([np.log1p(dt), np.log1p(dt_prev), -0.4 * mag, 0.4 * mag_err])
    cont = _apply_norm(cont, norm, inverse=False)
    one_hot = np.eye(3)[np.clip(band, 0, 2)]
    parts = [cont, one_hot]
    if extra is not None:
        extra = np.asarray(extra, dtype=np.float64)
        if extra.shape[0] != L:
            raise ValueError("extra channels must have one row per observation")
        parts.append(extra)
    return np.concatenate(parts, axis=1).astype(np.float32)


def set_photometry(x, mag, mag_err, norm=None):
    """Return a copy of x with the logflux / logflux_err channels replaced.

    Time channels, band one-hot and any metadata channels are untouched, so
    this is the right primitive for injecting flux into an existing cadence.
    """
    x = np.array(x, dtype=np.float32, copy=True)
    cont = _apply_norm(x[:, :4].astype(np.float64), norm, inverse=True)
    cont[:, 2] = -0.4 * np.asarray(mag, dtype=np.float64)
    cont[:, 3] = 0.4 * np.asarray(mag_err, dtype=np.float64)
    x[:, :4] = _apply_norm(cont, norm, inverse=False).astype(np.float32)
    return x


def recon_to_x(cont_hat, band_logits):
    """Assemble decoder outputs into native (B, L, 7) features (numpy).

    Args:
        cont_hat: (B, L, 4) tensor/array of reconstructed continuous channels
        band_logits: (B, L, 3) tensor/array of band logits
    """
    cont = _to_numpy(cont_hat).astype(np.float32)
    logits = _to_numpy(band_logits)
    band = logits.argmax(axis=-1)
    one_hot = np.eye(3, dtype=np.float32)[band]
    return np.concatenate([cont, one_hot], axis=-1)


def _to_numpy(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _slope(t, m, min_points=3, min_span=1.0):
    """Least-squares slope of m vs t (mag/day); NaN if too few points."""
    if len(t) < min_points:
        return np.nan
    span = t.max() - t.min()
    if span < min_span:
        return np.nan
    tc = t - t.mean()
    denom = np.sum(tc * tc)
    if denom <= 0:
        return np.nan
    return float(np.sum(tc * (m - m.mean())) / denom)


def compute_properties(t, mag, mag_err, band, color_window=7.0):
    """Compute discovery-notable properties of one light curve.

    The input does not need to be sorted; observations are sorted by time
    first so the result is invariant to row order (important for decoded
    light curves whose reconstructed times need not be monotonic).

    Returns:
        dict name -> float, NaN where a property is undefined (e.g. colour
        without both g and r detections, or a slope with < 3 points).
    """
    t = np.asarray(t, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    band = np.asarray(band, dtype=np.int64)
    out = {name: np.nan for name in PROPERTY_NAMES}
    if len(t) == 0:
        return out

    order = np.argsort(t, kind="stable")
    t, mag, band = t[order], mag[order], band[order]
    t = t - t[0]

    out["duration"] = float(t[-1] - t[0])
    out["mag_first"] = float(mag[0])

    ipk = int(np.argmin(mag))
    peak_mag = float(mag[ipk])
    t_pk = float(t[ipk])
    out["peak_mag"] = peak_mag
    out["t_peak"] = t_pk

    # Everything shape-related is measured in the band of the peak detection so
    # we never mix bands with different zero-points.
    pb = band[ipk]
    sel = band == pb
    tb, mb = t[sel], mag[sel]
    out["amplitude"] = float(mb.max() - mb.min())

    above = mb <= peak_mag + HALF_FLUX_MAG
    out["t_half_width"] = float(tb[above].max() - tb[above].min())

    pre = tb <= t_pk
    post = tb >= t_pk
    rise = _slope(tb[pre], mb[pre])
    out["rise_rate"] = -rise if np.isfinite(rise) else np.nan
    out["decline_rate"] = _slope(tb[post], mb[post])

    g = band == BAND_G
    r = band == BAND_R
    near = np.abs(t - t_pk) <= color_window
    if np.any(g & near) and np.any(r & near):
        out["color_peak"] = float(mag[g & near].mean() - mag[r & near].mean())
    if np.any(g) and np.any(r):
        out["mean_color"] = float(mag[g].mean() - mag[r].mean())

    return out


def properties_from_x(x, valid=None, norm=None, color_window=7.0):
    """Properties of one light curve in native feature layout.

    Args:
        x: (L, C>=7) features
        valid: optional (L,) boolean mask of real (non-padding) rows
        norm: optional (mean, std) normalization of the continuous channels
    """
    x = np.asarray(x)
    if valid is not None:
        x = x[np.asarray(valid, dtype=bool)]
    t, mag, mag_err, band = x_to_physical(x, norm)
    return compute_properties(t, mag, mag_err, band, color_window=color_window)


def properties_batch(x_batch, valid_batch=None, norm=None, color_window=7.0):
    """Properties for a batch: (B, L, C) -> (B, len(PROPERTY_NAMES)) float array."""
    x_batch = _to_numpy(x_batch)
    if valid_batch is not None:
        valid_batch = _to_numpy(valid_batch).astype(bool)
    rows = []
    for i in range(x_batch.shape[0]):
        valid = valid_batch[i] if valid_batch is not None else None
        p = properties_from_x(
            x_batch[i], valid=valid, norm=norm, color_window=color_window
        )
        rows.append([p[name] for name in PROPERTY_NAMES])
    return np.array(rows, dtype=np.float64).reshape(-1, len(PROPERTY_NAMES))
