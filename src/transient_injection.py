"""
Transient injection, null re-noising, source selection and synthetic data.

Three jobs, all in physical units and all pure numpy:

1. **Inject** a parametric transient (SN Ia / SN II / TDE / kilonova / flare
   presets) into an existing light curve *at its existing cadence*. Fluxes are
   added in linear flux space, magnitude errors are rescaled for the higher
   flux, and time / band / metadata channels are left untouched, so the
   injected sequence is a drop-in replacement for the model input.

2. **Select** subsets of a dataset: transients vs variables by label, single
   classes, or property thresholds (e.g. ``"amplitude>1.0"``).

3. **Synthesize** a small labelled dataset (SN Ia, SN II, TDE, CV outbursts,
   AGN random walk) in the NPZ layout produced by ``preprocess_alerts.py`` so
   the whole evaluation pipeline can run without access to real ZTF data.

Usage (synthetic dataset):
    python src/transient_injection.py --output-dir data_synth --n-per-class 400
"""

import argparse
import os
from pathlib import Path

import numpy as np

from transient_properties import (
    BAND_G,
    BAND_R,
    PROPERTY_NAMES,
    physical_to_x,
    set_photometry,
    x_to_physical,
)

# ---------------------------------------------------------------------------
# Transient shapes
# ---------------------------------------------------------------------------

# Timescale / colour ranges sampled uniformly per injection. Bazin rise/fall
# times are in days; colour is the transient's g - r at peak.
TRANSIENT_PRESETS = {
    "sn_ia": {
        "model": "bazin",
        "tau_rise": (3.0, 8.0),
        "tau_fall": (15.0, 40.0),
        "color": (-0.2, 0.4),
    },
    "sn_ii": {
        "model": "bazin",
        "tau_rise": (2.0, 6.0),
        "tau_fall": (40.0, 100.0),
        "color": (-0.1, 0.5),
    },
    "tde": {
        "model": "bazin",
        "tau_rise": (8.0, 25.0),
        "tau_fall": (40.0, 120.0),
        "color": (-0.6, -0.1),
    },
    "kilonova": {
        "model": "bazin",
        "tau_rise": (0.3, 1.0),
        "tau_fall": (1.5, 4.0),
        "color": (0.3, 1.5),
    },
    "flare": {
        "model": "gaussian",
        "sigma": (0.3, 2.0),
        "color": (-0.5, 0.0),
    },
}


def bazin_peak_offset(tau_rise, tau_fall):
    """Time from the Bazin reference t0 to the peak of the (unnormalized) shape."""
    return tau_rise * np.log(tau_fall / tau_rise - 1.0)


def _bazin_raw(s, tau_rise, tau_fall):
    return np.exp(-s / tau_fall) / (1.0 + np.exp(-s / tau_rise))


def transient_shape(t, params):
    """Normalized (peak = 1) transient flux at times t (days).

    params must contain "model" and "t_peak" plus model-specific keys:
        bazin:    tau_rise, tau_fall (tau_fall > tau_rise)
        gaussian: sigma
    """
    t = np.asarray(t, dtype=np.float64)
    model = params["model"]
    t_peak = params["t_peak"]
    if model == "bazin":
        tau_rise = float(params["tau_rise"])
        tau_fall = float(params["tau_fall"])
        if tau_fall <= tau_rise:
            raise ValueError("bazin requires tau_fall > tau_rise")
        t0 = t_peak - bazin_peak_offset(tau_rise, tau_fall)
        peak = _bazin_raw(bazin_peak_offset(tau_rise, tau_fall), tau_rise, tau_fall)
        return _bazin_raw(t - t0, tau_rise, tau_fall) / peak
    if model == "gaussian":
        sigma = float(params["sigma"])
        return np.exp(-0.5 * ((t - t_peak) / sigma) ** 2)
    raise ValueError(f"unknown transient model: {model}")


def sample_transient_params(preset, rng, t_peak=0.0, **overrides):
    """Draw concrete shape parameters from a preset's ranges."""
    if preset not in TRANSIENT_PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}; choose from {sorted(TRANSIENT_PRESETS)}"
        )
    spec = TRANSIENT_PRESETS[preset]
    params = {"preset": preset, "model": spec["model"], "t_peak": float(t_peak)}
    for key, val in spec.items():
        if key == "model":
            continue
        if key in overrides:
            params[key] = float(overrides[key])
        else:
            lo, hi = val
            params[key] = float(rng.uniform(lo, hi))
    for key, val in overrides.items():
        if key not in params:
            params[key] = float(val)
    if params["model"] == "bazin" and params["tau_fall"] <= params["tau_rise"]:
        params["tau_fall"] = 1.5 * params["tau_rise"]
    return params


def peak_mag_per_band(peak_mag_r, color):
    """Per-band peak magnitudes from an r-band peak and a g - r colour.

    The i band is placed half-way in colour space (r - i ~ 0.5 * (g - r)).
    """
    return np.array(
        [peak_mag_r + color, peak_mag_r, peak_mag_r - 0.5 * color], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Injection into an existing light curve
# ---------------------------------------------------------------------------


def host_reference_mag(x, norm=None, band=BAND_R):
    """Median magnitude of the host light curve in `band` (any band if absent)."""
    _, mag, _, bands = x_to_physical(x, norm)
    sel = bands == band
    if not np.any(sel):
        sel = np.ones_like(sel, dtype=bool)
    return float(np.median(mag[sel]))


def inject_transient(x, peak_mag, params, norm=None, rng=None, add_noise=False):
    """Add a transient to a light curve at its existing epochs.

    Args:
        x: (L, C>=7) native features of the host light curve
        peak_mag: r-band peak magnitude of the injected transient
        params: shape parameters from `sample_transient_params` (t_peak is
            measured in days since the host's first detection)
        norm: optional (mean, std) normalization of the continuous channels
        rng: numpy Generator/RandomState, only used when add_noise=True
        add_noise: re-draw photometric noise on the combined light curve

    Returns:
        x_new: (L, C) float32 copy of x with logflux / logflux_err replaced
        info: dict with the injected parameters, the peak signal-to-noise of
            the injection relative to the host's photometric noise, and the
            index of the epoch closest to the injected peak.
    """
    t, mag, mag_err, band = x_to_physical(x, norm)
    shape = transient_shape(t, params)
    pm = peak_mag_per_band(peak_mag, params.get("color", 0.0))
    f_tr = 10.0 ** (-0.4 * pm[band]) * shape
    f_host = 10.0 ** (-0.4 * mag)
    f_new = f_host + f_tr
    mag_new = -2.5 * np.log10(f_new)
    # Constant flux noise (sky-limited): sigma_mag scales as 1 / flux.
    err_new = mag_err * f_host / f_new
    if add_noise:
        if rng is None:
            rng = np.random.default_rng()
        mag_new = mag_new + rng.normal(0.0, err_new)

    sigma_flux = f_host * mag_err / 1.0857
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(sigma_flux > 0, f_tr / sigma_flux, np.inf)
    i_peak = int(np.argmin(np.abs(t - params["t_peak"])))

    info = dict(params)
    info.update(
        {
            "peak_mag_r": float(peak_mag),
            "snr_peak": float(np.max(snr)),
            "delta_mag_at_peak_epoch": float(mag[i_peak] - mag_new[i_peak]),
            "i_peak_epoch": i_peak,
        }
    )
    return set_photometry(x, mag_new, err_new, norm), info


def renoise(x, rng, norm=None):
    """Re-draw photometric noise: a null perturbation of the same light curve."""
    _, mag, mag_err, _ = x_to_physical(x, norm)
    return set_photometry(x, mag + rng.normal(0.0, mag_err), mag_err, norm)


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

COARSE_NAMES = {0: "SNIa", 1: "SNcc", 2: "Cataclysmic", 3: "AGN", 4: "TDE"}
COARSE_IDS = {v.lower(): k for k, v in COARSE_NAMES.items()}
TRANSIENT_COARSE = (0, 1, 4)
VARIABLE_COARSE = (2, 3)

_OPS = {
    "<=": np.less_equal,
    ">=": np.greater_equal,
    "<": np.less,
    ">": np.greater,
    "==": np.equal,
    "!=": np.not_equal,
}


def _property_filter(term, props):
    for op in ("<=", ">=", "!=", "==", "<", ">"):
        if op in term:
            name, value = term.split(op, 1)
            name = name.strip()
            if name not in PROPERTY_NAMES:
                raise ValueError(f"unknown property {name!r} in filter {term!r}")
            if props is None:
                raise ValueError(f"filter {term!r} needs computed properties")
            col = props[:, PROPERTY_NAMES.index(name)]
            with np.errstate(invalid="ignore"):
                return np.isfinite(col) & _OPS[op](col, float(value))
    raise ValueError(f"cannot parse selection term {term!r}")


def select_mask(selection, labels_coarse=None, labels_fine=None, props=None):
    """Boolean mask over sources for a comma-separated selection string.

    Terms (AND-ed together):
        all                 everything
        transient           coarse label in {SNIa, SNcc, TDE}
        variable            coarse label in {Cataclysmic, AGN}
        coarse:<name|id>    one coarse class (e.g. coarse:AGN, coarse:3)
        fine:<id>           one fine class id
        <prop><op><value>   property threshold, e.g. amplitude>1.0, t_peak<=10
    """
    n = None
    for arr in (labels_coarse, labels_fine, props):
        if arr is not None:
            n = len(arr)
            break
    if n is None:
        raise ValueError("select_mask needs labels or properties")
    mask = np.ones(n, dtype=bool)
    if labels_coarse is not None:
        labels_coarse = np.asarray(labels_coarse)
    if labels_fine is not None:
        labels_fine = np.asarray(labels_fine)

    for term in str(selection).split(","):
        term = term.strip()
        if not term or term == "all":
            continue
        if term in ("transient", "variable"):
            if labels_coarse is None:
                raise ValueError(f"selection {term!r} needs coarse labels")
            ids = TRANSIENT_COARSE if term == "transient" else VARIABLE_COARSE
            mask &= np.isin(labels_coarse, ids)
        elif term.startswith("coarse:"):
            if labels_coarse is None:
                raise ValueError("coarse: selection needs coarse labels")
            key = term.split(":", 1)[1].strip()
            cid = int(key) if key.lstrip("-").isdigit() else COARSE_IDS[key.lower()]
            mask &= labels_coarse == cid
        elif term.startswith("fine:"):
            if labels_fine is None:
                raise ValueError("fine: selection needs fine labels")
            mask &= labels_fine == int(term.split(":", 1)[1])
        else:
            mask &= _property_filter(term, props)
    return mask


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

MAG_LIMIT = 20.5
MAG_ERR_FLOOR = 0.03
MAG_ERR_CEIL = 1.0
# An epoch only counts as a detection if it is above the magnitude limit
# *and* reasonably measured (roughly SNR >= 2). Without this, the steeply
# rising noise model lets a wild noise draw on a far-below-limit epoch masquerade
# as an ultra-bright detection.
MAX_DETECTION_ERR = 0.5

# Synthetic class -> fine label used by the ZTF datasets
# (0 SN Ia, 3 SN II, 7 Cataclysmic, 8 AGN, 9 TDE; see AlertDataset.COARSE_MAP)
SYNTH_CLASSES = {"snia": 0, "snii": 3, "tde": 9, "cv": 7, "agn": 8}


def noise_model(mag):
    """ZTF-like magnitude error as a function of magnitude (clipped to a sane range)."""
    err = 0.1 * 10.0 ** (
        0.4 * (np.clip(np.asarray(mag, dtype=np.float64), -5.0, 40.0) - 19.0)
    )
    return np.clip(err, MAG_ERR_FLOOR, MAG_ERR_CEIL)


def ztf_cadence(rng, n_obs, span_days, mean_gap=2.5, p_g=0.45):
    """Random ZTF-like cadence: exponential gaps, g/r band assignment."""
    times = np.cumsum(rng.exponential(mean_gap, n_obs))
    times = times[times <= span_days]
    if len(times) < 5:
        times = np.sort(rng.uniform(0.0, span_days, max(5, n_obs // 2)))
    bands = rng.choice([BAND_G, BAND_R], size=len(times), p=[p_g, 1.0 - p_g])
    return times, bands


def _drw(times, rng, mag0, sigma, tau):
    """Damped random walk in magnitude (AGN-like stochastic variability)."""
    mags = np.empty(len(times))
    mags[0] = mag0 + rng.normal(0.0, sigma)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        a = np.exp(-dt / tau)
        mags[i] = (
            mag0
            + a * (mags[i - 1] - mag0)
            + rng.normal(0.0, sigma * np.sqrt(1.0 - a * a))
        )
    return mags


def synth_light_curve(kind, rng):
    """One synthetic light curve in physical units, after the detection cut.

    Returns (t, mag, mag_err, band, truth) or None if fewer than 3 epochs
    survive the magnitude limit. `truth` holds the generating parameters.
    """
    span = rng.uniform(40.0, 150.0)
    times, bands = ztf_cadence(rng, int(rng.integers(20, 80)), span)
    truth = {"kind": kind}

    if kind in ("snia", "snii", "tde"):
        preset = {"snia": "sn_ia", "snii": "sn_ii", "tde": "tde"}[kind]
        params = sample_transient_params(preset, rng, t_peak=rng.uniform(5.0, 40.0))
        peak_mag = rng.uniform(16.5, 19.5)
        pm = peak_mag_per_band(peak_mag, params["color"])
        flux = 10.0 ** (-0.4 * pm[bands]) * transient_shape(times, params)
        truth.update(params, peak_mag_r=peak_mag)
    elif kind == "cv":
        quiescent = rng.uniform(19.5, 21.5)
        flux = np.full(len(times), 10.0 ** (-0.4 * quiescent))
        n_out = int(rng.integers(1, 4))
        for _ in range(n_out):
            params = sample_transient_params(
                "kilonova",
                rng,
                t_peak=rng.uniform(0.0, span),
                color=rng.uniform(-0.3, 0.2),
            )
            params["tau_rise"] = rng.uniform(0.5, 2.0)
            params["tau_fall"] = rng.uniform(3.0, 12.0)
            pm = peak_mag_per_band(rng.uniform(15.5, 18.5), params["color"])
            flux = flux + 10.0 ** (-0.4 * pm[bands]) * transient_shape(times, params)
        truth.update(n_outbursts=n_out, quiescent_mag=quiescent)
    elif kind == "agn":
        mag0 = rng.uniform(17.0, 20.0)
        sigma = rng.uniform(0.15, 0.5)
        tau = rng.uniform(20.0, 150.0)
        mags = _drw(times, rng, mag0, sigma, tau)
        color = rng.uniform(0.0, 0.4)
        mags = mags + np.where(bands == BAND_G, color, 0.0)
        flux = 10.0 ** (-0.4 * mags)
        truth.update(mag0=mag0, sigma=sigma, tau=tau, color=color)
    else:
        raise ValueError(f"unknown synthetic class {kind!r}")

    mag_true = -2.5 * np.log10(np.clip(flux, 1e-12, None))
    mag_err = noise_model(mag_true)
    mag_obs = mag_true + rng.normal(0.0, mag_err)
    det = (mag_obs < MAG_LIMIT) & (mag_err <= MAX_DETECTION_ERR)
    if det.sum() < 3:
        return None
    t = times[det]
    t = t - t[0]
    return t, mag_obs[det], mag_err[det], bands[det], truth


def make_synthetic_dataset(
    output_dir,
    n_per_class=200,
    seed=0,
    classes=None,
    splits=(0.7, 0.15, 0.15),
    n_channels=37,
):
    """Write a synthetic dataset in preprocess_alerts.py NPZ layout.

    Files land in {output_dir}/{train,val,test}/{obj_id}.npz with keys
    ``x`` (L, n_channels) and ``label`` (fine label). Metadata channels are
    zero (no alert metadata for synthetic sources) and no stamps are written.

    Returns:
        dict split -> number of files written
    """
    classes = list(classes or SYNTH_CLASSES)
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    for kind in classes:
        label = SYNTH_CLASSES[kind]
        n_done, n_attempts = 0, 0
        while n_done < n_per_class and n_attempts < 20 * n_per_class:
            n_attempts += 1
            lc = synth_light_curve(kind, rng)
            if lc is None:
                continue
            t, mag, mag_err, band, _ = lc
            extra = np.zeros((len(t), n_channels - 7), dtype=np.float32)
            x = physical_to_x(t, mag, mag_err, band, extra=extra)
            u = rng.uniform()
            if u < splits[0]:
                split = "train"
            elif u < splits[0] + splits[1]:
                split = "val"
            else:
                split = "test"
            oid = f"SYN_{kind}_{n_done:05d}"
            np.savez(output_dir / split / f"{oid}.npz", x=x, label=np.int64(label))
            counts[split] += 1
            n_done += 1
    return counts


def _cli():
    parser = argparse.ArgumentParser(description="Generate a synthetic NPZ dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(SYNTH_CLASSES),
        choices=list(SYNTH_CLASSES),
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=37,
        help="Feature channels (37 = photometry + alert metadata layout, 7 = photometry only)",
    )
    args = parser.parse_args()
    counts = make_synthetic_dataset(
        args.output_dir,
        n_per_class=args.n_per_class,
        seed=args.seed,
        classes=args.classes,
        n_channels=args.n_channels,
    )
    for split, n in counts.items():
        print(f"  {split}: {n} files")
    print(f"Wrote {sum(counts.values())} sources to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    _cli()
