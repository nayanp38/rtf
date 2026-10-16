"""
Discovery-preservation evaluation of a trained RTF compressor.

Answers: does the latent fingerprint keep the information needed to *discover*
and characterise a transient? Three stages, all run on the eval split:

  recon   Decode every latent and re-measure the discovery properties (peak
          mag, t_peak, rise/decline rate, amplitude, half-max width, colours)
          on the decoded light curve. Agreement with the original curve is
          what survives compression *and* is recoverable by the decoder.

  probe   Fit linear (or MLP) probes from the latent to each property on the
          train split and score them on the eval split. A property can be
          preserved in the latent even when the decoder does not render it,
          so probe error <= reconstruction error is the expected ordering.
          Also probes binary discovery flags (transient vs variable, fast
          evolver).

  inject  Injection-recovery: add a parametric transient of controlled
          amplitude to host light curves, re-encode, and measure
            (a) latent displacement relative to a re-noised null,
            (b) detection completeness at a fixed false-positive rate,
            (c) a held-out probe's AUC for injected vs host, and
            (d) whether the injected peak is recovered in the decoded curve.

Usage:
    cd src

    # Trained model on real preprocessed data
    python eval_discovery.py \
        --checkpoint ../runs/ae_dim64_meta_gp_randtrunc/best_model.pt \
        --summary ../runs/ae_dim64_meta_gp_randtrunc/summary.json \
        --data-dir ../data_gp \
        --output-dir ../analysis/discovery \
        --inject-preset sn_ia --host-select variable

    # Pipeline smoke test: synthetic data + untrained model
    python eval_discovery.py --synthetic-dir ../data_synth --synthetic-n 50 \
        --latent-dim 16 --output-dir ../analysis/discovery_smoke
"""

import argparse
import json
import math
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import MetaNPZDataset, PhotoNPZDataset, collate_fn
from model import LightCurveCompressor
from transient_injection import (
    COARSE_NAMES,
    TRANSIENT_COARSE,
    TRANSIENT_PRESETS,
    VARIABLE_COARSE,
    host_reference_mag,
    inject_transient,
    make_synthetic_dataset,
    renoise,
    sample_transient_params,
    select_mask,
)
from transient_properties import (
    BAND_NAMES,
    PROPERTY_NAMES,
    PROPERTY_TOLERANCES,
    PROPERTY_UNITS,
    properties_from_x,
    recon_to_x,
    x_to_physical,
)

matplotlib.use("Agg")

BAND_COLORS = {0: "#1f77b4", 1: "#d62728", 2: "#ff7f0e"}
STAGES = ("recon", "probe", "inject")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def infer_config_from_state_dict(sd):
    """Recover LightCurveCompressor constructor args from a state dict.

    Everything except n_heads (which does not change parameter shapes) can be
    read off tensor shapes, so a checkpoint is loadable without summary.json.
    """
    cfg = {}
    if "bottleneck_proj.weight" in sd:
        cfg["mode"] = "ae"
        cfg["latent_dim"] = int(sd["bottleneck_proj.weight"].shape[0])
    elif "mu_proj.weight" in sd:
        cfg["mode"] = "vae"
        cfg["latent_dim"] = int(sd["mu_proj.weight"].shape[0])
    elif "pre_vq_proj.weight" in sd:
        cfg["mode"] = "vqvae"
        cfg["latent_dim"] = int(sd["pre_vq_proj.weight"].shape[0])
        cfg["num_codes"] = int(sd["vq.codebook.weight"].shape[0])
    else:
        raise ValueError("unrecognized checkpoint: no bottleneck projection found")

    cfg["in_channels"] = int(sd["in_proj.weight"].shape[1])
    cfg["d_model"] = int(sd["in_proj.weight"].shape[0])
    cfg["d_ff"] = int(sd["encoder.layers.0.linear1.weight"].shape[0])
    cfg["max_len"] = int(sd["pos_embed"].shape[1])

    def _n_layers(prefix):
        ids = [int(k.split(".")[2]) for k in sd if k.startswith(prefix)]
        return 1 + max(ids) if ids else 0

    cfg["enc_layers"] = _n_layers("encoder.layers.")
    cfg["dec_layers"] = _n_layers("decoder.layers.")
    cfg["gp_dim"] = (
        int(sd["gp_proj.0.weight"].shape[1]) if "gp_proj.0.weight" in sd else 0
    )
    cfg["num_classes"] = (
        int(sd["cls_head.3.weight"].shape[0]) if "cls_head.3.weight" in sd else 0
    )
    cfg["use_images"] = any(k.startswith("image_tower.") for k in sd)
    cfg["image_backbone"] = (
        "zoobot" if any(k.startswith("image_tower.cnn.stem") for k in sd) else "simple"
    )
    return cfg


def build_model(cfg, device="cpu"):
    model = LightCurveCompressor(
        mode=cfg["mode"],
        latent_dim=cfg["latent_dim"],
        in_channels=cfg.get("in_channels", 7),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 8),
        enc_layers=cfg.get("enc_layers", 4),
        dec_layers=cfg.get("dec_layers", 2),
        d_ff=cfg.get("d_ff", 512),
        max_len=cfg.get("max_len", 257),
        num_codes=cfg.get("num_codes", 512),
        use_images=cfg.get("use_images", False),
        image_backbone=cfg.get("image_backbone", "simple"),
        freeze_backbone=cfg.get("freeze_backbone"),
        gp_dim=cfg.get("gp_dim", 0),
        num_classes=cfg.get("num_classes", 0),
    )
    return model.to(device).eval()


def load_model(checkpoint, summary_path=None, device="cpu", overrides=None):
    """Load a trained compressor; config inferred from weights (+ summary.json)."""
    sd = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = infer_config_from_state_dict(sd)
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for key in ("n_heads", "image_backbone", "freeze_backbone"):
            if summary.get(key) is not None:
                cfg[key] = summary[key]
    for key, val in (overrides or {}).items():
        if val is not None:
            cfg[key] = val
    model = build_model(cfg, device)
    model.load_state_dict(sd)
    return model, cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def detect_npz_format(split_dir):
    """'meta' for preprocess_alerts.py NPZs (key x), 'photo' for legacy (key data)."""
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"no NPZ files in {split_dir}")
    with np.load(os.path.join(split_dir, files[0]), allow_pickle=True) as npz:
        keys = set(npz.keys())
    if "x" in keys:
        return "meta"
    if "data" in keys:
        return "photo"
    raise ValueError(f"unrecognized NPZ layout in {split_dir}: keys={sorted(keys)}")


def build_dataset(data_dir, split, cfg, max_detections=None, horizon=100.0):
    """Dataset for one split, matching the model's input layout.

    Returns (dataset, norm) where norm is (mean, std) applied to the
    continuous channels by the dataset (PhotoNPZDataset with stats) or None.
    """
    split_dir = os.path.join(data_dir, split)
    fmt = detect_npz_format(split_dir)
    if fmt == "meta":
        ds = MetaNPZDataset(
            split_dir,
            use_images=cfg.get("use_images", False),
            use_gp=cfg.get("gp_dim", 0) > 0,
            max_detections=max_detections,
        )
        norm = None
    else:
        if max_detections is not None:
            warnings.warn("--max-detections is ignored for photometry-only NPZ data")
        stats_path = os.path.join(data_dir, "feature_stats_day100.npz")
        if not os.path.exists(stats_path):
            stats_path = None
        ds = PhotoNPZDataset(
            split_dir,
            stats_path,
            horizon=horizon,
            use_metadata=cfg.get("in_channels", 7) == 13,
        )
        norm = (ds.mean, ds.std) if ds.mean is not None else None
    if ds.in_channels != cfg.get("in_channels", ds.in_channels):
        raise ValueError(
            f"model expects {cfg['in_channels']} input channels but {split_dir} "
            f"provides {ds.in_channels}"
        )
    return ds, norm


def iter_item_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield collate_fn(items[i : i + batch_size])


def _extra_args(batch, device):
    images = batch["images"].to(device) if "images" in batch else None
    gp = batch["gp_features"].to(device) if "gp_features" in batch else None
    return images, gp


# ---------------------------------------------------------------------------
# Encoding + property measurement
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_batches(model, batches, device="cpu", norm=None, keep_recon=False):
    """Encode, decode and measure properties for an iterable of collated batches.

    Returns a dict of arrays: z (N, D), props_orig / props_recon (N, P),
    recon_err (N,), labels_fine / labels_coarse (N,), n_det (N,), obj_ids
    (list) and, if keep_recon, recon_curves (list of (L, 7) native arrays of
    the decoded valid rows).
    """
    model.eval()
    out = {
        "z": [],
        "props_orig": [],
        "props_recon": [],
        "recon_err": [],
        "labels_fine": [],
        "labels_coarse": [],
        "n_det": [],
        "obj_ids": [],
    }
    recon_curves = []
    for batch in batches:
        x = batch["x"].to(device)
        pad = batch["pad_mask"].to(device)
        images, gp = _extra_args(batch, device)
        z = model.embed(x, pad, images, gp_features=gp)
        cont_hat, band_logits = model.decode(z)
        L = x.shape[1]
        cont_hat, band_logits = cont_hat[:, :L], band_logits[:, :L]

        valid = ~pad
        n_valid = valid.sum(dim=1).clamp(min=1).float()
        cont_err = ((cont_hat - x[..., :4]) ** 2 * valid.unsqueeze(-1)).sum(
            dim=(1, 2)
        ) / (n_valid * 4)
        band_target = x[..., 4:7].argmax(-1)
        ce = F.cross_entropy(
            band_logits.reshape(-1, 3), band_target.reshape(-1), reduction="none"
        ).reshape(x.shape[0], L)
        band_err = (ce * valid).sum(dim=1) / n_valid
        err = (cont_err + band_err).cpu().numpy()

        x_rec = recon_to_x(cont_hat, band_logits)
        x_np = x.cpu().numpy()
        valid_np = valid.cpu().numpy()
        z_np = z.cpu().numpy()
        for i in range(x.shape[0]):
            if batch["obj_ids"][i] == "BAD":
                continue
            v = valid_np[i]
            po = properties_from_x(x_np[i], v, norm)
            pr = properties_from_x(x_rec[i], v, norm)
            out["z"].append(z_np[i])
            out["props_orig"].append([po[k] for k in PROPERTY_NAMES])
            out["props_recon"].append([pr[k] for k in PROPERTY_NAMES])
            out["recon_err"].append(err[i])
            out["labels_fine"].append(int(batch["label_fine"][i]))
            out["labels_coarse"].append(int(batch["label_coarse"][i]))
            out["n_det"].append(int(v.sum()))
            out["obj_ids"].append(batch["obj_ids"][i])
            if keep_recon:
                recon_curves.append(x_rec[i][v])

    P = len(PROPERTY_NAMES)
    result = {
        "z": np.array(out["z"], dtype=np.float32).reshape(-1, model.latent_dim),
        "props_orig": np.array(out["props_orig"], dtype=np.float64).reshape(-1, P),
        "props_recon": np.array(out["props_recon"], dtype=np.float64).reshape(-1, P),
        "recon_err": np.array(out["recon_err"], dtype=np.float64),
        "labels_fine": np.array(out["labels_fine"], dtype=np.int64),
        "labels_coarse": np.array(out["labels_coarse"], dtype=np.int64),
        "n_det": np.array(out["n_det"], dtype=np.int64),
        "obj_ids": out["obj_ids"],
    }
    if keep_recon:
        result["recon_curves"] = recon_curves
    return result


def subset_encoding(enc, mask):
    mask = np.asarray(mask, dtype=bool)
    sub = {}
    for k, v in enc.items():
        if k == "obj_ids" or k == "recon_curves":
            sub[k] = [item for item, m in zip(v, mask) if m]
        else:
            sub[k] = v[mask]
    return sub


# ---------------------------------------------------------------------------
# Stage 1: reconstruction fidelity of discovery properties
# ---------------------------------------------------------------------------


def _pearson(a, b):
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def property_agreement(P_true, P_pred, names=PROPERTY_NAMES):
    """Per-property agreement metrics between two (N, P) property arrays."""
    res = {}
    for j, name in enumerate(names):
        a_all, b_all = P_true[:, j], P_pred[:, j]
        defined = np.isfinite(a_all)
        both = defined & np.isfinite(b_all)
        n = int(both.sum())
        entry = {
            "n": n,
            "coverage": float(both.sum() / max(int(defined.sum()), 1)),
            "unit": PROPERTY_UNITS[name],
            "tolerance": PROPERTY_TOLERANCES[name],
        }
        if n >= 2:
            a, b = a_all[both], b_all[both]
            err = b - a
            ss_tot = float(np.sum((a - a.mean()) ** 2))
            entry.update(
                {
                    "mae": float(np.mean(np.abs(err))),
                    "median_ae": float(np.median(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "bias": float(np.mean(err)),
                    "baseline_mae": float(np.mean(np.abs(a - np.median(a)))),
                    "frac_within_tol": float(
                        np.mean(np.abs(err) <= PROPERTY_TOLERANCES[name])
                    ),
                    "r2": float(1.0 - np.sum(err**2) / ss_tot)
                    if ss_tot > 0
                    else float("nan"),
                    "pearson": _pearson(a, b),
                }
            )
        else:
            for k in (
                "mae",
                "median_ae",
                "rmse",
                "bias",
                "baseline_mae",
                "frac_within_tol",
                "r2",
                "pearson",
            ):
                entry[k] = float("nan")
        res[name] = entry
    return res


def label_groups(labels_coarse):
    """Named boolean masks: all, transient, variable and each coarse class."""
    labels_coarse = np.asarray(labels_coarse)
    groups = {"all": np.ones(len(labels_coarse), dtype=bool)}
    if len(labels_coarse) == 0 or not np.any(labels_coarse >= 0):
        return groups
    groups["transient"] = np.isin(labels_coarse, TRANSIENT_COARSE)
    groups["variable"] = np.isin(labels_coarse, VARIABLE_COARSE)
    for cid, name in COARSE_NAMES.items():
        m = labels_coarse == cid
        if m.any():
            groups[name] = m
    return {k: v for k, v in groups.items() if v.sum() >= 2}


def grouped_agreement(P_true, P_pred, labels_coarse):
    return {
        g: property_agreement(P_true[m], P_pred[m])
        for g, m in label_groups(labels_coarse).items()
    }


# ---------------------------------------------------------------------------
# Stage 2: probes from the latent
# ---------------------------------------------------------------------------


def _make_regressor(kind, seed):
    if kind == "ridge":
        return RidgeCV(alphas=np.logspace(-2, 3, 11))
    if kind == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(128,),
            max_iter=1000,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"unknown probe kind {kind!r}")


def probe_properties(
    z_train, P_train, z_test, P_test, kind="ridge", seed=0, names=PROPERTY_NAMES
):
    """Regress each property from the latent; scores on the held-out split."""
    scaler = StandardScaler().fit(z_train)
    Ztr, Zte = scaler.transform(z_train), scaler.transform(z_test)
    res = {}
    for j, name in enumerate(names):
        mtr = np.isfinite(P_train[:, j])
        mte = np.isfinite(P_test[:, j])
        entry = {
            "n_train": int(mtr.sum()),
            "n_test": int(mte.sum()),
            "unit": PROPERTY_UNITS[name],
            "tolerance": PROPERTY_TOLERANCES[name],
        }
        if mtr.sum() < 20 or mte.sum() < 10:
            entry["skipped"] = True
            res[name] = entry
            continue
        reg = _make_regressor(kind, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(Ztr[mtr], P_train[mtr, j])
        pred = reg.predict(Zte[mte])
        y = P_test[mte, j]
        err = pred - y
        baseline = float(np.mean(np.abs(y - np.median(P_train[mtr, j]))))
        mae = float(np.mean(np.abs(err)))
        entry.update(
            {
                "skipped": False,
                "mae": mae,
                "median_ae": float(np.median(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "baseline_mae": baseline,
                "skill": float(1.0 - mae / baseline) if baseline > 0 else float("nan"),
                "r2": float(r2_score(y, pred)) if y.std() > 0 else float("nan"),
                "pearson": _pearson(y, pred),
                "frac_within_tol": float(
                    np.mean(np.abs(err) <= PROPERTY_TOLERANCES[name])
                ),
            }
        )
        res[name] = entry
    return res


def probe_binary(z_train, y_train, z_test, y_test, seed=0):
    """Logistic probe for a binary flag; returns AUC / balanced accuracy."""
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    entry = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pos_rate_test": float(y_test.mean()) if len(y_test) else float("nan"),
    }
    if (
        len(np.unique(y_train)) < 2
        or len(np.unique(y_test)) < 2
        or np.bincount(y_train).min() < 5
    ):
        entry["skipped"] = True
        return entry
    scaler = StandardScaler().fit(z_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(scaler.transform(z_train), y_train)
    Zte = scaler.transform(z_test)
    proba = clf.predict_proba(Zte)[:, 1]
    pred = (proba > 0.5).astype(int)
    entry.update(
        {
            "skipped": False,
            "auc": float(roc_auc_score(y_test, proba)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        }
    )
    return entry


def binary_tasks(enc_train, enc_test, fast_threshold=10.0):
    """Binary discovery flags derivable from labels / properties."""
    tasks = {}
    j_half = PROPERTY_NAMES.index("t_half_width")

    def _transient(enc):
        lc = enc["labels_coarse"]
        mask = np.isin(lc, TRANSIENT_COARSE + VARIABLE_COARSE)
        return mask, np.isin(lc, TRANSIENT_COARSE).astype(int)

    def _fast(enc):
        w = enc["props_orig"][:, j_half]
        return np.isfinite(w), (w < fast_threshold).astype(int)

    for name, fn in (("is_transient", _transient), ("fast_evolver", _fast)):
        mtr, ytr = fn(enc_train)
        mte, yte = fn(enc_test)
        tasks[name] = (mtr, ytr, mte, yte)
    return tasks


# ---------------------------------------------------------------------------
# Stage 3: injection-recovery
# ---------------------------------------------------------------------------


def _item_with_x(item, x_new):
    new = dict(item)
    new["x"] = torch.from_numpy(np.ascontiguousarray(x_new, dtype=np.float32))
    new["seq_len"] = int(x_new.shape[0])
    return new


def _physical_curve(x, norm):
    t, mag, mag_err, band = x_to_physical(x, norm)
    return {"t": t, "mag": mag, "mag_err": mag_err, "band": band}


def run_injection_recovery(
    model,
    hosts,
    device="cpu",
    norm=None,
    preset="sn_ia",
    delta_mags=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    rng=None,
    fpr=0.05,
    batch_size=128,
    n_examples=4,
    example_delta_mag=None,
    add_noise=False,
    t_peak_frac=(0.2, 0.8),
):
    """Injection-recovery experiment over a grid of injection amplitudes.

    `delta_mags` are the injected r-band peak magnitudes relative to the
    host's median r-band magnitude (positive = brighter than host; 0 means
    the transient doubles the host flux at peak).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    hosts = [h for h in hosts if h.get("obj_id") != "BAD"]
    N = len(hosts)
    if N < 4:
        raise ValueError("injection-recovery needs at least 4 host light curves")

    base = encode_batches(
        model, iter_item_batches(hosts, batch_size), device, norm, keep_recon=False
    )
    z_host = base["z"]
    null_items = [_item_with_x(h, renoise(h["x"].numpy(), rng, norm)) for h in hosts]
    z_null = encode_batches(
        model, iter_item_batches(null_items, batch_size), device, norm
    )["z"]
    d_null = np.linalg.norm(z_null - z_host, axis=1)
    threshold = float(np.quantile(d_null, 1.0 - fpr))
    nn = NearestNeighbors(n_neighbors=2).fit(z_host)
    nn_dist, _ = nn.kneighbors(z_host)
    scale = float(np.median(nn_dist[:, 1]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    j_peak = PROPERTY_NAMES.index("peak_mag")
    j_tpk = PROPERTY_NAMES.index("t_peak")
    delta_mags = [float(d) for d in delta_mags]
    if example_delta_mag is None:
        example_delta_mag = delta_mags[len(delta_mags) // 2]

    curves, examples = [], []
    for dm in delta_mags:
        inj_items, infos = [], []
        for h in hosts:
            x = h["x"].numpy()
            t, _, _, _ = x_to_physical(x, norm)
            duration = float(t.max() - t.min())
            t_peak = rng.uniform(*t_peak_frac) * duration if duration > 0 else 0.0
            params = sample_transient_params(preset, rng, t_peak=t_peak)
            ref = host_reference_mag(x, norm)
            x_new, info = inject_transient(
                x, ref - dm, params, norm=norm, rng=rng, add_noise=add_noise
            )
            inj_items.append(_item_with_x(h, x_new))
            infos.append(info)

        enc = encode_batches(
            model,
            iter_item_batches(inj_items, batch_size),
            device,
            norm,
            keep_recon=True,
        )
        z_inj = enc["z"]
        d = np.linalg.norm(z_inj - z_host, axis=1)
        detected = d > threshold
        # Stricter criterion: the injected source moves further than the typical
        # spacing between distinct sources, i.e. it leaves its host's neighbourhood.
        detected_nn = d > scale

        # Held-out probe: even-indexed hosts train, odd-indexed hosts test.
        idx = np.arange(N)
        tr, te = idx % 2 == 0, idx % 2 == 1
        probe = probe_binary(
            np.vstack([z_host[tr], z_inj[tr]]),
            np.r_[np.zeros(tr.sum()), np.ones(tr.sum())],
            np.vstack([z_host[te], z_inj[te]]),
            np.r_[np.zeros(te.sum()), np.ones(te.sum())],
        )

        # Recovery of the injected light curve's properties from the decoded curve
        agree = property_agreement(enc["props_orig"], enc["props_recon"])
        shift = property_agreement(base["props_orig"], enc["props_orig"])
        e_peak = np.abs(enc["props_recon"][:, j_peak] - enc["props_orig"][:, j_peak])
        e_tpk = np.abs(enc["props_recon"][:, j_tpk] - enc["props_orig"][:, j_tpk])
        recovered = (e_peak <= PROPERTY_TOLERANCES["peak_mag"]) & (
            e_tpk <= PROPERTY_TOLERANCES["t_peak"]
        )

        curves.append(
            {
                "delta_mag": dm,
                "snr_peak_median": float(np.median([i["snr_peak"] for i in infos])),
                "displacement_median": float(np.median(d)),
                "displacement_norm_median": float(np.median(d / scale)),
                "displacement_norm_p16": float(np.percentile(d / scale, 16)),
                "displacement_norm_p84": float(np.percentile(d / scale, 84)),
                "detection_efficiency": float(np.mean(detected)),
                "detection_efficiency_nn": float(np.mean(detected_nn)),
                "probe_auc": probe.get("auc", float("nan")),
                "probe_balanced_accuracy": probe.get("balanced_accuracy", float("nan")),
                "recovered_fraction": float(np.mean(recovered)),
                "recon": agree,
                "property_shift_mae": {k: v["mae"] for k, v in shift.items()},
            }
        )

        if math.isclose(dm, example_delta_mag) and not examples:
            for k in range(min(n_examples, N)):
                examples.append(
                    {
                        "obj_id": hosts[k]["obj_id"],
                        "delta_mag": dm,
                        "info": infos[k],
                        "host": _physical_curve(hosts[k]["x"].numpy(), norm),
                        "injected": _physical_curve(inj_items[k]["x"].numpy(), norm),
                        "decoded": _physical_curve(enc["recon_curves"][k], norm),
                        "detected": bool(detected[k]),
                    }
                )

    return {
        "preset": preset,
        "preset_spec": TRANSIENT_PRESETS[preset],
        "n_hosts": N,
        "fpr": fpr,
        "null_threshold": threshold,
        "null_displacement_median": float(np.median(d_null)),
        "latent_nn_scale": scale,
        "add_noise": add_noise,
        "curves": curves,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(v, width=8, prec=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'n/a':>{width}}"
    return f"{v:{width}.{prec}f}"


def print_recon_table(agreement, group="all"):
    res = agreement[group]
    print(f"\n  Reconstruction fidelity of discovery properties [{group}]")
    print(
        f"  {'property':14s} {'unit':8s} {'n':>5} {'cov':>5} {'MAE':>8} "
        f"{'medAE':>8} {'bias':>8} {'R2':>8} {'r':>8} {'within':>8}"
    )
    print("  " + "-" * 88)
    for name in PROPERTY_NAMES:
        m = res[name]
        print(
            f"  {name:14s} {m['unit']:8s} {m['n']:5d} {_fmt(m['coverage'], 5, 2)} "
            f"{_fmt(m['mae'])} {_fmt(m['median_ae'])} {_fmt(m['bias'])} "
            f"{_fmt(m['r2'])} {_fmt(m['pearson'])} "
            f"{_fmt(m['frac_within_tol'], prec=2)}"
        )
    print(
        "  cov = fraction of sources whose decoded curve still defines the property; "
        "within = |error| <= tolerance"
    )


def print_probe_table(probe_res, recon_res=None):
    print("\n  Latent probes (train split -> eval split)")
    print(
        f"  {'property':14s} {'n_test':>6} {'probeMAE':>9} {'reconMAE':>9} "
        f"{'baseMAE':>9} {'skill':>7} {'R2':>7} {'within':>7}"
    )
    print("  " + "-" * 82)
    for name in PROPERTY_NAMES:
        m = probe_res[name]
        if m.get("skipped"):
            print(f"  {name:14s} {m['n_test']:6d}   (skipped: too few defined values)")
            continue
        recon_mae = recon_res[name]["mae"] if recon_res else float("nan")
        print(
            f"  {name:14s} {m['n_test']:6d} {_fmt(m['mae'], 9)} {_fmt(recon_mae, 9)} "
            f"{_fmt(m['baseline_mae'], 9)} {_fmt(m['skill'], 7)} {_fmt(m['r2'], 7)} "
            f"{_fmt(m['frac_within_tol'], 7, 2)}"
        )


def print_binary_table(binary_res):
    print("\n  Binary discovery flags from the latent")
    for name, m in binary_res.items():
        if m.get("skipped"):
            print(f"    {name:14s} skipped (need both classes with >= 5 examples)")
        else:
            print(
                f"    {name:14s} AUC={m['auc']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
                f"(n_test={m['n_test']}, pos_rate={m['pos_rate_test']:.2f})"
            )


def print_injection_table(inj):
    print(
        f"\n  Injection-recovery: preset={inj['preset']}, hosts={inj['n_hosts']}, "
        f"FPR={inj['fpr']:.2f}, null threshold={inj['null_threshold']:.3g}, "
        f"latent NN scale={inj['latent_nn_scale']:.3g}"
    )
    print(
        f"  {'dmag':>5} {'SNR':>7} {'disp/NN':>8} {'det_null':>8} {'det_NN':>7} "
        f"{'probeAUC':>9} {'recov':>6} {'peakMAE':>8} {'tpkMAE':>7}"
    )
    print("  " + "-" * 74)
    for c in inj["curves"]:
        print(
            f"  {c['delta_mag']:5.1f} {_fmt(c['snr_peak_median'], 7, 1)} "
            f"{_fmt(c['displacement_norm_median'], 8, 2)} "
            f"{_fmt(c['detection_efficiency'], 8, 2)} "
            f"{_fmt(c['detection_efficiency_nn'], 7, 2)} {_fmt(c['probe_auc'], 9)} "
            f"{_fmt(c['recovered_fraction'], 6, 2)} "
            f"{_fmt(c['recon']['peak_mag']['mae'], 8)} "
            f"{_fmt(c['recon']['t_peak']['mae'], 7, 2)}"
        )
    print(
        "  det_null = latent shift beyond the re-noised null at the given FPR; "
        "det_NN = shift larger than the median nearest-neighbour spacing; "
        "recov = injected peak mag and t_peak both recovered within tolerance "
        "from the decoded curve"
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_recon_scatter(P_true, P_pred, path, title=""):
    n = len(PROPERTY_NAMES)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
    for j, (name, ax) in enumerate(zip(PROPERTY_NAMES, axes.flat)):
        a, b = P_true[:, j], P_pred[:, j]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 2:
            ax.scatter(a[m], b[m], s=6, alpha=0.4)
            lo, hi = np.nanpercentile(np.r_[a[m], b[m]], [1, 99])
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            mae = np.mean(np.abs(b[m] - a[m]))
            ax.text(0.05, 0.92, f"MAE={mae:.3g}", transform=ax.transAxes, fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "undefined on\ndecoded curves",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="gray",
            )
        ax.set_title(f"{name} [{PROPERTY_UNITS[name]}]", fontsize=9)
        ax.set_xlabel("original", fontsize=8)
        ax.set_ylabel("decoded", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_skill(recon_res, probe_res, path):
    """Skill = 1 - MAE / baseline MAE (median predictor). 1 = perfect, 0 = useless."""
    names = PROPERTY_NAMES
    recon_skill, probe_skill = [], []
    for name in names:
        r = recon_res[name]
        b = r["baseline_mae"]
        recon_skill.append(
            1 - r["mae"] / b if b and np.isfinite(b) and b > 0 else np.nan
        )
        p = probe_res.get(name, {}) if probe_res else {}
        probe_skill.append(p.get("skill", np.nan) if not p.get("skipped") else np.nan)
    xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(xs - 0.2, recon_skill, width=0.4, label="decoded curve")
    if probe_res:
        ax.bar(xs + 0.2, probe_skill, width=0.4, label="latent probe")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("skill = 1 - MAE / median-baseline MAE")
    ax.set_ylim(min(-0.5, np.nanmin(np.r_[recon_skill, probe_skill, 0]) - 0.1), 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_injection_curves(inj, path):
    c = inj["curves"]
    dm = np.array([e["delta_mag"] for e in c])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    ax.plot(
        dm,
        [e["detection_efficiency"] for e in c],
        "o-",
        label=f"latent shift > null (FPR={inj['fpr']:.0%})",
    )
    ax.plot(
        dm,
        [e["detection_efficiency_nn"] for e in c],
        "d-",
        label="latent shift > NN spacing",
    )
    ax.plot(dm, [e["probe_auc"] for e in c], "s--", label="held-out probe AUC")
    ax.plot(
        dm,
        [e["recovered_fraction"] for e in c],
        "^:",
        label="peak recovered in decoded curve",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("injected peak brightness relative to host [mag]")
    ax.set_ylabel("fraction")
    ax.set_title(f"Completeness ({inj['preset']})")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    med = np.array([e["displacement_norm_median"] for e in c])
    lo = np.array([e["displacement_norm_p16"] for e in c])
    hi = np.array([e["displacement_norm_p84"] for e in c])
    ax.plot(dm, med, "o-")
    ax.fill_between(dm, lo, hi, alpha=0.25)
    ax.axhline(
        inj["null_threshold"] / inj["latent_nn_scale"],
        color="k",
        ls="--",
        lw=1,
        label="null threshold",
    )
    ax.set_xlabel("injected peak brightness relative to host [mag]")
    ax.set_ylabel("latent displacement / NN spacing")
    ax.set_title("Latent sensitivity")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(dm, [e["recon"]["peak_mag"]["mae"] for e in c], "o-", label="peak_mag")
    ax.plot(dm, [e["recon"]["amplitude"]["mae"] for e in c], "s-", label="amplitude")
    ax.axhline(PROPERTY_TOLERANCES["peak_mag"], color="k", ls="--", lw=1)
    ax.set_xlabel("injected peak brightness relative to host [mag]")
    ax.set_ylabel("MAE of decoded property [mag]")
    ax.set_title("Brightness recovery")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(dm, [e["recon"]["t_peak"]["mae"] for e in c], "o-", label="t_peak")
    ax.plot(
        dm, [e["recon"]["t_half_width"]["mae"] for e in c], "s-", label="t_half_width"
    )
    ax.axhline(PROPERTY_TOLERANCES["t_peak"], color="k", ls="--", lw=1)
    ax.set_xlabel("injected peak brightness relative to host [mag]")
    ax.set_ylabel("MAE of decoded property [days]")
    ax.set_title("Timing recovery")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_injection_examples(examples, path):
    if not examples:
        return
    fig, axes = plt.subplots(
        len(examples), 3, figsize=(12, 2.8 * len(examples)), squeeze=False
    )
    for row, ex in zip(axes, examples):
        curves = [
            ("host", ex["host"]),
            ("injected", ex["injected"]),
            ("decoded", ex["decoded"]),
        ]
        ylim = None
        for ax, (label, cv) in zip(row, curves):
            for b in range(3):
                m = cv["band"] == b
                if m.any():
                    ax.errorbar(
                        cv["t"][m],
                        cv["mag"][m],
                        yerr=cv["mag_err"][m],
                        fmt="o",
                        ms=3,
                        color=BAND_COLORS[b],
                        label=f"ZTF-{BAND_NAMES[b]}",
                        alpha=0.8,
                    )
            if label != "host":
                ax.axvline(ex["info"]["t_peak"], color="gray", ls=":", lw=1)
            if ylim is None:
                allm = np.r_[ex["host"]["mag"], ex["injected"]["mag"]]
                ylim = (allm.max() + 0.3, allm.min() - 0.3)
            if label == "decoded":
                allm = np.r_[ex["injected"]["mag"], cv["mag"]]
                ax.set_ylim(allm.max() + 0.3, allm.min() - 0.3)
            else:
                ax.set_ylim(*ylim)
            ax.set_xlabel("days since first detection", fontsize=8)
            ax.set_ylabel("mag", fontsize=8)
            flag = (
                ""
                if label != "decoded"
                else (" [detected]" if ex["detected"] else " [missed]")
            )
            ax.set_title(f"{ex['obj_id']} {label}{flag}", fontsize=9)
        row[0].legend(fontsize=7)
    fig.suptitle(
        f"Injection examples: preset={examples[0]['info']['preset']}, "
        f"delta_mag={examples[0]['delta_mag']:+.1f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate how well the latent preserves discovery-notable properties"
    )
    p.add_argument("--checkpoint", default=None, help="Path to best_model.pt")
    p.add_argument("--summary", default=None, help="Path to summary.json (optional)")
    p.add_argument("--data-dir", default=None, help="Dir with {train,val,test}/*.npz")
    p.add_argument(
        "--synthetic-dir",
        default=None,
        help="Generate (if missing) and use a synthetic dataset here when no --data-dir",
    )
    p.add_argument(
        "--synthetic-n", type=int, default=200, help="Sources per synthetic class"
    )
    p.add_argument(
        "--synthetic-channels",
        type=int,
        default=37,
        help="Feature channels of generated synthetic data (37 metadata layout, 7 photometry only)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument(
        "--select",
        default="all",
        help="Subset for recon/probe metrics: all | transient | variable | coarse:AGN | fine:9 | amplitude>1 (comma = AND)",
    )
    p.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="Truncate sequences to first N epochs",
    )
    p.add_argument(
        "--horizon",
        type=float,
        default=100.0,
        help="Horizon (days) for photometry-only NPZ data",
    )
    p.add_argument("--probe", choices=["ridge", "mlp"], default="ridge")
    p.add_argument(
        "--fast-threshold",
        type=float,
        default=10.0,
        help="t_half_width (days) below which a source is a 'fast evolver'",
    )
    # injection
    p.add_argument(
        "--inject-preset", default="sn_ia", choices=sorted(TRANSIENT_PRESETS)
    )
    p.add_argument(
        "--host-select",
        default="variable",
        help="Selection for injection hosts (falls back to all)",
    )
    p.add_argument("--n-hosts", type=int, default=200)
    p.add_argument(
        "--delta-mags",
        nargs="+",
        type=float,
        default=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        help="Injected r-band peak brightness relative to host median mag (+ = brighter)",
    )
    p.add_argument(
        "--fpr",
        type=float,
        default=0.05,
        help="False-positive rate for the null-displacement detector",
    )
    p.add_argument(
        "--inject-noise",
        action="store_true",
        help="Re-draw photometric noise on injected curves",
    )
    p.add_argument("--n-examples", type=int, default=4)
    # untrained smoke-test model (only when no --checkpoint)
    p.add_argument("--mode", choices=["ae", "vae", "vqvae"], default="ae")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument(
        "--image-backbone", default=None, help="Override image backbone (simple/zoobot)"
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def _in_channels_of(data_dir, split):
    split_dir = os.path.join(data_dir, split)
    fmt = detect_npz_format(split_dir)
    if fmt == "photo":
        return 7
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".npz"))
    with np.load(os.path.join(split_dir, files[0]), allow_pickle=True) as npz:
        return int(npz["x"].shape[1])


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- data ---
    data_dir = args.data_dir
    if data_dir is None:
        if args.synthetic_dir is None:
            raise SystemExit("need --data-dir or --synthetic-dir")
        if not os.path.isdir(os.path.join(args.synthetic_dir, "train")):
            print(f"Generating synthetic dataset in {args.synthetic_dir} ...")
            counts = make_synthetic_dataset(
                args.synthetic_dir,
                n_per_class=args.synthetic_n,
                seed=args.seed,
                n_channels=args.synthetic_channels,
            )
            print("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        data_dir = args.synthetic_dir

    # --- model ---
    if args.checkpoint:
        model, cfg = load_model(
            args.checkpoint,
            args.summary,
            device=args.device,
            overrides={"image_backbone": args.image_backbone},
        )
        print(
            f"Loaded {cfg['mode']} dim={cfg['latent_dim']} in_channels={cfg['in_channels']} "
            f"images={cfg['use_images']} gp_dim={cfg['gp_dim']} from {args.checkpoint}"
        )
    else:
        print(
            "WARNING: no --checkpoint given; using an UNTRAINED model. "
            "Numbers below only exercise the pipeline."
        )
        cfg = {
            "mode": args.mode,
            "latent_dim": args.latent_dim,
            "in_channels": _in_channels_of(data_dir, args.eval_split),
            "d_model": 64,
            "n_heads": 4,
            "enc_layers": 2,
            "dec_layers": 1,
            "d_ff": 128,
            "use_images": False,
            "gp_dim": 0,
            "num_classes": 0,
        }
        model = build_model(cfg, args.device)
    if cfg.get("gp_dim", 0) > 0 and "inject" in args.stages:
        print(
            "NOTE: model uses GP features; injected light curves reuse the host's "
            "(now stale) GP features. Recompute them with lightcurve-fitting for a "
            "fully consistent injection test."
        )

    results = {"config": cfg, "args": vars(args), "data_dir": os.path.abspath(data_dir)}

    # --- encode eval split ---
    eval_ds, norm = build_dataset(
        data_dir, args.eval_split, cfg, args.max_detections, args.horizon
    )
    pin = args.device != "cpu"
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    print(f"\nEncoding {args.eval_split} split ({len(eval_ds)} sources) ...")
    enc_te = encode_batches(model, eval_loader, args.device, norm)
    sel = select_mask(
        args.select,
        enc_te["labels_coarse"],
        enc_te["labels_fine"],
        enc_te["props_orig"],
    )
    enc_sel = subset_encoding(enc_te, sel)
    print(
        f"  {len(enc_te['obj_ids'])} encoded, {int(sel.sum())} selected by '{args.select}', "
        f"median recon error={np.median(enc_te['recon_err']):.4f}"
    )
    results["data"] = {
        "eval_split": args.eval_split,
        "n_eval": len(enc_te["obj_ids"]),
        "n_selected": int(sel.sum()),
        "select": args.select,
        "median_n_det": float(np.median(enc_te["n_det"])),
        "median_recon_err": float(np.median(enc_te["recon_err"])),
        "class_counts": {
            COARSE_NAMES.get(int(c), str(int(c))): int(n)
            for c, n in zip(*np.unique(enc_te["labels_coarse"], return_counts=True))
        },
    }

    # --- stage 1: reconstruction fidelity ---
    recon_all = None
    if "recon" in args.stages:
        agreement = grouped_agreement(
            enc_sel["props_orig"], enc_sel["props_recon"], enc_sel["labels_coarse"]
        )
        results["recon"] = agreement
        recon_all = agreement["all"]
        print_recon_table(agreement, "all")
        for g in ("transient", "variable"):
            if g in agreement:
                print_recon_table(agreement, g)
        plot_recon_scatter(
            enc_sel["props_orig"],
            enc_sel["props_recon"],
            os.path.join(args.output_dir, "recon_properties.png"),
            title=f"Original vs decoded discovery properties ({args.select})",
        )

    # --- stage 2: latent probes ---
    probe_res = None
    if "probe" in args.stages:
        train_ds, norm_tr = build_dataset(
            data_dir, args.train_split, cfg, args.max_detections, args.horizon
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        print(
            f"\nEncoding {args.train_split} split ({len(train_ds)} sources) for probes ..."
        )
        enc_tr = encode_batches(model, train_loader, args.device, norm_tr)
        probe_res = probe_properties(
            enc_tr["z"],
            enc_tr["props_orig"],
            enc_sel["z"],
            enc_sel["props_orig"],
            kind=args.probe,
            seed=args.seed,
        )
        binary_res = {}
        for name, (mtr, ytr, mte, yte) in binary_tasks(
            enc_tr, enc_sel, args.fast_threshold
        ).items():
            binary_res[name] = probe_binary(
                enc_tr["z"][mtr], ytr[mtr], enc_sel["z"][mte], yte[mte], seed=args.seed
            )
        results["probe"] = {
            "kind": args.probe,
            "properties": probe_res,
            "binary": binary_res,
        }
        print_probe_table(probe_res, recon_all)
        print_binary_table(binary_res)
    if recon_all is not None:
        plot_skill(
            recon_all, probe_res, os.path.join(args.output_dir, "property_skill.png")
        )

    # --- stage 3: injection-recovery ---
    if "inject" in args.stages:
        host_mask = select_mask(
            args.host_select,
            enc_te["labels_coarse"],
            enc_te["labels_fine"],
            enc_te["props_orig"],
        )
        if host_mask.sum() < 4:
            print(
                f"\n  host selection '{args.host_select}' matched {int(host_mask.sum())} "
                "sources; falling back to all"
            )
            host_mask = np.ones(len(host_mask), dtype=bool)
        host_ids = set(np.array(enc_te["obj_ids"])[host_mask].tolist())
        ds_idx = [
            i for i, f in enumerate(getattr(eval_ds, "files", [])) if f.stem in host_ids
        ]
        rng.shuffle(ds_idx)
        hosts = [eval_ds[i] for i in ds_idx[: args.n_hosts]]
        print(
            f"\nInjection-recovery with {len(hosts)} hosts ('{args.host_select}'), "
            f"preset={args.inject_preset}, delta_mags={args.delta_mags}"
        )
        inj = run_injection_recovery(
            model,
            hosts,
            device=args.device,
            norm=norm,
            preset=args.inject_preset,
            delta_mags=args.delta_mags,
            rng=rng,
            fpr=args.fpr,
            batch_size=args.batch_size,
            n_examples=args.n_examples,
            add_noise=args.inject_noise,
        )
        print_injection_table(inj)
        plot_injection_curves(
            inj, os.path.join(args.output_dir, "injection_recovery.png")
        )
        plot_injection_examples(
            inj["examples"], os.path.join(args.output_dir, "injection_examples.png")
        )
        inj_out = dict(inj)
        inj_out["examples"] = [
            {k: v for k, v in ex.items() if k not in ("host", "injected", "decoded")}
            for ex in inj["examples"]
        ]
        results["injection"] = inj_out

    out_path = os.path.join(args.output_dir, "discovery_eval.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    print(f"\nSaved results to {out_path}")
    return results


if __name__ == "__main__":
    main()
