"""
Can shapes the classifier never saw be found in the latent?

The latent is trained image-only on circles vs. squares (as in
sanity_shapes.py). Four novel shapes -- triangle, ring, ellipse, cross -- are
then pushed through the frozen encoder and we ask, WITHOUT using their labels:

  * novelty:    does a kNN-distance score to the known-class latents flag them?
                (AUROC novel-vs-known, per shape); for classifier-trained
                latents the softmax confidence is the usual baseline
  * clustering: does k-means on the held-out latent give them their own
                clusters? (AMI, and per-shape majority-cluster purity)
  * ceiling:    a 6-way linear probe WITH labels -- is the information there?
  * geometry:   effective dimensionality (participation ratio) of the latent

across training regimes that keep different amounts of information in z:

  random   untrained simple CNN (control)
  cls      cross-entropy on cls_head(z) only            (the earlier baseline)
  recon    MSE of a small stamp decoder from z only     (pure autoencoder)
  both     cls + recon
  zoobot   frozen Galaxy Zoo ConvNeXt features -> random projection, no training

The stamp decoder lives only in this script; the encoder/latent is the real
LightCurveCompressor with constant dummy photometry.

Usage (from src/):
    python shape_novelty.py run  --regime cls  --seed 0 --out ../analysis/shapes/novelty
    python shape_novelty.py plot --out ../analysis/shapes/novelty --fig ../docs/figures/shapes
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from torch import nn

from sanity_shapes import STAMP, build_model, dummy_photometry, latent_with_grad

SHAPES = ["circle", "square", "triangle", "ring", "ellipse", "cross"]
KNOWN = [0, 1]
NOVEL = [2, 3, 4, 5]
REGIMES = ["random", "cls", "recon", "both", "zoobot", "zoobot_cls", "zoobot_both"]


def _losses(regime: str) -> str:
    """'zoobot_cls' -> 'cls', 'zoobot' -> 'none', 'random' -> 'none'."""
    base = regime.replace("zoobot_", "")
    return base if base in ("cls", "recon", "both") else "none"


# ---------------------------------------------------------------------------
# Stamps with six shapes, all area-matched to a circle of radius r
# ---------------------------------------------------------------------------


def _rot(dy, dx, theta):
    c, s = np.cos(theta), np.sin(theta)
    return c * dy - s * dx, s * dy + c * dx


def shape_mask(kind: str, yy, xx, cy, cx, r, theta):
    dy, dx = yy - cy, xx - cx
    if kind == "circle":
        return dy**2 + dx**2 <= r**2
    if kind == "square":
        s = r * np.sqrt(np.pi) / 2
        return (np.abs(dy) <= s) & (np.abs(dx) <= s)
    if kind == "ellipse":  # axis ratio 0.5, random orientation
        q = 0.5
        a, b = r / np.sqrt(q), r * np.sqrt(q)
        ry, rx = _rot(dy, dx, theta)
        return (rx / a) ** 2 + (ry / b) ** 2 <= 1
    if kind == "ring":  # annulus, inner radius = half outer
        R = r / np.sqrt(0.75)
        d2 = dy**2 + dx**2
        return (d2 <= R**2) & (d2 >= (R / 2) ** 2)
    if kind == "triangle":  # equilateral, random orientation
        side = r * np.sqrt(4 * np.pi / np.sqrt(3))
        ry, rx = _rot(dy, dx, theta)
        h = side * np.sqrt(3) / 2
        # centroid at origin: vertices at (0, 2h/3), (+-side/2, -h/3)
        inside = ry <= 2 * h / 3
        inside &= ry >= -h / 3
        inside &= np.abs(rx) <= (2 * h / 3 - ry) / np.sqrt(3)
        return inside
    if kind == "cross":  # two bars, length 4w, width w, random orientation
        w = r * np.sqrt(np.pi / 7)
        L = 4 * w
        ry, rx = _rot(dy, dx, theta)
        bar1 = (np.abs(rx) <= L / 2) & (np.abs(ry) <= w / 2)
        bar2 = (np.abs(ry) <= L / 2) & (np.abs(rx) <= w / 2)
        return bar1 | bar2
    raise ValueError(kind)


def make_stamps(
    n_per_shape: int,
    shapes=SHAPES,
    seed: int = 0,
    noise: float = 0.1,
    size: int = STAMP,
    min_r: int = 8,
    max_r: int = 16,
    jitter: int = 8,
    stats=None,
):
    """Return (images (N,3,size,size) float32, labels (N,) int64 into SHAPES, stats).

    Channels: science = shape + noise, template = noise, difference = sci - tmpl.
    ``stats`` = (mean, std) per channel; computed from this set when None so the
    training set defines the standardisation applied to held-out sets.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat([SHAPES.index(s) for s in shapes], n_per_shape).astype(np.int64)
    rng.shuffle(labels)
    yy, xx = np.mgrid[0:size, 0:size]
    images = np.zeros((len(labels), 3, size, size), dtype=np.float32)
    c0 = size // 2
    for i, lab in enumerate(labels):
        cy = c0 + rng.integers(-jitter, jitter + 1)
        cx = c0 + rng.integers(-jitter, jitter + 1)
        r = rng.uniform(min_r, max_r)
        theta = rng.uniform(0, np.pi)
        amp = rng.uniform(0.6, 1.4)
        mask = shape_mask(SHAPES[lab], yy, xx, cy, cx, r, theta)
        sci = amp * mask + rng.normal(0, noise, (size, size))
        tmpl = rng.normal(0, noise, (size, size))
        images[i, 0], images[i, 1], images[i, 2] = sci, tmpl, sci - tmpl
    if stats is None:
        mu = images.mean(axis=(0, 2, 3), keepdims=True)
        sd = images.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        stats = (mu, sd)
    images = (images - stats[0]) / stats[1]
    return images.astype(np.float32), labels, stats


# ---------------------------------------------------------------------------
# Stamp decoder (experiment-only) and training regimes
# ---------------------------------------------------------------------------


class StampDecoder(nn.Module):
    """z (latent_dim) -> (3, 63, 63). Only used to shape the latent here."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 64
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def forward(self, z):
        h = self.fc(z).view(z.shape[0], 128, 8, 8)
        return self.net(h)[:, :, :STAMP, :STAMP]


def train_regime(
    model,
    regime: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    recon_weight: float = 1.0,
    log_every: int = 10,
    targets: torch.Tensor | None = None,
):
    """``images`` feed the encoder; ``targets`` (default: images) are what the
    stamp decoder must reconstruct -- they differ when the encoder input is a
    cached feature tensor."""
    if targets is None:
        targets = images
    use_cls = regime in ("cls", "both")
    use_recon = regime in ("recon", "both")
    decoder = StampDecoder(model.latent_dim) if use_recon else None
    params = [p for p in model.parameters() if p.requires_grad]
    if decoder is not None:
        params += list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    n = images.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        if decoder is not None:
            decoder.train()
        perm = torch.randperm(n)
        tot = {"cls": 0.0, "recon": 0.0, "acc": 0.0}
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            img, lab = images[idx], labels[idx]
            x, pad = dummy_photometry(img.shape[0], model.in_channels)
            z = latent_with_grad(model, x, pad, img)
            loss = torch.zeros(())
            if use_cls:
                logits = model.cls_head(z)
                l_cls = F.cross_entropy(logits, lab)
                loss = loss + l_cls
                tot["cls"] += l_cls.item() * len(idx)
                tot["acc"] += (logits.argmax(-1) == lab).sum().item()
            if use_recon:
                l_rec = F.mse_loss(decoder(z), targets[idx])
                loss = loss + recon_weight * l_rec
                tot["recon"] += l_rec.item() * len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(
                f"  [{regime}] epoch {epoch:3d}  cls {tot['cls'] / n:.4f}  "
                f"recon {tot['recon'] / n:.4f}  acc {tot['acc'] / n:.3f}",
                flush=True,
            )
    return decoder


@torch.no_grad()
def cache_frozen_features(model, images: torch.Tensor, batch_size: int = 32):
    """Run the frozen Zoobot CNN once: (N,3,63,63) -> (N, num_features)."""
    tower = model.image_tower
    tower.eval()
    feats = []
    for i in range(0, images.shape[0], batch_size):
        x = F.interpolate(
            images[i : i + batch_size],
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        feats.append(tower.cnn(x))
    return torch.cat(feats)


@torch.no_grad()
def encode(model, images: torch.Tensor, batch_size: int = 128):
    """Latent and (if present) softmax confidence of the 2-way cls head."""
    model.eval()
    zs, confs = [], []
    for i in range(0, images.shape[0], batch_size):
        chunk = images[i : i + batch_size]
        x, pad = dummy_photometry(chunk.shape[0], model.in_channels)
        z = model.embed(x, pad, chunk)
        zs.append(z)
        confs.append(F.softmax(model.cls_head(z), -1).max(-1).values)
    return torch.cat(zs).numpy(), torch.cat(confs).numpy()


# ---------------------------------------------------------------------------
# Label-free evaluation of the latent
# ---------------------------------------------------------------------------


def participation_ratio(z: np.ndarray) -> float:
    zc = z - z.mean(0)
    ev = np.linalg.eigvalsh(np.cov(zc.T))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (np.sum(ev**2) + 1e-12))


def evaluate_latent(z_train, y_train, z_test, y_test, conf_test=None, seed=0) -> dict:
    """All metrics on the held-out latent. Only the probe uses test labels."""
    # Standardise with the known-class training latents
    mu, sd = z_train.mean(0), z_train.std(0) + 1e-6
    zt, ze = (z_train - mu) / sd, (z_test - mu) / sd
    is_novel = np.isin(y_test, NOVEL)
    out = {"participation_ratio": participation_ratio(z_test)}

    # --- novelty via kNN distance to known-class training latents ---
    nn_ = NearestNeighbors(n_neighbors=5).fit(zt)
    knn_score = nn_.kneighbors(ze)[0].mean(1)
    out["novelty_auroc_knn"] = float(roc_auc_score(is_novel, knn_score))
    out["novelty_auroc_knn_per_shape"] = {}
    for s in NOVEL:
        m = (y_test == s) | ~is_novel
        out["novelty_auroc_knn_per_shape"][SHAPES[s]] = float(
            roc_auc_score(y_test[m] == s, knn_score[m])
        )
    if conf_test is not None:
        # low confidence should mean novel
        out["novelty_auroc_conf"] = float(roc_auc_score(is_novel, -conf_test))
        out["novelty_auroc_conf_per_shape"] = {}
        for s in NOVEL:
            m = (y_test == s) | ~is_novel
            out["novelty_auroc_conf_per_shape"][SHAPES[s]] = float(
                roc_auc_score(y_test[m] == s, -conf_test[m])
            )

    # --- unsupervised clustering: does each shape get its own cluster? ---
    km = KMeans(n_clusters=len(SHAPES), n_init=10, random_state=seed).fit(ze)
    cl = km.labels_
    out["cluster_ami"] = float(adjusted_mutual_info_score(y_test, cl))
    out["cluster_per_shape"] = {}
    for s in range(len(SHAPES)):
        m = y_test == s
        counts = np.bincount(cl[m], minlength=len(SHAPES))
        c = int(counts.argmax())
        out["cluster_per_shape"][SHAPES[s]] = {
            "majority_cluster": c,
            "recall": float(counts[c] / m.sum()),  # fraction of shape in its cluster
            "purity": float((y_test[cl == c] == s).mean()),  # cluster owned by shape
        }
    contingency = np.zeros((len(SHAPES), len(SHAPES)), dtype=int)
    for s in range(len(SHAPES)):
        contingency[s] = np.bincount(cl[y_test == s], minlength=len(SHAPES))
    out["contingency"] = contingency.tolist()

    # --- information ceiling: labelled 6-way linear probe (5-fold CV) ---
    pred = cross_val_predict(
        LogisticRegression(max_iter=3000), ze, y_test, cv=5, method="predict"
    )
    out["probe6_acc"] = float((pred == y_test).mean())
    out["probe6_recall"] = {
        SHAPES[s]: float((pred[y_test == s] == s).mean()) for s in range(len(SHAPES))
    }
    # what do novel shapes get called by a circle/square-only probe?
    probe2 = LogisticRegression(max_iter=3000).fit(zt, y_train)
    p2 = probe2.predict(ze)
    out["novel_called_circle"] = {
        SHAPES[s]: float((p2[y_test == s] == 0).mean()) for s in NOVEL
    }
    return out


# ---------------------------------------------------------------------------
# One full run
# ---------------------------------------------------------------------------


def run(
    regime: str,
    seed: int,
    out: str,
    n_train: int = 2048,
    n_test: int = 256,
    epochs: int = 30,
    latent_dim: int = 16,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_img, tr_lab, stats = make_stamps(n_train // 2, ["circle", "square"], seed=seed)
    te_img, te_lab, _ = make_stamps(n_test, SHAPES, seed=1000 + seed, stats=stats)
    tr_img_t, tr_lab_t, te_img_t = map(torch.from_numpy, (tr_img, tr_lab, te_img))

    backbone = "zoobot" if regime.startswith("zoobot") else "simple"
    losses = _losses(regime)
    model = build_model(backbone=backbone, latent_dim=latent_dim)
    t0 = time.time()

    # For trained Zoobot regimes the backbone is frozen, so run it once and
    # train projection + bottleneck (+ heads) on cached features.
    tr_in, te_in = tr_img_t, te_img_t
    if regime.startswith("zoobot_"):
        tr_in = cache_frozen_features(model, tr_img_t)
        te_in = cache_frozen_features(model, te_img_t)
        tower = model.image_tower
        tower.forward = lambda feats: tower.proj(feats)
        print(f"  cached Zoobot features in {time.time() - t0:.0f}s", flush=True)

    if losses != "none":
        train_regime(model, losses, tr_in, tr_lab_t, epochs=epochs, targets=tr_img_t)
    z_tr, _ = encode(model, tr_in)
    z_te, conf = encode(model, te_in)
    metrics = evaluate_latent(
        z_tr, tr_lab, z_te, te_lab, conf if losses in ("cls", "both") else None, seed
    )
    result = {
        "regime": regime,
        "seed": seed,
        "n_train": n_train,
        "n_test_per_shape": n_test,
        "epochs": epochs if losses != "none" else 0,
        "latent_dim": latent_dim,
        "seconds": time.time() - t0,
        **metrics,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{regime}_seed{seed}.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(
        os.path.join(out, f"{regime}_seed{seed}_latents.npz"),
        z_test=z_te,
        y_test=te_lab,
        conf=conf,
        z_train=z_tr,
        y_train=tr_lab,
    )
    print(
        f"[{regime} seed={seed}] probe6 {metrics['probe6_acc']:.3f} | novelty AUROC "
        f"kNN {metrics['novelty_auroc_knn']:.3f} | AMI {metrics['cluster_ami']:.3f} | "
        f"PR {metrics['participation_ratio']:.2f} | {result['seconds']:.0f}s",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Reference palette, categorical slots 1-6, fixed order = SHAPES order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "o", "^", "D", "s", "P"]  # novel shapes get distinct markers


def _load_all(out):
    res = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".json"):
            with open(os.path.join(out, f)) as fh:
                r = json.load(fh)
            res.setdefault(r["regime"], []).append(r)
    return res


def make_plots(out: str, fig_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e5e5e5",
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )
    C_GREY = "#8a8a8a"
    res = _load_all(out)
    regimes = [r for r in REGIMES if r in res]
    saved = []

    def save(fig, name):
        p = os.path.join(fig_dir, name)
        fig.savefig(p)
        plt.close(fig)
        saved.append(p)

    def mean_std(regime, key, sub=None):
        vals = [r[key] if sub is None else r[key][sub] for r in res[regime]]
        return np.mean(vals), np.std(vals)

    # --- 10. the six shapes ---
    imgs, labs, _ = make_stamps(3, SHAPES, seed=3)
    fig, axes = plt.subplots(3, 6, figsize=(9, 4.6))
    for s in range(6):
        idx = np.where(labs == s)[0]
        for k in range(3):
            ax = axes[k, s]
            ax.imshow(imgs[idx[k], 0], cmap="gray", vmin=-1, vmax=4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if k == 0:
                ax.set_title(
                    SHAPES[s] + ("" if s in KNOWN else " (novel)"), fontsize=10
                )
    fig.suptitle("Known (circle, square) and novel shapes, science channel", y=1.0)
    save(fig, "10_novel_stamps.png")

    # --- 11. 2-D views of the held-out latent, seed 0, per regime ---
    fig, axes = plt.subplots(2, len(regimes), figsize=(2.7 * len(regimes), 5.6))
    axes = np.atleast_2d(axes)
    for j, regime in enumerate(regimes):
        d = np.load(os.path.join(out, f"{regime}_seed0_latents.npz"))
        z, y = d["z_test"], d["y_test"]
        z = (z - z.mean(0)) / (z.std(0) + 1e-6)
        views = [
            ("PCA", PCA(n_components=2).fit_transform(z)),
            (
                "t-SNE",
                TSNE(n_components=2, random_state=0, init="pca").fit_transform(z),
            ),
        ]
        for i, (name, e) in enumerate(views):
            ax = axes[i, j]
            for s in range(6):
                m = y == s
                ax.scatter(
                    e[m, 0],
                    e[m, 1],
                    s=7,
                    c=COLORS[s],
                    marker=MARKERS[s],
                    alpha=0.6,
                    lw=0,
                    label=SHAPES[s],
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{regime} — {name}", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=2, loc="best")
    fig.suptitle(
        "Held-out latent (256 stamps per shape); latent trained on circle/square only",
        y=1.0,
    )
    save(fig, "11_novel_latent_views.png")

    # --- 12. novelty AUROC per shape per regime ---
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(regimes))
    w = 0.18
    for k, s in enumerate(NOVEL):
        m = [mean_std(r, "novelty_auroc_knn_per_shape", SHAPES[s]) for r in regimes]
        ax.bar(
            x + (k - 1.5) * w,
            [v[0] for v in m],
            w,
            yerr=[v[1] for v in m],
            color=COLORS[s],
            label=SHAPES[s],
            capsize=2,
        )
    ax.axhline(0.5, color=C_GREY, ls=":", lw=1)
    ax.text(-0.45, 0.515, "chance", color=C_GREY, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC, novel shape vs known")
    ax.set_title(
        "Label-free novelty score: 5-NN distance to circle/square training latents"
    )
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "12_novelty_auroc.png")

    # --- 13. summary: probe ceiling, AMI, participation ratio ---
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, key, title, base in (
        (axes[0], "probe6_acc", "6-way probe accuracy (with labels)", 1 / 6),
        (axes[1], "cluster_ami", "k-means AMI vs true shape (no labels)", 0),
        (axes[2], "participation_ratio", "effective dimensionality of z", None),
    ):
        m = [mean_std(r, key) for r in regimes]
        ax.bar(
            x,
            [v[0] for v in m],
            0.55,
            yerr=[v[1] for v in m],
            color="#2a78d6",
            capsize=3,
        )
        if base is not None:
            ax.axhline(base, color=C_GREY, ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 8)
    save(fig, "13_novelty_summary.png")

    # --- 14. cluster contingency heatmaps (seed 0) ---
    fig, axes = plt.subplots(1, len(regimes), figsize=(2.9 * len(regimes), 3.2))
    axes = np.atleast_1d(axes)
    for ax, regime in zip(axes, regimes):
        r0 = next(r for r in res[regime] if r["seed"] == 0)
        C = np.array(r0["contingency"], dtype=float)
        C = C / C.sum(1, keepdims=True)
        # order clusters by the shape that owns them most, for readability
        order = np.argsort(C.argmax(0), kind="stable")
        ax.imshow(C[:, order], cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(6))
        ax.set_yticklabels(SHAPES, fontsize=8)
        ax.set_xticks(range(6))
        ax.set_xticklabels([f"c{o}" for o in order], fontsize=8)
        ax.set_xlabel("k-means cluster")
        ax.set_title(regime, fontsize=10)
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                v = C[i, order[j]]
                if v >= 0.15:
                    ax.text(
                        j,
                        i,
                        f"{v:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if v > 0.6 else "black",
                    )
    fig.suptitle(
        "Row-normalised: where each shape's held-out stamps land (seed 0)", y=1.03
    )
    save(fig, "14_novelty_clusters.png")
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "plot"])
    ap.add_argument("--regime", choices=REGIMES, default="cls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--out", default="../analysis/shapes/novelty")
    ap.add_argument("--fig", default="../docs/figures/shapes")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.cmd == "plot":
        for p in make_plots(args.out, args.fig):
            print("saved", p)
    else:
        run(args.regime, args.seed, args.out, n_train=args.n_train, epochs=args.epochs)


if __name__ == "__main__":
    main()
