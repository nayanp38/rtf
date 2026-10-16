"""
Cutout-only sanity check: can the latent separate circles from squares?

This bypasses photometry, metadata and GP features entirely. Every source gets
the same single dummy photometry token, so the transformer CLS output is a
constant and the latent becomes a pure function of the image tower:

    z = bottleneck( const + ImageTower(stamp) )

We then ask two questions:
  1. Untrained: are circle vs. square stamps already linearly separable in z?
     (Random CNN features usually are — this is the floor, not the goal.)
  2. Trained image-only with the joint classification head: does z become
     cleanly clustered by shape (probe accuracy, kNN accuracy, silhouette)?

Usage (from src/):
    python sanity_shapes.py                       # simple CNN, 512 stamps, 30 epochs
    python sanity_shapes.py --backbone zoobot     # needs timm + HF download
    python sanity_shapes.py --plot shapes.png     # PCA scatter before/after
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from model import LightCurveCompressor

STAMP = 63
CIRCLE, SQUARE = 0, 1
SHAPE_NAMES = {CIRCLE: "circle", SQUARE: "square"}


# ---------------------------------------------------------------------------
# Synthetic stamps
# ---------------------------------------------------------------------------


def make_shape_stamps(
    n: int,
    seed: int = 0,
    size: int = STAMP,
    noise: float = 0.1,
    min_r: int = 8,
    max_r: int = 16,
    jitter: int = 8,
):
    """Generate n (3, size, size) stamps, half circles and half squares.

    Shapes have matched area on average so that total flux alone does not
    separate the classes; radius/half-side, centre and brightness are random.

    Channels mimic (science, template, difference):
      science    = shape + noise
      template   = noise only
      difference = science - template
    Returns (images float32 (n,3,size,size), labels int64 (n,)).
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.int64)
    labels[n // 2 :] = SQUARE
    rng.shuffle(labels)

    yy, xx = np.mgrid[0:size, 0:size]
    images = np.zeros((n, 3, size, size), dtype=np.float32)
    c0 = size // 2

    for i, lab in enumerate(labels):
        cy = c0 + rng.integers(-jitter, jitter + 1)
        cx = c0 + rng.integers(-jitter, jitter + 1)
        amp = rng.uniform(0.6, 1.4)
        if lab == CIRCLE:
            r = rng.uniform(min_r, max_r)
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        else:
            # half-side chosen so mean area matches the circle: s = r*sqrt(pi)/2
            s = rng.uniform(min_r, max_r) * np.sqrt(np.pi) / 2
            mask = (np.abs(yy - cy) <= s) & (np.abs(xx - cx) <= s)
        shape = amp * mask.astype(np.float32)

        sci = shape + rng.normal(0, noise, (size, size)).astype(np.float32)
        tmpl = rng.normal(0, noise, (size, size)).astype(np.float32)
        images[i, 0] = sci
        images[i, 1] = tmpl
        images[i, 2] = sci - tmpl

    # Per-channel standardisation, as real stamps would be preprocessed
    mu = images.mean(axis=(0, 2, 3), keepdims=True)
    sd = images.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    images = (images - mu) / sd
    return images, labels


def dummy_photometry(n: int, in_channels: int = 7, device="cpu"):
    """One unpadded all-zero observation per source: identical for every row."""
    x = torch.zeros(n, 1, in_channels, device=device)
    pad_mask = torch.zeros(n, 1, dtype=torch.bool, device=device)
    return x, pad_mask


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def build_model(
    backbone: str = "simple",
    latent_dim: int = 16,
    d_model: int = 64,
    dropout: float = 0.1,
    mode: str = "ae",
):
    return LightCurveCompressor(
        mode=mode,
        latent_dim=latent_dim,
        in_channels=7,
        d_model=d_model,
        n_heads=4,
        enc_layers=1,
        dec_layers=1,
        d_ff=128,
        dropout=dropout,
        max_len=2,
        use_images=True,
        image_backbone=backbone,
        num_classes=2,
    )


@torch.no_grad()
def image_only_latent(model, images: torch.Tensor, batch_size: int = 256):
    """Deterministic latent for each stamp with photometry held constant."""
    model.eval()
    zs = []
    for i in range(0, images.shape[0], batch_size):
        chunk = images[i : i + batch_size]
        x, pad = dummy_photometry(chunk.shape[0], model.in_channels, chunk.device)
        zs.append(model.embed(x, pad, chunk))
    return torch.cat(zs).cpu().numpy()


@torch.no_grad()
def check_photometry_is_inert(model, n: int = 8) -> float:
    """Max spread of the latent across rows when *no* image is given.

    With identical dummy photometry every row must map to the same z, which
    confirms the image tower is the only thing that can vary the latent.
    """
    model.eval()
    x, pad = dummy_photometry(n, model.in_channels)
    z = model.embed(x, pad, None)
    return (z - z[0]).abs().max().item()


def latent_with_grad(model, x, pad_mask, images) -> torch.Tensor:
    """Differentiable latent. ``model.embed`` is wrapped in ``torch.no_grad``,
    so training must go through ``encode`` instead."""
    out = model.encode(x, pad_mask, images)
    return out[0]  # ae: z, vae: mu, vqvae: z_q (straight-through)


def train_image_only(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    log_every: int = 10,
    on_epoch=None,
):
    """Train ONLY the classification signal through the image tower.

    Loss = cross-entropy( cls_head(z) ). No reconstruction term: the dummy
    photometry is constant, so reconstructing it would be meaningless.

    ``on_epoch(epoch, mean_loss, train_acc)`` is called after every epoch.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    n = images.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot_loss, tot_correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            img, lab = images[idx], labels[idx]
            x, pad = dummy_photometry(img.shape[0], model.in_channels, img.device)
            z = latent_with_grad(model, x, pad, img)
            logits = model.cls_head(z)
            loss = F.cross_entropy(logits, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * img.shape[0]
            tot_correct += (logits.argmax(-1) == lab).sum().item()
        if log_every and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
            print(
                f"  epoch {epoch:3d}/{epochs}  loss {tot_loss / n:.4f}  "
                f"train acc {tot_correct / n:.3f}"
            )
        if on_epoch is not None:
            on_epoch(epoch, tot_loss / n, tot_correct / n)
    return model


# ---------------------------------------------------------------------------
# Latent separability metrics
# ---------------------------------------------------------------------------


def latent_metrics(z: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Linear probe, kNN, silhouette and class-centroid distance on latent z."""
    z_tr, z_te, y_tr, y_te = train_test_split(
        z, y, test_size=0.3, random_state=seed, stratify=y
    )
    probe = LogisticRegression(max_iter=2000).fit(z_tr, y_tr)
    knn = KNeighborsClassifier(n_neighbors=5).fit(z_tr, y_tr)

    mu0, mu1 = z[y == CIRCLE].mean(0), z[y == SQUARE].mean(0)
    within = 0.5 * (z[y == CIRCLE].std(0).mean() + z[y == SQUARE].std(0).mean())
    return {
        "probe_acc": float(probe.score(z_te, y_te)),
        "knn_acc": float(knn.score(z_te, y_te)),
        "silhouette": float(silhouette_score(z, y)),
        "centroid_sep": float(np.linalg.norm(mu0 - mu1) / (within + 1e-8)),
    }


def _fmt(m: dict) -> str:
    return (
        f"probe {m['probe_acc']:.3f} | kNN {m['knn_acc']:.3f} | "
        f"silhouette {m['silhouette']:+.3f} | centroid/within {m['centroid_sep']:.2f}"
    )


def plot_latents(z_before, z_after, y, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, z, title in zip(axes, (z_before, z_after), ("untrained", "trained")):
        p = PCA(n_components=2).fit_transform(z)
        for lab, c in ((CIRCLE, "tab:blue"), (SQUARE, "tab:orange")):
            m = y == lab
            ax.scatter(p[m, 0], p[m, 1], s=8, c=c, label=SHAPE_NAMES[lab], alpha=0.7)
        ax.set_title(f"latent PCA ({title})")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    n: int = 512,
    epochs: int = 30,
    backbone: str = "simple",
    latent_dim: int = 16,
    seed: int = 0,
    plot: str | None = None,
    verbose: bool = True,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    images_np, labels_np = make_shape_stamps(n, seed=seed)
    images = torch.from_numpy(images_np)
    labels = torch.from_numpy(labels_np)
    # Held-out stamps never seen in training, drawn with a different seed
    test_np, test_labels = make_shape_stamps(max(n // 2, 64), seed=seed + 1000)
    test_images = torch.from_numpy(test_np)

    model = build_model(backbone=backbone, latent_dim=latent_dim)
    n_params = sum(p.numel() for p in model.image_tower.parameters())
    if verbose:
        print(
            f"Backbone: {backbone} ({n_params / 1e6:.2f}M image-tower params), "
            f"latent_dim={latent_dim}, train stamps={n}, held-out={len(test_np)}"
        )

    spread = check_photometry_is_inert(model)
    if verbose:
        print(f"Latent spread with no image (should be ~0): {spread:.2e}")

    z0 = image_only_latent(model, test_images)
    before = latent_metrics(z0, test_labels, seed)
    if verbose:
        print(f"Untrained latent : {_fmt(before)}")

    t0 = time.time()
    if verbose:
        print(f"Training image-only for {epochs} epochs ...")
    train_image_only(
        model, images, labels, epochs=epochs, log_every=10 if verbose else 0
    )

    z1 = image_only_latent(model, test_images)
    after = latent_metrics(z1, test_labels, seed)
    if verbose:
        print(f"Trained latent   : {_fmt(after)}   ({time.time() - t0:.1f}s)")

    if plot:
        plot_latents(z0, z1, test_labels, plot)

    return {"photometry_spread": spread, "before": before, "after": after}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=512, help="number of training stamps")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--backbone", choices=["simple", "zoobot"], default="simple")
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default=None, help="save PCA scatter to this PNG")
    args = ap.parse_args()
    run(
        n=args.n,
        epochs=args.epochs,
        backbone=args.backbone,
        latent_dim=args.latent_dim,
        seed=args.seed,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
