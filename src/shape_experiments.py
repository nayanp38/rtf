"""
Sweeps for the cutout-only circle-vs-square baseline (see sanity_shapes.py).

Each experiment writes one JSON file with per-run metrics; ``plot`` turns the
JSON files into figures. Every run trains image-only (constant dummy
photometry) and scores the latent on a fixed held-out set of 512 stamps.

Usage (from src/):
    python shape_experiments.py curve     --out ../analysis/shapes
    python shape_experiments.py nstamps   --out ../analysis/shapes
    python shape_experiments.py latent    --out ../analysis/shapes
    python shape_experiments.py noise     --out ../analysis/shapes
    python shape_experiments.py mode      --out ../analysis/shapes
    python shape_experiments.py backbone  --out ../analysis/shapes   # needs timm
    python shape_experiments.py plot      --out ../analysis/shapes --fig ../docs/figures/shapes
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from sanity_shapes import (
    build_model,
    image_only_latent,
    latent_metrics,
    make_shape_stamps,
    train_image_only,
)

TEST_N = 512
TEST_SEED = 1000


def run_one(
    n: int = 512,
    epochs: int = 30,
    latent_dim: int = 16,
    backbone: str = "simple",
    mode: str = "ae",
    noise: float = 0.1,
    seed: int = 0,
    track_curve: bool = False,
    lr: float = 1e-3,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    imgs, labels = make_shape_stamps(n, seed=seed, noise=noise)
    test_imgs, test_labels = make_shape_stamps(TEST_N, seed=TEST_SEED, noise=noise)
    imgs, labels, test_imgs = map(torch.from_numpy, (imgs, labels, test_imgs))

    model = build_model(backbone=backbone, latent_dim=latent_dim, mode=mode)
    before = latent_metrics(image_only_latent(model, test_imgs), test_labels, seed)

    curve = []

    def on_epoch(epoch, loss, train_acc):
        if not track_curve:
            return
        m = latent_metrics(image_only_latent(model, test_imgs), test_labels, seed)
        curve.append({"epoch": epoch, "loss": loss, "train_acc": train_acc, **m})

    t0 = time.time()
    train_image_only(
        model, imgs, labels, epochs=epochs, lr=lr, log_every=0, on_epoch=on_epoch
    )
    after = latent_metrics(image_only_latent(model, test_imgs), test_labels, seed)
    return {
        "n": n,
        "epochs": epochs,
        "latent_dim": latent_dim,
        "backbone": backbone,
        "mode": mode,
        "noise": noise,
        "seed": seed,
        "before": before,
        "after": after,
        "curve": curve,
        "seconds": time.time() - t0,
    }


def _sweep(name, configs, out, seeds):
    results = []
    for cfg in configs:
        for seed in seeds:
            r = run_one(seed=seed, **cfg)
            results.append(r)
            print(
                f"[{name}] {cfg} seed={seed}: probe {r['before']['probe_acc']:.3f} "
                f"-> {r['after']['probe_acc']:.3f}, sil {r['after']['silhouette']:+.3f} "
                f"({r['seconds']:.0f}s)",
                flush=True,
            )
            with open(os.path.join(out, f"{name}.json"), "w") as f:
                json.dump(results, f, indent=1)
    return results


EXPERIMENTS = {
    "curve": lambda: [{"n": 512, "epochs": 40, "track_curve": True}],
    "nstamps": lambda: [{"n": n} for n in (64, 128, 256, 512, 1024, 2048)],
    "latent": lambda: [{"latent_dim": d} for d in (2, 4, 8, 16, 32, 64)],
    "noise": lambda: [{"noise": s} for s in (0.1, 0.25, 0.5, 1.0, 2.0)],
    # Same noise levels with 4x the training data: is the collapse a data
    # problem or a visibility problem?
    "noise_2048": lambda: [{"noise": s, "n": 2048} for s in (0.1, 0.25, 0.5, 1.0)],
    "mode": lambda: [{"mode": m} for m in ("ae", "vae", "vqvae")],
    # Zoobot (frozen ConvNeXt-pico at 224px) is ~70s/epoch on CPU, so keep short
    "backbone": lambda: [
        {"backbone": "simple", "epochs": 10},
        {"backbone": "zoobot", "epochs": 10},
    ],
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _load(out, name):
    path = os.path.join(out, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _agg(results, key, metric="probe_acc", stage="after"):
    """Group by config key -> (xs, mean, std)."""
    groups = {}
    for r in results:
        groups.setdefault(r[key], []).append(r[stage][metric])
    xs = sorted(groups)
    mean = np.array([np.mean(groups[x]) for x in xs])
    std = np.array([np.std(groups[x]) for x in xs])
    return xs, mean, std


def make_plots(out: str, fig_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
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
    # Categorical slots 1-2 of the reference palette + a neutral for "untrained"
    C_BLUE, C_ORANGE, C_GREY = "#2a78d6", "#eb6834", "#8a8a8a"
    saved = []

    def save(fig, name):
        path = os.path.join(fig_dir, name)
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    # --- 0. Example stamps -------------------------------------------------
    imgs, labels = make_shape_stamps(8, seed=7)
    fig, axes = plt.subplots(3, 8, figsize=(11, 4.2))
    for j in range(8):
        for i, ch in enumerate(("science", "template", "difference")):
            ax = axes[i, j]
            ax.imshow(imgs[j, i], cmap="gray", vmin=-1, vmax=4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if j == 0:
                ax.set_ylabel(ch)
            if i == 0:
                ax.set_title("circle" if labels[j] == 0 else "square", fontsize=10)
    fig.suptitle("Synthetic 63×63 stamps (area-matched, random size/position/flux)")
    save(fig, "00_stamps.png")

    # --- 1. Training curve ----------------------------------------------
    res = _load(out, "curve")
    if res:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for r in res:
            ep = [c["epoch"] for c in r["curve"]]
            axes[0].plot(ep, [c["loss"] for c in r["curve"]], color=C_BLUE, alpha=0.6)
            axes[1].plot(
                ep, [c["train_acc"] for c in r["curve"]], color=C_GREY, alpha=0.6
            )
            axes[1].plot(
                ep, [c["probe_acc"] for c in r["curve"]], color=C_BLUE, alpha=0.6
            )
            axes[2].plot(
                ep, [c["silhouette"] for c in r["curve"]], color=C_ORANGE, alpha=0.6
            )
        axes[0].set(title="Train loss (cross-entropy)", xlabel="epoch", yscale="log")
        axes[1].set(title="Accuracy", xlabel="epoch", ylim=(0.4, 1.02))
        axes[1].plot([], [], color=C_GREY, label="train (cls head)")
        axes[1].plot([], [], color=C_BLUE, label="held-out linear probe on z")
        axes[1].axhline(0.5, color=C_GREY, ls=":", lw=1)
        axes[1].legend(frameon=False, fontsize=9, loc="lower right")
        axes[2].set(title="Held-out silhouette of z", xlabel="epoch")
        axes[2].axhline(0, color=C_GREY, ls=":", lw=1)
        fig.suptitle(f"Image-only training, 512 stamps, {len(res)} seeds", y=1.02)
        save(fig, "01_training_curve.png")

    # --- 2..5. One-factor sweeps -------------------------------------------
    sweeps = [
        ("nstamps", "n", "training stamps", "02_data_efficiency.png", "log"),
        ("latent", "latent_dim", "latent dimension", "03_latent_dim.png", "log"),
        (
            "noise",
            "noise",
            "pixel noise σ (shape amp ≈ 0.6–1.4)",
            "04_noise.png",
            "log",
        ),
    ]
    for name, key, xlabel, fname, xscale in sweeps:
        res = _load(out, name)
        if not res:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, metric, title, base in (
            (axes[0], "probe_acc", "Held-out linear-probe accuracy", 0.5),
            (axes[1], "silhouette", "Held-out silhouette", 0.0),
        ):
            for stage, color, label, ls, mfc in (
                ("before", C_GREY, "untrained", "--", "white"),
                ("after", C_BLUE, "trained image-only", "-", None),
            ):
                xs, m, s = _agg(res, key, metric, stage)
                ax.errorbar(
                    xs,
                    m,
                    yerr=s,
                    marker="o",
                    ls=ls,
                    mfc=mfc,
                    color=color,
                    label=label,
                    capsize=3,
                    lw=1.8,
                )
            extra = _load(out, f"{name}_2048")
            if extra:
                xs2, m2, s2 = _agg(extra, key, metric, "after")
                ax.errorbar(
                    xs2,
                    m2,
                    yerr=s2,
                    marker="s",
                    color=C_ORANGE,
                    label="trained, 2048 stamps",
                    capsize=3,
                    lw=1.8,
                )
            ax.axhline(base, color=C_GREY, ls=":", lw=1)
            ax.set(title=title, xlabel=xlabel, xscale=xscale)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            ax.minorticks_off()
        axes[0].set_ylim(0.4, 1.02)
        axes[0].legend(frameon=False, fontsize=9)
        n_seeds = len({r["seed"] for r in res})
        fig.suptitle(f"mean ± std over {n_seeds} seeds", y=1.02)
        save(fig, fname)

    # --- 6. Categorical: bottleneck mode and backbone ----------------------
    for name, key, fname, title in (
        ("mode", "mode", "05_bottleneck_mode.png", "Bottleneck type"),
        ("backbone", "backbone", "06_backbone.png", "Image backbone (10 epochs)"),
    ):
        res = _load(out, name)
        if not res:
            continue
        cats = list(dict.fromkeys(r[key] for r in res))
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        for ax, metric, ylabel in (
            (axes[0], "probe_acc", "held-out probe accuracy"),
            (axes[1], "silhouette", "held-out silhouette"),
        ):
            x = np.arange(len(cats))
            for off, stage, color, label in (
                (-0.18, "before", C_GREY, "untrained"),
                (0.18, "after", C_BLUE, "trained"),
            ):
                vals = [[r[stage][metric] for r in res if r[key] == c] for c in cats]
                ax.bar(
                    x + off,
                    [np.mean(v) for v in vals],
                    0.34,
                    yerr=[np.std(v) for v in vals],
                    color=color,
                    label=label,
                    capsize=3,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(cats)
            ax.set_ylabel(ylabel)
            ax.grid(axis="x", visible=False)
        # Bars start at zero; chance is marked rather than used as the floor
        axes[0].set_ylim(0, 1.05)
        axes[0].axhline(0.5, color=C_GREY, ls=":", lw=1)
        axes[0].text(
            -0.45, 0.515, "chance", color=C_GREY, fontsize=8, va="bottom", ha="left"
        )
        axes[0].legend(frameon=False, fontsize=9, loc="upper left")
        fig.suptitle(title, y=1.02)
        save(fig, fname)

    # --- 7. Latent scatter before/after + per-class confusion --------------
    torch.manual_seed(0)
    imgs, labels = make_shape_stamps(512, seed=0)
    test_imgs, test_labels = make_shape_stamps(TEST_N, seed=TEST_SEED)
    imgs, labels, test_imgs = map(torch.from_numpy, (imgs, labels, test_imgs))
    model = build_model()
    z0 = image_only_latent(model, test_imgs)
    train_image_only(model, imgs, labels, epochs=30, log_every=0)
    z1 = image_only_latent(model, test_imgs)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, z, title in zip(axes, (z0, z1), ("untrained", "trained image-only")):
        p = PCA(n_components=2).fit(z)
        q = p.transform(z)
        for lab, c, nm in ((0, C_BLUE, "circle"), (1, C_ORANGE, "square")):
            m = test_labels == lab
            ax.scatter(q[m, 0], q[m, 1], s=7, c=c, alpha=0.65, label=nm, lw=0)
        ev = p.explained_variance_ratio_
        ax.set(
            title=f"latent PCA, {title}",
            xlabel=f"PC1 ({ev[0]:.0%} var)",
            ylabel=f"PC2 ({ev[1]:.0%} var)",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(frameon=False, markerscale=2.5)
    fig.suptitle("512 held-out stamps, 16-d latent", y=1.02)
    save(fig, "07_latent_pca.png")

    # Which held-out stamps does the trained latent still confuse? (by size)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    pred = cross_val_predict(LogisticRegression(max_iter=2000), z1, test_labels, cv=5)
    wrong = pred != test_labels
    # recover each stamp's blob area from the science channel (pixels > mid-level)
    sci = test_imgs[:, 0].numpy()
    area = (sci > (sci.min(axis=(1, 2), keepdims=True) + 1.0)).sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bins = np.quantile(area, np.linspace(0, 1, 6))
    for lab, c, nm in ((0, C_BLUE, "circle"), (1, C_ORANGE, "square")):
        m = test_labels == lab
        h_all, _ = np.histogram(area[m], bins)
        h_wrong, _ = np.histogram(area[m & wrong], bins)
        centers = 0.5 * (bins[1:] + bins[:-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(h_all > 0, h_wrong / np.maximum(h_all, 1), np.nan)
        ax.plot(centers, rate, marker="o", color=c, label=nm)
    ax.set(
        title="Probe error rate vs blob area (trained latent)",
        xlabel="blob area (pixels)",
        ylabel="error rate",
        ylim=(0, max(0.05, np.nanmax(rate) * 1.2) if np.isfinite(rate).any() else 1),
    )
    ax.legend(frameon=False)
    save(fig, "08_error_vs_size.png")

    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment", choices=list(EXPERIMENTS) + ["plot"])
    ap.add_argument("--out", default="../analysis/shapes")
    ap.add_argument("--fig", default="../docs/figures/shapes")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)

    if args.experiment == "plot":
        for p in make_plots(args.out, args.fig):
            print("saved", p)
        return
    _sweep(args.experiment, EXPERIMENTS[args.experiment](), args.out, range(args.seeds))


if __name__ == "__main__":
    main()
