"""
Single-panel figures for the non-ML astronomer talk (docs/slides/outline_astronomers.md).

Each slide of that talk gets exactly one figure cropped to the one panel that
carries its point, regenerated here from the saved sweep JSON and latents.

Usage (from src/):
    python shape_talk_figures.py --results ../docs/figures/shapes/results \
        --latents ../analysis/shapes/novelty --fig ../docs/figures/shapes/talk
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sanity_shapes import make_shape_stamps
from shape_novelty import COLORS, SHAPES, make_stamps

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a8a"
AQUA = "#1baf7a"


def style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e5e5e5",
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _load(results, name):
    with open(os.path.join(results, f"{name}.json")) as f:
        return json.load(f)


def _agg(rows, key):
    g = {}
    for r in rows:
        g.setdefault(r[key], []).append(r["after"]["probe_acc"])
    xs = sorted(g)
    return xs, [np.mean(g[x]) for x in xs], [np.std(g[x]) for x in xs]


def fig_stamp_triplet(fig_dir):
    imgs, _ = make_shape_stamps(8, seed=7)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for ax, ch, name in zip(axes, range(3), ("science", "template", "difference")):
        ax.imshow(imgs[0, ch], cmap="gray", vmin=-1, vmax=4)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.savefig(os.path.join(fig_dir, "talk_stamp_triplet.png"))
    plt.close(fig)


def fig_stamps_row(fig_dir):
    imgs, labels = make_shape_stamps(40, seed=7)
    pick = list(np.where(labels == 0)[0][:3]) + list(np.where(labels == 1)[0][:3])
    fig, axes = plt.subplots(1, 6, figsize=(9, 1.9))
    for ax, i in zip(axes, pick):
        ax.imshow(imgs[i, 0], cmap="gray", vmin=-1, vmax=4)
        ax.set_title("circle" if labels[i] == 0 else "square", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.savefig(os.path.join(fig_dir, "talk_stamps_row.png"))
    plt.close(fig)


def fig_novel_row(fig_dir):
    imgs, labels, _ = make_stamps(2, SHAPES, seed=3)
    fig, axes = plt.subplots(1, 6, figsize=(9, 2.1))
    for s, ax in enumerate(axes):
        i = np.where(labels == s)[0][0]
        ax.imshow(imgs[i, 0], cmap="gray", vmin=-1, vmax=4)
        title = SHAPES[s] if s < 2 else f"{SHAPES[s]}\n(never seen)"
        ax.set_title(title, fontsize=11, color="black" if s < 2 else ORANGE)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if s >= 2:
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(ORANGE)
                sp.set_linewidth(2)
    fig.savefig(os.path.join(fig_dir, "talk_novel_row.png"))
    plt.close(fig)


def fig_data_efficiency(results, fig_dir):
    xs, m, s = _agg(_load(results, "nstamps"), "n")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.errorbar(xs, m, yerr=s, marker="o", color=BLUE, lw=2, capsize=3, ms=7)
    ax.axhline(0.5, color=GREY, ls=":", lw=1.2)
    ax.text(xs[0], 0.515, "chance", color=GREY, fontsize=11)
    ax.set(
        xscale="log",
        xlabel="labelled training examples",
        ylabel="held-out accuracy",
        ylim=(0.4, 1.03),
        title="From scratch: circle vs. square",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    fig.savefig(os.path.join(fig_dir, "talk_data_efficiency.png"))
    plt.close(fig)


def fig_noise(results, fig_dir):
    xs, m, s = _agg(_load(results, "noise"), "noise")
    xs2, m2, s2 = _agg(_load(results, "noise_2048"), "noise")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.errorbar(
        xs,
        m,
        yerr=s,
        marker="o",
        color=BLUE,
        lw=2,
        capsize=3,
        ms=7,
        label="512 examples",
    )
    ax.errorbar(
        xs2,
        m2,
        yerr=s2,
        marker="s",
        color=ORANGE,
        lw=2,
        capsize=3,
        ms=7,
        label="2048 examples",
    )
    ax.axhline(0.5, color=GREY, ls=":", lw=1.2)
    ax.set(
        xscale="log",
        xlabel="pixel noise (fraction of blob brightness)",
        ylabel="held-out accuracy",
        ylim=(0.4, 1.03),
        title="From scratch: adding noise",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.legend(frameon=False)
    fig.savefig(os.path.join(fig_dir, "talk_noise.png"))
    plt.close(fig)


def fig_backbone(results, fig_dir):
    rows = _load(results, "backbone")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cats = ["simple", "zoobot"]
    names = ["from scratch", "Zoobot (Galaxy Zoo)"]
    x = np.arange(2)
    for off, stage, color, label in (
        (-0.19, "before", GREY, "before any training"),
        (0.19, "after", BLUE, "after training"),
    ):
        vals = [
            [r[stage]["probe_acc"] for r in rows if r["backbone"] == c] for c in cats
        ]
        bars = ax.bar(
            x + off,
            [np.mean(v) for v in vals],
            0.36,
            yerr=[np.std(v) for v in vals],
            color=color,
            label=label,
            capsize=3,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                np.mean(v) + 0.03,
                f"{np.mean(v):.0%}",
                ha="center",
                fontsize=11,
            )
    ax.axhline(0.5, color=GREY, ls=":", lw=1.2)
    ax.text(-0.55, 0.515, "chance", color=GREY, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("held-out accuracy")
    ax.set_title("Circle vs. square, 512 examples")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=11,
    )
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(fig_dir, "talk_backbone.png"))
    plt.close(fig)


def _latent(latents, regime):
    d = np.load(os.path.join(latents, f"{regime}_seed0_latents.npz"))
    z, y = d["z_test"], d["y_test"]
    return (z - z.mean(0)) / (z.std(0) + 1e-6), y


def _scatter(ax, e, y, label_clusters=True):
    for s in range(6):
        m = y == s
        ax.scatter(e[m, 0], e[m, 1], s=9, c=COLORS[s], alpha=0.65, lw=0)
        if label_clusters:
            cx, cy = np.median(e[m, 0]), np.median(e[m, 1])
            ax.text(
                cx,
                cy,
                SHAPES[s],
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.75,
                },
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def fig_tsne_pair(latents, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, regime, title in zip(
        axes,
        ("cls", "zoobot"),
        ("From scratch, trained on circle vs. square", "Zoobot, no training at all"),
    ):
        z, y = _latent(latents, regime)
        e = TSNE(n_components=2, random_state=0, init="pca").fit_transform(z)
        _scatter(ax, e, y, label_clusters=(regime == "zoobot"))
        ax.set_title(title, fontsize=12)
        if regime == "cls":
            ax.text(
                0.03,
                0.97,
                "all shapes mixed along one curve;\nonly squares are set apart",
                transform=ax.transAxes,
                fontsize=11,
                color="#3a4249",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": "white",
                    "ec": "#c9ced3",
                    "alpha": 0.95,
                },
            )
    fig.suptitle(
        "Fingerprints of 1536 held-out stamps: t-SNE view of the 16 numbers",
        fontsize=12,
        y=1.0,
    )
    fig.savefig(os.path.join(fig_dir, "talk_tsne_pair.png"))
    plt.close(fig)


def fig_pca_pair(latents, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, regime, title in zip(
        axes,
        ("zoobot", "zoobot_cls"),
        ("Zoobot, no training", "Zoobot, after learning circle vs. square"),
    ):
        z, y = _latent(latents, regime)
        p = PCA(n_components=2).fit(z)
        e = p.transform(z)
        _scatter(ax, e, y, label_clusters=False)
        ev = p.explained_variance_ratio_.sum()
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(
            f"these two axes hold {ev:.0%} of the fingerprint's spread",
            fontsize=10,
            color=GREY,
        )
    axes[1].text(
        0.03,
        0.04,
        "ring lands on top of circle",
        transform=axes[1].transAxes,
        fontsize=10,
        color=GREY,
        va="bottom",
    )
    fig.suptitle(
        "Fingerprints of 1536 held-out stamps: 2 largest principal components",
        fontsize=12,
        y=1.0,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLORS[s], ms=8, label=SHAPES[s])
        for s in range(6)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=6,
        frameon=False,
        fontsize=11,
    )
    fig.savefig(os.path.join(fig_dir, "talk_pca_pair.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Illustrations of the real inputs (mock data, for the background slides)
# ---------------------------------------------------------------------------


def _mock_field(seed=0, size=63, transient=True):
    """Host galaxy (elliptical exponential profile) + optional transient + noise.

    Returns (science, template, difference) with the difference built the way
    the pipeline would deliver it: transient + fresh noise, host cancelled.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cy, cx = size / 2 + 1, size / 2 - 2
    theta, q, re = 0.6, 0.55, 7.0
    dy, dx = yy - cy, xx - cx
    u = np.cos(theta) * dx + np.sin(theta) * dy
    v = -np.sin(theta) * dx + np.cos(theta) * dy
    r = np.sqrt(u**2 + (v / q) ** 2)
    host = 2.2 * np.exp(-r / re)
    # a faint neighbouring star
    star = 1.6 * np.exp(-((yy - 12) ** 2 + (xx - 50) ** 2) / (2 * 1.4**2))
    ty, tx = cy - 6, cx + 7
    psf = np.exp(-((yy - ty) ** 2 + (xx - tx) ** 2) / (2 * 1.6**2))
    src = 2.4 * psf if transient else 0 * psf
    noise = 0.12
    sci = host + star + src + rng.normal(0, noise, (size, size))
    tmpl = host + star + rng.normal(0, noise / 2, (size, size))
    diff = src + rng.normal(0, noise * 1.2, (size, size))
    return sci, tmpl, diff, (ty, tx), (cy, cx)


def fig_triplet_explained(fig_dir):
    sci, tmpl, diff, _, _ = _mock_field(seed=1)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, img, title, sub in zip(
        axes,
        (sci, tmpl, diff),
        ("science", "template", "difference"),
        ("tonight", "stacked earlier images", "what changed"),
    ):
        ax.imshow(img, cmap="gray", vmin=-0.4, vmax=3.0, origin="lower")
        ax.set_title(f"{title}\n{sub}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0].annotate(
        "transient",
        xy=(37, 25),
        xytext=(44, 50),
        color=ORANGE,
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5},
    )
    axes[0].annotate(
        "host galaxy",
        xy=(27, 36),
        xytext=(4, 55),
        color="white",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5},
    )
    axes[1].text(3, 4, "no transient yet", color="white", fontsize=11, va="bottom")
    axes[2].text(3, 4, "host cancels out", color="white", fontsize=11, va="bottom")
    fig.text(0.353, 0.52, "$-$", fontsize=26, ha="center", va="center")
    fig.text(0.665, 0.52, "$=$", fontsize=26, ha="center", va="center")
    fig.subplots_adjust(wspace=0.18)
    fig.savefig(os.path.join(fig_dir, "talk_triplet_explained.png"))
    plt.close(fig)


def fig_light_curve(fig_dir):
    """Mock SN-like light curve in g/r/i with irregular cadence, in magnitudes."""
    rng = np.random.default_rng(3)

    def bazin(t, a, t0, rise, fall):
        return a * np.exp(-(t - t0) / fall) / (1 + np.exp(-(t - t0) / rise))

    bands = {
        "g": {"c": BLUE, "a": 1.0, "off": 0.0, "fall": 18.0},
        "r": {"c": ORANGE, "a": 1.1, "off": 0.15, "fall": 26.0},
        "i": {"c": "#008300", "a": 0.9, "off": 0.35, "fall": 32.0},
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    t_grid = np.linspace(-15, 80, 400)
    limit = 20.8  # detection limit (mag)
    for name, b in bands.items():
        # irregular cadence: ~every 3 nights with a 12-night weather gap
        nights = np.arange(-8, 80, 3) + rng.uniform(-1, 1, 30)
        nights = nights[(nights < 22) | (nights > 34)]
        nights = nights[rng.random(len(nights)) < 0.75]
        f_true = bazin(t_grid, b["a"], 0.0, 3.5, b["fall"])
        m_true = 18.0 + b["off"] - 2.5 * np.log10(np.clip(f_true, 1e-3, None))
        f = bazin(nights, b["a"], 0.0, 3.5, b["fall"])
        err_f = 0.04 + 0.02 * rng.random(len(nights))
        f_obs = f + rng.normal(0, err_f)
        det = f_obs > 3 * err_f
        m = 18.0 + b["off"] - 2.5 * np.log10(np.clip(f_obs[det], 1e-3, None))
        m_err = 1.086 * err_f[det] / f_obs[det]
        ax.plot(t_grid, m_true, color=b["c"], lw=1, alpha=0.35)
        ax.errorbar(
            nights[det],
            m,
            yerr=m_err,
            fmt="o",
            color=b["c"],
            ms=5,
            capsize=2,
            label=f"{name} band",
        )
    ax.axhline(limit, color=GREY, ls=":", lw=1.2)
    ax.text(-14, limit - 0.08, "detection limit", color=GREY, fontsize=10)
    ax.axvspan(22, 34, color=GREY, alpha=0.12, lw=0)
    ax.text(28, 17.6, "weather\ngap", color=GREY, fontsize=10, ha="center")
    ax.annotate(
        "first detection",
        xy=(-8.5, 20.0),
        xytext=(-14, 21.15),
        fontsize=10,
        va="bottom",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
    )
    ax.annotate(
        "peak",
        xy=(4, 18.4),
        xytext=(10, 17.4),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
    )
    ax.set(
        xlabel="days since first detection",
        ylabel="magnitude (brighter $\\uparrow$)",
        ylim=(21.5, 17.0),
        xlim=(-15, 80),
        title="A supernova-like light curve",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.savefig(os.path.join(fig_dir, "talk_light_curve.png"))
    plt.close(fig)


def fig_metadata_annotated(fig_dir):
    sci, _, _, (ty, tx), (cy, cx) = _mock_field(seed=1)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.imshow(sci, cmap="gray", vmin=-0.4, vmax=3.0, origin="lower")
    # white margins around the image so the boxes never cover the pixels
    ax.set_xlim(-48, 110)
    ax.set_ylim(-4, 70)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    box = {"boxstyle": "round,pad=0.35", "fc": "white", "ec": ORANGE, "lw": 1.2}
    arrow = {"arrowstyle": "->", "color": ORANGE, "lw": 1.5}
    ax.annotate(
        'nearest catalogued source\n  distance 2.1"   r = 18.4\n  star/galaxy score 0.08\n  (a galaxy)',
        xy=(cx, cy),
        xytext=(-47, 60),
        fontsize=10,
        bbox=box,
        arrowprops=arrow,
        va="top",
    )
    ax.annotate(
        'the detection\n  PSF width 2.0"  (point-like)\n  real/bogus score 0.94\n  4 detections so far',
        xy=(tx, ty),
        xytext=(67, 50),
        fontsize=10,
        bbox=box,
        arrowprops=arrow,
        va="top",
    )
    ax.annotate(
        'nearby star\n  distance 29"   r = 16.9',
        xy=(50, 12),
        xytext=(67, 14),
        fontsize=10,
        bbox=box,
        arrowprops=arrow,
        va="top",
    )
    ax.set_title("What the pipeline already knows about this spot", fontsize=12)
    fig.savefig(os.path.join(fig_dir, "talk_metadata_annotated.png"))
    plt.close(fig)


def fig_alert_flood(fig_dir):
    """The problem slide: alert flood -> broker filter -> served / discarded."""
    from matplotlib.patches import FancyBboxPatch, Polygon

    rng = np.random.default_rng(5)
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    PANEL = "#f3f6f9"
    RULE = "#d9dee3"

    def box(x, y, w, h, fc=PANEL, ec=RULE):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                fc=fc, ec=ec, lw=1.2, zorder=1,
            )
        )

    # --- tonight's alerts: a dense flood ---
    box(0.25, 0.35, 3.7, 3.4)
    n = 3000
    xs = rng.uniform(0.35, 3.85, n)
    ys = rng.uniform(0.45, 3.65, n)
    ax.scatter(xs, ys, s=5, c=BLUE, alpha=0.45, lw=0, zorder=2)
    ax.text(2.1, 4.0, "Tonight: ~10 million alerts", ha="center", fontsize=13,
            fontweight="bold", color="#1f2a33")
    ax.text(2.1, 0.12, "~100 MB/s, every night", ha="center", fontsize=11,
            color=GREY)

    # --- the broker funnel ---
    ax.add_patch(
        Polygon([(4.0, 3.3), (6.4, 2.65), (6.4, 1.55), (4.0, 0.8)],
                closed=True, fc="#e4e8ec", ec=RULE, lw=1.2, zorder=1)
    )
    ax.text(5.2, 2.1, "filter", ha="center", va="center", fontsize=12,
            color="#1f2a33", zorder=3)

    # --- what reaches you ---
    box(6.6, 2.35, 3.65, 1.45)
    kx = rng.uniform(6.75, 10.1, 30)
    ky = rng.uniform(2.5, 3.65, 30)
    ax.scatter(kx, ky, s=14, c=BLUE, lw=0, zorder=2)
    ax.text(8.42, 4.0, "Processed alerts: ~1%", ha="center", fontsize=13,
            fontweight="bold", color="#1f2a33")
    ax.text(8.42, 2.15, "~1 MB/s, one HD video stream", ha="center", fontsize=11,
            color=GREY)

    # --- the discarded majority, with the one that mattered ---
    box(6.6, 0.35, 3.65, 1.35, fc="#fafafa")
    dx = rng.uniform(6.75, 10.1, 900)
    dy = rng.uniform(0.45, 1.6, 900)
    ax.scatter(dx, dy, s=5, c="#b8bec4", alpha=0.6, lw=0, zorder=2)
    ax.scatter([8.9], [1.0], marker="*", s=260, c=AQUA, ec="white", lw=0.8,
               zorder=4)
    ax.text(8.42, 0.12, "discarded: ~99%", ha="center", fontsize=11, color=GREY)
    ax.annotate(
        "missed discovery",
        xy=(8.9, 1.0), xytext=(5.55, 0.45), fontsize=11, color=AQUA,
        ha="center", va="center",
        arrowprops={"arrowstyle": "->", "color": AQUA, "lw": 1.5,
                    "connectionstyle": "arc3,rad=-0.25"},
        zorder=5,
    )

    # arrows: flood -> funnel -> kept / discarded
    ax.annotate("", xy=(6.6, 3.08), xytext=(6.4, 2.6),
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.5})
    ax.annotate("", xy=(6.6, 1.1), xytext=(6.4, 1.6),
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.5})
    fig.savefig(os.path.join(fig_dir, "talk_alert_flood.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../docs/figures/shapes/results")
    ap.add_argument("--latents", default="../analysis/shapes/novelty")
    ap.add_argument("--fig", default="../docs/figures/shapes/talk")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    style()
    fig_alert_flood(a.fig)
    fig_triplet_explained(a.fig)
    fig_light_curve(a.fig)
    fig_metadata_annotated(a.fig)
    fig_stamp_triplet(a.fig)
    fig_stamps_row(a.fig)
    fig_novel_row(a.fig)
    fig_data_efficiency(a.results, a.fig)
    fig_noise(a.results, a.fig)
    fig_backbone(a.results, a.fig)
    fig_tsne_pair(a.latents, a.fig)
    fig_pca_pair(a.latents, a.fig)
    print("saved", sorted(os.listdir(a.fig)))


if __name__ == "__main__":
    main()
