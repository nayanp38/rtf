"""Cutout-only sanity check: circle vs. square separability in the latent.

Photometry is held constant (one dummy zero token per source), so the latent
depends only on the image tower. See src/sanity_shapes.py for the full CLI.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanity_shapes import (
    CIRCLE,
    SQUARE,
    build_model,
    check_photometry_is_inert,
    image_only_latent,
    latent_metrics,
    make_shape_stamps,
    train_image_only,
)


class TestShapeStamps:
    def test_shapes_and_balance(self):
        imgs, labels = make_shape_stamps(64, seed=0)
        assert imgs.shape == (64, 3, 63, 63)
        assert imgs.dtype == np.float32
        assert (labels == CIRCLE).sum() == (labels == SQUARE).sum() == 32

    def test_channel_semantics_noise_free(self):
        # With no noise the template is flat and difference == science exactly,
        # which survives per-channel standardisation.
        imgs, _ = make_shape_stamps(8, seed=0, noise=0.0)
        sci, tmpl, diff = imgs[:, 0], imgs[:, 1], imgs[:, 2]
        assert np.abs(tmpl).max() < 1e-3
        np.testing.assert_allclose(diff, sci, atol=1e-4)
        # Every stamp has a blob: two distinct intensity levels in science.
        for s in sci:
            assert (s > s.min() + 0.5).any()


class TestImageOnlyLatent:
    def test_photometry_branch_is_constant(self):
        model = build_model()
        assert check_photometry_is_inert(model) == 0.0

    def test_latent_varies_with_image(self):
        model = build_model(latent_dim=8)
        imgs, _ = make_shape_stamps(16, seed=0)
        z = image_only_latent(model, torch.from_numpy(imgs))
        assert z.shape == (16, 8)
        assert z.std(0).mean() > 0

    def test_gradient_reaches_image_tower(self):
        """Guards against training through the no-grad ``embed`` path."""
        model = build_model()
        imgs, labels = make_shape_stamps(32, seed=0)
        train_image_only(
            model,
            torch.from_numpy(imgs),
            torch.from_numpy(labels),
            epochs=1,
            log_every=0,
        )
        grads = [p.grad for p in model.image_tower.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)


class TestCircleVsSquare:
    """~30s on CPU: trains the simple CNN image-only on 512 stamps for 30 epochs.

    Seeds 0-2 give held-out probe accuracy 0.82-0.94 (0.99 with 1024 stamps);
    the untrained latent sits at chance.
    """

    def test_trained_latent_separates_shapes(self):
        torch.manual_seed(0)
        np.random.seed(0)
        imgs, labels = make_shape_stamps(512, seed=0)
        test_imgs, test_labels = make_shape_stamps(256, seed=1000)
        model = build_model()

        z0 = image_only_latent(model, torch.from_numpy(test_imgs))
        before = latent_metrics(z0, test_labels)

        train_image_only(
            model,
            torch.from_numpy(imgs),
            torch.from_numpy(labels),
            epochs=30,
            log_every=0,
        )
        z1 = image_only_latent(model, torch.from_numpy(test_imgs))
        after = latent_metrics(z1, test_labels)

        assert after["probe_acc"] > 0.75, after
        assert after["probe_acc"] > before["probe_acc"]
        assert after["silhouette"] > before["silhouette"]
