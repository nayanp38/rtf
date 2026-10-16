"""Tests for the novel-shape discovery experiment (src/shape_novelty.py)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanity_shapes import build_model
from shape_novelty import (
    NOVEL,
    SHAPES,
    encode,
    evaluate_latent,
    make_stamps,
    shape_mask,
    train_regime,
)


class TestStamps:
    def test_all_shapes_area_matched(self):
        yy, xx = np.mgrid[0:63, 0:63]
        for kind in SHAPES:
            for r, theta in ((8, 0.0), (12, 0.7), (16, 2.1)):
                area = shape_mask(kind, yy, xx, 31, 31, r, theta).sum()
                assert abs(area / (np.pi * r * r) - 1) < 0.12, (kind, r, area)

    def test_labels_and_shared_standardisation(self):
        tr, tr_lab, stats = make_stamps(4, ["circle", "square"], seed=0)
        te, te_lab, _ = make_stamps(2, SHAPES, seed=1, stats=stats)
        assert tr.shape == (8, 3, 63, 63) and te.shape == (12, 3, 63, 63)
        assert set(tr_lab.tolist()) == {0, 1}
        assert sorted(set(te_lab.tolist())) == list(range(6))
        # the held-out set is standardised with the training stats, not its own
        assert abs(float(te.mean())) > 1e-3 or abs(float(te.std()) - 1) > 1e-3


class TestLatentEvaluation:
    def test_metrics_separate_perfect_from_random(self):
        rng = np.random.default_rng(0)
        y_tr = np.repeat([0, 1], 50)
        y_te = np.repeat(range(6), 30)
        # perfectly clustered latent: each shape at its own corner
        centres = np.eye(6, 8)
        z_tr = centres[y_tr] + 0.01 * rng.normal(size=(100, 8))
        z_te = centres[y_te] + 0.01 * rng.normal(size=(180, 8))
        good = evaluate_latent(z_tr, y_tr, z_te, y_te)
        assert good["probe6_acc"] > 0.95
        assert good["novelty_auroc_knn"] > 0.95
        assert good["cluster_ami"] > 0.9
        # pure noise latent: everything at chance
        z_tr, z_te = rng.normal(size=(100, 8)), rng.normal(size=(180, 8))
        bad = evaluate_latent(z_tr, y_tr, z_te, y_te)
        assert bad["probe6_acc"] < 0.4
        assert abs(bad["novelty_auroc_knn"] - 0.5) < 0.15
        assert bad["cluster_ami"] < 0.1
        for s in NOVEL:
            assert SHAPES[s] in good["novelty_auroc_knn_per_shape"]

    def test_train_and_encode_roundtrip(self):
        model = build_model(latent_dim=8)
        imgs, labs, _ = make_stamps(16, ["circle", "square"], seed=0)
        imgs_t, labs_t = torch.from_numpy(imgs), torch.from_numpy(labs)
        decoder = train_regime(model, "both", imgs_t, labs_t, epochs=1, log_every=0)
        assert decoder is not None
        z, conf = encode(model, imgs_t)
        assert z.shape == (32, 8) and conf.shape == (32,)
        assert np.all((conf >= 0.5) & (conf <= 1.0))
