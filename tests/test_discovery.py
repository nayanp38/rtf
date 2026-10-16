"""Tests for the discovery-preservation evaluation pipeline.

Covers transient_properties (property extraction + feature-layout round trips),
transient_injection (flux-space injection, nulls, selection, synthetic data)
and eval_discovery (encoding, agreement metrics, probes, injection-recovery,
checkpoint config inference, and the CLI end to end).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import eval_discovery as ed
from dataset import MetaNPZDataset, collate_fn
from model import LightCurveCompressor
from transient_injection import (
    SYNTH_CLASSES,
    TRANSIENT_PRESETS,
    bazin_peak_offset,
    host_reference_mag,
    inject_transient,
    make_synthetic_dataset,
    peak_mag_per_band,
    renoise,
    sample_transient_params,
    select_mask,
    synth_light_curve,
    transient_shape,
)
from transient_properties import (
    PROPERTY_NAMES,
    compute_properties,
    physical_to_x,
    properties_batch,
    properties_from_x,
    recon_to_x,
    set_photometry,
    x_to_physical,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bazin_curve(rng, t_peak=20.0, peak_mag=17.5, color=0.2, n=60, span=90.0):
    """Noiseless two-band Bazin light curve with a known peak."""
    t = np.sort(rng.uniform(0.0, span, n))
    t[0] = 0.0
    band = rng.choice([0, 1], size=n)
    params = {
        "model": "bazin",
        "t_peak": t_peak,
        "tau_rise": 5.0,
        "tau_fall": 25.0,
        "color": color,
    }
    pm = peak_mag_per_band(peak_mag, color)
    flux = 10.0 ** (-0.4 * pm[band]) * transient_shape(t, params)
    mag = -2.5 * np.log10(flux)
    mag_err = np.full(n, 0.05)
    return t, mag, mag_err, band, params


def tiny_model(mode="ae", in_channels=37, latent_dim=8, **kw):
    return LightCurveCompressor(
        mode=mode,
        latent_dim=latent_dim,
        in_channels=in_channels,
        d_model=32,
        n_heads=4,
        enc_layers=1,
        dec_layers=1,
        d_ff=64,
        **kw,
    ).eval()


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    counts = make_synthetic_dataset(out, n_per_class=12, seed=3)
    assert sum(counts.values()) == 12 * len(SYNTH_CLASSES)
    return out


# ---------------------------------------------------------------------------
# transient_properties
# ---------------------------------------------------------------------------


class TestFeatureLayout:
    def test_roundtrip_physical_to_x(self):
        rng = np.random.default_rng(0)
        t, mag, mag_err, band, _ = bazin_curve(rng)
        x = physical_to_x(t, mag, mag_err, band)
        assert x.shape == (len(t), 7)
        assert x.dtype == np.float32
        t2, mag2, err2, band2 = x_to_physical(x)
        assert np.allclose(t2, t, atol=1e-3)
        assert np.allclose(mag2, mag, atol=1e-4)
        assert np.allclose(err2, mag_err, atol=1e-5)
        assert np.array_equal(band2, band)

    def test_roundtrip_with_norm_and_extra(self):
        rng = np.random.default_rng(1)
        t, mag, mag_err, band, _ = bazin_curve(rng, n=20)
        norm = (np.array([1.0, 0.5, -7.0, 0.02]), np.array([1.5, 0.7, 0.4, 0.01]))
        extra = rng.normal(size=(20, 30))
        x = physical_to_x(t, mag, mag_err, band, extra=extra, norm=norm)
        assert x.shape == (20, 37)
        assert np.allclose(x[:, 7:], extra, atol=1e-5)
        t2, mag2, _, _ = x_to_physical(x, norm)
        assert np.allclose(mag2, mag, atol=1e-3)
        assert np.allclose(t2, t, atol=1e-2)

    def test_set_photometry_only_touches_flux_channels(self):
        rng = np.random.default_rng(2)
        t, mag, mag_err, band, _ = bazin_curve(rng, n=15)
        extra = rng.normal(size=(15, 30))
        x = physical_to_x(t, mag, mag_err, band, extra=extra)
        x_new = set_photometry(x, mag - 1.0, mag_err * 0.5)
        assert x_new is not x
        assert np.array_equal(x_new[:, :2], x[:, :2])
        assert np.array_equal(x_new[:, 4:], x[:, 4:])
        _, mag2, err2, _ = x_to_physical(x_new)
        assert np.allclose(mag2, mag - 1.0, atol=1e-4)
        assert np.allclose(err2, mag_err * 0.5, atol=1e-5)

    def test_recon_to_x(self):
        cont = torch.randn(2, 5, 4)
        logits = torch.zeros(2, 5, 3)
        logits[..., 1] = 1.0
        x = recon_to_x(cont, logits)
        assert x.shape == (2, 5, 7)
        assert np.allclose(x[..., :4], cont.numpy())
        assert (x[..., 4:].argmax(-1) == 1).all()

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            x_to_physical(np.zeros((5, 4)))
        with pytest.raises(ValueError):
            physical_to_x([1.0, 0.5], [18, 18], [0.1, 0.1], [0, 1])


class TestProperties:
    def test_known_bazin_peak(self):
        rng = np.random.default_rng(0)
        t, mag, mag_err, band, _ = bazin_curve(rng, t_peak=20.0, peak_mag=17.5)
        p = compute_properties(t, mag, mag_err, band)
        assert set(p) == set(PROPERTY_NAMES)
        # The brightest band is r (colour > 0); peak within ~1.5 days of truth
        # given a random cadence, and within 0.05 mag of the true r peak.
        assert abs(p["t_peak"] - 20.0) < 3.0
        assert abs(p["peak_mag"] - 17.5) < 0.08
        assert p["rise_rate"] > 0
        assert p["decline_rate"] > 0
        assert p["amplitude"] > 1.0
        assert p["duration"] == pytest.approx(t.max() - t.min())
        assert p["mag_first"] == pytest.approx(mag[0])
        assert abs(p["color_peak"] - 0.2) < 0.15
        assert abs(p["mean_color"] - 0.2) < 0.15
        assert 0 < p["t_half_width"] < p["duration"]

    def test_order_invariance(self):
        rng = np.random.default_rng(4)
        t, mag, mag_err, band, _ = bazin_curve(rng)
        p1 = compute_properties(t, mag, mag_err, band)
        perm = rng.permutation(len(t))
        p2 = compute_properties(t[perm], mag[perm], mag_err[perm], band[perm])
        for k in PROPERTY_NAMES:
            assert np.isclose(p1[k], p2[k], equal_nan=True)

    def test_flat_curve(self):
        t = np.linspace(0, 50, 30)
        mag = np.full(30, 18.0)
        p = compute_properties(t, mag, np.full(30, 0.1), np.zeros(30, dtype=int))
        assert p["amplitude"] == 0.0
        assert p["rise_rate"] == 0.0 or np.isnan(p["rise_rate"])
        assert np.isnan(p["color_peak"])  # single band -> no colour
        assert p["t_half_width"] == pytest.approx(50.0)

    def test_undefined_cases(self):
        p = compute_properties([], [], [], [])
        assert all(np.isnan(v) for v in p.values())
        p = compute_properties([0.0, 0.1], [18.0, 17.0], [0.1, 0.1], [0, 1])
        assert np.isnan(p["rise_rate"])  # < 3 points, span < 1 day
        assert np.isfinite(p["peak_mag"])

    def test_properties_from_x_with_padding(self):
        rng = np.random.default_rng(5)
        t, mag, mag_err, band, _ = bazin_curve(rng, n=25)
        x = physical_to_x(t, mag, mag_err, band)
        x_pad = np.zeros((40, 7), dtype=np.float32)
        x_pad[:25] = x
        valid = np.zeros(40, dtype=bool)
        valid[:25] = True
        p_ref = compute_properties(t, mag, mag_err, band)
        p = properties_from_x(x_pad, valid=valid)
        for k in PROPERTY_NAMES:
            assert np.isclose(p[k], p_ref[k], atol=1e-3, equal_nan=True)
        P = properties_batch(x_pad[None], valid[None])
        assert P.shape == (1, len(PROPERTY_NAMES))
        assert np.isclose(P[0, PROPERTY_NAMES.index("peak_mag")], p_ref["peak_mag"])


# ---------------------------------------------------------------------------
# transient_injection
# ---------------------------------------------------------------------------


class TestTransientShapes:
    @pytest.mark.parametrize("preset", sorted(TRANSIENT_PRESETS))
    def test_shape_peaks_at_one(self, preset):
        rng = np.random.default_rng(0)
        params = sample_transient_params(preset, rng, t_peak=30.0)
        t = np.linspace(-20, 200, 20001)
        f = transient_shape(t, params)
        assert np.isclose(f.max(), 1.0, atol=1e-3)
        assert abs(t[np.argmax(f)] - 30.0) < 0.05
        assert (f >= 0).all()

    def test_bazin_peak_offset(self):
        assert bazin_peak_offset(5.0, 25.0) == pytest.approx(5.0 * np.log(4.0))

    def test_sample_overrides_and_validation(self):
        rng = np.random.default_rng(0)
        p = sample_transient_params("sn_ia", rng, t_peak=3.0, tau_rise=4.0, color=0.1)
        assert p["tau_rise"] == 4.0 and p["color"] == 0.1 and p["t_peak"] == 3.0
        p = sample_transient_params("sn_ia", rng, tau_rise=50.0, tau_fall=10.0)
        assert p["tau_fall"] > p["tau_rise"]
        with pytest.raises(ValueError):
            sample_transient_params("nope", rng)


class TestInjection:
    def test_flux_addition(self):
        rng = np.random.default_rng(0)
        L = 40
        t = np.sort(rng.uniform(0, 60, L))
        t[0] = 0
        band = rng.choice([0, 1], size=L)
        mag = np.full(L, 19.0)
        err = np.full(L, 0.1)
        extra = rng.normal(size=(L, 30))
        x = physical_to_x(t, mag, err, band, extra=extra)
        params = sample_transient_params("sn_ia", rng, t_peak=30.0, color=0.0)
        x_inj, info = inject_transient(x, 19.0, params)  # delta_mag = 0 at peak

        assert x_inj.shape == x.shape
        assert np.array_equal(x_inj[:, :2], x[:, :2])  # cadence untouched
        assert np.array_equal(x_inj[:, 4:], x[:, 4:])  # bands + metadata untouched
        _, mag_new, err_new, _ = x_to_physical(x_inj)
        expected = -2.5 * np.log10(
            10 ** (-0.4 * 19.0) * (1 + transient_shape(t, params))
        )
        assert np.allclose(mag_new, expected, atol=1e-4)
        assert (mag_new <= mag + 1e-6).all()  # only ever brighter
        assert (err_new <= err + 1e-6).all()  # errors shrink with flux
        assert info["snr_peak"] > 5
        assert info["peak_mag_r"] == 19.0
        assert info["preset"] == "sn_ia"

        # The injected peak is measurable with the property code.
        p = properties_from_x(x_inj)
        assert abs(p["t_peak"] - 30.0) < 3.0
        assert abs(p["peak_mag"] - (19.0 - 2.5 * np.log10(2))) < 0.05

    def test_injection_with_norm(self):
        rng = np.random.default_rng(1)
        t, mag, err, band, _ = bazin_curve(rng, n=30)
        norm = (np.array([0.5, 0.2, -7.2, 0.02]), np.array([1.0, 0.5, 0.3, 0.01]))
        x = physical_to_x(t, mag, err, band, norm=norm)
        params = sample_transient_params("kilonova", rng, t_peak=10.0)
        x_inj, _ = inject_transient(x, 16.0, params, norm=norm)
        _, mag_a, _, _ = x_to_physical(x, norm)
        _, mag_b, _, _ = x_to_physical(x_inj, norm)
        assert (mag_b <= mag_a + 1e-5).all()
        assert mag_b.min() < mag_a.min()

    def test_add_noise_is_reproducible(self):
        rng = np.random.default_rng(0)
        t, mag, err, band, _ = bazin_curve(rng, n=30)
        x = physical_to_x(t, mag, err, band)
        params = sample_transient_params("flare", rng, t_peak=10.0)
        a, _ = inject_transient(
            x, 17.0, params, rng=np.random.default_rng(7), add_noise=True
        )
        b, _ = inject_transient(
            x, 17.0, params, rng=np.random.default_rng(7), add_noise=True
        )
        c, _ = inject_transient(x, 17.0, params, add_noise=False)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_renoise_and_reference_mag(self):
        rng = np.random.default_rng(0)
        t, mag, err, band, _ = bazin_curve(rng, n=30)
        x = physical_to_x(t, mag, err, band)
        x_null = renoise(x, np.random.default_rng(1))
        _, mag_n, err_n, _ = x_to_physical(x_null)
        assert np.array_equal(x_null[:, :2], x[:, :2])
        assert np.allclose(err_n, err, atol=1e-5)
        z = (mag_n - mag) / err
        assert np.abs(z).max() < 5.0
        assert np.abs(z).mean() > 0.2
        ref = host_reference_mag(x)
        assert ref == pytest.approx(np.median(mag[band == 1]))
        # Falls back to all bands if r is missing
        x_g = physical_to_x(t, mag, err, np.zeros_like(band))
        assert host_reference_mag(x_g) == pytest.approx(np.median(mag))


class TestSelection:
    def test_label_and_property_terms(self):
        lc = np.array([0, 1, 2, 3, 4, -1])
        lf = np.array([0, 3, 7, 8, 9, -1])
        props = np.full((6, len(PROPERTY_NAMES)), np.nan)
        props[:, PROPERTY_NAMES.index("amplitude")] = [2.0, 0.5, 3.0, 0.1, 1.0, np.nan]
        assert select_mask("all", lc).sum() == 6
        assert select_mask("transient", lc).tolist() == [1, 1, 0, 0, 1, 0]
        assert select_mask("variable", lc).tolist() == [0, 0, 1, 1, 0, 0]
        assert select_mask("coarse:AGN", lc).tolist() == [0, 0, 0, 1, 0, 0]
        assert select_mask("coarse:4", lc).tolist() == [0, 0, 0, 0, 1, 0]
        assert select_mask("fine:7", lc, lf).tolist() == [0, 0, 1, 0, 0, 0]
        assert select_mask("amplitude>=1", lc, lf, props).tolist() == [1, 0, 1, 0, 1, 0]
        assert select_mask("transient, amplitude>1.5", lc, lf, props).tolist() == [
            1, 0, 0, 0, 0, 0,
        ]  # fmt: skip

    def test_errors(self):
        with pytest.raises(ValueError):
            select_mask("transient", props=np.zeros((3, len(PROPERTY_NAMES))))
        with pytest.raises(ValueError):
            select_mask(
                "bogus>1", np.zeros(3), props=np.zeros((3, len(PROPERTY_NAMES)))
            )
        with pytest.raises(ValueError):
            select_mask("all")


class TestSyntheticData:
    @pytest.mark.parametrize("kind", sorted(SYNTH_CLASSES))
    def test_synth_light_curve(self, kind):
        rng = np.random.default_rng(11)
        lc = None
        for _ in range(20):
            lc = synth_light_curve(kind, rng)
            if lc is not None:
                break
        assert lc is not None
        t, mag, _, _, truth = lc
        assert t[0] == 0 and (np.diff(t) >= 0).all()
        assert len(t) >= 3 and (mag < 20.5).all()
        assert truth["kind"] == kind

    def test_synthetic_magnitudes_are_physical(self):
        """Regression: a wild noise draw on a far-below-limit epoch must not
        become an ultra-bright 'detection' (this once produced mag ~ -2500)."""
        rng = np.random.default_rng(0)
        for _ in range(300):
            for kind in SYNTH_CLASSES:
                lc = synth_light_curve(kind, rng)
                if lc is None:
                    continue
                _, mag, err, _, _ = lc
                assert (mag > 10.0).all() and (mag < 20.5).all()
                assert (err > 0).all() and (err <= 0.5).all()

    def test_dataset_loads_with_meta_dataset(self, synth_dir):
        ds = MetaNPZDataset(str(synth_dir / "train"))
        assert len(ds) > 0
        assert ds.in_channels == 37
        item = ds[0]
        assert item["x"].shape[1] == 37
        assert item["label_fine"] in set(SYNTH_CLASSES.values())
        batch = collate_fn([ds[i] for i in range(min(4, len(ds)))])
        assert batch["x"].shape[2] == 37
        for split in ("train", "val", "test"):
            assert any((synth_dir / split).glob("*.npz"))


# ---------------------------------------------------------------------------
# eval_discovery
# ---------------------------------------------------------------------------


class TestConfigInference:
    @pytest.mark.parametrize("mode", ["ae", "vae", "vqvae"])
    def test_roundtrip_through_state_dict(self, mode, tmp_path):
        model = tiny_model(
            mode=mode, in_channels=37, latent_dim=8, gp_dim=12, num_classes=5
        )
        model.vq = getattr(model, "vq", None) or None
        path = tmp_path / "m.pt"
        torch.save(model.state_dict(), path)
        loaded, cfg = ed.load_model(str(path), device="cpu", overrides={"n_heads": 4})
        assert cfg["mode"] == mode
        assert cfg["latent_dim"] == 8
        assert cfg["in_channels"] == 37
        assert cfg["d_model"] == 32
        assert cfg["d_ff"] == 64
        assert cfg["enc_layers"] == 1 and cfg["dec_layers"] == 1
        assert cfg["gp_dim"] == 12
        assert cfg["num_classes"] == 5
        assert cfg["use_images"] is False
        x = torch.randn(2, 6, 37)
        x[..., 4:7] = 0
        x[..., 4] = 1
        pad = torch.zeros(2, 6, dtype=torch.bool)
        gp = torch.randn(2, 12)
        torch.manual_seed(0)
        z1 = model.embed(x, pad, gp_features=gp)
        z2 = loaded.embed(x, pad, gp_features=gp)
        assert torch.allclose(z1, z2, atol=1e-6)

    def test_summary_overrides_n_heads(self, tmp_path):
        model = tiny_model()
        torch.save(model.state_dict(), tmp_path / "m.pt")
        (tmp_path / "summary.json").write_text('{"n_heads": 4, "mode": "ae"}')
        _, cfg = ed.load_model(str(tmp_path / "m.pt"), str(tmp_path / "summary.json"))
        assert cfg["n_heads"] == 4


class TestMetrics:
    def test_property_agreement_perfect_and_offset(self):
        rng = np.random.default_rng(0)
        P = rng.normal(size=(50, len(PROPERTY_NAMES)))
        res = ed.property_agreement(P, P)
        for name in PROPERTY_NAMES:
            assert res[name]["mae"] == 0
            assert res[name]["frac_within_tol"] == 1.0
            assert res[name]["r2"] == pytest.approx(1.0)
            assert res[name]["coverage"] == 1.0
        res = ed.property_agreement(P, P + 10.0)
        for name in PROPERTY_NAMES:
            assert res[name]["bias"] == pytest.approx(10.0)
            assert res[name]["frac_within_tol"] == 0.0

    def test_property_agreement_handles_nans(self):
        P = np.ones((10, len(PROPERTY_NAMES)))
        Q = P.copy()
        Q[:5, 0] = np.nan
        res = ed.property_agreement(P, Q)
        assert res[PROPERTY_NAMES[0]]["n"] == 5
        assert res[PROPERTY_NAMES[0]]["coverage"] == 0.5
        assert np.isnan(res[PROPERTY_NAMES[0]]["r2"])  # zero variance
        res = ed.property_agreement(np.full((3, len(PROPERTY_NAMES)), np.nan), P[:3])
        assert res[PROPERTY_NAMES[0]]["n"] == 0
        assert np.isnan(res[PROPERTY_NAMES[0]]["mae"])

    def test_label_groups(self):
        groups = ed.label_groups(np.array([0, 0, 1, 3, 3, 4, 4]))
        assert set(groups) >= {"all", "transient", "variable", "SNIa", "AGN", "TDE"}
        assert "SNcc" not in groups  # only one member
        assert groups["variable"].sum() == 2
        assert list(ed.label_groups(np.array([-1, -1]))) == ["all"]

    def test_probes_recover_linear_signal(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(300, 8))
        w = rng.normal(size=8)
        P = np.tile((z @ w)[:, None], (1, len(PROPERTY_NAMES)))
        P[:, 1] = np.nan  # a property that is never defined
        res = ed.probe_properties(z[:200], P[:200], z[200:], P[200:])
        assert res[PROPERTY_NAMES[0]]["r2"] > 0.99
        assert res[PROPERTY_NAMES[0]]["skill"] > 0.9
        assert res[PROPERTY_NAMES[1]]["skipped"] is True
        y = (z[:, 0] > 0).astype(int)
        b = ed.probe_binary(z[:200], y[:200], z[200:], y[200:])
        assert b["auc"] > 0.95
        assert b["skipped"] is False
        b = ed.probe_binary(z[:20], np.zeros(20), z[20:40], y[20:40])
        assert b["skipped"] is True

    def test_probe_mlp_runs(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(120, 4))
        P = np.tile(z[:, :1], (1, len(PROPERTY_NAMES)))
        res = ed.probe_properties(z[:80], P[:80], z[80:], P[80:], kind="mlp")
        assert np.isfinite(res[PROPERTY_NAMES[0]]["mae"])
        with pytest.raises(ValueError):
            ed.probe_properties(z[:80], P[:80], z[80:], P[80:], kind="svm")


class TestPipeline:
    def test_encode_and_injection(self, synth_dir):
        torch.manual_seed(0)
        model = tiny_model()
        cfg = {"in_channels": 37, "use_images": False, "gp_dim": 0}
        ds, norm = ed.build_dataset(str(synth_dir), "train", cfg)
        assert norm is None
        loader = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=collate_fn)
        enc = ed.encode_batches(model, loader, "cpu", norm)
        N = len(ds)
        assert enc["z"].shape == (N, 8)
        assert enc["props_orig"].shape == (N, len(PROPERTY_NAMES))
        assert enc["props_recon"].shape == (N, len(PROPERTY_NAMES))
        assert np.isfinite(enc["z"]).all()
        assert (enc["recon_err"] >= 0).all()
        assert len(enc["obj_ids"]) == N
        assert np.isfinite(enc["props_orig"][:, PROPERTY_NAMES.index("peak_mag")]).all()

        mask = enc["labels_coarse"] == 3
        sub = ed.subset_encoding(enc, mask)
        assert sub["z"].shape[0] == mask.sum()
        assert len(sub["obj_ids"]) == mask.sum()

        hosts = [ds[i] for i in range(8)]
        inj = ed.run_injection_recovery(
            model,
            hosts,
            norm=norm,
            preset="sn_ia",
            delta_mags=[0.0, 2.0],
            rng=np.random.default_rng(0),
            n_examples=2,
        )
        assert inj["n_hosts"] == 8
        assert len(inj["curves"]) == 2
        c = inj["curves"][1]
        assert 0 <= c["detection_efficiency"] <= 1
        assert c["snr_peak_median"] > inj["curves"][0]["snr_peak_median"]
        assert "peak_mag" in c["recon"]
        assert len(inj["examples"]) == 2
        for ex in inj["examples"]:
            assert {"host", "injected", "decoded"} <= set(ex)
            # Injection only ever brightens the host
            assert (ex["injected"]["mag"] <= ex["host"]["mag"] + 1e-5).all()
        with pytest.raises(ValueError):
            ed.run_injection_recovery(
                model, hosts[:2], preset="sn_ia", delta_mags=[0.0]
            )

    def test_cli_end_to_end(self, tmp_path):
        out = tmp_path / "out"
        results = ed.main(
            [
                "--synthetic-dir",
                str(tmp_path / "synth"),
                "--synthetic-n",
                "16",
                "--latent-dim",
                "8",
                "--output-dir",
                str(out),
                "--n-hosts",
                "10",
                "--delta-mags",
                "0",
                "2",
                "--batch-size",
                "32",
                "--seed",
                "1",
            ]
        )
        assert (out / "discovery_eval.json").exists()
        for png in (
            "recon_properties.png",
            "property_skill.png",
            "injection_recovery.png",
            "injection_examples.png",
        ):
            assert (out / png).exists(), png
        assert set(results) >= {"config", "data", "recon", "probe", "injection"}
        assert results["data"]["n_eval"] > 0
        assert "all" in results["recon"]
        assert results["injection"]["preset"] == "sn_ia"
        assert len(results["injection"]["curves"]) == 2
        # JSON must be valid (no NaN literals)
        import json

        with open(out / "discovery_eval.json") as f:
            loaded = json.load(f)
        assert loaded["config"]["latent_dim"] == 8

    def test_cli_stage_subset_and_selection(self, tmp_path):
        synth = tmp_path / "synth"
        make_synthetic_dataset(synth, n_per_class=14, seed=5)
        out = tmp_path / "out"
        results = ed.main(
            [
                "--data-dir",
                str(synth),
                "--latent-dim",
                "8",
                "--output-dir",
                str(out),
                "--stages",
                "recon",
                "--select",
                "transient,duration>5",
                "--seed",
                "2",
            ]
        )
        assert "probe" not in results and "injection" not in results
        assert results["data"]["n_selected"] <= results["data"]["n_eval"]
        assert results["data"]["n_selected"] > 0
        assert "variable" not in results["recon"]  # selection removed variables
