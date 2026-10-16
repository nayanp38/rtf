# Discovery-preservation evaluation: design, implementation, and where to go next

This document explains the evaluation pipeline in `src/eval_discovery.py`,
`src/transient_properties.py` and `src/transient_injection.py`: why it is
shaped the way it is, what alternatives were on the table, how each piece
works line by line, how to read its outputs, and a proposal for extending it
to out-of-distribution (OOD) transients.

Contents

1. [The question the pipeline answers](#1-the-question-the-pipeline-answers)
2. [Design decisions and the alternatives](#2-design-decisions-and-the-alternatives)
3. [Implementation walkthrough](#3-implementation-walkthrough)
4. [Reading the outputs (worked example)](#4-reading-the-outputs-worked-example)
5. [Extension: out-of-distribution transients](#5-extension-out-of-distribution-transients)
6. [Known limitations and quick wins](#6-known-limitations-and-quick-wins)

---

## 1. The question the pipeline answers

RTF compresses a multi-band alert light curve (plus metadata, stamps, GP
features) into a fixed-length latent `z`. The repo already measured two
things about `z`:

* **Reconstruction error** (`evaluate_physical.py`): decode `z`, compare the
  decoded magnitudes/times to the input. MSE in normalised units; median mag
  residual in physical units.
* **Classification** (`linear_probe.py`, `eval_early.py`): fit a logistic
  regression from `z` to the 5 coarse / 10 fine classes.

Neither answers the question you actually care about — *would an astronomer
(or a broker filter) still be able to make a discovery from `z`?* — because:

* MSE is an average over every epoch of every source. It is dominated by the
  bulk of ordinary points and says nothing about the one property that
  triggers a follow-up decision (how bright did it get? when? how fast?).
* Classification accuracy tests only *class-discriminative* information for
  five classes that the model was trained on. A latent can classify perfectly
  and still have thrown away the peak time, the colour, or a second peak.
* Nothing tested *sensitivity*: if something new happens inside a known
  source (a nuclear transient in an AGN, a rebrightening in a supernova), does
  `z` move? Does the decoded curve show it?

The pipeline therefore defines a small set of **discovery-notable properties**
and asks three questions about them, in order of increasing demand:

| Stage | Question | What "good" looks like |
|---|---|---|
| `recon` | If I decode `z`, can I re-measure the property? | Decoded property ≈ original property, within a physically motivated tolerance |
| `probe` | Is the property *in* `z` at all, even if the decoder does not render it? | A cheap regressor from `z` beats a "predict the median" baseline |
| `inject` | If a transient appears in a known source, does `z` notice, and does the decoded curve show it? | Latent displacement, detection completeness and decoded recovery all rise with injected amplitude |

The three are deliberately complementary. `probe` skill ≫ `recon` skill means
the information survives the encoder but the decoder cannot use it; both low
means the bottleneck destroyed it; `inject` tells you about *unexpected*
signals rather than the population average.

---

## 2. Design decisions and the alternatives

Each subsection lists the options considered, their trade-offs, and the choice.

### 2.1 What to measure

| Option | Pros | Cons |
|---|---|---|
| (a) Raw reconstruction MSE (already exists) | Simple; it is the training loss | Averages over everything; normalised units; dominated by time channels; not tied to any decision |
| (b) Class accuracy (already exists) | Tied to broker use | Only 5/10 classes; cannot say *what* was lost; needs labels |
| (c) **Hand-defined discovery properties** (chosen) | Interpretable, physical units, tolerances map to science requirements, works without labels, identical extractor on original and decoded curves cancels most estimator noise | Hand-picked (may miss things); the extractors are themselves noisy estimators on sparse cadence (e.g. `t_peak` of a flat AGN is noise-driven) |
| (d) Downstream task suite (rare-class classifiers, anomaly AUROC) | Closest to end use | Needs rare-class labels; conflates the latent with the probe model; expensive |
| (e) Mutual-information estimates between `z` and the raw curve | Principled | Hard to estimate in 64 dims; uninterpretable for an astronomer |

Choice: (c) as the backbone, with a light version of (d) as the binary probes
(`is_transient`, `fast_evolver`) and, in §5, novelty detection.

The ten properties (`PROPERTY_NAMES`) were chosen to cover every axis of a
follow-up decision: *brightness* (`mag_first` = discovery mag, `peak_mag`),
*timing* (`t_peak`, `duration`), *speed* (`rise_rate`, `decline_rate`,
`t_half_width`), *variability* (`amplitude`) and *colour* (`color_peak`,
`mean_color`). Each has a unit and a tolerance (`PROPERTY_TOLERANCES`) so the
"fraction recovered within tolerance" metric has a physical meaning
(0.2 mag ≈ a few times the photometric error, 2 days ≈ one ZTF cadence gap).

Rejected properties: `n_det` (reconstructed exactly by construction — the
decoder emits one row per input row), band fractions (already covered by
`band_acc` in training), and periodicity (not a transient property).

### 2.2 How to test "preserved": decode, probe, or both

* **Decode-and-remeasure** needs no extra training and is the strictest test
  ("can a human read it off the reconstruction?"). Its weakness: a poor
  decoder hides a good latent.
* **Probe from `z`** isolates the latent from the decoder and can find
  information the decoder ignores. Its weaknesses: you must train something
  (so you need a train split and you inherit the probe's capacity and
  overfitting behaviour), and "present" is not the same as "usable".
* **Both** (chosen) gives the gap analysis described in §1. The cost is one
  extra encoding pass over the train split.

An option I considered but did not implement: *retrieval consistency* — do a
source's nearest neighbours in latent space share its properties? This is the
similarity-search use case in the README and is proposed in §5.4.

### 2.3 Which probe

| Option | Pros | Cons |
|---|---|---|
| **Ridge (chosen default, `RidgeCV`)** | Cheap, deterministic, measures *linear* decodability which is what a downstream linear head would use | Misses non-linear encodings |
| MLP (`--probe mlp`) | Finds non-linear structure | Higher variance, slower, results depend on training details |
| kNN regression | Non-parametric | Scale-sensitive, weak in 64 dims with 12k points |
| Gradient boosting | Strong | Heavy dependency; overkill for a diagnostic |

Two details matter more than the model choice:

1. **A baseline.** Every property error is reported next to the MAE of
   predicting the training-set median (`baseline_mae`) and summarised as
   `skill = 1 − MAE/baseline_MAE`. Skill 1 is perfect, 0 is "no better than a
   constant", negative is worse than a constant. This is what makes properties
   in days, magnitudes and mag/day comparable on one plot.
2. **Regularisation chosen by cross-validation.** The first version used
   `Ridge(alpha=1)`. On the untrained-model smoke test the probes extrapolated
   wildly (MAE of 141 mag on `mag_first`) because degenerate latents get blown
   up by standardisation. `RidgeCV(alphas=logspace(-2, 3, 11))` fixed that and
   costs nothing.

### 2.4 Injection: where, how, and how much

**Where to inject.**

| Option | Pros | Cons |
|---|---|---|
| **Into real host light curves at their existing epochs** (chosen) | Realistic cadence and noise; metadata and stamps stay consistent; models the actual discovery problem ("something new happened in a source we already know") | ZTF `magpsf` is *difference* photometry, so an SN "host" is the SN itself — hosts default to `variable` sources (AGN, CVs) whose difference flux is a sensible baseline |
| Fully synthetic hosts | Total control | Unrealistic cadence and noise; zero metadata makes them trivially separable |
| Add new epochs / resample cadence | Can test the transient's own cadence | Changing sequence length moves `z` by itself — a confound |

**How to add flux.** Magnitudes are not additive, so the injection is done in
linear flux: `f_new = f_host + f_transient`, `mag_new = −2.5 log10 f_new`. The
magnitude error is rescaled as `σ_new = σ_host · f_host / f_new`, i.e. the flux
error is held fixed (sky-limited regime) so the brighter point gets a smaller
magnitude error. The alternative — leaving `σ` untouched — produces bright
points with implausibly large errors, which the model has never seen.

**How much.** Amplitude is expressed as `delta_mag`, the injected r-band peak
relative to the host's median r-band magnitude (`+` = brighter than host,
0 = doubles the host flux at peak). Alternatives were an absolute peak
magnitude (does not adapt to host brightness; a 19th-mag transient is huge in
a 21st-mag host and invisible in a 16th-mag one) or a direct SNR (needs the
noise model up front). `delta_mag` is host-relative and the pipeline records
the resulting peak SNR anyway (`snr_peak`), so completeness can be plotted
against either.

**Shape.** Bazin `exp(−s/τ_fall) / (1 + exp(−s/τ_rise))` for everything
SN/TDE/KN-like and a Gaussian for flares. Bazin has an analytic peak
(§3.2), so the injected peak magnitude is exact. The repo's
`generate_synthetic.py` uses lightcurve-fitting's richer models, but that
dependency is not installable locally and the analytic normalisation is worth
more here than fidelity. Presets carry *ranges* of timescales and colours so a
batch of injections is diverse rather than one fixed template.

**Timing.** `t_peak` is drawn uniformly in `[0.2, 0.8]` of the host duration so
the injected rise and decline both fall inside the observed window.

### 2.5 What counts as "detected" in latent space

Four criteria are reported side by side, because each answers a different
question and each fails differently:

| Criterion | Question | Failure mode seen in practice |
|---|---|---|
| `det_null`: shift > (1−FPR) quantile of shifts caused by re-drawing photometric noise | Could this move be noise? | For a deterministic AE the null is *tiny* (0.04 vs a nearest-neighbour spacing of 0.4), so everything is "detected" — true but uninformative |
| `det_NN`: shift > median nearest-neighbour spacing among hosts | Did the source leave its neighbourhood? | Depends on eval-set density; added after `det_null` saturated |
| `probe_auc`: logistic host-vs-injected on even-indexed hosts, scored on odd-indexed hosts | Could a linear detector be built? | Probe capacity / small-sample variance |
| `recov`: injected `peak_mag` and `t_peak` both recovered from the decoded curve within tolerance | Does the transient survive to the reconstruction? | Strict; sensitive to decoder band/time errors |

### 2.6 Data handling: reuse the repo's datasets

The model must see *exactly* what it was trained on: per-sample robust scaling
of the 30 metadata channels, "most recent valid stamp" selection, GP-key
remapping. Re-implementing that in a new loader is a correctness risk, so
`build_dataset` wraps `MetaNPZDataset` / `PhotoNPZDataset` and auto-detects the
NPZ layout. Injection then edits only channels 2–3 (`logflux`, `logflux_err`)
of the loaded item, which the metadata scaling never touches. For the legacy
photometry-only layout, whose continuous channels are normalised by
`feature_stats_day100.npz`, a `norm=(mean, std)` pair is threaded through every
conversion so injection happens in physical units and is re-normalised after.

### 2.7 Model configuration: infer from weights

`summary.json` written by `train.py` does not record `num_classes`,
`image_backbone` or `freeze_backbone`, and `eval_early.py` needs CLI overrides
for them. Every architectural choice except `n_heads` changes a tensor shape,
so `infer_config_from_state_dict` reads the config off the checkpoint
(`bottleneck_proj` ⇒ AE, `mu_proj` ⇒ VAE, `pre_vq_proj` ⇒ VQ-VAE;
`in_proj.weight` ⇒ `in_channels`/`d_model`; `gp_proj.0.weight` ⇒ `gp_dim`;
`cls_head.3.weight` ⇒ `num_classes`; `image_tower.cnn.stem*` ⇒ Zoobot). The
summary is consulted only for `n_heads`. Any checkpoint from any run is
loadable with one flag.

### 2.8 A synthetic dataset

There is no ZTF data on a laptop and CI has none either, so the pipeline ships
a generator with five classes mapped onto the repo's fine labels
(`snia→0, snii→3, tde→9, cv→7, agn→8`, which `COARSE_MAP` turns into the five
coarse classes). Its only job is to make the whole pipeline — including
`train.py` — runnable and testable end to end; it is not a physics simulator.

One lesson from building it: the ZTF-like error model
`σ = 0.1·10^(0.4(m−19))` explodes for faint true magnitudes (m≈30 gives
σ≈2500 mag). A wild noise draw then *passes* the `mag_obs < 20.5` detection cut
and produces a "detection" at magnitude −2500. The first synthetic training
run had a validation MSE of 265 because of three such files. The fix caps the
error model and requires `σ ≤ 0.5` for a detection; a regression test pins
this. The repo's own `generate_synthetic.py` uses the same error model and cut
and is likely affected the same way.

### 2.9 Things deliberately left out

* **Recomputing GP features for injected curves.** Needs lightcurve-fitting.
  The pipeline prints a warning and reuses the host's GP vector.
* **Modifying stamps.** The injection is photometric; the host stamp remains
  the correct context image for "something happened at this position".
* **Early-epoch sweeps.** Use `--max-detections N` and loop, as
  `eval_early.py` does.
* **Retrieval metrics and novelty detection.** Proposed in §5.

---

## 3. Implementation walkthrough

### 3.1 `transient_properties.py`

**Feature layout ↔ physical units.** The model's native row is
`[log1p(dt), log1p(dt_prev), logflux, logflux_err, g, r, i, meta...]` with
`logflux = −0.4·magpsf` and `logflux_err = 0.4·sigmapsf` (see
`preprocess_alerts.py`). The inverse is therefore

```python
t       = expm1(clip(log1p_dt, 0, None))   # days since first detection
mag     = -2.5 * logflux                    # -2.5 * (-0.4 m) = m
mag_err = 2.5 * |logflux_err|
band    = argmax(one_hot)
```

The clip matters only for *decoded* rows, whose log-times can be slightly
negative. `_apply_norm` undoes/applies the optional `(mean, std)` on the four
continuous channels; `physical_to_x` is the exact inverse and appends any
extra channels untouched; `set_photometry` copies a row block and overwrites
only columns 2–3 — the primitive that injection and re-noising are built on.
`recon_to_x` turns decoder outputs `(cont_hat, band_logits)` into the same
layout by one-hot encoding `argmax(band_logits)`.

**`compute_properties(t, mag, mag_err, band)`.** Steps, in order:

1. Sort by time (stable) and re-reference `t` to `t[0]`. Row order therefore
   never matters, which is required for decoded curves whose reconstructed
   times are not monotonic.
2. `duration = t[-1] − t[0]`, `mag_first = mag[0]`.
3. Peak = brightest row over all bands: `peak_mag = min(mag)`, `t_peak` its
   time.
4. Everything shape-related is measured **in the peak band only** so that
   bands with different zero-points are never mixed:
   `amplitude = max − min` in that band;
   `t_half_width` = span of epochs with `mag ≤ peak_mag + 2.5·log10(2)`
   (0.753 mag = half the peak flux);
   `rise_rate = −slope(mag vs t)` on epochs `≤ t_peak` (so positive means
   brightening) and `decline_rate = +slope` on epochs `≥ t_peak` (positive
   means fading). `_slope` is an unweighted least-squares fit that returns
   NaN with fewer than 3 points or a span under 1 day — a slope from two
   points 0.02 days apart is noise.
5. Colours: `color_peak` = mean g − mean r among epochs within ±7 days of the
   peak; `mean_color` over the whole curve. NaN if either band is absent.

NaN is the "undefined" signal throughout; every downstream metric masks on
finiteness and reports how many rows survived (`n`) and, for decoded curves,
the fraction of originally-defined values that are still defined
(`coverage`).

`properties_from_x(x, valid, norm)` applies a padding mask then the two
functions above; `properties_batch` stacks the result into an `(N, 10)` array
in `PROPERTY_NAMES` order, which is the representation used everywhere else.

### 3.2 `transient_injection.py`

**Bazin normalisation.** For `f(s) = e^{−s/τf} / (1 + e^{−s/τr})`, setting
`d ln f/ds = 0` gives `σ(−s/τr) = τr/τf`, hence the peak is at

```
s* = τr · ln(τf/τr − 1)         (requires τf > τr)
```

`transient_shape` shifts the curve so that its peak lands exactly on the
requested `t_peak` and divides by `f(s*)`, so the returned shape has peak 1 and
`peak_mag` is exact. Presets that could violate `τf > τr` are repaired in
`sample_transient_params` (`τf ← 1.5 τr`).

**Presets** (`TRANSIENT_PRESETS`) hold ranges; `sample_transient_params(preset,
rng, t_peak, **overrides)` draws one concrete parameter set and lets tests pin
any value. `peak_mag_per_band` converts an r-band peak and a g−r colour into
per-band peaks (i placed at `r − 0.5·(g−r)`).

**`inject_transient(x, peak_mag, params, norm, rng, add_noise)`:**

```python
t, mag, mag_err, band = x_to_physical(x, norm)
f_tr   = 10 ** (-0.4 * pm[band]) * transient_shape(t, params)   # per-band peak flux × shape
f_host = 10 ** (-0.4 * mag)
f_new  = f_host + f_tr
mag_new = -2.5 * log10(f_new)
err_new = mag_err * f_host / f_new          # constant flux error ⇒ σ_mag ∝ 1/flux
snr     = f_tr / (f_host * mag_err / 1.0857) # 1.0857 = 2.5/ln 10 converts σ_mag → σ_f/f
```

`add_noise=True` re-draws Gaussian noise with the *new* errors (off by default:
the host's own noise realisation is already in `f_host`). The returned `info`
carries the parameters, `snr_peak`, and the actual brightening at the epoch
nearest the peak, which is what the examples plot annotates.

`renoise(x, rng)` is the null: same curve, noise re-drawn with the original
errors. `host_reference_mag` is the host's median r-band magnitude (any band if
r is missing) — the zero point of `delta_mag`.

**`select_mask(selection, labels_coarse, labels_fine, props)`** parses a
comma-separated AND of terms: `all`, `transient` (coarse ∈ {SNIa, SNcc, TDE}),
`variable` (∈ {Cataclysmic, AGN}), `coarse:<name|id>`, `fine:<id>`, and
property thresholds `name<op>value` on the `(N, 10)` property array (NaN rows
never match). It raises rather than silently returning everything when a term
needs labels or properties that were not supplied.

**Synthetic generator.** `ztf_cadence` draws exponential gaps (mean 2.5 d)
and g/r bands; `synth_light_curve(kind, rng)` builds fluxes per class (Bazin
transients on no host — difference photometry; a CV as a faint quiescent level
plus 1–3 fast Bazin outbursts; an AGN as a damped random walk in magnitude
`m_i = m0 + a(m_{i−1} − m0) + N(0, σ√(1−a²))`, `a = e^{−Δt/τ}`, with a
constant g−r offset), applies the capped error model and the two-part
detection cut, and returns times re-referenced to the first *detection* —
exactly what `preprocess_alerts.py` does. `make_synthetic_dataset` writes
`{split}/{oid}.npz` with `x (L, 37)` (zero metadata) and `label`, using
uncompressed `np.savez` because `MetaNPZDataset` drops files under 200 bytes.

### 3.3 `eval_discovery.py`

**Model loading.** `infer_config_from_state_dict` (§2.7) → `build_model` →
`load_state_dict`. `load_model` layers `summary.json` fields and CLI overrides
on top. Without `--checkpoint`, `main` builds a small random model and prints a
loud warning — the smoke-test path used by the CLI test.

**Data.** `detect_npz_format` peeks at the first file (`x` ⇒ preprocessed
layout, `data` ⇒ legacy photometry). `build_dataset` returns the dataset and
the `norm` pair (None for the preprocessed layout) and refuses to continue if
the dataset's channel count does not match the model's.

**`encode_batches(model, batches, device, norm, keep_recon)`** is the one
place the model runs. Per batch:

```python
z = model.embed(x, pad, images, gp_features=gp)        # deterministic latent (mu for VAE, z_q for VQ)
cont_hat, band_logits = model.decode(z)[:, :L]          # decoder emits max_len rows; keep L
recon_err = masked MSE on channels 0:4 + masked band cross-entropy   # same formula as model.reconstruction_error, computed inline to avoid a second encode
x_rec = recon_to_x(cont_hat, band_logits)
for each item: props_orig = properties_from_x(x[i], valid[i], norm)
               props_recon = properties_from_x(x_rec[i], valid[i], norm)
```

It skips the `"BAD"` placeholder items the datasets emit on load errors and
returns arrays (`z`, `props_orig`, `props_recon`, `recon_err`, labels,
`n_det`, `obj_ids`) plus, if asked, the decoded valid rows for plotting.
`subset_encoding` applies a boolean mask to all of them.

**Stage 1 — `property_agreement(P_true, P_pred)`** per property: mask to rows
finite in both, then MAE, median AE, RMSE, bias (`pred − true`), `baseline_mae`
(MAE of predicting the median of the true values), `frac_within_tol`,
`R² = 1 − Σerr²/Σ(true − mean)²` (NaN if the truth has no variance), Pearson
`r`, `n`, and `coverage`. `label_groups` builds masks for `all`, `transient`,
`variable` and every coarse class with ≥ 2 members; `grouped_agreement` runs
the metric per group.

**Stage 2 — probes.** `probe_properties` standardises `z` on the train split,
fits one `RidgeCV` (or `MLPRegressor`) per property on rows where the property
is defined, and scores on the eval split: MAE, RMSE, `R²` (sklearn), Pearson,
`frac_within_tol`, `baseline_mae` and `skill`. Properties with < 20 training
or < 10 test values are marked `skipped`. `probe_binary` is a balanced
logistic regression reporting AUC and balanced accuracy, skipped when either
split lacks both classes or a class has < 5 training examples.
`binary_tasks` defines `is_transient` (from coarse labels, restricted to rows
that are transient or variable) and `fast_evolver` (`t_half_width <
--fast-threshold`, default 10 d).

**Stage 3 — `run_injection_recovery`.**

1. Encode the hosts once (`z_host`) and their re-noised copies (`z_null`);
   `threshold = quantile(‖z_null − z_host‖, 1 − FPR)`.
2. `scale` = median distance from each host to its nearest *other* host
   (`NearestNeighbors(n_neighbors=2)`), the unit of the "disp/NN" column.
3. For each `delta_mag`: draw a preset parameter set and a `t_peak` per host,
   set `peak_mag = host_reference_mag − delta_mag`, inject, encode with
   `keep_recon=True`. Then
   `d = ‖z_inj − z_host‖`, `det_null = d > threshold`, `det_NN = d > scale`;
   a logistic probe trained on `[z_host, z_inj]` of even-indexed hosts and
   scored on odd-indexed hosts (so no host contributes to both);
   `property_agreement(props(injected input), props(decoded injected))` for
   decoded recovery and `property_agreement(props(host), props(injected
   input))` to record how much the injection actually moved each property
   (`property_shift_mae` — a sanity check that the injection was visible in
   the input at all); `recovered = |Δpeak_mag| ≤ 0.2 ∧ |Δt_peak| ≤ 2 d`.
4. At one amplitude (the middle of the grid by default) keep the first
   `n_examples` hosts' physical curves — host, injected, decoded — for the
   example figure.

**Reporting.** `print_*_table` render the console tables; `plot_recon_scatter`
(original vs decoded per property with the identity line), `plot_skill`
(decoded-curve skill next to probe skill), `plot_injection_curves`
(completeness, latent sensitivity with the 16–84 % band, brightness and timing
recovery vs amplitude) and `plot_injection_examples`. `_jsonable` converts
numpy types and turns NaN into `null` so the JSON is standards-compliant.

**CLI.** `--stages` selects `recon probe inject` (default all; the eval split
is always encoded). `--select` restricts recon/probe metrics; `--host-select`
picks injection hosts (default `variable`, falling back to all with a message
if fewer than four match). `--synthetic-dir` generates data when `--data-dir`
is absent. `--max-detections` truncates sequences for early-epoch tests.

### 3.4 Tests (`tests/test_discovery.py`)

* **Layout round trips**, with and without normalisation and metadata, and
  the guarantee that `set_photometry` touches only columns 2–3.
* **Known-peak recovery**: a noiseless Bazin curve with `t_peak = 20`,
  `peak_mag = 17.5`, colour 0.2 must be measured back within cadence/colour
  tolerances; order invariance; flat-curve and degenerate-input behaviour;
  padding masks.
* **Shapes**: every preset peaks at exactly 1 at the requested time; the
  Bazin offset formula; parameter repair.
* **Injection algebra**: with `delta_mag = 0` the flux exactly doubles at
  peak, cadence/bands/metadata are untouched, errors shrink, SNR > 5, and the
  injected peak is recoverable by the property code; behaviour under
  normalisation; reproducible noise; `renoise` stays within 5σ.
* **Selection grammar** and its error cases.
* **Synthetic data**: loadable by `MetaNPZDataset` with 37 channels; and the
  regression test that no synthetic magnitude is brighter than 10 or has an
  error above 0.5.
* **Config inference** round-trips AE/VAE/VQ-VAE with GP and class heads and
  reproduces the original model's embeddings bit-for-bit.
* **Metrics**: perfect and offset agreement, NaN masking, group construction,
  probes recovering a planted linear signal, MLP path, skip logic.
* **Pipeline**: `encode_batches` shapes, `run_injection_recovery` structure
  and monotone SNR, and the CLI end to end (all four figures + valid JSON), plus
  a stage/selection subset run.

---

## 4. Reading the outputs (worked example)

Numbers below are from a 40-epoch CPU run of `train.py` (`--mode ae
--use-metadata --latent-dims 16`) on 2 000 synthetic sources — a deliberately
weak model, which makes the diagnostics easy to see.

**Recon table, `[all]`.** `duration` R² = 0.91 (time structure survives);
`peak_mag` MAE 0.60 mag overall but R² = 0.81 / r = 0.97 on `[transient]` and
bias **+0.95 mag on `[variable]`** — the decoder makes variables fainter at
peak, i.e. it smooths outbursts away. `t_peak` R² ≈ 0. `color_peak`
**coverage 0.10**: the decoder emits essentially one band (band accuracy 58 %),
so colour is undefined for 90 % of decoded curves. That is a band-reconstruction
problem, not a flux problem.

**Probe table.** `peak_mag` skill 0.51, `duration` 0.76, `t_half_width` 0.23;
`t_peak`, `amplitude` and both colours ≈ 0. `is_transient` AUC 0.56,
`fast_evolver` 0.66. The skill plot shows probe ≥ decoded everywhere, with
the largest gaps on `mag_first`, `t_half_width` and `mean_color` — information
that is in `z` but that the decoder does not render.

**Injection table (SN Ia into 120 variable hosts).**

```
 dmag   SNR  disp/NN  det_null  det_NN  probeAUC  recov  peakMAE  tpkMAE
 -1.0   7.6    0.42     1.00     0.02    0.574    0.03    0.850   21.8
  0.0  18.8    0.91     1.00     0.37    0.643    0.02    0.727   19.7
  1.0  49.7    1.61     1.00     0.88    0.742    0.01    0.673   19.7
  3.0 295.2    3.79     1.00     1.00    0.947    0.00    0.864   16.6
```

`det_null` saturates (the noise null is tiny); `det_NN` gives a proper
completeness curve, 50 % at `delta_mag ≈ +0.2`; the probe AUC rises smoothly;
**`recov` stays at ~0** and the decoded `t_peak` error is ~20 days regardless
of amplitude. The example figure makes the mechanism obvious: every decoded
curve is the same smooth single-band hump, peaking around day 15–20 whatever
the input. The latent *notices* the transient (it moves ~4 neighbour spacings
at `+3 mag`) but the decoder has collapsed onto a class template. For a
compression algorithm whose success criterion is discovery, that is the
headline finding, and no MSE or accuracy number would have shown it.

---

## 5. Extension: out-of-distribution transients

The model knows five coarse classes. Real discovery is disproportionately
about what it has *not* seen: kilonovae, GRB afterglows, fast blue optical
transients (FBOTs), superluminous supernovae, Ca-rich / Iax supernovae,
luminous red novae, microlensing events, stellar flares. An OOD transient
should (1) be flagged as unusual, (2) have its properties preserved so a human
can characterise it, (3) *not* be silently assimilated into a known class's
template, and (4) be retrievable — its nearest neighbours should be other
examples of the same new thing. Each of those is a measurable quantity.

### 5.1 Where OOD light curves come from

Three sources, in increasing order of realism, and all three should be used:

1. **Analytic presets** (extend `TRANSIENT_PRESETS`): afterglow — power-law
   decay `f ∝ ((t − t0)/1 d)^{−α}` with `α ∈ [0.8, 1.6]` after a ≤ 1-day rise;
   FBOT — Bazin with `τ_rise ∈ [0.5, 2]`, `τ_fall ∈ [3, 8]`, colour
   `[−0.6, −0.2]`, peak `[15, 18]`; SLSN — Bazin with `τ_rise ∈ [20, 40]`,
   `τ_fall ∈ [60, 150]`; microlensing — Paczyński
   `A(u) = (u² + 2)/(u√(u² + 4))`, `u(t) = √(u0² + ((t − t0)/tE)²)`,
   *achromatic* (colour 0) and symmetric; stellar flare — instantaneous rise,
   exponential decay with `τ ∈ [0.02, 0.3]` d, very blue. Cheap, runs locally,
   and — crucially — can be **injected into real hosts**, which keeps real
   metadata and stamps (see the confound in §5.5).
2. **Physical models with real cadence**: the repo already has
   `generate_synthetic.py` (lightcurve-fitting `Afterglow`, `MetzgerKN`,
   `Magnetar`, `ShockCooling`) and `generate_surveysim.py` (survey-sim
   kilonova / afterglow / FBOT populations on real ZTF cadence). Both write
   label 5 ("other") in the training layout. Run them on OzSTAR with the
   noise-model fix from §2.8 applied to `generate_synthetic.py`.
3. **Real held-out classes (gold standard)**: leave-one-class-out. Train with
   `--exclude-classes TDE` (a small addition to `train.py`/`MetaNPZDataset`:
   drop files whose coarse label is excluded) and treat the real TDEs as OOD.
   Repeat for Cataclysmic and for the SN subtypes at the fine level
   (e.g. hold out SN IIn or SN Ib/c). This is the only variant with real
   noise, real metadata and real stamps on the OOD side.

### 5.2 Metrics

**(1) Novelty detection.** Score every eval source with three detectors that
need no OOD labels at training time: reconstruction error
(`model.reconstruction_error`, already an "anomaly score" in the README);
latent density — `k`-NN distance to the *training* latents, or a Gaussian /
GMM log-likelihood on standardised `z`; and, when a classification head
exists, `1 − max softmax` or the predictive entropy. Report AUROC and
**recall at 1 % and 0.1 % false-positive rate** (the numbers that matter for
an alert stream) for OOD vs in-distribution test sources. Add a *baseline*
detector built on the 10 discovery properties (or the 114 GP features) with the
same k-NN scoring: if the latent is not a better novelty detector than
hand-crafted features, that is worth knowing.

**(2) Property preservation off-distribution.** Run the `recon` and `probe`
stages with `--select` restricted to the OOD set but with probes trained on
the in-distribution train split. Compare skill per property ID vs OOD. A large
drop on, say, `t_half_width` for kilonovae means the latent encodes speed in a
class-conditional way and the linear readout learned on supernovae does not
transfer — a direct measure of how much a new class would need new probes.

**(3) Assimilation ("template collapse").** Decode the OOD latents and ask
which known class the *decoded* curve looks like: fit a classifier on the
decoded properties of in-distribution test sources, apply it to decoded OOD
curves, and report the confusion. Complement with the per-class `bias` column
of the recon table on OOD groups (a +10-day `t_half_width` bias on kilonovae
means the decoder stretches them into SN-like widths). The §4 example already
shows the failure mode on in-distribution data; OOD is where it is most
dangerous because nothing else will flag it.

**(4) Retrieval.** For each OOD source, the fraction of its `k = 10` nearest
neighbours (cosine, in the pooled ID + OOD test latents) that are the same OOD
type (k-NN purity) and mean average precision. Compare to the same metric on
the raw property vector. This is the README's "similarity search" use case
evaluated exactly where it is needed — rare classes with too few examples to
train a classifier.

**(5) Early epochs.** All of the above at `--max-detections 3, 5, 10`.
A kilonova exists for a handful of epochs; if it is only distinguishable with
30 detections it was never discoverable.

**(6) Injection completeness for OOD shapes** into real hosts:
`--inject-preset kilonova|afterglow|fbot|microlens|flare` — already supported
for the two existing presets, and the natural first experiment because it
needs no new data.

### 5.3 Implementation sketch

* `transient_injection.py`: add the presets above and two shape functions
  (`powerlaw`, `paczynski`); each needs a peak-normalisation and a `t_peak`
  convention (afterglow and flare peak at `t0`; microlensing at `t0` with
  `u0` as the amplitude knob).
* `eval_ood.py` (or an `ood` stage): inputs `--id-data-dir`, `--ood-dir`
  (NPZ dirs in the training layout, possibly several with a name each) and
  the checkpoint. It reuses `encode_batches`, `probe_properties`,
  `property_agreement`, and adds: `novelty_scores(model, enc_train, enc_test)`
  returning the three scores; `novelty_metrics(scores, is_ood)` → AUROC,
  recall@FPR; `knn_purity(z, labels, k)`; `assimilation_confusion(props_decoded_id,
  labels_id, props_decoded_ood)`. Outputs: ROC curves, a recall@FPR table per
  OOD family, ID-vs-OOD skill bars, purity table, and a UMAP of the ID test
  latents with OOD sources overlaid (reuse `visualize.py`).
* `train.py`: `--exclude-classes` (coarse or fine), recorded in
  `summary.json`, plus a `slurm/train_holdout.sh` that loops over classes.
* Tests: planted OOD cluster in a toy latent gives AUROC ≈ 1 and purity ≈ 1;
  Paczyński amplitude formula; power-law normalisation.

### 5.4 The order to do it in

1. `--inject-preset kilonova` on real hosts with the existing checkpoints
   (zero new code, immediate answer to "would we notice a KN in a known AGN?").
2. Novelty metrics + retrieval on synthetic OOD from `generate_synthetic.py` /
   survey-sim (a day of code; needs OzSTAR data).
3. Leave-one-class-out retraining (a few GPU-hours per held-out class; the
   result that belongs in the paper).

### 5.5 Pitfalls to design around

* **Synthetic OOD is "easy OOD".** Synthetic sources have zero metadata
  channels and no stamps; a model can flag them as novel from the metadata
  alone, which says nothing about the light curve. Mitigations: inject OOD
  *shapes* into real hosts (keeps real metadata), fill synthetic metadata by
  sampling real alerts at similar magnitude, or use real held-out classes.
  Always report the novelty metrics with the metadata channels zeroed for
  *both* ID and OOD as a control.
* **Leakage through "other".** `slurm/train_with_synthetic.sh` merges
  label-5 synthetic sources into training; any model trained that way has seen
  those OOD families and they are no longer OOD for it.
* **Cadence confounds.** OOD populations from survey-sim have their own
  detection statistics (kilonovae have 3–6 epochs); compare against ID sources
  truncated to the same `n_det` distribution, or the "novelty" you measure is
  just sequence length.
* **Probe transfer vs probe refit.** For (2) report both the ID-trained probe
  applied to OOD *and* a probe refit on OOD with cross-validation; the gap
  between them is the class-conditionality of the encoding.

---

## 6. Known limitations and quick wins

* `t_peak` and `amplitude` are ill-posed for stochastic variables (the argmin of
  noisy magnitudes). A smoothed estimate (binned or GP-smoothed curve before
  measuring) would make the `[variable]` rows more meaningful; the same
  smoother must then be applied to decoded curves.
* Colour metrics are limited by band reconstruction. An alternative flag,
  `--decoded-flux-only`, would measure decoded magnitudes on the *original*
  times and bands (positions correspond one-to-one), isolating flux
  reconstruction from band/time reconstruction. The current strict choice is
  the right default for "can a human read the decoded curve", but the
  flux-only variant is the right diagnostic for "which head is failing".
* The noise-redraw null is the correct null hypothesis but is nearly
  degenerate for a deterministic AE; the NN-spacing criterion is the useful
  one in practice. A third option — a *sequence* null that drops or perturbs
  one random epoch — would test robustness to cadence rather than to noise.
* GP features are stale on injected curves; wire in lightcurve-fitting on the
  cluster to recompute them per injection.
* Everything here has run only on synthetic data. The first real-data run
  should be the strongest checkpoint (`ae_dim64_meta_img_ft_gp_randtrunc_cls0.5`)
  on `data_gp`, with `--host-select variable`, then the kilonova preset, then
  `--max-detections 3 5 10`.
