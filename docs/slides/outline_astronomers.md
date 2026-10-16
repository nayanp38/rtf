# Planning outline: "Can a compressed alert still surprise us?"

Talk for astronomers with no ML background and no prior exposure to RTF.
Target 15 minutes plus questions, 12 slides, 3 backup slides.

**One-sentence version.** We are compressing every LSST alert into a 256-byte
fingerprint; a toy experiment with circles and squares shows that a network
pre-trained on Galaxy Zoo keeps unknown objects discoverable in that fingerprint,
while a network trained from scratch learns only the classes it was shown.

**Narrative arc.** Problem (too many alerts) → idea (fingerprints) → the tension
(classify vs. discover) → a toy we can fully understand → three results that
escalate → what it means for the real survey.

**Rules for this deck.**
- Say "fingerprint", never "latent". Say "trained to recognise", never "loss".
- One figure per slide, cropped to the one panel that carries the point.
- Numbers only when they change the conclusion; round aggressively.
- No regime names (`zoobot_cls` etc.) anywhere on a slide.

---

## Slide 1 — Title

**Title:** Can a compressed alert still surprise us?
**Subtitle:** Fingerprinting LSST alerts, and a test with circles and squares.
**Visual:** one real ZTF cutout triplet (science / template / difference).

Speaker note: say in one breath that this is a status report on an experiment
inside a larger compression project, and that the result is about *what the
compression forgets*.

## Slide 2 — The problem: 10 million alerts a night

**Message:** LSST will issue ~10M alerts per night; brokers keep a small
fraction and discard the rest.

**Visual:** a bar: raw stream ~100 MB/s vs. what a broker can actually serve
(~1 MB/s, "an HD video stream").

**Talking points**
- Every alert is a light curve so far, a 63×63 pixel cutout triplet, and some
  catalogue metadata.
- Filtering early means the thing you did not know to look for never reaches you.

## Slide 3 — The idea: a 256-byte fingerprint

**Message:** compress each alert into a short list of numbers that preserves
what matters, and distribute *that* instead.

**Visual:** alert (light curve + cutouts) → box labelled "encoder" → 64 numbers.
Beside it, the analogy: a star's full spectrum → (colour, magnitude) on a CMD.
The fingerprint is the same idea with more numbers, learned from data.

**Talking points**
- Trained on 18K labelled ZTF transients (SN Ia, SN II, TDE, AGN, CVs…).
- Current result on real data: 87% classification accuracy from 256 bytes, vs.
  94% for the full hand-built pipeline. Mention once; not the focus today.
- Things you can do with fingerprints: classify, find look-alikes by distance,
  flag outliers.

## Slide 4 — The tension: recognise vs. discover

**Message:** a fingerprint built to recognise known classes may throw away
exactly the information needed to notice something new.

**Visual:** two cartoon fingerprint spaces. Left: tidy clusters for the known
classes, an unknown object landing *inside* one of them. Right: looser
clusters, the unknown object sitting alone.

**Talking points**
- The network only keeps what its training objective rewards.
- If the objective is "is this an SN Ia or a TDE?", nothing rewards keeping
  "this looks like neither".
- This is the question the rest of the talk tests.

## Slide 5 — Why a toy: circles and squares

**Message:** before trusting this on real transients, test the image part
alone on a problem whose answer we know.

**Visual:** the stamp grid (`00_stamps.png`, science row only, 6 examples).

**Talking points**
- Light-curve input switched off, so the fingerprint depends only on the cutout.
- Synthetic stamps: circles and squares of equal area, random size, position,
  brightness, with noise. Nothing but shape distinguishes them.
- If the fingerprint cannot separate these, nothing downstream is meaningful.

## Slide 6 — Two ways to build the image encoder

**Message:** train a small network from scratch, or reuse a network that has
already looked at a million galaxies.

**Visual:** two boxes side by side. "From scratch: 0.25 M parameters, starts
knowing nothing." "Zoobot: 8.5 M parameters, pre-trained on Galaxy Zoo
morphology votes, weights frozen."

**Talking points**
- Zoobot is a public model (Walmsley et al.); we never retrain its weights. We
  only train the small read-out that maps its output into our 16 numbers.
- Analogy: a first-year student vs. someone who has classified galaxies for a
  decade. Both get the same homework.

## Slide 7 — Result 1: from scratch works, but is hungry and fragile

**Message:** the from-scratch encoder separates circles from squares, but only
with ~1000 labelled examples and clean images.

**Visual:** `02_data_efficiency.png`, left panel only (accuracy vs. number of
training stamps), with a second small inset from `04_noise.png` left panel.

**Talking points**
- Chance is 50%. 256 examples: 62%. 1000 examples: 99%.
- Add modest pixel noise and 512 examples is no longer enough; 2000 fixes it.
- Translation to the survey: rare classes do not have 1000 labelled examples.

## Slide 8 — Result 2: Zoobot already knows

**Message:** with no training at all, the Galaxy Zoo network's fingerprint
already separates circles from squares at 95%.

**Visual:** `06_backbone.png`, left panel: four bars (from-scratch before /
after, Zoobot before / after).

**Talking points**
- "Before" means a random read-out on top of frozen Zoobot features.
- This is the data-efficiency argument in one picture.

## Slide 9 — The discovery test: shapes it has never seen

**Message:** now show the encoder four shapes it was never trained on, and ask
whether we can spot them as *new* without telling it.

**Visual:** `10_novel_stamps.png` (one row: circle, square | triangle, ring,
ellipse, cross, the last four labelled "never seen in training").

**Talking points**
- Two label-free questions, in plain words:
  1. Is this fingerprint far from every circle and square we trained on?
     (nearest-neighbour distance; report as "how often a new shape scores as
     more unusual than a known one", 50% = useless, 100% = perfect)
  2. Do the new shapes form their own groups if we cluster blindly?
- Also, as a ceiling: if we *did* have labels, could a straight line separate
  all six? That tells us whether the information is there at all.

## Slide 10 — Result 3: from scratch cannot; Zoobot can

**Message:** the from-scratch fingerprint files every new shape as "circle",
confidently. The untrained Zoobot fingerprint finds all four.

**Visual:** `11_novel_latent_views.png`, cropped to two t-SNE panels: the
class-trained from-scratch network ("cls") and untrained Zoobot. Label the
islands with shape names by hand.

**Talking points**
- From scratch: new-shape detection 68%, six-shape separability 43%.
  Triangles, rings, ellipses are all called circles with full confidence.
- Zoobot, no training: detection 99.8%, six-shape separability 99.5%.
  Ring and cross form their own clusters; triangle and ellipse are close.
- Say explicitly: this panel is a 2-D picture of a 16-number fingerprint; the
  numbers are computed on all 16.

## Slide 11 — The catch: teaching it to classify squashes the fingerprint

**Message:** once the Zoobot read-out is trained to recognise circle vs.
square, the fingerprint collapses onto a line and loses part of what it knew.

**Visual:** `11_novel_latent_views.png`, two PCA panels: untrained Zoobot
(cloud) vs. Zoobot after classification training (the V).

**Talking points**
- Six-shape separability drops from 99.5% to 85%. New-shape detection holds up
  better (96%) because each shape still lands on its own stretch of the line,
  but ring now sits on top of circle.
- The classification objective rewards one direction and lets the others fade.
  This is slide 4's tension, measured.
- Adding a "reconstruct the image" task alongside recovers most of it (97%).

## Slide 12 — What this means, and what is next

**Message:** discoverability comes from the pre-trained morphology network, and
the classification objective is the main thing that erodes it.

**Three takeaways**
1. Use a Galaxy Zoo pre-trained encoder for cutouts; from-scratch is not viable
   for rare classes.
2. Be deliberate about how hard the fingerprint is pushed to classify; keep a
   reconstruction-type objective in the mix.
3. Measure discoverability on real alerts next, with held-out classes (e.g.
   train without TDEs, see if TDEs look unusual).

**Honest caveats (say them, one line each)**
- Synthetic shapes are easy; real stamps have hosts, PSF blur, faint sources.
- Not yet tested: whether an ordinary ImageNet-pre-trained network does as well.
  If it does, the win is "pre-training", not "Galaxy Zoo".

---

## Backup slides

**B1 — The seven fingerprint variants.** The full results table from the
technical deck (regime, labelled separability, detection, clustering,
effective dimensions). Use only if asked how the from-scratch results depend
on the objective.

**B2 — How the metrics are defined.** Nearest-neighbour novelty score and
AUROC; k-means and AMI; linear probe; participation ratio. Keep to one line
each.

**B3 — Noise and data in detail.** `04_noise.png` full figure with the
512-vs-2048 comparison, for the "is it visibility or data?" question.

---

## Translation table for the speaker

| Technical term | Say instead |
|---|---|
| latent / embedding | fingerprint |
| encoder / backbone | the image network |
| frozen | weights never changed; used as published |
| training with a classification loss | teaching it to recognise circle vs. square |
| linear probe | "could a straight line separate them if we had labels" |
| AUROC | "how often a new shape scores as more unusual than a known one" |
| k-means / AMI | blind grouping; how well the groups match the true shapes |
| participation ratio | how many independent numbers the fingerprint really uses |
| t-SNE / PCA | a 2-D map of a 16-number space (say which, and that t-SNE exaggerates gaps) |

## What was dropped from the technical deck, and why

- Latent-dimension and bottleneck-type sweeps: null results; one sentence in
  backup if asked.
- The `embed()` no-grad bug: engineering detail.
- Training curves and silhouette: replaced by the single accuracy-vs-data plot.
- The reconstruction-only regime for the from-scratch network: it failed for a
  reason (pixel error rewards position and size, not shape) that needs too
  much explanation for the payoff.
- The seven-regime table: moved to backup; the story needs three variants only
  (from scratch, Zoobot untrained, Zoobot after classification training).

## Timing

| Slides | Minutes |
|---|---|
| 1–4 problem, idea, tension | 4 |
| 5–6 the toy and the two encoders | 2.5 |
| 7–8 results 1 and 2 | 3 |
| 9–11 discovery test and the catch | 4 |
| 12 takeaways and caveats | 1.5 |

## Figure sources

All in `docs/figures/shapes/`: `00_stamps.png`, `02_data_efficiency.png`,
`04_noise.png`, `06_backbone.png`, `10_novel_stamps.png`,
`11_novel_latent_views.png`. Slides 2–4 need new simple diagrams (bar,
pipeline box, two cartoon fingerprint spaces). Technical deck for reference:
`docs/slides/technical/shape_baseline_slides.pdf`.
