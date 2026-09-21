# VEIL

**Make machine perception measurable.**

A research and testing platform for measuring how computer-vision systems
perceive the physical world — and the apparatus to turn a result into a
garment whose label can be checked.

> VEIL measures detector behaviour under stated conditions. It does not
> provide, and does not claim, invisibility from surveillance. Every result is
> specific to the model, dataset, transformations and — for physical tests —
> the camera, distance, angle and lighting it was measured on.

---

## Why this exists

The nearest shipped comparison in this space publishes one sentence of
evidence: *"tested against YOLO."* No baseline. No control. No conditions. No
sample count.

That sentence is unfalsifiable, and it is the industry norm. VEIL's premise is
that a garment making a perception claim should ship with the experiment
attached — and that the experiment should be built to *fail* when the claim is
weak.

Most of this codebase is machinery for killing your own results:

| check | what it kills |
|---|---|
| **control arm** | "the pattern worked" when a rectangle covering your chest did it |
| **control variance** | an effect smaller than the spread between random patterns |
| **paired McNemar** | a difference that could be chance, tested against *every* control draw |
| **baseline headroom** | a "drop" measured on a detector that never saw the subject |
| **dataset qualification** | imagery with no room for a drop, *before* compute is spent |
| **manufacturing re-measure** | an effect that exists in float pixels but not in yarn |
| **physical arms** | a camera measurement where nobody recorded what was worn |

Each one was added because it caught a real, already-reported result of ours
and downgraded it.

---

## Quick start

```bash
git clone <this repo> && cd VEIL
uv venv --python 3.12 .venv            # or python -m venv .venv
.venv/bin/pip install -e ".[dev,camera]"

# Seeds an org, runs a full experiment, writes a JSON + PDF report.
.venv/bin/python scripts/demo.py                                        # seconds, offline
.venv/bin/python scripts/demo.py --detector fasterrcnn-mobilenet-320    # real weights
```

API and dashboard:

```bash
.venv/bin/veil init
.venv/bin/veil create-org "Acme" "you@example.com"    # prints an API key once
.venv/bin/uvicorn veil.api.main:app --reload          # http://localhost:8000/docs

cd web && npm install && npm run dev                  # http://localhost:3000
```

Paste the key on **Settings** in the dashboard. Docker: `cd infrastructure &&
docker compose up --build`.

Scripts add the repo root to `sys.path`, so they run from a checkout whether
or not the package is installed — but use `.venv/bin/python`, not a system or
conda `python`.

> **macOS: a silently broken editable install.** If `import veil` fails after
> `pip install -e .` *succeeded*, run
> `ls -lO .venv/lib/python3.12/site-packages/*.pth`. When those files carry the
> `hidden` flag, CPython's `site.py` skips them by design and the install does
> nothing, with no error. Clear it with
> `chflags nohidden .venv/lib/python*/site-packages/*.pth`.

---

## What it measures

```
authorized imagery
   │
   ├─ qualify ─────── is there headroom? (per image, before spending compute)
   │
   ├─ baseline sweep ──────────────── detector over the transformation grid
   ├─ placement ───────────────────── from the detector's own baseline box
   ├─ optimization (EOT) ──────────── gradient, or evolution if no gradients
   ├─ control sweeps × N ──────────── unoptimized patterns, identical in every
   │                                  way except the optimization
   ├─ candidate sweep ─────────────── the optimized pattern
   ├─ transfer ────────────────────── same pattern, detectors it never saw
   │
   ├─ attribution ─── occlusion vs. pattern, + exact paired McNemar
   ├─ manufacture ─── quantize to yarn/ink, re-measure, yarn card + artwork
   └─ report ─────── JSON + PDF, limitations section not optional
```

### The objective

```
P* = argmin_P  E_θ[ s(f(T(P, θ))) ] + λ_tv·TV(P) + λ_nps·NPS(P)
```

`P` the pattern, `f` an authorized detector, `s` its confidence for the target
label, `T` the render + camera chain, `θ` a draw from the transformation
distribution. `TV` and `NPS` penalise unprintable colour and detail.

The **expectation** is the point. Minimising `s` for one static view yields a
pattern that works in that view and nowhere else. A different draw is used
each step, so the pattern must survive a distribution.

### The claim that counts

```
total drop        = baseline_rate − candidate_rate    ← what a vendor quotes
occlusion drop    = baseline_rate − control_rate      ← the patch
attributable drop = control_rate  − candidate_rate    ← the only real claim
vs_best_control   = best_control_rate − candidate_rate ← the stricter test
```

Plus an **exact paired McNemar** test — the arms share images and
transformation points, so they are paired — reported as the *worst* p-value
across every control draw (an intersection-union test: conservative by
construction).

Detection rates carry Wilson 95% intervals. A robustness axis nothing
exercised reads `null`, never `0`. Below 20 samples per arm the verdict is
`inconclusive` rather than a number dressed as a finding.

---

## Measured results

Everything below was produced by running this code. Nothing is estimated.

### The optimizer works

First dataset to pass qualification (procedurally generated figures, baseline
0.889), `fasterrcnn-mobilenet-320`, 18 transformation points × 6 images:

| arm | detection rate |
|---|---|
| baseline | 0.880 (n=108) |
| control — 3 unoptimized draws | 0.389, spread 0.093 |
| **candidate** | **0.046** |

**+0.296 over the best control draw · p < 0.0001 · significant against every
draw.** Three times the control spread and three times the measured physical
noise floor.

### It survives manufacturing

`POST /patterns/{id}/manufacture` quantizes to a production spec and
re-measures against the same detector, dataset and grid:

| route | detection | effect retained |
|---|---|---|
| knit 3 st/cm | 0.056 | **97%** |
| knit 5 st/cm (12-gauge) | 0.065 | 95% |
| knit 8 st/cm | 0.083 | 89% |
| print 300 dpi | 0.037 | 103% |

Six yarns, 200 × 150 stitches at 30 × 40 cm, with a yarn card the mill can
quote from.

### Higher resolution made it worse

Same everything, only `pattern_size` changed:

| pattern_size | candidate | vs. best control | worst p | verdict |
|---|---|---|---|---|
| **96** | **0.046** | **+0.296** | **<0.0001** | significant |
| 192 | 0.157 | +0.083 | 0.064 | **not significant** |

Doubling the resolution tripled the detection rate and lost significance. The
manufacturing table points the same way (coarse knit retains more than fine).
The consistent reading: **high-frequency detail does not survive the
transformation distribution** — rotation, scale, lighting and cloth
deformation destroy it before the detector looks. Start coarse; it is cheaper
*and* better.

Two independent pieces of evidence, not a proven law.

### It does not transfer to a photorealistic person

The pattern above, measured on a photorealistic image of a person wearing a
hoodie (AI-generated, detected as `person` at 0.999 — ample headroom), same
detector, same 18-point grid, same three-arm discipline:

| arm | detection rate |
|---|---|
| baseline | 1.000 |
| control × 3 | 1.000, 1.000, 1.000 |
| **candidate** | **1.000** |

**Attributable effect: +0.000.** Not a weak effect — a complete null, at every
transformation point.

This is the most consequential result in the project. The +0.296 above was
measured on crude procedural figures sitting at a 0.88 baseline. A real person
is detected at 0.999, and a chest patch covering ~12% of the frame does not
move a detection with that much margin. Removing the subject's head entirely
only took it from 0.999 to 0.813.

**The synthetic result does not generalise.** Everything the optimizer
achieved was against a weak detector looking at a marginal subject. The honest
state of the science is that VEIL has not yet produced a pattern that affects
a realistic person, and the measured evidence says the problem is
substantially harder than the synthetic run suggested.

**Coverage is not the limiting factor.** Sweeping the patch from a chest
print to the whole frame on the same subject:

| coverage | % of frame | control | candidate | difference |
|---|---|---|---|---|
| 0.50 × 0.24 (chest print) | 12% | 1.000 | 1.000 | +0.000 |
| 0.90 × 0.45 (whole hoodie) | 40% | 1.000 | 1.000 | +0.000 |
| 1.00 × 0.60 | 60% | 1.000 | 1.000 | +0.000 |
| 1.00 × 0.80 | 80% | 0.333 | 0.000 | +0.333 |
| 1.00 × 1.00 | 100% | 0.000 | 0.000 | +0.000 |

Nothing happens anywhere in the wearable range. The only separation appears at
80% of the *frame* covered — where the head is obscured and even a random
pattern already halves detection on its own. That is occlusion, not an
adversarial effect, and it is not a garment.

**Optimizing directly against the realistic subject works — and overfits.**
120 iterations against a photorealistic full-body image took the loss from
1.000 to 0.015 in 66 seconds:

| test | control | optimized | reading |
|---|---|---|---|
| same image, same detector | 1.000 | **0.056** | the optimizer can break a 1.000 detection |
| held-out image | 1.000 | 1.000 | overfit to one image |
| unseen detectors (×2) | 1.000 | 1.000 | no transfer |
| after 5 st/cm knit | — | 1.000 | quantization destroys it |

So the capability exists and the generalisation does not. One training image
guarantees overfitting; one detector guarantees no transfer; and an effect
that lives in exact pixel values does not survive being knitted (unlike the
coarse synthetic-figure pattern, which kept 97%). The known remedies are all
expressible in this pipeline: many varied subjects, an ensemble of detectors
in the loss, and optimizing *under* the production constraint rather than
quantizing afterwards.

A separately generated "adversarial-looking" camo pattern (rendered by an
image model rather than optimized) was also tested: `person` at 0.999, 0.989
and 0.967 across the three detectors. **Looking like digital camouflage is not
the same as being adversarial**, and no visual inspection can tell the
difference — only a measurement can.

### First physical three-arm session — the simulation predicted it

Printed sheets (17.8 × 23.7 cm, 300 dpi, scale verified against a ruler bar)
held on the chest at 1.78 m — the visual angle of a 30 cm print at 3 m —
60 frames per arm, one session, laptop camera, `fasterrcnn-mobilenet-320`:

| arm | detection | Wilson 95% | mean confidence |
|---|---|---|---|
| baseline | 60/60 = 1.000 | [0.940, 1.000] | 0.9990 |
| control (unoptimized sheet) | 60/60 = 1.000 | [0.940, 1.000] | 0.9989 |
| candidate (`1f4d8a32`, knit artwork) | 60/60 = 1.000 | [0.940, 1.000] | 0.9987 |

**Attributable effect: +0.000.** Confidence moved by 0.0001.

The digital result on a realistic subject was +0.000, so this was the
predicted outcome, and it was written down *before* the test was run. That is
the useful part: the simulation's null held in the physical world, which is
the first evidence that a digital measurement here says something about a
physical one. It is evidence in one direction only — a digital null predicting
a physical null is a much weaker check than a digital effect surviving print.

### The physical noise floor

Two 60-frame camera runs, operator-confirmed that nothing changed between
them:

| quantity | spread |
|---|---|
| detection rate | **0.100** |
| mean confidence | **0.402** |

This is the bar any physical claim must clear. It is currently **larger than
any effect VEIL has measured physically** — and it is why all three arms must
be captured back to back in one session.

---

## Honest status

- **No garment exists.** Nothing has been knitted or printed.
- **Every positive result is on procedurally generated figures** that only
  `fasterrcnn-mobilenet-320` detects at all. RetinaNet and Faster R-CNN-R50
  correctly refuse to call them people. They demonstrate the pipeline; they
  are not a product claim and they do not transfer.
- **Transfer to realistic subjects is measured, and it fails.** +0.000 on a
  photorealistic person at a 0.999 baseline. The positive result exists only
  on crude procedural figures.
- **Physical robustness is measured, and it is zero**: one three-arm session,
  candidate indistinguishable from control and from baseline.
- **Video is not swept.** Datasets accept MP4; the runner skips non-images.
- **One placement per experiment**, from the first baseline detection.
- **Cloth deformation is sinusoidal**, a proxy for wrinkle warping — not cloth
  simulation.

Blocking item, in order of everything else: **a real photographic dataset**,
consented, passing qualification.

---

## API

Base `/api/v1`. Auth: `X-API-Key` or `Authorization: Bearer`. Interactive docs
at `/docs`.

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness, no auth |
| POST/GET | `/projects` | projects |
| POST | `/projects/{id}/artifacts` | upload authorized imagery (multipart) |
| POST/GET | `/datasets` | versioned sets, with provenance and annotations |
| POST | `/datasets/{id}/qualify` | **baseline-sweep before spending compute** |
| GET | `/detectors` | the allowed-detector catalogue (read-only by design) |
| POST/GET | `/experiments` | experiments |
| POST | `/experiments/{id}/run` | queue a run → 202 |
| GET | `/runs/{id}` | status, log, pinned versions |
| GET | `/experiments/{id}/results` | evaluations, attribution, transfer |
| GET | `/patterns/{id}/image` | the pattern PNG |
| POST | `/patterns/{id}/manufacture` | **quantize, re-measure, yarn card** |
| GET | `/artifacts/{id}/download` | artwork and imagery |
| POST/GET | `/physical-tests` | camera measurements (arm required) |
| POST/GET | `/garments` | printed items |
| GET | `/garments/{sku}/provenance` | what a QR tag resolves to |
| POST | `/experiments/{id}/report` | JSON + PDF report |
| GET | `/reports/{id}/pdf` | the PDF |

Errors: `400` invalid · `401` no/bad key · `404` not found *or not yours* ·
`409` run already active, or report before any completed run · `413` too large
· `415` unrecognised file content · `422` schema violation.

---

## Architecture

One Python package, one Next.js app. They change in the same commit and deploy
to the same process, so splitting them would add ceremony to buy a boundary
that does not exist.

```
veil/
  config · models · schemas · db · storage · events · jobs · cli
  api/      deps (auth + tenant scoping) · main · routes/
  ml/
    detectors/   base contract · torchvision · colour-blob · registry
    simulation/  deterministic differentiable transforms · renderer · pipeline
    patterns/    generator · EOT optimizer · constraints · manufacture
    evaluation/  detection · metrics · robustness · comparison · significance
    scenes.py    procedural figures for development
    qualify.py   pre-flight dataset check
    runner.py    the experiment engine
  reports/  builder (JSON) · render (PDF)
web/        Next.js dashboard (12 routes)
scripts/    demo · seed_scenes · physical_test · live_view · print_artwork
```

**One dialect.** Everything between renderer and detector speaks
`torch.Tensor`, `float32`, `[B,3,H,W]`, `[0,1]`. The simulation transforms are
**differentiable**, so the code that optimizes a pattern is the code that
evaluates it. Training on one pipeline and reporting on another would make the
numbers meaningless.

**Reproducible by schema.** Every run pins seed, git sha, detector weights,
dataset version and full configuration *before* work starts. Same seed →
identical metrics *and* byte-identical pattern artifacts (tested).

**Tenancy.** Every user-reachable row carries `organization_id`; every query
starts at `db.scoped()`; every single-row load goes through `deps.fetch`, which
returns **404, not 403**, for another org's row — existence is information.

**No user-supplied models, ever.** The detector registry is code.
`torch.load` on an untrusted file is arbitrary code execution, so there is no
endpoint to add one and `POST /detectors` returns 405.

### Deliberately not built

| not built | why | when |
|---|---|---|
| Celery + Redis | runs are minutes on one box; state is in `experiment_runs` | when runs must survive a deploy |
| **Alembic** | was pre-release | **now blocking** — real data exists, one migration was hand-written |
| Kubernetes | one box runs everything | more than one box |
| MLflow / W&B | runs already pin every version | comparing hundreds of runs visually |
| Payments | not in scope | when there is something to sell |

---

## Physical testing

```bash
# Print at exact size, with a ruler bar to verify it.
.venv/bin/python scripts/print_artwork.py --api-key "$KEY" \
  --artwork "$ARTWORK_ID" --width-cm 21 --dpi 300

# Live camera, detection overlay, rolling rate, three arms.
.venv/bin/python scripts/live_view.py --api-key "$KEY" \
  --pattern 1f4d8a32 --artwork "$ARTWORK_ID" \
  --distance 1.78 --angle 0 --lighting "office, ~400 lux" --environment "lab"
```

`b`/`c`/`k` select arm · `SPACE` record · `s` submit · `q` quit. Nothing
reaches the API until `s`, and every record carries its arm — only `candidate`
records score.

**Size is part of the experiment.** A 30 cm print at 3 m subtends ~5.7°. A
pattern shown smaller is a postage stamp and the detector ignores it for
reasons unrelated to the pattern. **Run all three arms back to back**; drift
between sessions exceeds the effect.

---

## Development

```bash
.venv/bin/python -m pytest -q                 # 107 tests
.venv/bin/python -m pytest -q -m "not slow"   # skip tests needing model weights
cd web && npx tsc --noEmit && npm run build
```

Tests assert behaviour, not shape: tenant isolation, seed reproducibility down
to byte-identical artifacts, that a useless optimizer is *reported* as useless,
that zero discordant pairs give `p = 1.0` rather than "highly significant",
that a knit chart only ever uses loaded yarns, and that report PDFs contain
the numbers rather than merely being valid PDFs.

Design notes and the decision record live in `docs/` on disk and are not
tracked in this repository.

## Licence & scope

Detector weights are third-party (torchvision, BSD-3-Clause, COCO). VEIL runs
only against open models it is authorised to test, on data the operator has
recorded a basis for holding. Datasets containing identifiable people require
a recorded consent basis before they can be created.
# VEIL
