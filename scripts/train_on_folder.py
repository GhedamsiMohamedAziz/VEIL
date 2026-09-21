#!/usr/bin/env python3
"""Train a pattern on a folder of subject images and test it on held-out ones.

    .venv/bin/python scripts/train_on_folder.py ~/Downloads/veil-subjects

One command for the question that matters: does a pattern trained on some
subjects work on subjects it has never seen? Each image gets the print placed
on its own torso (from the detector's box), a share of the images is held out
and never optimized against, and the result is reported against unoptimized
controls with the paired test - the same discipline as the platform's runs.

Images should show a person with a PLAIN top. A pattern already on the
garment is just background the optimizer has to fight.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random

import torch
import torch.nn.functional as F

from veil.ml.detectors import registry
from veil.ml.evaluation.comparison import compare
from veil.ml.evaluation.detection import sweep
from veil.ml.evaluation.metrics import summarize
from veil.ml.patterns import constraints, generator
from veil.ml.patterns.manufacture import ProductionSpec, to_artwork
from veil.ml.patterns.optimizer import OptimizationConfig, optimize
from veil.ml.patterns.serialization import image_to_tensor, to_png
from veil.ml.simulation.renderer import Placement
from veil.ml.simulation.transforms import TransformSpec

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load(path: Path, height: int, width: int) -> torch.Tensor:
    tensor = image_to_tensor(path.read_bytes(), size=None)
    return F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--detector", default="fasterrcnn-mobilenet-320")
    parser.add_argument("--label", default="person")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--holdout", type=float, default=0.3,
                        help="share of usable images never optimized against")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--coverage", type=float, default=0.45,
                        help="fraction of the detected box the print covers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("var/trained"))
    args = parser.parse_args()

    paths = sorted(p for p in args.folder.expanduser().iterdir()
                   if p.suffix.lower() in EXTENSIONS)
    if not paths:
        sys.exit(f"no images in {args.folder}")
    detector = registry.get(args.detector)

    # 1. Qualify: keep images where the detector confidently sees the target,
    #    and take the print placement from its own box.
    print(f"qualifying {len(paths)} image(s) against {args.detector}")
    usable = []
    for path in paths:
        image = load(path, args.height, args.width)
        hits = [d for d in detector.predict(image, threshold=args.threshold)[0]
                if d.label == args.label]
        if not hits:
            print(f"  skip  {path.name}  (no {args.label!r} at {args.threshold})")
            continue
        best = max(hits, key=lambda d: d.score)
        placement = Placement.from_detection(best, (args.height, args.width), args.coverage)
        usable.append((path, image, placement))
        print(f"  ok    {path.name}  score {best.score:.3f}")
    if len(usable) < 2:
        sys.exit("need at least 2 usable images (1 to train on, 1 held out)")

    # 2. Split. The held-out images are the whole point: a pattern that only
    #    works on images it was optimized against is an overfit, not a garment.
    random.Random(args.seed).shuffle(usable)
    n_test = max(1, round(len(usable) * args.holdout))
    test, train = usable[:n_test], usable[n_test:]
    print(f"\ntrain on {len(train)}, hold out {len(test)}: "
          f"{', '.join(p.name for p, _, _ in test)}")
    if len(train) < 5:
        print("WARNING: fewer than 5 training subjects - expect the pattern to "
              "overfit them. 15-20 varied images is a sensible minimum.")

    spec = TransformSpec(rotation_deg=[-15.0, 0.0, 15.0], scale=[0.9, 1.0, 1.1],
                         translate=[0.0], perspective=[0.0], brightness=[0.8, 1.0, 1.2],
                         contrast=[1.0], blur_sigma=[0.0, 0.8], noise_std=[0.0],
                         deformation=[0.0, 0.03], mode="grid", max_samples=24)

    train_images = torch.cat([image for _, image, _ in train])
    train_places = [placement for _, _, placement in train]
    config = OptimizationConfig(target_label=args.label, pattern_size=96,
                                iterations=args.iterations, batch_transforms=3,
                                learning_rate=0.08, tv_weight=0.02, nps_weight=0.02,
                                seed=args.seed)

    def progress(step: int, record: dict) -> None:
        if step % 25 == 0 or step == args.iterations - 1:
            print(f"  iter {step:4d}  loss {record['loss']:.4f}", flush=True)

    print(f"\noptimizing ({args.iterations} iterations)")
    result = optimize(detector, train_images, train_places, spec, config, on_iteration=progress)

    # 3. Measure, three arms, on train and on held-out separately.
    controls = [constraints.quantize_to_palette(
        generator.initialize(96, "palette_noise", seed=args.seed + 100_000 + i))
        for i in range(3)]

    def measure(subset, pattern):
        images = torch.cat([image for _, image, _ in subset])
        places = [placement for _, _, placement in subset] if pattern is not None else None
        return sweep(detector, images, spec, args.label, pattern=pattern, placement=places,
                     seed=args.seed, threshold=args.threshold,
                     image_ids=[p.name for p, _, _ in subset])

    knit_spec = ProductionSpec(method="knit", stitches_per_cm=5, width_cm=30, height_cm=40)
    knitted = F.interpolate(to_artwork(result.pattern, knit_spec).unsqueeze(0),
                            size=(96, 96), mode="area").squeeze(0)

    print()
    verdict = None
    for name, subset in (("TRAIN (seen)", train), ("HELD-OUT (never seen)", test)):
        baseline = measure(subset, None)
        draws = [measure(subset, c) for c in controls]
        candidate = measure(subset, result.pattern)
        pooled = [r for d in draws for r in d]
        report = compare(baseline, candidate, pooled, draws)["attribution"]
        knit_rate = summarize(measure(subset, knitted))["detection_rate"]
        print(f"{name}  (n={len(candidate)} per arm)")
        print(f"  baseline  {summarize(baseline)['detection_rate']:.3f}")
        print(f"  control   {[round(summarize(d)['detection_rate'], 3) for d in draws]}")
        print(f"  candidate {summarize(candidate)['detection_rate']:.3f}"
              f"    after 5 st/cm knit: {knit_rate:.3f}")
        print(f"  vs best control {report.get('attributable_vs_best_control', 0):+.3f}   "
              f"worst p {report['significance']['worst_p_value']:.4f}")
        print(f"  -> {report['verdict']}\n")
        verdict = report

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pattern.png").write_bytes(to_png(result.pattern))
    (args.out / "pattern-knit.png").write_bytes(to_png(to_artwork(result.pattern, knit_spec)))
    print(f"saved {args.out}/pattern.png and pattern-knit.png")
    print("The HELD-OUT block is the result. The TRAIN block only shows the "
          "optimizer ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
