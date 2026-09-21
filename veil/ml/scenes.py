"""Procedural human-figure scenes for development and qualification.

Real photographs of consenting people are the right input (docs/product.md).
Until a dataset exists, development needs subjects a detector *reliably*
sees, because an optimizer cannot be tested on images with no headroom - the
crude rectangle used earlier qualified at 0.556 and told us nothing.

These are explicitly synthetic and every dataset built from them says so.
They are for exercising the pipeline and for answering "does the optimizer
beat its control at all", not for product claims.

**Measured scope, so nobody over-reads them** (24 figures, threshold 0.3):

| detector | detection rate | mean score |
|---|---|---|
| fasterrcnn-mobilenet-320 | 0.958 | 0.927 |
| retinanet-resnet50 | 0.000 | 0.000 |
| fasterrcnn-resnet50 | 0.000 | 0.000 |

The stronger models correctly refuse to call these people. That makes the
figures usable for testing the optimizer against the one detector that does
see them, and **useless for transfer studies** - a transfer result needs a
subject every detector agrees is a person, which means photographs.
"""

from __future__ import annotations

import io
import math
import random

from PIL import Image, ImageDraw, ImageFilter


def _body_palette(rng: random.Random) -> dict[str, tuple[int, int, int]]:
    skin = rng.choice([(226, 190, 160), (198, 154, 120), (150, 110, 82),
                       (104, 74, 56), (240, 212, 186)])
    shirt = tuple(rng.randint(40, 210) for _ in range(3))
    trousers = tuple(rng.randint(30, 110) for _ in range(3))
    return {"skin": skin, "shirt": shirt, "trousers": trousers}


def _background(draw: ImageDraw.ImageDraw, size: int, rng: random.Random) -> None:
    """A soft vertical gradient plus a horizon line: enough structure that the
    figure is not floating on a flat field, which flattens detector scores."""
    top = tuple(rng.randint(150, 220) for _ in range(3))
    bottom = tuple(rng.randint(90, 150) for _ in range(3))
    for y in range(size):
        t = y / max(size - 1, 1)
        draw.line(
            [(0, y), (size, y)],
            fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)),
        )
    horizon = int(size * rng.uniform(0.62, 0.78))
    draw.rectangle([0, horizon, size, size],
                   fill=tuple(rng.randint(70, 120) for _ in range(3)))


def _figure_geometry(rng: random.Random, size: int, height_fraction: float):
    """(figure_h, head_r, cx, top_y). Shared by `person` and `torso_box` so the
    annotation cannot drift from the drawing; call it at the same point of the
    RNG stream in both - after the background and the palette."""
    figure_h = size * height_fraction * rng.uniform(0.9, 1.05)
    head_r = figure_h / 15.0
    cx = size * rng.uniform(0.4, 0.6)
    top_y = (size - figure_h) * rng.uniform(0.35, 0.75)
    return figure_h, head_r, cx, top_y


def person(
    size: int = 480,
    seed: int = 0,
    *,
    height_fraction: float = 0.72,
) -> Image.Image:
    """One synthetic standing figure, proportioned on a ~7.5-head canon."""
    rng = random.Random(seed)
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    _background(draw, size, rng)
    colors = _body_palette(rng)
    figure_h, head_r, cx, top_y = _figure_geometry(rng, size, height_fraction)

    head_cy = top_y + head_r
    neck_y = head_cy + head_r * 0.95
    shoulder_y = neck_y + head_r * 0.55
    hip_y = shoulder_y + figure_h * 0.33
    foot_y = top_y + figure_h
    shoulder_w = head_r * 3.2
    hip_w = shoulder_w * 0.82

    lean = rng.uniform(-0.06, 0.06)

    def x(fraction: float, y: float) -> float:
        return cx + fraction + (y - shoulder_y) * lean

    # legs
    for side in (-1, 1):
        knee_x = x(side * hip_w * 0.30 + side * rng.uniform(0, 8), hip_y)
        ankle_x = knee_x + side * rng.uniform(-6, 10)
        draw.line([(x(side * hip_w * 0.32, hip_y), hip_y), (knee_x, (hip_y + foot_y) / 2),
                   (ankle_x, foot_y)],
                  fill=colors["trousers"], width=int(head_r * 0.95), joint="curve")
        draw.ellipse([ankle_x - head_r * 0.5, foot_y - head_r * 0.22,
                      ankle_x + head_r * 0.62, foot_y + head_r * 0.26],
                     fill=(45, 42, 40))

    # Torso: tapered to a waist rather than a rectangle, with rounded
    # shoulders. The waist is what makes the outline read as a body.
    waist_y = shoulder_y + (hip_y - shoulder_y) * 0.55
    waist_w = shoulder_w * 0.70
    draw.polygon(
        [(x(-shoulder_w / 2, shoulder_y), shoulder_y),
         (x(shoulder_w / 2, shoulder_y), shoulder_y),
         (x(waist_w / 2, waist_y), waist_y),
         (x(hip_w / 2, hip_y), hip_y),
         (x(-hip_w / 2, hip_y), hip_y),
         (x(-waist_w / 2, waist_y), waist_y)],
        fill=colors["shirt"],
    )
    for side in (-1, 1):
        draw.ellipse([x(side * shoulder_w / 2, shoulder_y) - head_r * 0.42,
                      shoulder_y - head_r * 0.38,
                      x(side * shoulder_w / 2, shoulder_y) + head_r * 0.42,
                      shoulder_y + head_r * 0.46], fill=colors["shirt"])

    # Arms, held slightly clear of the torso. This is a narrow band: flush
    # arms merge into the torso and the silhouette stops reading as a person
    # (0.97 -> 0.00 on some seeds), while arms spread wide read as a kite
    # (COCO "kite" at 0.888). Swept empirically; 0.10-0.30 gives 20/20.
    for side in (-1, 1):
        swing = rng.uniform(0.10, 0.30) * side
        elbow_y = shoulder_y + figure_h * 0.17
        wrist_y = shoulder_y + figure_h * 0.33
        elbow_x = x(side * shoulder_w * 0.58, elbow_y) + swing * head_r * 1.6
        wrist_x = elbow_x + side * rng.uniform(2, 12) + swing * head_r * 1.2
        draw.line([(x(side * shoulder_w * 0.44, shoulder_y), shoulder_y),
                   (elbow_x, elbow_y), (wrist_x, wrist_y)],
                  fill=colors["shirt"], width=int(head_r * 0.66), joint="curve")
        draw.ellipse([wrist_x - head_r * 0.3, wrist_y - head_r * 0.28,
                      wrist_x + head_r * 0.34, wrist_y + head_r * 0.36],
                     fill=colors["skin"])

    # neck and head
    draw.rectangle([x(-head_r * 0.34, neck_y), neck_y - 2,
                    x(head_r * 0.34, neck_y), shoulder_y + 2], fill=colors["skin"])
    draw.ellipse([x(-head_r, head_cy) - 0, head_cy - head_r * 1.12,
                  x(head_r, head_cy), head_cy + head_r * 1.12], fill=colors["skin"])
    hair = tuple(max(c - rng.randint(60, 110), 12) for c in colors["skin"])
    draw.chord([x(-head_r, head_cy), head_cy - head_r * 1.12,
                x(head_r, head_cy), head_cy + head_r * 0.6], 180, 360, fill=hair)

    return image.filter(ImageFilter.GaussianBlur(radius=0.6))


def person_png(size: int = 480, seed: int = 0) -> bytes:
    buffer = io.BytesIO()
    person(size=size, seed=seed).save(buffer, format="PNG")
    return buffer.getvalue()


def torso_box(size: int = 480, seed: int = 0) -> tuple[float, float, float, float]:
    """Approximate ground-truth box for the figure, for annotated datasets."""
    rng = random.Random(seed)
    # Replay what person() draws first: the background and palette consume
    # random numbers, and skipping them boxed a figure that was never drawn.
    _background(ImageDraw.Draw(Image.new("RGB", (size, size))), size, rng)
    _body_palette(rng)
    figure_h, head_r, cx, top_y = _figure_geometry(rng, size, 0.72)
    half_w = head_r * 3.2  # measured against the drawn figure: arms reach past the shoulders
    return (cx - half_w, top_y, cx + half_w, top_y + figure_h)
