#!/usr/bin/env python3
"""Run a physical test: a real camera, a real printed pattern, real numbers.

    python scripts/physical_test.py --experiment <id> --pattern <id> \
        --api-key $VEIL_API_KEY --distance 3.0 --angle 15 \
        --lighting "office fluorescent" --environment "lab" --frames 60

Captures N frames, runs the experiment's detector on each, and posts one
PhysicalTest record with the measured detection rate *and* the conditions it
was measured under. Conditions are required arguments on purpose: a physical
number without them is not a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from a checkout whether or not the package is installed: sys.path[0] is
# scripts/, so the repo root has to be added explicitly. Harmless when veil is
# installed, and it survives an editable install that silently did nothing
# (macOS sets UF_HIDDEN on .pth files, which CPython's site.py then skips).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import sys
import urllib.error
import urllib.request

import numpy as np
import torch


def call(url: str, api_key: str, payload: dict | None = None) -> dict:
    """One JSON request. piggy: urllib, not a dependency for two calls."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"X-API-Key": api_key, **({"Content-Type": "application/json"} if data else {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        sys.exit(f"{exc.code} from {url}\n{detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach {url}: {exc.reason}\n"
                 "Is the API running?  uvicorn veil.api.main:app")


def capture(camera_index: int, frames: int, warmup: int = 10):
    try:
        import cv2
    except ImportError:  # pragma: no cover
        sys.exit("opencv is required: pip install 'veil[camera]'")

    capture_device = cv2.VideoCapture(camera_index)
    if not capture_device.isOpened():
        hint = ""
        if sys.platform == "darwin":
            hint = (
                "\nOn macOS the terminal needs camera access: System Settings > "
                "Privacy & Security > Camera, enable your terminal app, then "
                "restart it. Grant this yourself - it is a system permission."
            )
        sys.exit(f"cannot open camera {camera_index}.{hint}")
    try:
        for _ in range(warmup):  # let auto-exposure settle
            capture_device.read()
        width = int(capture_device.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture_device.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture_device.get(cv2.CAP_PROP_FPS)) or 0.0
        collected = []
        for _ in range(frames):
            ok, frame = capture_device.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            collected.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0)
        return collected, f"{width}x{height}", fps
    finally:
        capture_device.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000/api/v1")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--arm", required=True, choices=("baseline", "control", "candidate"),
        help="what was on the subject: nothing (baseline), an unoptimized "
             "print (control), or the optimized one (candidate). Only "
             "candidate records count towards physical robustness.",
    )
    parser.add_argument("--pattern")
    parser.add_argument("--garment")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-name", default="default camera")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--distance", type=float, required=True, help="metres to subject")
    parser.add_argument("--angle", type=float, required=True, help="degrees off-axis")
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    for name, value in (("--experiment", args.experiment), ("--pattern", args.pattern)):
        if value and (value.startswith("<") or value.endswith(">")):
            sys.exit(f"{name} looks like a placeholder ({value!r}). Pass a real id - "
                     f"GET {args.api}/experiments lists them.")

    experiment = call(f"{args.api}/experiments/{args.experiment}", args.api_key)
    target_label = experiment["configuration"].get("target_label", "person")

    from veil.ml.detectors import registry

    detector = registry.get(experiment["detector_id"])
    frames, resolution, fps = capture(args.camera_index, args.frames)
    if not frames:
        sys.exit("captured no frames")

    size = frames[0].shape[-2:]
    batch = torch.stack([f for f in frames if f.shape[-2:] == size])
    detected = 0
    scores: list[float] = []
    for i in range(0, batch.shape[0], 4):  # small batches keep memory flat
        for dets in detector.predict(batch[i : i + 4], threshold=args.threshold):
            hits = [d for d in dets if d.label == target_label]
            detected += bool(hits)
            scores.append(max((d.score for d in hits), default=0.0))

    total = len(scores)
    result = {
        "detected": detected > 0,
        "detection_rate": detected / total,
        "detection_count": detected,
        "frames_analyzed": total,
        "max_score": max(scores),
        "mean_score": sum(scores) / total,
        "target_label": target_label,
        "threshold": args.threshold,
    }
    print(f"[{args.arm}] {detected}/{total} frames contained '{target_label}' "
          f"(rate {result['detection_rate']:.3f}, mean score {result['mean_score']:.3f})")

    recorded = call(f"{args.api}/physical-tests", args.api_key, {
        "experiment_id": args.experiment, "arm": args.arm, "pattern_id": args.pattern,
        "garment_id": args.garment, "camera": args.camera_name,
        "resolution": resolution, "fps": fps, "distance_m": args.distance,
        "angle_deg": args.angle, "lighting": args.lighting,
        "environment": args.environment, "frame_count": total,
        "result": result, "notes": args.notes,
    })
    print("recorded physical test", recorded["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
