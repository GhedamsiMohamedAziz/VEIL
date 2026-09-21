#!/usr/bin/env python3
"""Capture your own training photographs with the camera used for testing.

    .venv/bin/python scripts/capture_subjects.py ~/Downloads/veil-subjects

Press g and walk away: after a countdown it takes one photo every few
seconds, with a prompt on screen telling you what to change before each one.
Variety is the whole point - a pattern trained on twenty near-identical
frames overfits exactly like a pattern trained on one.

Using the same camera for training and for the physical test removes the
domain gap between them. Wear a PLAIN top: the print is composited onto the
torso, and a pattern already there is noise the optimizer has to fight.

Everything stays on this machine. Nothing is uploaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import time

import numpy as np
import torch

PROMPTS = [
    "face the camera, arms down",
    "step BACK one pace",
    "turn slightly LEFT",
    "turn slightly RIGHT",
    "step back again - full body in frame",
    "arms crossed",
    "hands in pockets",
    "step CLOSER two paces",
    "shift to the LEFT of the frame",
    "shift to the RIGHT of the frame",
    "one arm raised",
    "lean on one leg",
    "step back, turn a quarter left",
    "quarter right",
    "walk-in pose, mid stride",
    "face the camera, far back",
    "closer, arms down",
    "look away from the camera",
    "hands behind your back",
    "any natural pose",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=4.0,
                        help="seconds between shots, to change pose")
    parser.add_argument("--countdown", type=float, default=8.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--detector", default="fasterrcnn-mobilenet-320")
    parser.add_argument("--label", default="person")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv is required: pip install 'veil[camera]'")
    from veil.ml.detectors import registry

    folder = args.folder.expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    # Highest index, not a count: deleting early photos must never make new ones overwrite later ones.
    existing = max((int(m.group(1)) for p in folder.glob("subject-*.jpg")
                    if (m := re.fullmatch(r"subject-(\d+)", p.stem))), default=0)
    print(f"saving to {folder}  (numbering continues after {existing})")
    print("loading detector…")
    detector = registry.get(args.detector)

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        hint = ("\nmacOS: System Settings > Privacy & Security > Camera, enable "
                "your terminal, then restart it." if sys.platform == "darwin" else "")
        sys.exit(f"cannot open camera {args.camera_index}.{hint}")

    cv2.namedWindow("VEIL capture", cv2.WINDOW_NORMAL)
    print("\n  g   start: countdown, then one photo every "
          f"{args.interval:.0f}s ({args.count} photos)")
    print("  x   stop the sequence")
    print("  q   quit\n")

    next_shot, taken, saved, flash_until = 0.0, 0, existing, 0.0
    score, frame_index = 0.0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        clean = frame.copy()
        height, width = frame.shape[:2]
        now = time.time()

        if frame_index % 3 == 0:  # the overlay only needs a rough live score
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                found = [d for d in detector.predict(tensor.unsqueeze(0), threshold=0.3)[0]
                         if d.label == args.label]
            score = max((d.score for d in found), default=0.0)
            box = max(found, key=lambda d: d.score).box if found else None
        if score and box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 200, 90), 2)

        lines = [f"{args.label} {score:.2f}   saved {saved - existing}/{args.count}"]
        if next_shot:
            remaining = max(next_shot - now, 0.0)
            prompt = PROMPTS[taken % len(PROMPTS)]
            lines += [f"NEXT: {prompt}", f"photo in {remaining:3.1f}s"]
            cv2.putText(frame, f"{remaining:.0f}", (width // 2 - 40, height // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, (60, 200, 235), 8, cv2.LINE_AA)
            if now >= next_shot:
                saved += 1
                taken += 1
                path = folder / f"subject-{saved:03d}.jpg"
                cv2.imwrite(str(path), clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
                kept = "" if score >= 0.5 else "  (detector unsure - may be skipped in training)"
                print(f"  saved {path.name}  {args.label} {score:.2f}{kept}", flush=True)
                flash_until = now + 0.25
                next_shot = (now + args.interval) if taken < args.count else 0.0
                if not next_shot:
                    print(f"done: {taken} photos in {folder}", flush=True)
        else:
            lines.append("press g to start")

        if now < flash_until:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (255, 255, 255), 24)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (560, 18 + 30 * len(lines)), (18, 17, 21), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (232, 234, 240), 2, cv2.LINE_AA)
        cv2.imshow("VEIL capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g") and not next_shot:
            taken = 0
            next_shot = time.time() + args.countdown
            print(f"starting in {args.countdown:.0f}s…", flush=True)
        elif key == ord("x"):
            next_shot = 0.0
            print("stopped", flush=True)

    capture.release()
    cv2.destroyAllWindows()
    print(f"\n{saved - existing} new photo(s). Train with:")
    print(f"  .venv/bin/python scripts/train_on_folder.py {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
