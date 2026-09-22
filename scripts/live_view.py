#!/usr/bin/env python3
"""Live physical testing: camera, detector overlay, and a pattern to hold up.

    python scripts/live_view.py --api-key "$VEIL_API_KEY" \
        --experiment "$EXPERIMENT_ID" --pattern 1f4d8a32 \
        --distance 3.0 --angle 0 --lighting "office, ~400 lux" --environment "lab"

Two windows: the camera with live detection boxes and a rolling detection
rate, and the pattern itself, sized so it can be shown on a second screen or
printed.

Built for one person working alone: press b, c or k and walk away. A
countdown gives you time to get into position, exactly --frames frames are
recorded, and the measurement is submitted automatically. Frames captured
while you walk to and from the keyboard never enter the result.

Every submission carries the arm it was recorded under - a physical number
without that is uninterpretable (ADR-008).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import atexit
import json
import time
import urllib.error
import urllib.request
from collections import deque

import numpy as np
import torch

ARMS = {ord("b"): "baseline", ord("c"): "control", ord("k"): "candidate"}
ARM_COLOR = {"baseline": (232, 134, 79), "control": (122, 158, 63), "candidate": (47, 127, 207)}


class ApiError(RuntimeError):
    pass


def call(url: str, api_key: str, payload: dict | None = None, *, fatal: bool = True) -> dict:
    """`fatal=False` raises ApiError instead of exiting - for calls made while
    a measurement is in hand, where exiting would throw it away."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"X-API-Key": api_key}
    if data:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        message = f"{exc.code} from {url}\n{exc.read().decode(errors='replace')[:400]}"
    except urllib.error.URLError as exc:
        message = f"cannot reach {url}: {exc.reason}\nIs the API running?"
    if fatal:
        sys.exit(message)
    raise ApiError(message)


def resolve_pattern(api: str, api_key: str, prefix: str) -> dict:
    """Accept an id prefix, so you can paste the short form you see in the UI."""
    patterns = call(f"{api}/patterns", api_key)
    matches = [p for p in patterns if p["id"].startswith(prefix)]
    if not matches:
        sys.exit(f"no pattern starting with {prefix!r}; {len(patterns)} exist")
    if len(matches) > 1:
        sys.exit(f"{prefix!r} matches {len(matches)} patterns; use more characters")
    return matches[0]


def fetch_image(api: str, api_key: str, path: str):
    import cv2

    request = urllib.request.Request(f"{api}{path}", headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = np.frombuffer(response.read(), dtype=np.uint8)
    except urllib.error.HTTPError as exc:
        sys.exit(f"{exc.code} from {api}{path}\n{exc.read().decode(errors='replace')[:400]}")
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach {api}{path}: {exc.reason}\nIs the API running?")
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"{api}{path} did not return a decodable image")
    return image


def hud(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    import cv2

    pad, line_h = 10, 26
    height = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (430, height), (18, 17, 21), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    for index, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + line_h * (index + 1) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000/api/v1")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--pattern", required=True, help="pattern id or unique prefix")
    parser.add_argument("--experiment", help="defaults to the pattern's experiment")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-name", default="default camera")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--detect-every", type=int, default=1,
                        help="run the detector every Nth frame (raise if the feed lags)")
    parser.add_argument("--window", type=int, default=60,
                        help="frames in the rolling detection-rate window")
    parser.add_argument("--frames", type=int, default=60,
                        help="frames recorded per measurement")
    parser.add_argument("--save-frames", type=Path, default=None,
                        help="folder to save BASELINE-arm frames into, as training "
                             "images. Only baseline: in the other arms a sheet of "
                             "paper covers the torso the print is composited onto.")
    parser.add_argument("--save-every", type=int, default=3,
                        help="keep every Nth recorded frame (consecutive frames "
                             "are near-duplicates)")
    parser.add_argument("--countdown", type=float, default=8.0,
                        help="seconds between pressing an arm key and recording, "
                             "to walk back and get into position")
    parser.add_argument("--pattern-window-px", type=int, default=520)
    parser.add_argument("--artwork", help="artifact id of production artwork to display "
                                          "instead of the raw pattern - this is what a "
                                          "garment actually carries")
    parser.add_argument("--artwork-cm", type=float, default=30.0,
                        help="finished width of the artwork on the garment, for the "
                             "display-size guidance printed at startup")
    parser.add_argument("--distance", type=float, required=True, help="metres to subject")
    parser.add_argument("--angle", type=float, required=True, help="degrees off-axis")
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv is required: pip install 'veil[camera]'")

    pattern = resolve_pattern(args.api, args.api_key, args.pattern)
    experiment_id = args.experiment or pattern["experiment_id"]
    experiment = call(f"{args.api}/experiments/{experiment_id}", args.api_key)
    target_label = experiment["configuration"].get("target_label", "person")

    from veil.ml.detectors import registry

    print(f"pattern    {pattern['id']}  v{pattern['version']}")
    print(f"experiment {experiment['name']}  detector={experiment['detector_id']}  "
          f"target={target_label!r}")
    print("loading detector…")
    detector = registry.get(experiment["detector_id"])

    if args.artwork:
        pattern_image = fetch_image(args.api, args.api_key,
                                    f"/artifacts/{args.artwork}/download")
        source = f"production artwork {args.artwork[:8]}"
    else:
        pattern_image = fetch_image(args.api, args.api_key,
                                    f"/patterns/{pattern['id']}/image")
        source = "raw optimized pattern"
    side = args.pattern_window_px
    rows, cols = pattern_image.shape[:2]
    scale = side / max(rows, cols)
    pattern_display = cv2.resize(pattern_image, (int(cols * scale), int(rows * scale)),
                                 interpolation=cv2.INTER_NEAREST)

    capture = cv2.VideoCapture(args.camera_index)
    # Released on every way out (exception, sys.exit, q), not only the last line.
    atexit.register(cv2.destroyAllWindows)
    atexit.register(capture.release)
    if not capture.isOpened():
        hint = ("\nmacOS: System Settings > Privacy & Security > Camera, enable "
                "your terminal, then restart it." if sys.platform == "darwin" else "")
        sys.exit(f"cannot open camera {args.camera_index}.{hint}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_reported = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0

    cv2.namedWindow("VEIL live", cv2.WINDOW_NORMAL)
    cv2.namedWindow("VEIL pattern", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VEIL pattern", pattern_display.shape[1], pattern_display.shape[0])
    cv2.imshow("VEIL pattern", pattern_display)

    # The pattern must subtend the same visual angle a garment print would, or
    # the test measures a postage stamp. A 30 cm print at 3 m is ~5.7 degrees.
    print(f"\ndisplaying:  {source}  ({cols}x{rows} -> "
          f"{pattern_display.shape[1]}x{pattern_display.shape[0]} px)")
    print(f"SIZE MATTERS: to stand in for a {args.artwork_cm:.0f} cm print at "
          f"{args.distance:.1f} m, the pattern must be physically that wide.")
    print("  - print it at that size, or")
    print(f"  - show it on a screen ~{args.artwork_cm:.0f} cm wide and keep the "
          f"camera at {args.distance:.1f} m, or")
    print("  - move the camera proportionally closer for a smaller display.")
    print("A pattern shown smaller than a real print is not the same experiment.")

    print("\n  b / c / k   measure baseline / control / candidate:")
    print(f"              {args.countdown:.0f}s countdown -> {args.frames} frames -> auto-submit")
    print("  x           cancel the current countdown or recording")
    print("  q           quit\n")

    save_dir = args.save_frames.expanduser() if args.save_frames else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"baseline frames will be saved to {save_dir}")
    session_tag, saved_count = "", 0
    take_files: list[Path] = []  # frames of the take in progress, deleted if it is cancelled

    arm = "baseline"
    rolling: deque[bool] = deque(maxlen=args.window)
    recording = False
    countdown_until = 0.0
    recorded: list[tuple[bool, float]] = []
    history: list[str] = []

    def submit() -> str:
        total = len(recorded)
        hits = sum(1 for hit, _ in recorded if hit)
        scores = [score for _, score in recorded]
        payload = {
            "experiment_id": experiment_id, "arm": arm, "pattern_id": pattern["id"],
            "camera": args.camera_name, "resolution": f"{width}x{height}",
            "fps": fps_reported, "distance_m": args.distance, "angle_deg": args.angle,
            "lighting": args.lighting, "environment": args.environment,
            "frame_count": total, "notes": args.notes,
            "result": {
                "detected": hits > 0, "detection_rate": hits / total,
                "detection_count": hits, "frames_analyzed": total,
                "max_score": max(scores), "mean_score": sum(scores) / total,
                "target_label": target_label, "threshold": args.threshold,
            },
        }
        try:
            created = call(f"{args.api}/physical-tests", args.api_key, payload, fatal=False)
        except ApiError as exc:
            # The take cost a walk across the room: keep it, and keep going.
            kept = Path(f"physical-test-{time.strftime('%Y%m%d-%H%M%S')}.json")
            kept.write_text(json.dumps(payload, indent=2))
            print(f"NOT SUBMITTED ({exc})\n  measurement kept in {kept}; "
                  "POST it to /physical-tests once the API is back", flush=True)
            return f"{arm:9} {hits}/{total} NOT SUBMITTED -> {kept.name}"
        line = (f"{arm:9} {hits}/{total} = {hits / total:.3f}  "
                f"conf {sum(scores) / total:.3f}  [{created['id'][:8]}]")
        print("SUBMITTED " + line, flush=True)
        return line
    detections: list = []
    frame_index = 0
    fps_clock, fps_count, fps_live = time.time(), 0, 0.0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        fps_count += 1
        if time.time() - fps_clock >= 1.0:
            fps_live = fps_count / (time.time() - fps_clock)
            fps_clock, fps_count = time.time(), 0

        if countdown_until and time.time() >= countdown_until:
            countdown_until = 0.0
            recording, recorded = True, []
            # One tag per recording: frames of one take are near-duplicates, so
            # training must hold out whole takes, never individual frames.
            session_tag, saved_count = time.strftime("%H%M%S"), 0
            take_files = []
            print(f"recording {arm}…", flush=True)

        if frame_index % max(args.detect_every, 1) == 0:
            if (recording and save_dir and arm == "baseline"
                    and len(recorded) % max(args.save_every, 1) == 0):
                saved_count += 1
                take_files.append(save_dir / f"rec-{session_tag}-{saved_count:03d}.jpg")
                cv2.imwrite(str(take_files[-1]), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                found = detector.predict(tensor.unsqueeze(0), threshold=args.threshold)[0]
            detections = [d for d in found if d.label == target_label]
            hit = bool(detections)
            rolling.append(hit)
            if recording:
                recorded.append((hit, max((d.score for d in detections), default=0.0)))
                if len(recorded) >= args.frames:
                    recording = False
                    history.append(submit())
                    recorded = []
                    if save_dir and arm == "baseline":
                        print(f"  saved {saved_count} frame(s) as rec-{session_tag}-*.jpg",
                              flush=True)

        colour = ARM_COLOR[arm]
        for detection in detections:
            x1, y1, x2, y2 = (int(v) for v in detection.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{detection.label} {detection.score:.2f}",
                        (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colour, 2, cv2.LINE_AA)

        rate = (sum(rolling) / len(rolling)) if rolling else 0.0
        lines = [
            (f"arm: {arm.upper()}", colour),
            (f"rolling detection ({len(rolling)}f): {rate * 100:5.1f}%", (232, 234, 240)),
            (f"{'DETECTED' if detections else 'no ' + target_label}"
             f"  score {max((d.score for d in detections), default=0.0):.2f}",
             (80, 210, 120) if not detections else (90, 120, 240)),
            (f"{width}x{height}  {fps_live:.1f} fps  detect 1/{args.detect_every}",
             (150, 150, 165)),
        ]
        if countdown_until:
            remaining = max(countdown_until - time.time(), 0.0)
            lines.append((f"GET IN POSITION  {remaining:4.1f}s", (60, 200, 235)))
            cv2.putText(frame, f"{remaining:.0f}", (width // 2 - 40, height // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, (60, 200, 235), 8, cv2.LINE_AA)
        elif recording:
            captured = sum(1 for hit, _ in recorded if hit)
            lines.append((f"REC {len(recorded)}/{args.frames}  {captured} hits - HOLD STILL",
                          (60, 60, 235)))
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (60, 60, 235), 10)
        for entry in history[-4:]:
            lines.append((entry, (150, 150, 165)))
        hud(frame, lines)

        cv2.imshow("VEIL live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in ARMS and not recording:
            arm = ARMS[key]
            rolling.clear()
            countdown_until = time.time() + args.countdown
            print(f"{arm}: {args.countdown:.0f}s to get into position…", flush=True)
        elif key == ord("x"):
            if countdown_until or recording:
                print("cancelled", flush=True)
            if recording:  # a stub take would count as a whole held-out take in training
                for path in take_files:
                    path.unlink(missing_ok=True)
                take_files = []
            countdown_until, recording, recorded = 0.0, False, []

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
