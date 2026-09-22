#!/usr/bin/env python3
"""Export a pattern as a PDF printed at an exact physical size.

    python scripts/print_artwork.py --api-key "$VEIL_API_KEY" \
        --artwork <artifact-id> --width-cm 21 --out var/test-sheet.pdf

Physical size is part of the experiment: a pattern shown smaller than the
finished print does not stand in for a garment. This prints at a measured
width with a centimetre scale bar, so the size can be verified on paper
rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import io
import urllib.request

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def fetch(url: str, api_key: str) -> bytes:
    request = urllib.request.Request(url, headers={"X-API-Key": api_key})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000/api/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--artwork", help="artifact id of production artwork")
    parser.add_argument("--pattern", help="pattern id, if not using --artwork")
    parser.add_argument("--file", type=Path, help="a local PNG instead of an API artifact")
    parser.add_argument("--title", default="", help="line printed under the sheet")
    parser.add_argument("--width-cm", type=float, default=21.0)
    parser.add_argument("--out", type=Path, default=Path("var/veil-test-sheet.pdf"))
    parser.add_argument("--dpi", type=int, default=300,
                        help="target print resolution. Each production unit is "
                             "expanded to whole pixels with nearest-neighbour, so "
                             "stitch blocks print crisp instead of blurred. This "
                             "does not add detail - detail comes from pattern_size.")
    args = parser.parse_args()

    if not (args.artwork or args.pattern or args.file):
        sys.exit("pass --artwork, --pattern or --file")
    if args.file:
        data = args.file.read_bytes()
    else:
        path = (f"/artifacts/{args.artwork}/download" if args.artwork
                else f"/patterns/{args.pattern}/image")
        data = fetch(f"{args.api}{path}", args.api_key)

    source = Image.open(io.BytesIO(data)).convert("RGB")
    pixel_w, pixel_h = source.size
    width = args.width_cm * cm
    height = width * pixel_h / pixel_w

    page_w, page_h = A4
    if height > page_h - 6 * cm:
        height = page_h - 6 * cm
        width = height * pixel_w / pixel_h
        print(f"scaled to fit A4: {width / cm:.1f} cm wide")

    # Expand each production unit to whole pixels so the printer receives hard
    # edges. Smoothing a knit chart is wrong: each block is one stitch.
    target_px = max(int(width / cm * args.dpi / 2.54), pixel_w)
    factor = max(round(target_px / pixel_w), 1)
    crisp = source.resize((pixel_w * factor, pixel_h * factor), Image.NEAREST)
    buffer = io.BytesIO()
    crisp.save(buffer, format="PNG")
    buffer.seek(0)
    image = ImageReader(buffer)
    effective_dpi = crisp.width / (width / cm / 2.54)
    print(f"upscaled x{factor} -> {crisp.width}x{crisp.height} px "
          f"({effective_dpi:.0f} dpi, nearest-neighbour)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(args.out), pagesize=A4)
    x = (page_w - width) / 2
    y = page_h - height - 3 * cm
    # Nearest-neighbour: the blocks are stitches, not a photograph.
    pdf.drawImage(image, x, y, width, height, preserveAspectRatio=False)
    pdf.rect(x, y, width, height)

    # A scale bar, so the printed size can be checked with a ruler instead of
    # trusting the printer's "fit to page".
    bar_y = y - 1.2 * cm
    pdf.setFont("Helvetica", 8)
    pdf.line(x, bar_y, x + 10 * cm, bar_y)
    for i in range(11):
        pdf.line(x + i * cm, bar_y, x + i * cm, bar_y - (0.25 if i % 5 else 0.45) * cm)
    pdf.drawString(x, bar_y - 0.9 * cm,
                   "10 cm - measure this with a ruler. If it is not 10 cm, "
                   "reprint at 100% scale (no 'fit to page').")
    pdf.drawString(x, bar_y - 1.5 * cm,
                   f"{args.title + ' - ' if args.title else ''}"
                   f"{args.artwork or args.pattern or args.file.name} printed {width / cm:.1f} x "
                   f"{height / cm:.1f} cm from {pixel_w}x{pixel_h} production units.")
    pdf.drawString(x, bar_y - 2.1 * cm,
                   f"To stand in for a 30 cm garment print at 3.0 m, view this "
                   f"sheet from {3.0 * (width / cm) / 30:.1f} m.")
    pdf.showPage()
    pdf.save()
    print(f"wrote {args.out}  ({width / cm:.1f} x {height / cm:.1f} cm)")
    print(f"view distance for a 30 cm print at 3 m: {3.0 * (width / cm) / 30:.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
