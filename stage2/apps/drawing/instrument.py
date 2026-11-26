"""stage2/apps/drawing/instrument.py - raster analyzer.

Analyzes the rendered PNG using Pillow and numpy.  Reports
stated measurements only; it does not say what the picture
shows.
"""

import numpy as np
from PIL import Image

METHOD = (
    "image opened with Pillow and converted to RGB; background "
    "taken as the most common color; dominant colors are the "
    "most frequent colors by pixel count; ink coverage is the "
    "share of pixels differing from the background, overall and "
    "per quadrant (top-left, top-right, bottom-left, "
    "bottom-right); bounding box is the smallest rectangle "
    "containing all non-background pixels"
)


def _dominant_colors(pixels, max_colors=8):
    colors, counts = np.unique(
        pixels.reshape(-1, pixels.shape[-1]), axis=0, return_counts=True
    )
    total = int(counts.sum()) if counts.size else 0
    order = np.argsort(-counts)
    result = []
    for index in order[:max_colors]:
        color = [int(c) for c in colors[index]]
        count = int(counts[index])
        result.append(
            {
                "color": color,
                "hex": "#%02x%02x%02x" % tuple(color),
                "count": count,
                "share": (count / total) if total else 0.0,
            }
        )
    return result


def analyze_image(path):
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = np.asarray(rgb)

    colors = _dominant_colors(pixels)
    background = colors[0]["color"] if colors else [255, 255, 255]
    background_array = np.array(background, dtype=np.int16)
    non_background = np.any(
        np.abs(pixels.astype(np.int16) - background_array) > 0, axis=2
    )
    total_pixels = width * height if width and height else 0
    ink_count = int(np.count_nonzero(non_background))
    ink_coverage = (ink_count / total_pixels) if total_pixels else 0.0

    mid_x = width // 2
    mid_y = height // 2
    quadrants = {}
    regions = {
        "top_left": non_background[:mid_y, :mid_x],
        "top_right": non_background[:mid_y, mid_x:],
        "bottom_left": non_background[mid_y:, :mid_x],
        "bottom_right": non_background[mid_y:, mid_x:],
    }
    for name, region in regions.items():
        region_total = int(region.size)
        quadrants[name] = {
            "ink_count": int(np.count_nonzero(region)),
            "coverage": (
                int(np.count_nonzero(region)) / region_total
                if region_total
                else 0.0
            ),
        }

    if ink_count:
        rows = np.where(np.any(non_background, axis=1))[0]
        cols = np.where(np.any(non_background, axis=0))[0]
        bounding_box = {
            "min_x": int(cols[0]),
            "min_y": int(rows[0]),
            "max_x": int(cols[-1]),
            "max_y": int(rows[-1]),
            "width": int(cols[-1] - cols[0] + 1),
            "height": int(rows[-1] - rows[0] + 1),
        }
    else:
        bounding_box = {
            "min_x": None,
            "min_y": None,
            "max_x": None,
            "max_y": None,
            "width": 0,
            "height": 0,
        }

    if colors:
        summary = (
            "Image %d by %d pixels. Dominant color %s with share %.4f. "
            "Ink coverage %.4f overall. Non-background bounding box "
            "x %s to %s, y %s to %s. It does not say what the picture shows."
            % (
                width,
                height,
                colors[0]["hex"],
                colors[0]["share"],
                ink_coverage,
                bounding_box["min_x"],
                bounding_box["max_x"],
                bounding_box["min_y"],
                bounding_box["max_y"],
            )
        )
    else:
        summary = "Image %d by %d pixels with no colors recorded." % (
            width,
            height,
        )

    measurements = {
        "instrument": "raster analyzer",
        "method": METHOD,
        "width": int(width),
        "height": int(height),
        "dimensions": [int(width), int(height)],
        "background_color": background,
        "dominant_colors": colors,
        "ink_count": ink_count,
        "ink_coverage": ink_coverage,
        "quadrants": quadrants,
        "bounding_box": bounding_box,
    }
    full = {
        "instrument": "raster analyzer",
        "method": METHOD,
        "render_path": str(path),
        "measurements": measurements,
        "interpretation": None,
    }
    return {
        "summary": summary,
        "measurements": measurements,
        "full": full,
        "render_path": str(path),
        "instrument": "raster analyzer",
        "interpretation": None,
    }


def perceive_image(path):
    return analyze_image(path)
