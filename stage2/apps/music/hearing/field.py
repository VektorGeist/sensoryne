"""stage2/apps/music/hearing/field.py - the place-by-time field.

Renders a reading's tonotopic places over time as an image,
and reads that image back into numbers: peak band, motion,
and brightness.  The reading is taken from the image array,
not from the frames directly.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .cochlea import PLACE_COUNT, PLACE_FREQS

__all__ = [
    "BAND_PLACES",
    "place_time_field",
    "normalize_field",
    "render_field_image",
    "read_field_image",
    "field_reading",
    "save_field_image",
]

# Places grouped into low, mid, and high bands by index.
BAND_PLACES = {
    "low": list(range(0, 11)),
    "mid": list(range(11, 22)),
    "high": list(range(22, PLACE_COUNT)),
}

_EPS = 1e-12


def place_time_field(frames):
    """Return the place-by-time field as a (places, frames) array.

    Each column is one frame's place vector.  An empty reading
    yields a field with no columns.
    """
    if not frames:
        return np.zeros((PLACE_COUNT, 0), dtype=np.float64)
    field = np.asarray([frame["place"] for frame in frames], dtype=np.float64).T
    if field.shape[0] != PLACE_COUNT:
        raise ValueError(
            "frame place vectors must have %d places, got shape %r"
            % (PLACE_COUNT, field.shape)
        )
    return field


def normalize_field(field):
    """Scale a field so its peak is 1.0.  A silent field stays zero."""
    field = np.asarray(field, dtype=np.float64)
    if field.size == 0:
        return field
    peak = float(np.max(field))
    if peak <= _EPS:
        return np.zeros_like(field)
    return field / peak


def render_field_image(field):
    """Render a place-by-time field as a grayscale image.

    Rows are places, low frequencies at the bottom; columns are
    time.  Returns a PIL Image in mode L.
    """
    normalized = normalize_field(field)
    if normalized.size == 0 or normalized.shape[1] == 0:
        array = np.zeros((PLACE_COUNT, 1), dtype=np.uint8)
        return Image.fromarray(array, mode="L")
    # Flip vertically so low places sit at the bottom of the image.
    flipped = np.flipud(normalized)
    array = np.clip(flipped * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def read_field_image(image):
    """Read a rendered field image back into numbers.

    ``image`` may be a PIL Image or a 2-D array.  Returns the
    peak band and its place frequency, the motion across time,
    and the brightness, all computed from the image pixels.
    """
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    else:
        array = np.asarray(image, dtype=np.float64)
        if not array.size:
            array = np.zeros((PLACE_COUNT, 1), dtype=np.float64)
        elif float(np.max(array)) > 1.0:
            array = array / 255.0
    if array.ndim != 2:
        raise ValueError("field image must be two-dimensional")

    # The image is stored with low places at the bottom; flip it
    # back so row 0 is the lowest place.
    field = np.flipud(array)
    if field.shape[0] != PLACE_COUNT and field.size:
        # Resample rows to the place count by repeating or
        # truncating, so a resized image still reads.
        indices = np.linspace(0, field.shape[0] - 1, PLACE_COUNT).astype(int)
        field = field[indices]

    row_means = np.mean(field, axis=1) if field.size else np.zeros(PLACE_COUNT)
    peak_place = int(np.argmax(row_means)) if row_means.size else 0

    band_means = {}
    for band, places in BAND_PLACES.items():
        band_means[band] = float(np.mean(row_means[places])) if places else 0.0
    peak_band = max(band_means, key=lambda band: band_means[band]) if band_means else None

    if field.shape[1] > 1:
        motion = float(np.mean(np.abs(np.diff(field, axis=1))))
    else:
        motion = 0.0

    overall = float(np.mean(field)) if field.size else 0.0
    high_places = BAND_PLACES["high"]
    high_mean = float(np.mean(row_means[high_places])) if high_places else 0.0
    brightness = high_mean / overall if overall > _EPS else 0.0

    return {
        "peak_band": peak_band,
        "peak_place": peak_place,
        "peak_place_hz": float(PLACE_FREQS[peak_place]),
        "band_means": band_means,
        "motion": motion,
        "brightness": float(brightness),
        "overall": overall,
    }


def field_reading(frames):
    """Render the place-by-time field of a reading as an image
    and read that image back.  Returns the reading numbers and
    the image array."""
    field = place_time_field(frames)
    image = render_field_image(field)
    reading = read_field_image(image)
    reading["image"] = np.asarray(image, dtype=np.float64) / 255.0
    return reading


def save_field_image(frames, path):
    """Render the place-by-time field of a reading to a PNG file."""
    image = render_field_image(place_time_field(frames))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(target))
    return str(target)
