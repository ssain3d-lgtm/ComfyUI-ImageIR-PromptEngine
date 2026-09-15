"""Turn a ComfyUI IMAGE into bytes a vision API will accept.

ComfyUI hands around a float tensor shaped [batch, height, width, channel] with
values in 0..1. Vision endpoints want a base64 image with a media type. This
module is the whole of that conversion, and it is kept out of the backend
package because it is the one place that touches ComfyUI's data model.

Two implementations of the same thing, chosen at call time: numpy and Pillow
when they are importable — which inside ComfyUI they always are — and a pure
standard-library path when they are not. The fallback is not defensive
programming for its own sake; it is what lets the encoder be tested, and the
analyzer be exercised end to end, on a machine with no ML stack installed.
"""

from __future__ import annotations

import struct
import zlib

from .backend.base import ImagePayload

# How much resolution each analysis depth is worth. More pixels cost latency and
# tokens on every request, and past roughly a thousand pixels on the long side
# the marginal detail a describing model recovers falls off sharply.
DETAIL_MAX_SIDE = {
    "fast": 512,
    "balanced": 768,
    "detailed": 1152,
}


class ImageError(ValueError):
    """The IMAGE input was not something that can be encoded."""


def encode_png(width: int, height: int, rgb: bytes) -> bytes:
    """A minimal, correct PNG from packed 8-bit RGB rows.

    Standard library only. PNG is a handful of length-tagged, CRC'd chunks
    around a zlib stream, which is far less code than carrying an image library
    into a package whose runtime dependency list is deliberately empty.
    """
    if len(rgb) != width * height * 3:
        raise ImageError(f"expected {width * height * 3} bytes of RGB, got {len(rgb)}")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    # Filter byte 0 (None) in front of every scanline: the rows are already
    # small after downscaling, and skipping the filter search keeps this fast.
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride : (y + 1) * stride] for y in range(height))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _scaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_side or longest == 0:
        return width, height
    scale = max_side / longest
    return max(1, int(width * scale)), max(1, int(height * scale))


def _first_frame(image):
    """The first still of whatever ComfyUI passed.

    A batch is the normal case — one image is a batch of one — and describing
    more than the first frame is a different feature than this one.
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    array = image
    if hasattr(array, "detach"):  # a torch tensor
        array = array.detach().cpu()
        if np is not None:
            array = array.numpy()
        else:
            array = array.tolist()

    # Select a nested-list batch before numpy conversion: later frames may have
    # different dimensions, and only the first frame is part of this API.
    if isinstance(array, (list, tuple)) and array and _depth(array) == 4:
        array = array[0]

    if np is not None:
        array = np.asarray(array)
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3:
            raise ImageError(f"expected an IMAGE shaped [batch, height, width, channel], got shape {array.shape}")
        return array

    # Pure-Python fallback: nested lists.
    while isinstance(array, (list, tuple)) and array and isinstance(array[0], (list, tuple)) and _depth(array) == 4:
        array = array[0]
    if _depth(array) != 3:
        raise ImageError("expected an IMAGE shaped [batch, height, width, channel]")
    return array


def _depth(value) -> int:
    depth = 0
    while isinstance(value, (list, tuple)):
        depth += 1
        if not value:
            break
        value = value[0]
    return depth


def to_rgb(image, max_side: int = 768) -> tuple[int, int, bytes]:
    """(width, height, packed RGB) for the first frame, downscaled to fit.

    Nearest-neighbour sampling: the destination is a description, not a print,
    and a describing model does not reward the extra cost of a better kernel.
    """
    frame = _first_frame(image)

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        height, width = frame.shape[0], frame.shape[1]
        new_width, new_height = _scaled_size(width, height, max_side)
        if (new_width, new_height) != (width, height):
            rows = (np.arange(new_height) * height // new_height).clip(0, height - 1)
            cols = (np.arange(new_width) * width // new_width).clip(0, width - 1)
            frame = frame[rows][:, cols]
        channels = frame.shape[2]
        if channels < 3:
            raise ImageError(f"expected at least 3 channels, got {channels}")
        # Alpha is dropped rather than composited: these models read the colour
        # image, and inventing a background to flatten onto would be a visual
        # fact the source never carried.
        frame = frame[:, :, :3]
        if frame.dtype.kind == "f":
            frame = np.floor(np.clip(frame, 0.0, 1.0) * 255.0 + 0.5)
        data = frame.astype(np.uint8).tobytes()
        return new_width, new_height, data

    height = len(frame)
    width = len(frame[0]) if height else 0
    # The same guard the numpy path applies. Without it a single-channel input —
    # a MASK wired into an IMAGE socket, which happens — surfaces as an
    # IndexError from deep inside the resampling loop.
    if height and width and len(frame[0][0]) < 3:
        raise ImageError(f"expected at least 3 channels, got {len(frame[0][0])}")
    new_width, new_height = _scaled_size(width, height, max_side)
    out = bytearray()
    for y in range(new_height):
        source_row = frame[(y * height) // new_height if new_height else 0]
        for x in range(new_width):
            pixel = source_row[(x * width) // new_width if new_width else 0]
            for channel in range(3):
                value = pixel[channel]
                if isinstance(value, float):
                    value = int(max(0.0, min(1.0, value)) * 255.0 + 0.5)
                out.append(int(max(0, min(255, value))))
    return new_width, new_height, bytes(out)


def image_to_payload(image, *, detail: str = "balanced", max_side: int | None = None) -> ImagePayload:
    """A ComfyUI IMAGE as an ``ImagePayload`` ready for any backend."""
    if image is None:
        raise ImageError("no IMAGE was connected")
    limit = max_side if max_side is not None else DETAIL_MAX_SIDE.get(detail, 768)
    width, height, rgb = to_rgb(image, max_side=limit)
    if width == 0 or height == 0:
        raise ImageError("the IMAGE has no pixels")
    return ImagePayload(data=encode_png(width, height, rgb), media_type="image/png")
