"""Encoding a ComfyUI IMAGE, on a machine with no ML stack installed.

These run the pure-standard-library path. That path exists so the encoder and
everything downstream of it can be tested at all without numpy, torch or
Pillow — which is also why this file can assert real PNG bytes rather than
trusting a mock.
"""

import struct
import unittest
import zlib

import harness  # noqa: F401

from imageir.imaging import DETAIL_MAX_SIDE, ImageError, encode_png, image_to_payload, to_rgb

# ComfyUI's IMAGE layout: [batch, height, width, channel], floats in 0..1.
TINY = [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]]


def png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


def png_pixels(data: bytes) -> bytes:
    """Decode the IDAT back to raw RGB, filter bytes removed."""
    width, height = png_size(data)
    offset = 8
    idat = b""
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        if tag == b"IDAT":
            idat += data[offset + 8 : offset + 8 + length]
        offset += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 3
    return b"".join(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)] for y in range(height))


class EncodeTests(unittest.TestCase):
    def test_the_output_is_a_real_png(self):
        data = encode_png(2, 1, bytes([255, 0, 0, 0, 255, 0]))
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(png_size(data), (2, 1))

    def test_the_pixels_survive_the_round_trip(self):
        pixels = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        self.assertEqual(png_pixels(encode_png(2, 2, pixels)), pixels)

    def test_the_chunk_crcs_are_correct(self):
        data = encode_png(1, 1, bytes([1, 2, 3]))
        offset = 8
        while offset < len(data):
            length = struct.unpack(">I", data[offset : offset + 4])[0]
            body = data[offset + 4 : offset + 8 + length]
            stored = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])[0]
            self.assertEqual(stored, zlib.crc32(body) & 0xFFFFFFFF)
            offset += 12 + length

    def test_a_wrong_sized_buffer_is_refused(self):
        with self.assertRaises(ImageError):
            encode_png(4, 4, b"\x00\x01\x02")


class ConversionTests(unittest.TestCase):
    def test_floats_become_bytes(self):
        width, height, rgb = to_rgb(TINY, max_side=64)
        self.assertEqual((width, height), (2, 2))
        self.assertEqual(rgb[:3], bytes([255, 0, 0]))

    def test_out_of_range_floats_are_clamped_not_wrapped(self):
        image = [[[[2.0, -1.0, 0.5]]]]
        self.assertEqual(to_rgb(image, max_side=8)[2], bytes([255, 0, 128]))

    def test_the_first_frame_of_a_batch_is_used(self):
        batch = [TINY[0], [[[[0.0, 0.0, 0.0]]]][0]]
        self.assertEqual(to_rgb(batch, max_side=64)[:2], (2, 2))

    def test_an_alpha_channel_is_dropped_rather_than_composited(self):
        # Flattening onto an invented background would add a visual fact.
        rgba = [[[[1.0, 0.0, 0.0, 0.0]]]]
        self.assertEqual(to_rgb(rgba, max_side=8)[2], bytes([255, 0, 0]))

    def test_too_few_channels_is_an_error(self):
        with self.assertRaises(ImageError):
            to_rgb([[[[0.5, 0.5]]]], max_side=8)

    def test_a_large_image_is_downscaled_to_the_limit(self):
        wide = [[[[0.5, 0.5, 0.5]] * 40 for _ in range(10)]]
        width, height, rgb = to_rgb(wide, max_side=8)
        self.assertEqual(width, 8)
        self.assertEqual(height, 2)
        self.assertEqual(len(rgb), width * height * 3)

    def test_a_small_image_is_left_at_its_own_size(self):
        self.assertEqual(to_rgb(TINY, max_side=512)[:2], (2, 2))


class PayloadTests(unittest.TestCase):
    def test_a_payload_carries_png_bytes_and_a_media_type(self):
        payload = image_to_payload(TINY, detail="balanced")
        self.assertEqual(payload.media_type, "image/png")
        self.assertEqual(payload.data[:8], b"\x89PNG\r\n\x1a\n")

    def test_the_data_url_is_what_an_openai_request_wants(self):
        self.assertTrue(image_to_payload(TINY).data_url.startswith("data:image/png;base64,"))

    def test_each_detail_level_has_its_own_resolution_budget(self):
        self.assertEqual(sorted(DETAIL_MAX_SIDE), ["balanced", "detailed", "fast"])
        self.assertLess(DETAIL_MAX_SIDE["fast"], DETAIL_MAX_SIDE["balanced"])
        self.assertLess(DETAIL_MAX_SIDE["balanced"], DETAIL_MAX_SIDE["detailed"])

    def test_detail_chooses_the_encoded_size(self):
        big = [[[[0.5, 0.5, 0.5]] * 2000 for _ in range(4)]]
        self.assertEqual(png_size(image_to_payload(big, detail="fast").data)[0], DETAIL_MAX_SIDE["fast"])

    def test_an_explicit_limit_overrides_the_detail_level(self):
        big = [[[[0.5, 0.5, 0.5]] * 2000 for _ in range(4)]]
        self.assertEqual(png_size(image_to_payload(big, max_side=32).data)[0], 32)

    def test_a_missing_image_is_a_clear_error(self):
        with self.assertRaises(ImageError) as caught:
            image_to_payload(None)
        self.assertIn("no IMAGE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
