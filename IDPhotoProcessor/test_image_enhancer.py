import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ImageEnhancer import ImageEnhancer


def test_auto_enhance_brightens_face_without_changing_blue_background():
    image = np.full((100, 100, 3), (210, 110, 40), dtype=np.uint8)
    image[30:70, 35:65] = (55, 55, 55)

    result = ImageEnhancer().auto_enhance(image, (35, 30, 30, 40))

    assert result[50, 50].mean() > image[50, 50].mean()
    assert np.array_equal(result[5, 5], image[5, 5])


def test_auto_enhance_without_face_does_not_adjust_global_brightness():
    image = np.full((60, 60, 3), (210, 110, 40), dtype=np.uint8)

    result = ImageEnhancer().auto_enhance(image)

    assert abs(float(result.mean()) - float(image.mean())) < 2


def test_auto_enhance_without_face_keeps_small_photo_detail_unchanged():
    base = np.full((60, 60, 3), (150, 130, 110), dtype=np.uint8)
    noise = np.random.default_rng(1).integers(-10, 11, (60, 60, 1), dtype=np.int16)
    image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    result = ImageEnhancer().auto_enhance(image)

    mean_change = float(np.abs(result.astype(np.int16) - image.astype(np.int16)).mean())
    assert mean_change < 1


def test_face_brightness_adjustment_fades_out_at_the_expanded_region_edge():
    image = np.full((100, 100, 3), (210, 110, 40), dtype=np.uint8)
    image[30:70, 35:65] = (55, 55, 55)

    result = ImageEnhancer()._gently_adjust_face_brightness(image, (35, 30, 30, 40))

    # The expanded region ends at y=82. Its final in-region row must already
    # be close to the untouched row below, preventing a visible horizontal seam.
    final_row_delta = float((result[81].astype(np.int16) - image[81].astype(np.int16)).mean())
    outside_row_delta = float((result[82].astype(np.int16) - image[82].astype(np.int16)).mean())
    assert abs(final_row_delta - outside_row_delta) <= 3


def test_portrait_denoise_smooths_face_without_changing_background():
    image = np.full((100, 100, 3), (210, 110, 40), dtype=np.uint8)
    for row in range(30, 70):
        for column in range(35, 65):
            image[row, column] = (60 + (row + column) % 40,) * 3

    result = ImageEnhancer().denoise_portrait_region(image, (35, 30, 30, 40))

    assert result[35:65, 40:60].std() < image[35:65, 40:60].std()
    assert np.array_equal(result[5, 5], image[5, 5])


def test_portrait_denoise_without_face_keeps_image_unchanged():
    image = np.full((60, 60, 3), (210, 110, 40), dtype=np.uint8)

    result = ImageEnhancer().denoise_portrait_region(image, None)

    assert np.array_equal(result, image)
