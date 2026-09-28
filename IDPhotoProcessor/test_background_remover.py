import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from BackgroundRemover import BackgroundRemover


def test_polish_mask_keeps_humanseg_transition_alpha():
    mask = np.zeros((21, 21), dtype=np.uint8)
    mask[4:17, 4:17] = 255
    mask[4, 5:16] = 128
    mask[16, 5:16] = 128

    result = BackgroundRemover()._polish_mask(mask)

    assert 0 < int(result[4, 10]) < 255


def test_replace_background_discards_uncertain_light_edge_pixels():
    source_background = np.array([255, 255, 255], dtype=np.float32)
    foreground = np.array([30, 60, 100], dtype=np.float32)
    target_background = np.array([219, 142, 67], dtype=np.uint8)

    mask = np.zeros((17, 17), dtype=np.uint8)
    mask[2:15, 2:15] = 128
    mask[3:14, 3:14] = 220
    mask[4:13, 4:13] = 255
    alpha = mask.astype(np.float32) / 255.0
    image = (foreground * alpha[..., np.newaxis] + source_background * (1.0 - alpha[..., np.newaxis])).astype(np.uint8)

    result = BackgroundRemover().replace_background(image, mask, "blue", feather_radius=5)

    # The original partial-alpha edge contains white source-background pixels.
    # It must become the requested background rather than a white halo.
    assert np.abs(result[2, 8].astype(np.int16) - target_background.astype(np.int16)).max() <= 5
    assert np.abs(result[8, 8].astype(np.int16) - foreground.astype(np.int16)).max() <= 1
