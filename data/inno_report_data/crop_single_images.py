#!/usr/bin/env python3
"""Batch-remove invalid black borders from endoscopy images.

The source directory is read-only. The relative directory structure is copied
to a separate output directory. Images for which no valid crop is detected are
copied byte-for-byte instead of being re-encoded.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_INPUT_DIR = Path("/media/maxlin/SATA/ReportData/test_single_images")
DEFAULT_OUTPUT_DIR = Path("/media/maxlin/SATA/ReportData/test_single_images_cropped")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def crop_invalid_region(
    image: np.ndarray, ignore_square: bool = True
) -> Tuple[np.ndarray, Optional[Sequence[int]]]:
    """Crop the valid endoscopy region and return ROI as ``[x, y, width, height]``."""

    height, width = image.shape[:2]
    if ignore_square and 0.8 <= float(height) / width <= 1.2:
        return image, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_size = max(3, min(height, width) // 100)
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.medianBlur(gray, blur_size)

    channels = image.astype(np.int32)
    gap = 5
    mask = (
        (np.abs(channels[:, :, 0] - channels[:, :, 1]) > gap)
        | (np.abs(channels[:, :, 1] - channels[:, :, 2]) > gap)
        | (np.abs(channels[:, :, 0] - channels[:, :, 2]) > gap)
    ).astype(np.uint8)

    element_size = max(3, min(height, width) // 200)
    if element_size % 2 == 0:
        element_size += 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (element_size, element_size))
    mask = cv2.erode(cv2.dilate(mask, element), element)
    if not mask.any():
        mask = np.ones_like(mask)

    enhanced = cv2.bitwise_and(blurred, blurred, mask=mask)
    threshold = max(10, int(np.percentile(enhanced[mask > 0], 5)))
    binary = (enhanced > threshold).astype(np.uint8) * 255
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return image, None

    x, y, crop_width, crop_height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    valid_scope = (
        0.7 < float(crop_width) / max(crop_height, 1) < 1.5
        and min(crop_width, crop_height) > min(height, width) / 2.0
    )
    if not valid_scope:
        return image, None

    return image[y : y + crop_height, x : x + crop_width], [x, y, crop_width, crop_height]


def discover_images(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def write_image(path: Path, image: np.ndarray) -> None:
    parameters = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif path.suffix.lower() == ".png":
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    if not cv2.imwrite(str(path), image, parameters):
        raise IOError("OpenCV failed to write image: {}".format(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove black borders from a directory of endoscopy images.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output image. By default existing files are skipped.",
    )
    parser.add_argument(
        "--no-ignore-square",
        action="store_true",
        help="Also attempt cropping images whose aspect ratio is between 0.8 and 1.2.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError("Input directory does not exist: {}".format(input_dir))
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must not be the input directory or one of its subdirectories.")

    images = list(discover_images(input_dir))
    if not images:
        raise RuntimeError("No supported images found under {}".format(input_dir))

    cropped_count = 0
    unchanged_count = 0
    skipped_count = 0
    failures = []
    ignore_square = not args.no_ignore_square

    print("Input: {}".format(input_dir))
    print("Output: {}".format(output_dir))
    print("Images: {:,}; ignore_square={}".format(len(images), ignore_square))

    for index, source in enumerate(images, start=1):
        relative_path = source.relative_to(input_dir)
        destination = output_dir / relative_path
        if destination.exists() and not args.overwrite:
            skipped_count += 1
            print("[{}/{}] SKIP {}".format(index, len(images), relative_path))
            continue

        try:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV cannot decode image")

            original_height, original_width = image.shape[:2]
            cropped, roi = crop_invalid_region(image, ignore_square=ignore_square)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if roi is None:
                shutil.copy2(str(source), str(destination))
                unchanged_count += 1
                result = "UNCHANGED {}x{}".format(original_width, original_height)
            else:
                write_image(destination, cropped)
                cropped_count += 1
                result = "CROPPED {}x{} -> {}x{} roi={}".format(
                    original_width,
                    original_height,
                    cropped.shape[1],
                    cropped.shape[0],
                    list(roi),
                )

            print("[{}/{}] {}: {}".format(index, len(images), result, relative_path))
        except Exception as error:  # continue so one corrupt image does not discard the whole batch
            failures.append((relative_path, str(error)))
            print("[{}/{}] ERROR {}: {}".format(index, len(images), relative_path, error), file=sys.stderr)

    print(
        "Summary: total={:,}, cropped={:,}, unchanged={:,}, skipped={:,}, failed={:,}".format(
            len(images), cropped_count, unchanged_count, skipped_count, len(failures)
        )
    )
    if failures:
        print("Failed images:", file=sys.stderr)
        for path, error in failures:
            print("  {}: {}".format(path, error), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
