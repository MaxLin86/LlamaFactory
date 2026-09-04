#!/usr/bin/env python3
"""Draw normalized xyxy bounding boxes on one image for visual verification."""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]
BOX_PATTERN = re.compile(r"[\[【]\s*([^\]】]+?)\s*[\]】]")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
COLORS = (
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
)


def parse_bbox_text(text: str) -> List[BBox]:
    """Extract ``[x_min, y_min, x_max, y_max]`` groups from model output."""

    boxes: List[BBox] = []
    for content in BOX_PATTERN.findall(text):
        numbers = NUMBER_PATTERN.findall(content)
        if len(numbers) != 4:
            continue
        boxes.append(tuple(float(value) for value in numbers))  # type: ignore[arg-type]

    # Also accept a plain input such as: 0.1, 0.2, 0.8, 0.9
    if not boxes:
        numbers = NUMBER_PATTERN.findall(text)
        if len(numbers) == 4:
            boxes.append(tuple(float(value) for value in numbers))  # type: ignore[arg-type]

    return boxes


def validate_bbox(box: Sequence[float], index: int) -> BBox:
    if len(box) != 4:
        raise ValueError("Box {} must contain exactly four values.".format(index))

    x_min, y_min, x_max, y_max = (float(value) for value in box)
    if not all(np.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        raise ValueError("Box {} contains a non-finite value: {}".format(index, list(box)))
    if not all(0.0 <= value <= 1.0 for value in (x_min, y_min, x_max, y_max)):
        raise ValueError("Box {} is outside the normalized range [0, 1]: {}".format(index, list(box)))
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            "Box {} must satisfy x_min < x_max and y_min < y_max: {}".format(index, list(box))
        )
    return x_min, y_min, x_max, y_max


def normalized_to_pixels(box: BBox, width: int, height: int) -> Tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = box
    return (
        round(x_min * (width - 1)),
        round(y_min * (height - 1)),
        round(x_max * (width - 1)),
        round(y_max * (height - 1)),
    )


def draw_boxes(image: np.ndarray, boxes: Sequence[BBox]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    result = image.copy()
    height, width = result.shape[:2]
    thickness = max(2, round(max(width, height) / 500))
    font_scale = max(0.55, max(width, height) / 1400.0)
    pixel_boxes: List[Tuple[int, int, int, int]] = []

    for index, box in enumerate(boxes, start=1):
        pixel_box = normalized_to_pixels(box, width, height)
        pixel_boxes.append(pixel_box)
        x_min, y_min, x_max, y_max = pixel_box
        color = COLORS[(index - 1) % len(COLORS)]
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, thickness, cv2.LINE_AA)

        label = "BOX {}".format(index)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_top = max(0, y_min - text_height - baseline - 6)
        label_bottom = min(height - 1, label_top + text_height + baseline + 6)
        label_right = min(width - 1, x_min + text_width + 10)
        cv2.rectangle(result, (x_min, label_top), (label_right, label_bottom), color, cv2.FILLED)
        cv2.putText(
            result,
            label,
            (x_min + 5, label_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return result, pixel_boxes


def resize_for_display(image: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / max(width, height))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters: List[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif path.suffix.lower() == ".png":
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(str(path), image, parameters):
        raise IOError("OpenCV failed to write image: {}".format(path))


def read_interactive_text() -> str:
    print(
        "请粘贴包含坐标的模型回答，支持单框或多框；输入空行后开始绘制：",
        file=sys.stderr,
    )
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw normalized xyxy [x_min, y_min, x_max, y_max] boxes on one image."
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--bbox",
        action="append",
        nargs=4,
        type=float,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Normalized xyxy box. Repeat --bbox to draw multiple boxes.",
    )
    parser.add_argument(
        "--text",
        help="Model response containing one or more bracketed coordinate groups.",
    )
    parser.add_argument("--output", type=Path, help="Optional path for the rendered full-resolution image.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the preview window.")
    parser.add_argument("--display-max-edge", type=int, default=1080, help="Maximum preview-window image edge.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError("Input image does not exist: {}".format(image_path))
    if args.display_max_edge <= 0:
        raise ValueError("--display-max-edge must be greater than zero.")

    raw_boxes: List[Sequence[float]] = list(args.bbox or [])
    if args.text:
        raw_boxes.extend(parse_bbox_text(args.text))
    if not raw_boxes:
        raw_boxes.extend(parse_bbox_text(read_interactive_text()))
    if not raw_boxes:
        raise ValueError("No valid four-value coordinate group was found.")

    boxes = [validate_bbox(box, index) for index, box in enumerate(raw_boxes, start=1)]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV cannot decode image: {}".format(image_path))

    rendered, pixel_boxes = draw_boxes(image, boxes)
    height, width = image.shape[:2]
    print("Image: {} ({}x{})".format(image_path, width, height))
    for index, (normalized, pixels) in enumerate(zip(boxes, pixel_boxes), start=1):
        print("BOX {}: normalized={} -> pixels={}".format(index, list(normalized), list(pixels)))

    output_path: Optional[Path] = args.output
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        write_image(output_path, rendered)
        print("Saved: {}".format(output_path))

    if not args.no_show:
        preview = resize_for_display(rendered, args.display_max_edge)
        cv2.imshow("Normalized XYXY Verification - press any key to close", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif output_path is None:
        print("Nothing was displayed or saved: remove --no-show or provide --output.", file=sys.stderr)
        return 2

    return 0


# 使用方法：
#
# 1. 交互输入坐标（推荐）：
#    python data/inno_report_data/plot_single_images.py '/完整路径/image.jpg'
#
#    然后粘贴模型回答，例如：
#    有2个目标：
#    目标1：[0.1, 0.2, 0.5, 0.6]
#    目标2：[0.55, 0.1, 0.9, 0.4]
#    最后再输入一个空行，即可弹出绘制结果。
#
# 2. 直接输入一个坐标框：
#    python data/inno_report_data/plot_single_images.py '/完整路径/image.jpg' \
#        --bbox 0.1 0.2 0.5 0.6
#
# 3. 直接输入多个坐标框：
#    python data/inno_report_data/plot_single_images.py '/完整路径/image.jpg' \
#        --bbox 0.1 0.2 0.5 0.6 \
#        --bbox 0.55 0.1 0.9 0.4
#
# 4. 显示并保存全分辨率绘制结果：
#    python data/inno_report_data/plot_single_images.py '/完整路径/image.jpg' \
#        --bbox 0.1 0.2 0.5 0.6 \
#        --output '/完整路径/result.jpg'
#
# 坐标格式：归一化 xyxy [x_min, y_min, x_max, y_max]，各值范围为 0～1。
# 原始图片不会被修改；只有指定 --output 时才会另外保存绘制结果。
if __name__ == "__main__":
    raise SystemExit(main())
