# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for plotting normalized xyxy boxes returned by WebChat."""

import re
from typing import Any

from PIL import Image, ImageDraw, ImageFont


_BOX_PATTERN = re.compile(r"[\[【]\s*([^\]】]+?)\s*[\]】]")
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_COLORS = ("#00ff00", "#ff9900", "#ff00ff", "#00ffff", "#ff3333", "#3388ff")
_TEXT = {
    "en": {
        "no_image": "Please upload an image first.",
        "no_answer": "No assistant response is available.",
        "no_box": "No valid normalized xyxy box was found in the latest response.",
        "summary": "Found {count} box(es). Image size: {width} x {height}.",
    },
    "ru": {
        "no_image": "Сначала загрузите изображение.",
        "no_answer": "Ответ ассистента отсутствует.",
        "no_box": "В последнем ответе не найден корректный нормализованный xyxy-бокс.",
        "summary": "Найдено боксов: {count}. Размер изображения: {width} x {height}.",
    },
    "zh": {
        "no_image": "请先上传图像。",
        "no_answer": "当前还没有模型回答。",
        "no_box": "最新一条模型回答中未找到有效的归一化 xyxy 坐标。",
        "summary": "共找到 {count} 个目标框；图像分辨率：{width} x {height}。",
    },
    "ko": {
        "no_image": "먼저 이미지를 업로드하세요.",
        "no_answer": "어시스턴트 응답이 없습니다.",
        "no_box": "최신 응답에서 유효한 정규화 xyxy 상자를 찾지 못했습니다.",
        "summary": "{count}개의 상자를 찾았습니다. 이미지 크기: {width} x {height}.",
    },
    "ja": {
        "no_image": "先に画像をアップロードしてください。",
        "no_answer": "アシスタントの回答がありません。",
        "no_box": "最新の回答に有効な正規化 xyxy 座標が見つかりません。",
        "summary": "{count}個のボックスを検出しました。画像サイズ: {width} x {height}。",
    },
}


def _local_text(lang: str) -> dict[str, str]:
    return _TEXT.get(lang, _TEXT["en"])


def _latest_assistant_text(messages: list[dict[str, Any]] | None) -> str | None:
    for message in reversed(messages or []):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _parse_normalized_boxes(text: str) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for content in _BOX_PATTERN.findall(text):
        values = _NUMBER_PATTERN.findall(content)
        if len(values) != 4:
            continue

        x_min, y_min, x_max, y_max = (float(value) for value in values)
        if (
            0.0 <= x_min < x_max <= 1.0
            and 0.0 <= y_min < y_max <= 1.0
        ):
            boxes.append((x_min, y_min, x_max, y_max))

    return boxes


def plot_latest_response_boxes(
    image: Image.Image | None,
    messages: list[dict[str, Any]] | None,
    lang: str,
) -> tuple[Image.Image | None, str]:
    """Draw boxes from the latest assistant response on a copy of ``image``."""
    text = _local_text(lang)
    if image is None:
        return None, text["no_image"]

    response = _latest_assistant_text(messages)
    if response is None:
        return None, text["no_answer"]

    boxes = _parse_normalized_boxes(response)
    if not boxes:
        return None, text["no_box"]

    rendered = image.convert("RGB").copy()
    width, height = rendered.size
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    line_width = max(2, round(max(width, height) / 500))
    details = [text["summary"].format(count=len(boxes), width=width, height=height)]

    for index, box in enumerate(boxes, start=1):
        x_min, y_min, x_max, y_max = box
        pixel_box = (
            round(x_min * (width - 1)),
            round(y_min * (height - 1)),
            round(x_max * (width - 1)),
            round(y_max * (height - 1)),
        )
        color = _COLORS[(index - 1) % len(_COLORS)]
        draw.rectangle(pixel_box, outline=color, width=line_width)

        label = f"BOX {index}"
        label_box = draw.textbbox((pixel_box[0], pixel_box[1]), label, font=font, stroke_width=1)
        label_height = label_box[3] - label_box[1] + 6
        label_width = label_box[2] - label_box[0] + 8
        label_y = max(0, pixel_box[1] - label_height)
        draw.rectangle(
            (pixel_box[0], label_y, min(width - 1, pixel_box[0] + label_width), label_y + label_height),
            fill=color,
        )
        draw.text((pixel_box[0] + 4, label_y + 2), label, fill="black", font=font, stroke_width=1)
        normalized = [round(value, 4) for value in box]
        details.append(f"BOX {index}: normalized={normalized} -> pixels={list(pixel_box)}")

    return rendered, "\n".join(details)
