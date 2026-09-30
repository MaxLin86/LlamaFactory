#!/usr/bin/env python3
"""Interactively evaluate a directory of upper- or lower-GI images.

The model is loaded once.  For every image, this module runs an independent
CADE query and a complete cumulative multi-turn conversation, then shows the
answers and any normalized xyxy boxes in a non-modal Matplotlib window.
"""

from __future__ import annotations

import argparse
import queue
import re
import sys
import textwrap
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "examples/inference/qwen3vl_lora_sft_inno.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_IMAGE_DIRS = {
    "gs": Path("/media/maxlin/SATA/ReportData/test_single_images_cropped/ec05_02_20250217"),
    "cs": Path("/media/maxlin/SATA/ReportData/test_single_cs_select"),
}
BOX_PATTERN = re.compile(r"[\[【]\s*([^\]】]+?)\s*[\]】]")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

GS_QUESTIONS: tuple[str, ...] = (
    "请进行CADE：图像上是否有病变存在？若有，请指出病变位置。",
    "这是肠镜图像还是胃镜图像？",
    "该图像所处的具体位置是什么？",
    "该图像是什么光源？",
    "该图像是否有染色？",
    "该图像是否有手术帽？",
    "该图像中是否有器械？",
    "该图像是否开启放大？",
    "该图像质量如何？",
    "该图像在体内还是体外？",
)

# Union of the questions found in the cs_det_1cls and cs_cadx_2cls preview
# conversations.  Pathology is intentionally absent; morphology is retained.
CS_QUESTIONS: tuple[str, ...] = (
    "请进行CADE：图像上是否有病变存在？若有，请指出病变位置。",
    "请进行CADX：判断图像中肠息肉的NICE分型；如有多个目标，请逐一回答。",
    (
        "请进行CADX：判断图像中的肠息肉属于肿瘤性病变还是非肿瘤性病变；"
        "如有多个目标，请逐一回答。"
    ),
    "这是肠镜图像还是胃镜图像？",
    "该图像所处的具体位置是什么？",
    "该图像是什么光源？",
    "该图像是否有染色？",
    "该图像是否有手术帽？",
    "该图像中是否有器械？",
    "该图像是否开启放大？",
    "该图像质量如何？",
    "该图像在体内还是体外？",
    "该病变的形态学表现是什么？",
)


@dataclass
class TurnResult:
    number: int
    question: str
    answer: str


@dataclass
class ImageResult:
    image_path: Path
    direct_cade: str
    multi_turn: list[TurnResult]
    direct_boxes: list[tuple[float, float, float, float]]
    multi_turn_boxes: list[tuple[float, float, float, float]]


class PreviewStopped(Exception):
    """Raised when the user closes or quits the interactive preview."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="逐图评测上/下消化道 VLM，对比直接 CADE 与累积多轮对话。",
    )
    parser.add_argument("scope", choices=("gs", "cs"), help="gs=上消化道，cs=下消化道。")
    parser.add_argument(
        "image_dir",
        metavar="图片目录",
        type=Path,
        nargs="?",
        help="可选；不填时根据 gs/cs 使用脚本内置的默认目录。",
    )
    return parser


def discover_images(directory: Path) -> list[Path]:
    iterator = directory.rglob("*")
    return sorted(
        (path.resolve() for path in iterator if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES),
        key=lambda path: str(path).casefold(),
    )


def load_chat_model() -> Any:
    from omegaconf import OmegaConf

    from llamafactory.chat import ChatModel

    if not DEFAULT_CONFIG.is_file():
        raise FileNotFoundError(f"推理配置不存在：{DEFAULT_CONFIG}")
    config = OmegaConf.to_container(OmegaConf.load(DEFAULT_CONFIG), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"推理 YAML 顶层必须是对象：{DEFAULT_CONFIG}")
    return ChatModel(config)


def ask(
    model: Any,
    messages: list[dict[str, str]],
    image_path: Path,
) -> str:
    responses = model.chat(messages=messages, images=[str(image_path)])
    if not responses:
        return "[模型未返回内容]"
    return responses[0].response_text.strip()


def extract_valid_boxes(text: str) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for content in BOX_PATTERN.findall(text):
        numbers = NUMBER_PATTERN.findall(content)
        if len(numbers) != 4:
            continue
        box = tuple(float(value) for value in numbers)
        x_min, y_min, x_max, y_max = box
        if (
            all(0.0 <= value <= 1.0 for value in box)
            and x_min < x_max
            and y_min < y_max
        ):
            boxes.append(box)
    return boxes


def evaluate_image(
    model: Any,
    image_path: Path,
    questions: Sequence[tuple[int, str]],
) -> ImageResult:
    direct_cade = ask(
        model,
        [
            {
                "role": "user",
                "content": "请进行CADE：图像上是否有病变存在？若有，请指出病变位置。",
            }
        ],
        image_path,
    )

    messages: list[dict[str, str]] = []
    turns: list[TurnResult] = []
    multi_cade_answers: list[str] = []
    for number, question in questions:
        messages.append({"role": "user", "content": question})
        answer = ask(model, messages, image_path)
        messages.append({"role": "assistant", "content": answer})
        turns.append(TurnResult(number, question, answer))
        # Identify CADE by the question itself, never by its position.  The
        # question lists above can therefore be freely reordered or extended.
        if question.startswith("请进行CADE："):
            multi_cade_answers.append(answer)

    return ImageResult(
        image_path=image_path,
        direct_cade=direct_cade,
        multi_turn=turns,
        direct_boxes=extract_valid_boxes(direct_cade),
        multi_turn_boxes=extract_valid_boxes("\n".join(multi_cade_answers)),
    )


def _load_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_prediction_boxes(image: Image.Image, result: ImageResult) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    width, height = rendered.size
    line_width = max(2, round(max(width, height) / 350))
    font = _load_cjk_font(max(14, round(max(width, height) / 45)))

    # Draw the wider multi-turn box first and the thinner direct box second.
    # If both predictions are identical, the green outer edge and red inner
    # edge remain visible instead of one rectangle completely hiding another.
    groups = (
        (result.multi_turn_boxes, (30, 225, 30), "M", line_width + 3),
        (result.direct_boxes, (255, 45, 45), "D", line_width),
    )
    for boxes, color, prefix, box_width in groups:
        for index, box in enumerate(boxes, start=1):
            x_min, y_min, x_max, y_max = box
            pixel_box = (
                round(x_min * (width - 1)),
                round(y_min * (height - 1)),
                round(x_max * (width - 1)),
                round(y_max * (height - 1)),
            )
            draw.rectangle(pixel_box, outline=color, width=box_width)
            label = f"{prefix}{index}"
            text_box = draw.textbbox((pixel_box[0], pixel_box[1]), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_y = (
                min(height - text_height - 8, pixel_box[1] + box_width + 2)
                if prefix == "M"
                else max(0, pixel_box[1] - text_height - 8)
            )
            draw.rectangle(
                (pixel_box[0], label_y, pixel_box[0] + text_width + 8, label_y + text_height + 8),
                fill=color,
            )
            draw.text((pixel_box[0] + 4, label_y + 2), label, fill=(0, 0, 0), font=font)
    return rendered


def wrap_text(text: str, width: int = 46, subsequent_indent: str = "") -> str:
    lines: list[str] = []
    for raw_line in str(text).splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                raw_line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=True,
                subsequent_indent=subsequent_indent,
            )
            or [""]
        )
    return "\n".join(lines)


def make_result_text(result: ImageResult) -> tuple[str, str]:
    direct = f"【直接 CADE（红框 D，{len(result.direct_boxes)} 个）】\n" + wrap_text(result.direct_cade)
    turns: list[str] = [f"【逐轮对话（CADE 绿框 M，{len(result.multi_turn_boxes)} 个）】"]
    for turn in result.multi_turn:
        turns.append(f"\n{turn.number}. 问：{wrap_text(turn.question, subsequent_indent='   ')}")
        turns.append(f"   答：{wrap_text(turn.answer, subsequent_indent='      ')}")
    return direct, "\n".join(turns)


def show_result(result: ImageResult, index: int, total: int) -> None:
    try:
        import matplotlib

        if str(matplotlib.get_backend()).casefold() == "agg":
            matplotlib.use("Qt5Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("需要可交互的 Matplotlib/Qt 后端才能显示评测窗口。") from error

    with Image.open(result.image_path) as source:
        rendered = draw_prediction_boxes(source, result)

    font_path = next(
        (
            path
            for path in (
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            )
            if path.is_file()
        ),
        None,
    )
    mpl_font = FontProperties(fname=str(font_path)) if font_path else None
    direct_text, turns_text = make_result_text(result)

    figure = plt.figure(figsize=(22, 12))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.18, 1.0),
        height_ratios=(0.22, 0.78),
        left=0.025,
        right=0.985,
        top=0.90,
        bottom=0.07,
        wspace=0.025,
        hspace=0.035,
    )
    image_axis = figure.add_subplot(grid[:, 0])
    direct_axis = figure.add_subplot(grid[0, 1])
    turns_axis = figure.add_subplot(grid[1, 1])

    image_axis.imshow(rendered)
    image_axis.axis("off")
    image_axis.set_title(
        f"原图 + 预测框（红=D:直接CADE，绿=M:多轮CADE）\n{result.image_path.name}",
        fontproperties=mpl_font,
        fontsize=15,
    )
    for axis in (direct_axis, turns_axis):
        axis.axis("off")
    direct_axis.text(
        0,
        1,
        direct_text,
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.25,
        fontproperties=mpl_font,
        transform=direct_axis.transAxes,
    )
    line_count = max(1, len(turns_text.splitlines()))
    turn_font_size = max(8.5, min(12.5, 500.0 / line_count))
    turns_axis.text(
        0,
        1,
        turns_text,
        va="top",
        ha="left",
        fontsize=turn_font_size,
        linespacing=1.18,
        fontproperties=mpl_font,
        transform=turns_axis.transAxes,
    )
    figure.suptitle(
        f"{result.image_path.name}    {index}/{total}    |    空格 / Enter / → / N：下一张    Q / Esc：结束",
        fontproperties=mpl_font,
        fontsize=17,
    )
    progress_axis = figure.add_axes((0.1, 0.025, 0.8, 0.018))
    progress_axis.barh([0], [index], color="#2b8cbe")
    progress_axis.set_xlim(0, max(1, total))
    progress_axis.set_yticks([])
    progress_axis.set_xticks([])
    progress_axis.set_title(f"{index}/{total}", fontproperties=mpl_font, fontsize=10, pad=1)

    try:
        from matplotlib.backends.qt_compat import QtCore

        window = figure.canvas.manager.window
        window.setWindowModality(QtCore.Qt.NonModal)
        window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, False)
        window.show()
    except (AttributeError, ImportError):
        pass

    pressed: dict[str, str | None] = {"key": None}

    def on_key(event: Any) -> None:
        key = str(event.key or "").casefold()
        if key in {" ", "space", "enter", "return", "right", "n", "q", "escape"}:
            pressed["key"] = key

    connection = figure.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)
    while pressed["key"] is None and plt.fignum_exists(figure.number):
        plt.pause(0.05)
    figure.canvas.mpl_disconnect(connection)
    key = (pressed["key"] or "closed").casefold()
    plt.close(figure)
    if key in {"q", "escape", "closed"}:
        raise PreviewStopped()


def print_result(result: ImageResult, index: int, total: int) -> None:
    print(f"\n{'=' * 88}\n[{index}/{total}] {result.image_path}")
    print(f"\n[直接 CADE：解析到 {len(result.direct_boxes)} 个坐标框]\n{result.direct_cade}")
    print(f"\n[逐轮对话：解析到 {len(result.multi_turn_boxes)} 个 CADE 坐标框]")
    for turn in result.multi_turn:
        print(f"{turn.number}. 问：{turn.question}\n   答：{turn.answer}")


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_dir = (args.image_dir or DEFAULT_IMAGE_DIRS[args.scope]).expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"图片目录不存在：{image_dir}")

    images = discover_images(image_dir)
    if not images:
        raise FileNotFoundError(f"未在目录中找到可识别的图片：{image_dir}")

    question_texts = GS_QUESTIONS if args.scope == "gs" else CS_QUESTIONS
    questions = list(enumerate(question_texts, start=1))
    print(f"图片目录：{image_dir}")
    print(f"数据类型：{args.scope} ({'上消化道' if args.scope == 'gs' else '下消化道'})")
    print(f"推理配置：{DEFAULT_CONFIG}")
    print(f"图片数量：{len(images)}")
    print("正在加载模型……", flush=True)
    model = load_chat_model()

    total = len(images)
    result_queue: queue.Queue[tuple[int, Path, ImageResult | None, Exception | None]] = queue.Queue()
    stop_event = threading.Event()

    def inference_worker() -> None:
        """Continuously infer later images while the current result is displayed."""
        for worker_index, worker_image_path in enumerate(images, start=1):
            if stop_event.is_set():
                break
            print(
                f"\n[后台推理 {worker_index}/{total}] {worker_image_path.name}",
                flush=True,
            )
            try:
                worker_result = evaluate_image(model, worker_image_path, questions)
            except Exception as error:
                result_queue.put((worker_index, worker_image_path, None, error))
                if "does not support image input" in str(error):
                    stop_event.set()
                    break
            else:
                result_queue.put((worker_index, worker_image_path, worker_result, None))

    worker = threading.Thread(target=inference_worker, name="vlm-evaluation-worker", daemon=True)
    worker.start()

    for _ in images:
        index, image_path, result, error = result_queue.get()
        if error is not None:
            print(
                f"[{index}/{total}] 推理失败：{image_path}\n{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            if "does not support image input" in str(error):
                print("请检查推理 YAML 中的 template；已停止后续图片。", file=sys.stderr)
                return 1
            continue
        assert result is not None
        print_result(result, index, total)
        try:
            show_result(result, index, total)
        except PreviewStopped:
            stop_event.set()
            print(f"已在 {index}/{total} 结束评测。")
            return 0

    worker.join()
    print(f"\n评测完成，共 {total} 张。")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
