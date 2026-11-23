#!/usr/bin/env python3
"""Evaluate case-level multi-image reports without modifying the dataset.

在 llama_factory 环境中运行：

    # 从训练时真正使用的 train 缓存中随机抽取10例，检查已训练病例是否学会。
    CUDA_VISIBLE_DEVICES=0 python data/inno_report_data/eval_case_images.py train

    # 从训练时真正使用的 validation 缓存中随机抽取10例。
    CUDA_VISIBLE_DEVICES=0 python data/inno_report_data/eval_case_images.py validation

    # 之前鹰眼JSON方案的5胃镜、5肠镜，直接读取选图器的select/。
    CUDA_VISIBLE_DEVICES=0 python data/inno_report_data/eval_case_images.py eagle

    # 只核对病例清单，不加载模型、不打开窗口、不使用GPU。
    python data/inno_report_data/eval_case_images.py train --list-only
    python data/inno_report_data/eval_case_images.py validation --list-only
    python data/inno_report_data/eval_case_images.py eagle --list-only

模型路径、像素范围、生成长度在独立的推理YAML中调整；可用--config切换。
数据路径、鹰眼病例清单等默认值集中在文件末尾的build_parser上方。
validation不是独立test：这些病例训练期间已用于计算eval_loss。
鹰眼只读取selected.txt登记的选中图片，绝不读取all/或montage.jpg作为输入。
同一病例的全部图片一次传入；GT、think、文件名和选图标签不传给模型。
后台连续推理，左侧图片、右侧上方模型回答和下方GT；点击缩略图可放大。
关闭窗口会终止本脚本创建的模型子进程，释放其显存。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import random
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    key: str
    scope: str
    images: tuple[str, ...]
    gt: str | None = None

    @property
    def question(self) -> str:
        exam = {"gs": "胃镜", "cs": "肠镜"}[self.scope]
        return f"请综合该病例的{exam}图像，描述镜检所见，并给出诊断结论。"

    @property
    def prompt(self) -> str:
        # 与data_generate_case_images.build_sample完全相同的图文顺序。
        return self.question + "\n<image>" * len(self.images)


def read_config(path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"推理配置顶层必须是对象：{path}")
    return config


def image_paths(values: list[str], root: Path) -> tuple[str, ...]:
    paths = tuple(str((root / value).resolve()) for value in values)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("病例没有图片或图片重复，请核对输入记录。")
    for value in paths:
        if not Path(value).is_file():
            raise FileNotFoundError(f"病例图片不存在：{value}")
    return paths


def load_cached_cases(dataset_root: Path, tokenized_root: Path, seed: int, split: str) -> list[EvalCase]:
    from datasets import load_from_disk

    if split not in ("train", "validation"):
        raise ValueError(f"不支持的缓存划分：{split}")
    label = "训练集" if split == "train" else "验证集"
    # memory-map，只取10条记录的images列；不读取全部Token或重建随机划分。
    dataset = load_from_disk(str(tokenized_root / split), keep_in_memory=False)
    if len(dataset) < CASE_COUNT:
        raise ValueError(f"{label}只有{len(dataset)}例，不足{CASE_COUNT}例。")
    indices = sorted(random.Random(seed).sample(range(len(dataset)), CASE_COUNT))
    selected = dataset.select_columns(["images"]).select(indices)
    wanted = [image_paths(row["images"], dataset_root) for row in selected]
    targets = set(wanted)
    if len(targets) != len(wanted):
        raise ValueError(f"抽到的{label}记录存在重复病例，请核对缓存。")
    requested_keys = {Path(paths[0]).parent.parent.relative_to(dataset_root.resolve()).as_posix() for paths in wanted}

    found: dict[tuple[str, ...], EvalCase] = {}
    # 逐行读取中间记录，仅保留抽中的病例；不加载整份conversations.json。
    with (dataset_root / "prepared_cases.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["case_key"] not in requested_keys:
                continue
            paths = tuple(str((dataset_root / value).resolve()) for value in record["images"])
            if paths not in targets:
                continue
            if paths in found:
                raise ValueError(f"中间记录中出现重复病例：{record['case_key']}")
            found[paths] = EvalCase(
                record["case_key"],
                record["scope"],
                paths,
                json.dumps(record["gt"], ensure_ascii=False, indent=2),
            )
    if targets - found.keys():
        raise ValueError(f"{label}缓存与prepared_cases.jsonl不匹配，不能确认病例身份和GT；请核对默认数据路径。")
    print(f"{label}共{len(dataset)}例；seed={seed}；抽取行号（从0开始）：{indices}")
    return [found[paths] for paths in wanted]


def load_eagle_cases(root: Path) -> list[EvalCase]:
    cases = []
    for scope, case_ids in EAGLE_CASE_IDS.items():
        for case_id in case_ids:
            directory = root / case_id
            report = directory / "selected.txt"
            lines = report.read_text(encoding="utf-8").splitlines()
            payloads = [line[len(EXPORT_PREFIX) :] for line in lines if line.startswith(EXPORT_PREFIX)]
            if len(payloads) != 1:
                raise ValueError(f"选图结果缺少唯一的文件登记，不能猜测输入图片：{report}")
            record = json.loads(payloads[0])
            source_dir = Path(record["case_dir"])
            if source_dir.parent.name != scope or source_dir.name != case_id:
                raise ValueError(f"选图结果的病例或检查类型不匹配：{report}")
            selected = []
            for value in record["files"]:
                relative = Path(value)
                if not relative.parts or relative.parts[0] != "select":
                    continue
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"选图文件登记路径越界：{value}")
                selected.append(value)
            # 保留登记顺序，即选图器输出的001、002……顺序，不能按原文件名重新排序。
            paths = image_paths(selected, directory)
            gt_file = source_dir / "reports" / f"struct_{case_id}.json"
            gt = None
            if gt_file.is_file():
                gt = json.dumps(json.loads(gt_file.read_text(encoding="utf-8-sig")), ensure_ascii=False, indent=2)
            cases.append(EvalCase(f"{scope}/{case_id}", scope, paths, gt))
    return cases


def infer_case(model: Any, case: EvalCase) -> dict[str, Any]:
    started = time.monotonic()
    # 每个病例都新建消息，不继承上一病例的任何对话，更不会把GT放入上下文。
    responses = model.chat(
        messages=[{"role": "user", "content": case.prompt}],
        images=list(case.images),
    )
    if not responses:
        raise RuntimeError("模型没有返回内容。")
    response = responses[0]
    return {
        "answer": response.response_text.strip(),
        "finish_reason": response.finish_reason,
        "seconds": time.monotonic() - started,
    }


def inference_worker(cases: list[EvalCase], config_path: str, output: Any, stop: Any) -> None:
    """Load one model in a dedicated process that can be stopped with the viewer."""
    try:
        src = str(PROJECT_ROOT / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from llamafactory.chat import ChatModel

        output.put({"kind": "status", "text": "正在加载病例模型……"})
        model = ChatModel(read_config(Path(config_path)))
        for index, case in enumerate(cases):
            if stop.is_set():
                break
            output.put({"kind": "status", "text": f"后台推理 {index + 1}/{len(cases)}：{case.key}"})
            try:
                result = infer_case(model, case)
            except Exception:
                output.put({"kind": "result", "index": index, "error": traceback.format_exc()})
                # 真正抛出的OOM与可恢复的分配警告不同；记录本例失败后清理空闲缓存。
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                output.put({"kind": "result", "index": index, **result})
        output.put({"kind": "done"})
    except Exception:
        output.put({"kind": "fatal", "error": traceback.format_exc()})


def format_answer(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = "\n".join(candidate.splitlines()[1:-1])
    try:
        parsed = json.loads(candidate)
    except ValueError:
        return text
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def configure_ui_fonts(root: Any) -> str:
    """Choose a CJK family supported by Tk itself, including X11 bitmap fonts."""
    import tkinter.font as tkfont
    from tkinter import ttk

    # 部分Conda Tk没有Xft支持，系统fc-list中的Noto等字体并不代表Tk可以使用。
    # 本机这类Tk实际只识别song ti / fangsong ti，不能退回不含中文的fixed。
    available = {family.casefold(): family for family in tkfont.families(root)}
    candidates = (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "Droid Sans Fallback",
        "SimHei",
        "SimSun",
        "song ti",
        "fangsong ti",
    )
    family = next((available[name.casefold()] for name in candidates if name.casefold() in available), None)
    if family is None:
        raise RuntimeError("当前Tk没有识别到中文字体，不能正常显示评测窗口；请检查Tk的字体支持。")

    # 只修正正文font不够：按钮、图片说明和标签页默认使用这些命名字体。
    for name in tkfont.names(root):
        if name.startswith("Tk"):
            tkfont.nametofont(name, root=root).configure(family=family, size=12)
    root.option_add("*Font", (family, 12))
    ttk.Style(root).configure(".", font=(family, 12))
    print(f"界面中文字体：{family}", flush=True)
    return family


class CaseViewer:
    """Show scrollable images and reports in a non-modal, non-topmost Tk window."""

    def __init__(self, cases: list[EvalCase], config_path: Path, route: str):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.tk, self.ttk = tk, ttk
        self.cases, self.current = cases, 0
        self.results: dict[int, dict[str, Any]] = {}
        self.done, self.closed = False, False
        self.fatal_error = ""
        self.exited_at: float | None = None
        self.photos: list[Any] = []
        self.image_layout: tuple[int, int] | None = None
        self.root = tk.Tk()
        self.root.title("病例多图评测")
        self.root.attributes("-topmost", False)
        width = min(1800, int(self.root.winfo_screenwidth() * 0.94))
        height = min(1050, int(self.root.winfo_screenheight() * 0.90))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(width, 1000), min(height, 650))
        font = configure_ui_fonts(self.root)

        self.heading = tk.StringVar()
        ttk.Label(self.root, textvariable=self.heading, font=(font, 14), padding=10).pack(fill="x")
        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=10)
        left, right = ttk.Frame(panes), ttk.Frame(panes)
        panes.add(left, weight=3)
        panes.add(right, weight=2)
        self.canvas = tk.Canvas(left, highlightthickness=0)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.tiles = ttk.Frame(self.canvas)
        self.tile_window = self.canvas.create_window((0, 0), window=self.tiles, anchor="nw")
        self.tiles.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self.resize_images)
        self.bind_image_scroll(self.canvas)
        self.bind_image_scroll(self.tiles)

        self.question_label = ttk.Label(right, font=(font, 12), wraplength=500, padding=(12, 8))
        self.question_label.pack(fill="x")
        reports = ttk.Panedwindow(right, orient="vertical")
        reports.pack(fill="both", expand=True, padx=(8, 0))
        answer_panel = ttk.LabelFrame(reports, text="模型回答", padding=4)
        reference_panel = ttk.LabelFrame(reports, text="参考 GT", padding=4)
        reports.add(answer_panel, weight=1)
        reports.add(reference_panel, weight=1)
        # 上下同列，同时可见；分隔条可拖动，每块文字各自滚动。
        text_options = {
            "wrap": "word",
            "font": (font, 14),
            "padx": 14,
            "pady": 12,
            "width": 40,
            "height": 8,
            "spacing1": 4,  # 每个逻辑行上方留白。
            "spacing2": 6,  # 自动折行之间留白。
            "spacing3": 4,  # 每个逻辑行下方留白。
        }
        self.answer = ScrolledText(answer_panel, **text_options)
        self.reference = ScrolledText(reference_panel, **text_options)
        self.answer.pack(fill="both", expand=True)
        self.reference.pack(fill="both", expand=True)
        right.bind("<Configure>", lambda event: self.question_label.configure(wraplength=max(150, event.width - 30)))

        footer = ttk.Frame(self.root, padding=10)
        footer.pack(fill="x")
        self.previous_button = ttk.Button(footer, text="上一例", command=lambda: self.navigate(-1))
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(footer, text="下一例", command=lambda: self.navigate(1))
        self.next_button.pack(side="left", padx=10)
        ttk.Button(footer, text="结束并释放模型", command=self.close).pack(side="right")
        self.progress = ttk.Progressbar(footer, maximum=len(cases), length=180)
        self.progress.pack(side="left", padx=10)
        self.status = tk.StringVar(value="正在启动模型进程……")
        ttk.Label(self.root, textvariable=self.status, padding=(10, 0, 10, 8)).pack(fill="x")
        self.route_label = ROUTE_LABELS[route]
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.update_idletasks()
        self.render_case()

        context = mp.get_context("spawn")
        self.output, self.stop = context.Queue(), context.Event()
        self.process = context.Process(
            target=inference_worker,
            args=(cases, str(config_path), self.output, self.stop),
            daemon=True,
            name="case-vlm-inference",
        )
        self.process.start()
        self.root.after(100, self.poll)

    @staticmethod
    def set_text(widget: Any, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def resize_images(self, event: Any) -> None:
        self.canvas.itemconfigure(self.tile_window, width=event.width)
        columns = min(3, max(1, event.width // 240))
        layout = (columns, max(100, event.width // columns - 20))
        if layout != self.image_layout:
            self.image_layout = layout
            if hasattr(self, "route_label"):
                self.render_images()

    def bind_image_scroll(self, widget: Any) -> None:
        # 仅绑定左侧控件，不抢占右侧文本框或其他应用的滚轮。
        for event, units in (("<Button-4>", -3), ("<Button-5>", 3)):
            widget.bind(event, lambda _event, step=units: self.canvas.yview_scroll(step, "units"))
        widget.bind("<MouseWheel>", lambda event: self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"))

    def render_images(self) -> None:
        from PIL import Image, ImageTk

        for child in self.tiles.winfo_children():
            child.destroy()
        self.photos.clear()
        columns, cell_width = self.image_layout or (3, 260)
        for column in range(3):
            self.tiles.columnconfigure(column, weight=1 if column < columns else 0)
        for number, value in enumerate(self.cases[self.current].images):
            frame = self.ttk.Frame(self.tiles, padding=5)
            frame.grid(row=number // columns, column=number % columns, sticky="nsew")
            self.bind_image_scroll(frame)
            try:
                with Image.open(value) as source:
                    image = source.convert("RGB")
                image.thumbnail((cell_width, 230))  # 只缩小，不放大小图。
                photo = ImageTk.PhotoImage(image, master=self.root)
                self.photos.append(photo)
                button = self.ttk.Button(frame, image=photo, command=lambda path=value: self.enlarge(path))
                button.pack()
                self.bind_image_scroll(button)
            except Exception as error:
                self.ttk.Label(frame, text=f"图片显示失败：{error}", wraplength=cell_width).pack()
            caption = self.ttk.Label(
                frame,
                text=f"{number + 1}. {Path(value).name}",
                wraplength=cell_width,
                anchor="center",
                justify="center",
            )
            caption.pack(fill="x", pady=(4, 8))
            self.bind_image_scroll(caption)

    def enlarge(self, path: str) -> None:
        from PIL import Image, ImageTk

        window = self.tk.Toplevel(self.root)
        window.title(Path(path).name)
        window.attributes("-topmost", False)
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((self.root.winfo_screenwidth() - 120, self.root.winfo_screenheight() - 160))
        photo = ImageTk.PhotoImage(image, master=window)
        label = self.ttk.Label(window, image=photo)
        label.image = photo
        label.pack()
        # 不调用grab_set、wait_window或focus_force，仍可切回其他应用。

    def render_case(self) -> None:
        case = self.cases[self.current]
        self.root.title(f"病例评测 {self.current + 1}/{len(self.cases)} — {case.key}")
        self.heading.set(
            f"{self.route_label}    {self.current + 1}/{len(self.cases)}    {case.key}    {len(case.images)}张"
        )
        self.question_label.configure(text=case.question)
        self.previous_button.configure(state="normal" if self.current else "disabled")
        self.next_button.configure(state="normal" if self.current + 1 < len(self.cases) else "disabled")
        self.canvas.yview_moveto(0)
        self.render_images()
        self.render_answer()
        self.set_text(
            self.reference, case.gt or "未提供该病例的结构化报告GT。\n鹰眼_info.json是选图元数据，不作为诊断结论。"
        )

    def render_answer(self) -> None:
        result = self.results.get(self.current)
        if result is None:
            self.set_text(self.answer, self.fatal_error or "等待后台推理……\n可以先查看左侧图片或切换病例。")
        elif "error" in result:
            self.set_text(self.answer, "本病例推理失败：\n\n" + result["error"])
        else:
            text = format_answer(result["answer"])
            if result["finish_reason"] == "length":
                text += "\n\n【回答达到max_new_tokens上限，可能被截断；可提高推理YAML中的该值后重测。】"
            text += f"\n\n—— 推理耗时 {result['seconds']:.1f} 秒 ——"
            self.set_text(self.answer, text)

    def navigate(self, delta: int) -> None:
        target = self.current + delta
        if 0 <= target < len(self.cases):
            self.current = target
            self.render_case()

    def poll(self) -> None:
        if self.closed:
            return
        try:
            while True:
                event = self.output.get_nowait()
                kind = event["kind"]
                if kind == "status":
                    self.status.set(event["text"])
                    print(event["text"], flush=True)
                elif kind == "result":
                    index = event["index"]
                    self.results[index] = event
                    self.progress["value"] = len(self.results)
                    print(
                        f"\n[{index + 1}/{len(self.cases)}] {self.cases[index].key}\n"
                        + event.get("error", event.get("answer", "")),
                        flush=True,
                    )
                    if index == self.current:
                        self.render_answer()
                elif kind == "fatal":
                    self.fatal_error = "模型进程启动或运行失败：\n\n" + event["error"]
                    self.status.set("推理已停止，请查看右侧错误或终端。")
                    self.done = True
                    self.render_answer()
                    print(self.fatal_error, file=sys.stderr, flush=True)
                elif kind == "done":
                    failed = sum("error" in result for result in self.results.values())
                    self.status.set(
                        f"后台推理完成：成功{len(self.results) - failed}例，失败{failed}例。可继续逐例查看。"
                    )
                    self.done = True
        except queue.Empty:
            pass
        # 异常退出（例如系统杀进程）没有队列消息，不能无限停留在“等待”。
        if not self.process.is_alive() and not self.done:
            if self.exited_at is None:
                self.exited_at = time.monotonic()
            elif time.monotonic() - self.exited_at > 1:
                self.fatal_error = f"模型进程意外退出，退出码：{self.process.exitcode}；请查看终端日志。"
                self.status.set(self.fatal_error)
                self.done = True
                self.render_answer()
        self.root.after(150, self.poll)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "process"):
            self.stop.set()
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.output.close()
            self.output.cancel_join_thread()
        self.root.destroy()

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self.close()


# 默认路径和选项集中在parser附近；修改模型只需修改独立的推理YAML。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "examples/inference/qwen3vl_lora_sft_inno_case.yaml"
DEFAULT_DATASET_ROOT = Path("/media/maxlin/SATA/ReportData/Max_V1_case_dataV2")
DEFAULT_TOKENIZED_ROOT = Path(
    "/media/maxlin/SATA/ReportData/tokenized/max_case_v2_qwen3vl2b_nothink_min256_max512_cut5120_val10pct_seed2026"
)
DEFAULT_EAGLE_ROOT = Path("/home/maxlin/桌面/innoaitools/inference/report/selector_results")
DEFAULT_SEED = 2026
CASE_COUNT = 10
ROUTE_LABELS = {
    "train": "训练集抽样（已参与训练）",
    "validation": "验证集抽样（非独立test）",
    "eagle": "鹰眼JSON选图",
}
EXPORT_PREFIX = "# report-selector-files-v1: "
EAGLE_CASE_IDS = {
    "gs": ("583_295", "583_562", "585_10", "585_92", "585_130"),
    "cs": ("450_335", "450_389", "450_433", "450_465", "476_342"),
}  # 使用原来的5+5；不自动混入额外的583_562_新。


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("route", choices=tuple(ROUTE_LABELS), help="训练/验证缓存抽10例，或鹰眼JSON选图5胃镜+5肠镜")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="病例版推理YAML，默认使用本脚本对应配置")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="训练/验证集抽样随机种子，默认2026；eagle固定病例，不使用此值"
    )
    parser.add_argument("--list-only", action="store_true", help="只输出病例清单，不加载模型、不打开窗口")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.route in ("train", "validation"):
            cases = load_cached_cases(DEFAULT_DATASET_ROOT, DEFAULT_TOKENIZED_ROOT, args.seed, args.route)
            if args.route == "train":
                print("注意：这些病例已参与训练，仅用于检查拟合程度，不能作为泛化能力指标。")
            else:
                print("注意：这是训练期间使用的验证集，不是独立测试集。")
        else:
            cases = load_eagle_cases(DEFAULT_EAGLE_ROOT)
        counts = Counter(case.scope for case in cases)
        print(f"共{len(cases)}例：胃镜{counts['gs']}例，肠镜{counts['cs']}例。")
        for index, case in enumerate(cases, 1):
            print(f"{index:2}/{len(cases)}  {case.key}  图片={len(case.images)}  GT={'有' if case.gt else '无'}")
        if args.list_only:
            return 0
        config_path = args.config.expanduser().resolve()
        config = read_config(config_path)
        print(
            f"推理配置：{config_path}\n基座：{config.get('model_name_or_path')}\n"
            f"LoRA：{config.get('adapter_name_or_path')}",
            flush=True,
        )
        CaseViewer(cases, config_path, args.route).run()
    except KeyboardInterrupt:
        print("\n已结束评测。")
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
