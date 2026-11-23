#!/usr/bin/env python3
"""Generate one multi-image LLaMA-Factory sample per endoscopy case (Python 3.8+).

Usage (all paths below have defaults)::

    # Inspect sources without loading models or writing anything.
    python data/inno_report_data/data_generate_case_images.py --dry-run

    # Small real-model test: at most one eligible case of each scope.
    python data/inno_report_data/data_generate_case_images.py --limit 1 --output-root /tmp/case_dataset_test

    # Inspect the first valid conversation in the terminal, confirm, then generate all cases.
    python data/inno_report_data/data_generate_case_images.py

    # Include predicted position/lesion hints, keeping the original output filenames.
    python data/inno_report_data/data_generate_case_images.py --with-image-hints

The original images and reports are READ ONLY. A persistent isolated Python
worker calls the existing report_images_selector.py API; its ONNX/TRT models
are loaded once per examination type. Training images use the exact crop boxes
used during selection, never the selector's unprocessed export copies.

Output layout::

    Max_V1_case_dataV2/
        all_conversations.json          # Images -> structured report.
        all_conversations_think.json    # Images -> <think>...</think> + report.
        dataset_info.json               # max_v1_case / max_v1_case_think
        prepared_cases.jsonl            # Streaming metadata for export/token statistics.
        manifest.json / failures.jsonl / generation.log
        gs|cs/<doctor>/<category>/<case>/
            images/                     # Selected, cropped, no annotations.
            reports/                    # Original reports, copied verbatim.
            selection.json              # Selector result and audit information.
            info.txt                    # Original case metadata, if present.

All training image paths are relative to the output root. Move the whole
directory and use its new location as dataset_dir (media_dir defaults to it).
Cases without a valid struct_<case>.json are logged and excluded. Missing or
empty think files exclude a case only from the thinking variant. The thinking
text is preserved, not checked or rewritten as image-grounded reasoning.
Cases whose reports/*.json or reports/*.txt (including think files) contain
EXCLUDED_REPORT_KEYWORDS are excluded before selection.

Every generation run rebuilds the complete output. If the destination contains
anything (including hidden files), generation only proceeds after an explicit
y/yes confirmation to clear it. Refusal, empty input, EOF or a noninteractive
terminal stops the program without changing the destination or loading models.
--dry-run only inspects sources and never clears or writes the output.
After preparing the first valid case, both its normal and available thinking
conversations are printed in the terminal. Enter y/yes to process all remaining
cases without further prompts. Refusal, EOF, Ctrl+C or a noninteractive terminal
stops generation, preserving the first case and intermediate records but not
publishing final conversation files. No preview images or files are generated.

--with-image-hints adds predicted position/lesion hints to the human input of
both exports. Filenames, JSON structure and dataset registrations remain
all_conversations.json / all_conversations_think.json and max_v1_case /
max_v1_case_think. Use separate output directories to distinguish versions.
Each image's hints come only from auxiliary model outputs, never reports, filenames or category
directories. Unaccepted positions are unknown; no detection does NOT mean no
disease. Hinted models should receive the same hints at inference, including
real deployment, not just validation. Use a separate tokenized_path when
training on the hinted exports; existing tokenized datasets are not updated.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO


LOGGER = logging.getLogger("case_dataset_builder")


@dataclass(frozen=True)
class Case:
    scope: str
    root: Path
    directory: Path

    @property
    def key(self) -> str:
        return (Path(self.scope) / self.directory.relative_to(self.root)).as_posix()

    @property
    def report_path(self) -> Path:
        return self.directory / "reports" / ("struct_" + self.directory.name + ".json")

    @property
    def think_path(self) -> Path:
        return self.directory / "reports" / ("think_" + self.directory.name + ".txt")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def text_report_keyword_matches(value: Any, name: str) -> list[dict[str, str]]:
    """Search decoded JSON/text content, never source paths or image names."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return [{"file": name, "keyword": keyword} for keyword in EXCLUDED_REPORT_KEYWORDS if keyword in text]


def report_keyword_matches(reports_dir: Path) -> list[dict[str, str]]:
    """Inspect report JSON/TXT files; one returned match per file and keyword."""
    matches = []
    if not reports_dir.is_dir():
        return matches
    for path in sorted(reports_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".json", ".txt"):
            continue
        raw = path.read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16")
        else:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("gb18030")
        content = text
        if path.suffix.lower() == ".json":
            try:
                content = json.loads(text)  # Also detect escaped Chinese: \u8d85\u58f0.
            except ValueError:
                pass  # A damaged raw report can still contain an excluded word.
        matches.extend(text_report_keyword_matches(content, path.relative_to(reports_dir).as_posix()))
    return matches


def cached_report_keyword_matches(record: dict[str, Any], output: Path) -> list[dict[str, str]]:
    """Check local report snapshots/metadata for export and token statistics."""
    matches = []
    if record.get("reports_dir"):
        reports_dir = output_file(output, record["reports_dir"])
        if not reports_dir.is_dir():
            raise FileNotFoundError("病例报告副本不存在，无法检查排除词：" + str(reports_dir))
        matches.extend(report_keyword_matches(reports_dir))
    for field in ("gt_raw", "gt", "think"):
        if record.get(field):
            matches.extend(text_report_keyword_matches(record[field], "cache:" + field))
    return matches


class ReportKeywordExcluded(ValueError):
    def __init__(self, matches: list[dict[str, str]]):
        self.matches = matches
        detail = "、".join(sorted({item["file"] for item in matches}))
        words = "、".join(sorted({item["keyword"] for item in matches}))
        super().__init__(f"报告包含排除词“{words}”：{detail}")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def normalize_report(value: Any) -> dict[str, Any]:
    """Keep all GT values; fold older flat organ fields under 镜检所见."""
    if not isinstance(value, dict) or not value or "诊断结论" not in value:
        raise ValueError("GT必须是包含诊断结论的非空JSON对象")
    if "镜检所见" in value:
        normalized = dict(value)
    else:
        findings = {key: item for key, item in value.items() if key != "诊断结论"}
        normalized = {"镜检所见": findings, "诊断结论": value["诊断结论"]}
    if not normalized["镜检所见"] or not normalized["诊断结论"]:
        raise ValueError("镜检所见或诊断结论为空")
    # Check serializability and catch accidental NaN/Infinity in annotations.
    json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    return normalized


def thinking_block(raw: str) -> Optional[str]:
    """Keep the thought body, adding exactly one pair of think delimiters."""
    text = raw.lstrip("\ufeff").strip()
    if not text:
        return None
    if "<think>" in text or "</think>" in text:
        if text.count("<think>") != 1 or text.count("</think>") != 1:
            raise ValueError("think文件的<think>/</think>标签不配对或存在多组")
        before, rest = text.split("<think>", 1)
        body = rest.split("</think>", 1)[0]
        if before.strip():
            raise ValueError("think文件在<think>之前存在额外正文")
        text = body.strip()
    return "<think>\n" + text + "\n</think>" if text else None


def discover_cases(scope: str, root: Path) -> Iterator[Case]:
    """Stop at case boundaries; never recursively walk thousands of images."""
    pending = [root]
    while pending:
        directory = pending.pop()
        if (directory / "images").is_dir() or (directory / "reports").is_dir():
            yield Case(scope, root, directory)
            continue
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
        pending.extend(reversed([path for path in entries if path.is_dir() and not path.is_symlink()]))


def collect_cases(args: argparse.Namespace) -> list[Case]:
    cases = []
    for scope, source in (("gs", args.gs_root), ("cs", args.cs_root)):
        if args.scope not in ("all", scope):
            continue
        root = source.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError("原始病例目录不存在：" + str(root))
        counts = Counter()
        eligible = 0
        for case in discover_cases(scope, root):
            counts["directories"] += 1
            matches = report_keyword_matches(case.directory / "reports")
            if matches:
                counts["excluded_report_keyword"] += 1
            has_inputs = case.report_path.is_file() and (case.directory / "images").is_dir()
            counts["with_gt_and_images" if has_inputs else "missing_gt_or_images"] += 1
            if has_inputs and not matches:
                eligible += 1
            # Full generation records all failures; limited tests spend their
            # case budget only on nonexcluded cases with GT and images.
            if not args.limit or (has_inputs and not matches and eligible <= args.limit):
                cases.append(case)
            counts["with_think"] += case.think_path.is_file()
        print(f"{scope} 病例统计：{json.dumps(dict(counts), ensure_ascii=False)}", flush=True)
    return cases


def validate_output(output: Path, source_roots: Sequence[Path]) -> None:
    # Generation can remove output after explicit confirmation. Never allow an
    # original-data directory, its ancestor, a root, or this script's repository.
    protected = [path.resolve() for path in source_roots]
    protected += [DEFAULT_GS_ROOT.resolve(), DEFAULT_CS_ROOT.resolve(), Path(__file__).resolve().parents[2]]
    if output == Path(output.anchor) or output == Path.home().resolve():
        raise ValueError("输出不能使用磁盘根目录或用户主目录")
    for source in protected:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"输出目录与原始数据/代码目录重叠：{output}；{source}")


def confirm_in_terminal(prompt: str) -> bool:
    """Only an explicit y/yes from an interactive terminal approves an action."""
    if not sys.stdin.isatty():
        print("当前终端无法交互确认，已终止。", flush=True)
        return False
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def prepare_output_directory(output: Path) -> bool:
    """Clear a nonempty destination only after an explicit interactive yes."""
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError("目标输出路径不是目录：" + str(output))
        if any(output.iterdir()):
            print(f"目标目录非空：{output}\n继续将清空其中全部内容，包括图片、报告、JSON和日志。", flush=True)
            if not confirm_in_terminal("确认清空目标目录并重新全部生成？[y/N]: "):
                print("未确认清空，已终止；目标目录未修改。")
                return False
            shutil.rmtree(str(output))
    output.mkdir(parents=True, exist_ok=True)
    return True


class TimestampFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        return "\n".join(f"[{stamp}] {record.levelname} {line}" for line in record.getMessage().splitlines())


class ConsoleFormatter(TimestampFormatter):
    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "plain_output", False):
            return record.getMessage()
        return super().format(record)


def configure_logging(output: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)
    log_path = output / "generation.log"
    for handler in (logging.StreamHandler(), logging.FileHandler(str(log_path), mode="w", encoding="utf-8")):
        handler.setFormatter(TimestampFormatter() if isinstance(handler, logging.FileHandler) else ConsoleFormatter())
        LOGGER.addHandler(handler)
    LOGGER.info("开始病例数据全部生成：output=%s", output)


def load_selector(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("case_report_images_selector", str(path))
    if spec is None or spec.loader is None:
        raise ImportError("无法加载选图器：" + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def export_selected(selector: Any, result: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    """Export exactly the images seen by selector models, without label overlays."""
    selected = result["selected"]
    if not selected:
        raise ValueError("选图器未选出任何可用图片；不能构造空图病例")
    destination.mkdir(parents=True, exist_ok=True)
    exported = []
    seen = set()
    for number, record in enumerate(selected, 1):
        source = Path(record["path"]).resolve()
        if source in seen:
            raise ValueError("选图器重复返回同一图片：" + str(source))
        seen.add(source)
        image = selector._read_image(record)["image"]
        height, width = image.shape[:2]
        if record["image_size"] != [width, height]:
            raise ValueError("选图后原图尺寸发生变化：" + str(source))
        box = record["crop_box"]
        if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in box):
            raise ValueError("选图器返回非法crop_box：" + repr(box))
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("裁图框超出原图：" + str(source))
        target = destination / (f"{number:03d}_" + source.name)
        if box == [0, 0, width, height]:
            shutil.copy2(str(source), str(target))
        else:
            selector._write_image(target, image[y1:y2, x1:x2])
        exported.append(
            {
                "file_name": target.name,
                "source": str(source),
                "crop_xyxy": box,
                "original_size": [width, height],
                "saved_size": [x2 - x1, y2 - y1],
                "region": record.get("region"),
                "source_type": record.get("source"),
                "selection_reasons": record.get("selection_reasons", []),
                "lesion_detected": record.get("lesion_detected"),
            }
        )
    return exported


def selector_worker(args: argparse.Namespace) -> int:
    # Keep a dedicated protocol FD. Redirect both Python and native TRT stdout
    # to stderr so CUDA diagnostic output cannot corrupt JSON responses.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    selector = load_selector(args.selector)
    config = selector.SelectorConfig(
        max_images_gs=args.max_images_gs,
        max_images_cs=args.max_images_cs,
        workers=args.workers,
    )
    models = {}
    for line in sys.stdin:
        request = json.loads(line)
        if request["action"] == "close":
            break
        scope = request["scope"]
        try:
            if scope not in models:
                models[scope] = selector.load_report_models(config, exam_type=scope, image_only=True)
        except Exception:
            protocol.write(json.dumps({"ok": False, "fatal": True, "error": traceback.format_exc()}) + "\n")
            continue
        try:
            result = selector.select_case_images(
                Path(request["case_dir"]) / "images",
                models=models[scope],
                exam_type=scope,
                config=config,
            )
            if len(result["selected"]) > (args.max_images_gs if scope == "gs" else args.max_images_cs):
                raise ValueError("选图器返回的图片数量超过配置上限")
            exported = export_selected(selector, result, Path(request["destination"]))
            protocol.write(
                json.dumps({"ok": True, "result": result, "images": exported}, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        except Exception:
            protocol.write(json.dumps({"ok": False, "fatal": False, "error": traceback.format_exc()}) + "\n")
    return 0


class CaseSelectionError(RuntimeError):
    pass


class SelectorWorker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.process = None
        self.log_thread = None
        self.stderr_tail = deque(maxlen=25)

    def __enter__(self) -> SelectorWorker:
        for path in (self.args.selector, self.args.selector_python):
            if not path.is_file():
                raise FileNotFoundError("选图器或其Python环境不存在：" + str(path))
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for key in list(env):
            if key.startswith(("PYDEVD_", "DEBUGPY_")):
                env.pop(key, None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        library_dirs = [
            "/home/maxlin/TensorRT-8.2.0.6/lib",
            "/usr/local/cuda-11.1/lib64",
            "/home/maxlin/onnxruntime-linux-x64-gpu-1.9.0/lib",
        ]
        env["LD_LIBRARY_PATH"] = ":".join(library_dirs + [env.get("LD_LIBRARY_PATH", "")])
        command = [
            str(self.args.selector_python),
            "-B",
            "-u",
            str(Path(__file__).resolve()),
            "--selector-worker",
            "--selector",
            str(self.args.selector),
            "--workers",
            str(self.args.workers),
            "--max-images-gs",
            str(self.args.max_images_gs),
            "--max-images-cs",
            str(self.args.max_images_cs),
        ]
        LOGGER.info("启动独立选图进程：%s；选图器：%s", self.args.selector_python, self.args.selector)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )

        def drain_stderr() -> None:
            for line in self.process.stderr:
                line = line.rstrip()
                if line:
                    self.stderr_tail.append(line)
                    LOGGER.info("[selector] %s", line)

        self.log_thread = threading.Thread(target=drain_stderr, daemon=True)
        self.log_thread.start()
        return self

    def select(self, case: Case, destination: Path) -> dict[str, Any]:
        request = {
            "action": "select",
            "scope": case.scope,
            "case_dir": str(case.directory),
            "destination": str(destination),
        }
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except OSError as exc:
            raise RuntimeError("选图进程通信失败：" + "\n".join(self.stderr_tail)) from exc
        if not line:
            self.process.wait()
            self.log_thread.join(timeout=2)
            raise RuntimeError(
                "选图进程退出，code={}：\n{}".format(self.process.returncode, "\n".join(self.stderr_tail))
            )
        response = json.loads(line)
        if not response["ok"]:
            error = RuntimeError if response.get("fatal") else CaseSelectionError
            raise error(response["error"])
        return response

    def __exit__(self, *_exc: Any) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"action":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            if self.log_thread is not None:
                self.log_thread.join(timeout=2)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None:
                    stream.close()


def output_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    resolved = candidate.resolve()
    if root not in resolved.parents or candidate.is_symlink():
        raise ValueError("中间缓存中的文件路径越界或为符号链接：" + relative)
    return candidate


def image_hint(prediction: dict[str, Any], scope: str) -> dict[str, Any]:
    """Extract model predictions only; missing inference is never a negative."""
    accepted_field, position_field = {
        "gs": ("gqc36_accepted", "part"),
        "cs": ("QC01_accepted", "cecum_class"),
    }[scope]
    accepted = prediction.get(accepted_field)
    lesion_detected = prediction.get("lesion_detected")
    if not isinstance(accepted, bool) or not isinstance(lesion_detected, bool):
        raise ValueError("选图记录缺少部位/病变模型结果，不能从来源、文件名或报告猜测图片提示")
    position = prediction.get(position_field) if accepted else None
    if position is not None:
        if not isinstance(position, str) or not position.strip():
            raise ValueError("选图记录的预测部位必须是非空字符串或null")
        position = position.strip()
    return {"position": position, "lesion_detected": lesion_detected}


def image_hints_from_selection(record: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Match selected predictions by source path, preserving exported image order."""
    selected = selection["selected"]
    by_source = {str(Path(item["path"]).resolve()): item for item in selected}
    if len(by_source) != len(selected):
        raise ValueError("选图记录中原图路径重复：" + record["case_key"])
    images, image_info = record["images"], record["image_info"]
    if len(images) != len(image_info) or len(images) != len(selected):
        raise ValueError("图片与选图记录的数量不匹配：" + record["case_key"])
    hints = []
    seen = set()
    for image, info in zip(images, image_info):
        source = str(Path(info["source"]).resolve())
        if Path(image).name != info["file_name"] or source not in by_source or source in seen:
            raise ValueError("无法将图片与选图预测结果一一对应：" + record["case_key"])
        seen.add(source)
        hints.append(image_hint(by_source[source], record["scope"]))
    return hints


def load_image_hints(record: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    """Read new cached hints or recover old-cache hints from selection.json."""
    hints = record.get("image_hints")
    if hints is None:
        selection = read_json(output_file(output, record["selection_file"]))
        hints = image_hints_from_selection(record, selection)
    return hints


def build_case_prompt(scope: str, image_count: int, image_hints: Optional[list[dict[str, Any]]] = None) -> str:
    """Shared prompt layout; passing no hints preserves the original input exactly."""
    exam = {"gs": "胃镜", "cs": "肠镜"}[scope]
    prompt = f"请综合该病例的{exam}图像，描述镜检所见，并给出诊断结论。"
    if image_hints is None:
        return prompt + "\n<image>" * image_count
    if len(image_hints) != image_count:
        raise ValueError("逐图提示的数量与图片数量不一致")
    prompt += "\n以下信息为辅助模型预测，可能存在误差。"
    for number, hint in enumerate(image_hints, 1):
        position, detected = hint["position"], hint["lesion_detected"]
        if position is not None and (
            not isinstance(position, str) or not position.strip() or "<image>" in position or "\n" in position
        ):
            raise ValueError("逐图提示包含非法部位文字")
        if not isinstance(detected, bool):
            raise ValueError("逐图病变提示必须是明确的模型检测结果，不能缺省为阴性")
        lesion = "；检出疑似病变" if detected else ""
        prompt += f"\n\n图{number}：预测部位：{position or '未确定'}{lesion}。\n<image>"
    return prompt


def build_sample(
    record: dict[str, Any], *, include_thinking: bool = False, include_image_hints: bool = False
) -> Optional[dict[str, Any]]:
    images = record["images"]
    if not images:
        raise ValueError("病例没有选中图片：" + record["case_key"])
    prompt = build_case_prompt(record["scope"], len(images), record["image_hints"] if include_image_hints else None)
    answer = json.dumps(record["gt"], ensure_ascii=False, indent=2, allow_nan=False)
    if include_thinking:
        block = thinking_block(record.get("think", ""))
        if block is None:
            return None
        answer = block + "\n\n" + answer
    if "<image>" in answer:
        raise ValueError("报告或think包含<image>占位符，会破坏多模态图文对齐")
    return {
        "images": list(images),
        "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}],
    }


def prepare_case(case: Case, output: Path, worker: SelectorWorker) -> dict[str, Any]:
    matches = report_keyword_matches(case.directory / "reports")
    if matches:
        raise ReportKeywordExcluded(matches)
    raw_gt = read_json(case.report_path)
    gt = normalize_report(raw_gt)
    if "镜检所见" not in raw_gt:
        LOGGER.warning("[%s] 已将顶层部位字段整理到镜检所见；原始GT保留", case.key)
    think = case.think_path.read_bytes().decode("utf-8-sig") if case.think_path.is_file() else ""
    think_error = None
    try:
        thinking_block(think)
    except ValueError as exc:
        think_error = str(exc)
        LOGGER.warning("[%s] think格式异常，仅生成普通版：%s；原文仍保存", case.key, think_error)
    if not think.strip():
        LOGGER.warning("[%s] 缺少或空白think，只纳入普通报告版", case.key)
    target = output / case.key
    # A rejected or interrupted case must not leave half a case in the dataset.
    with tempfile.TemporaryDirectory(prefix=".preparing_case_", dir=str(output)) as temporary:
        staging = Path(temporary) / "case"
        staging.mkdir()
        # Reports remain byte-identical; never pass report text into selection.
        shutil.copytree(str(case.directory / "reports"), str(staging / "reports"))
        if (case.directory / "info.txt").is_file():
            shutil.copy2(str(case.directory / "info.txt"), str(staging / "info.txt"))
        response = worker.select(case, staging / "images")
        atomic_json(staging / "selection.json", response["result"])
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
    images = [(target / "images" / image["file_name"]).relative_to(output).as_posix() for image in response["images"]]
    record = {
        "schema_version": SCHEMA_VERSION,
        "case_key": case.key,
        "case_id": case.directory.name,
        "scope": case.scope,
        "source_dir": str(case.directory),
        "images": images,
        "image_info": response["images"],
        "gt_raw": raw_gt,
        "gt": gt,
        "think": think,
        "think_error": think_error,
        "reports_dir": (target / "reports").relative_to(output).as_posix(),
        "selection_file": (target / "selection.json").relative_to(output).as_posix(),
        "selection_counts": response["result"]["counts"],
        "coverage": response["result"].get("coverage"),
    }
    record["image_hints"] = image_hints_from_selection(record, response["result"])
    build_sample(record)
    return record


@contextlib.contextmanager
def atomic_array(path: Path) -> Iterator[TextIO]:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("[\n")
        yield stream
        stream.write("\n]\n")
    os.replace(str(temporary), str(path))


def write_sample(stream: TextIO, sample: dict[str, Any], index: int) -> None:
    if index:
        stream.write(",\n")
    # Image lists and each message stay on one line; newlines in the response
    # remain escaped, so editors can fold one complete case at a time.
    stream.write(
        '  {\n    "images": ' + json.dumps(sample["images"], ensure_ascii=False) + ',\n    "conversations": [\n'
    )
    stream.write(",\n".join("      " + json.dumps(message, ensure_ascii=False) for message in sample["conversations"]))
    stream.write("\n    ]\n  }")


def cache_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if record.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("不支持的病例缓存版本")
                yield record
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"病例缓存第{number}行损坏：{exc}") from exc


def confirm_first_sample(record: dict[str, Any], *, include_image_hints: bool = False) -> bool:
    """Show actual conversations once, requiring approval before the next case."""
    sections = [
        "首个有效样本：" + record["case_key"],
        "images:\n" + json.dumps(record["images"], ensure_ascii=False, indent=2),
    ]
    hint_label = "有提示" if include_image_hints else "无提示"
    for name, thinking in (("普通版", False), ("think版", True)):
        sample = (
            None
            if thinking and record.get("think_error")
            else build_sample(record, include_thinking=thinking, include_image_hints=include_image_hints)
        )
        sections.append(f"========== {name}（{hint_label}） conversations ==========")
        if sample is None:
            sections.append("本病例没有可用的think，仅纳入普通版。")
            continue
        for message in sample["conversations"]:
            sections.append(message["from"] + ":\n" + message["value"])
    # One continuous terminal block without prefixes; the file log still
    # records every line with timestamps using its regular formatter.
    LOGGER.info("\n%s\n", "\n\n".join(sections), extra={"plain_output": True})
    return confirm_in_terminal("确认首个样本的对话内容正确，并继续生成全部剩余病例？[y/N]: ")


def write_conversations(output: Path, *, include_image_hints: bool = False) -> dict[str, Any]:
    normal_path = output / "all_conversations.json"
    think_path = output / "all_conversations_think.json"
    counts = Counter({"cases": 0, "images": 0, "thinking_cases": 0, "excluded_report_keyword": 0})
    excluded_reports = []
    by_scope = Counter()
    seen_keys = set()
    with atomic_array(normal_path) as normal_stream, atomic_array(think_path) as think_stream:
        for record in cache_records(output / CACHE_NAME):
            if record["case_key"] in seen_keys:
                raise ValueError("缓存包含重复病例：" + record["case_key"])
            seen_keys.add(record["case_key"])
            matches = cached_report_keyword_matches(record, output)
            if matches:
                counts["excluded_report_keyword"] += 1
                excluded_reports.append({"case": record["case_key"], "matches": matches})
                LOGGER.info("[%s] 导出对话时跳过：%s", record["case_key"], ReportKeywordExcluded(matches))
                continue
            for image in record["images"]:
                if not output_file(output, image).is_file():
                    raise FileNotFoundError("缓存引用的训练图片不存在：" + image)
            if include_image_hints:
                record["image_hints"] = load_image_hints(record, output)
            write_sample(normal_stream, build_sample(record, include_image_hints=include_image_hints), counts["cases"])
            counts["cases"] += 1
            counts["images"] += len(record["images"])
            thought_sample = (
                None
                if record.get("think_error")
                else build_sample(record, include_thinking=True, include_image_hints=include_image_hints)
            )
            if thought_sample:
                write_sample(think_stream, thought_sample, counts["thinking_cases"])
                counts["thinking_cases"] += 1
            by_scope[record["scope"]] += 1
    if not counts["cases"]:
        raise ValueError("没有可导出的病例，请检查failures.jsonl和generation.log")
    info_path = output / "dataset_info.json"
    info = {}
    for name, path in (("max_v1_case", normal_path), ("max_v1_case_think", think_path)):
        info[name] = {
            "file_name": path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        }
    atomic_json(info_path, info)
    summary = {
        **dict(counts),
        "cases_by_scope": dict(by_scope),
        "conversations": normal_path.name,
        "thinking_conversations": think_path.name,
        "excluded_report_keywords": list(EXCLUDED_REPORT_KEYWORDS),
        "excluded_report_cases": excluded_reports,
        "with_image_hints": include_image_hints,
    }
    LOGGER.info(
        "对话已保存：%s",
        json.dumps(
            {key: value for key, value in summary.items() if key != "excluded_report_cases"}, ensure_ascii=False
        ),
    )
    return summary


def run_generation(args: argparse.Namespace, output: Path, cases: Sequence[Case]) -> int:
    """Generate into the empty output directory prepared by main()."""
    configure_logging(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "discovered": len(cases),
        "source_roots": {"gs": str(args.gs_root.resolve()), "cs": str(args.cs_root.resolve())},
        "selector": str(args.selector.resolve()),
        "selector_python": str(args.selector_python),
        "selector_sha256": hashlib.sha256(args.selector.read_bytes()).hexdigest(),
        "max_images": {"gs": args.max_images_gs, "cs": args.max_images_cs},
        "workers": args.workers,
        "with_image_hints": args.with_image_hints,
        "first_sample_confirmed": False,
        "excluded_report_keywords": list(EXCLUDED_REPORT_KEYWORDS),
    }
    atomic_json(output / "manifest.json", manifest)
    counts = Counter({"prepared": 0, "skipped": 0, "excluded_report_keyword": 0})
    excluded_reports = []
    try:
        from tqdm import tqdm

        with contextlib.ExitStack() as stack:
            cache = stack.enter_context((output / CACHE_NAME).open("w", encoding="utf-8"))
            failures = stack.enter_context((output / "failures.jsonl").open("w", encoding="utf-8"))
            worker = stack.enter_context(SelectorWorker(args))
            progress = stack.enter_context(tqdm(cases, desc="病例选图与生成", unit="例", dynamic_ncols=True))
            for case in progress:
                try:
                    if not (case.directory / "images").is_dir():
                        raise ValueError("缺少images目录")
                    record = prepare_case(case, output, worker)
                except (FileNotFoundError, ValueError, CaseSelectionError) as exc:
                    counts["skipped"] += 1
                    failure = {"case": case.key, "source": str(case.directory), "error": str(exc)}
                    if isinstance(exc, ReportKeywordExcluded):
                        counts["excluded_report_keyword"] += 1
                        failure.update(reason="excluded_report_keyword", matches=exc.matches)
                        excluded_reports.append({"case": case.key, "matches": exc.matches})
                    LOGGER.warning("[%s] 已跳过：%s", case.key, exc)
                    failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failures.flush()
                    continue
                cache.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                cache.flush()
                counts["prepared"] += 1
                LOGGER.info(
                    "[%s] 输入%d张 → 保存%d张；已完成%d/%d例",
                    case.key,
                    record["selection_counts"]["input"],
                    len(record["images"]),
                    counts["prepared"],
                    len(cases),
                )
                if counts["prepared"] == 1:
                    progress.clear()
                    if not confirm_first_sample(record, include_image_hints=args.with_image_hints):
                        manifest["status"] = "cancelled_by_user"
                        LOGGER.info("未确认首个样本，已停止；保留首例及中间记录，不生成最终对话文件。")
                        return 0
                    manifest["first_sample_confirmed"] = True
                    LOGGER.info("首个样本已确认，连续生成剩余全部病例，不再逐例询问。")
                    progress.refresh()
        summary = write_conversations(output, include_image_hints=args.with_image_hints)
        manifest.update(summary)
        manifest["status"] = "completed_with_skips" if counts["skipped"] else "completed"
        return 0
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        LOGGER.error("生成未完成：%s", traceback.format_exc())
        raise
    finally:
        manifest.update(dict(counts))
        manifest["excluded_report_cases"] = excluded_reports
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        atomic_json(output / "manifest.json", manifest)


# 常用配置集中放在 parser 前；直接运行或 VS Code 调试均使用这些默认值。
DEFAULT_GS_ROOT = Path("/media/maxlin/SATA/ReportData/D4/胃镜_拆分")
DEFAULT_CS_ROOT = Path("/media/maxlin/SATA/ReportData/D4/肠镜_拆分")
DEFAULT_OUTPUT_ROOT = Path("/media/maxlin/SATA/ReportData/Max_V1_case_dataV2")
DEFAULT_SELECTOR = Path("/home/maxlin/桌面/innoaitools/inference/report/report_images_selector.py")
DEFAULT_SELECTOR_PYTHON = Path("/home/maxlin/anaconda3/bin/python")
DEFAULT_MAX_IMAGES_GS = 15
DEFAULT_MAX_IMAGES_CS = 12
DEFAULT_WORKERS = 2
DEFAULT_WITH_IMAGE_HINTS = False  # True时普通/think版均带逐图提示；文件名和数据集注册名不变。
EXCLUDED_REPORT_KEYWORDS = ("超声",)  # reports目录任意JSON/TXT命中即整例排除，含think。
CACHE_NAME = "prepared_cases.jsonl"
SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="病例级多图数据全部生成；非空目录须确认清空，首个有效样本的对话须确认后才继续。原始数据只读。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="病例数据集输出目录")
    parser.add_argument("--gs-root", type=Path, default=DEFAULT_GS_ROOT, help="胃镜病例根目录")
    parser.add_argument("--cs-root", type=Path, default=DEFAULT_CS_ROOT, help="肠镜病例根目录")
    parser.add_argument("--scope", choices=("all", "gs", "cs"), default="all", help="生成的病例类型")
    parser.add_argument(
        "--with-image-hints",
        action="store_true",
        default=DEFAULT_WITH_IMAGE_HINTS,
        help="在普通/think版输入中加入逐图预测部位+疑似病变提示；文件名不变，用输出目录区分；训练须换tokenized_path",
    )
    parser.add_argument("--max-images-gs", type=int, default=DEFAULT_MAX_IMAGES_GS, help="每例胃镜图片上限，至少12")
    parser.add_argument("--max-images-cs", type=int, default=DEFAULT_MAX_IMAGES_CS, help="每例肠镜图片上限，至少10")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="选图器病例内线程数；病例间串行，模型复用"
    )
    parser.add_argument("--selector", type=Path, default=DEFAULT_SELECTOR, help="现有病例选图器脚本")
    parser.add_argument(
        "--selector-python", type=Path, default=DEFAULT_SELECTOR_PYTHON, help="装有ONNX/TensorRT的独立Python >=3.8"
    )
    parser.add_argument("--limit", type=int, default=0, help="每种镜检最多取多少个具备GT的病例；0为全部")
    parser.add_argument("--dry-run", action="store_true", help="只统计病例，不加载模型、不写任何输出")
    parser.add_argument("--selector-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selector_worker:
        return selector_worker(args)
    output = args.output_root.expanduser().resolve()
    if args.limit < 0 or args.workers < 1:
        parser.error("limit须>=0，workers须>=1")
    if args.max_images_gs < 12 or args.max_images_cs < 10:
        parser.error("胃镜max-images-gs至少12，肠镜max-images-cs至少10，须保留选图器保底位")
    validate_output(output, [args.gs_root, args.cs_root])
    args.selector = args.selector.expanduser().resolve()
    args.selector_python = args.selector_python.expanduser().resolve()
    if not args.dry_run:
        # Check the input/runtime paths before any cleanup. Ask about an
        # existing output before scanning cases, logging or starting workers.
        for scope, source in (("gs", args.gs_root), ("cs", args.cs_root)):
            if args.scope in ("all", scope) and not source.expanduser().is_dir():
                raise FileNotFoundError("原始病例目录不存在：" + str(source))
        for path in (args.selector, args.selector_python):
            if not path.is_file():
                raise FileNotFoundError("选图器或Python环境不存在：" + str(path))
        if not prepare_output_directory(output):
            return 0
    cases = collect_cases(args)
    print(
        f"本次计划处理{len(cases)}个病例；胃镜最多{args.max_images_gs}张/例，肠镜最多{args.max_images_cs}张/例。",
        flush=True,
    )
    if args.dry_run:
        return 0
    if not cases:
        raise ValueError("未发现可处理病例")
    return run_generation(args, output, cases)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中止；独立选图进程已关闭。", file=sys.stderr)
        raise SystemExit(130) from None
