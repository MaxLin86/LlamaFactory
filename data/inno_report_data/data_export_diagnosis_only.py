#!/usr/bin/env python3
r"""Export diagnosis-only case conversations without reselecting images.

从病例数据的中间记录导出「仅诊断结论」普通版，不重新选图或运行模型。

直接运行（默认路径集中在文件末尾）：
    python data/inno_report_data/data_export_diagnosis_only.py

指定其他源/目标目录：
    python data/inno_report_data/data_export_diagnosis_only.py \
        --dataset-root /media/maxlin/SATA/ReportData/Max_V1_case_dataV2 \
        --output-root /media/maxlin/SATA/ReportData/Max_V1_case_dataV2_diagonly

沿用源 manifest.json 的 with_image_hints 设置，以及原有病例/图片顺序。
只将问题改为「请综合该病例的胃镜/肠镜图像，给出诊断结论。」，
回答仅保留原 GT 的「诊断结论」，不改写、映射或归并诊断。
不导出原 think 文本：它不属于本次仅结论普通版训练目标。

输出 all_conversations.json、dataset_info.json、prepared_cases.jsonl、
export_summary.json、generation.log 和 gs/cs 下的选中图片。
图片优先使用硬链接（不依赖源路径继续存在），跨文件系统时退回复制。
硬链接共享文件内容，因此两边图片均只读使用，切勿原地编辑。
原始报告、V2 JSON、图片内容均不会改动。
目标非空时必须交互确认清空；拒绝、EOF 或非交互终端均不继续。
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from data_generate_case_images import (
    CACHE_NAME,
    EXCLUDED_REPORT_KEYWORDS,
    SCHEMA_VERSION,
    ConsoleFormatter,
    TimestampFormatter,
    atomic_array,
    atomic_json,
    build_sample,
    cache_records,
    cached_report_keyword_matches,
    load_image_hints,
    output_file,
    prepare_output_directory,
    read_json,
    validate_output,
    write_sample,
)


LOGGER = logging.getLogger("export_diagnosis_only")


def diagnosis_question(scope: str) -> str:
    exam = {"gs": "胃镜", "cs": "肠镜"}[scope]
    return f"请综合该病例的{exam}图像，给出诊断结论。"


def build_diagnosis_sample(record: dict[str, Any], *, include_image_hints: bool = False) -> dict[str, Any]:
    """Keep diagnosis values verbatim, including the existing JSON answer layout."""
    diagnosis = record["gt"]["诊断结论"]
    if not diagnosis or (isinstance(diagnosis, str) and not diagnosis.strip()):
        raise ValueError("诊断结论为空：" + record["case_key"])
    reduced = dict(record, gt={"诊断结论": diagnosis})
    sample = build_sample(reduced, include_image_hints=include_image_hints)
    # Keep image tokens and all per-image predictions exactly as the shared builder.
    _, separator, images_and_hints = sample["conversations"][0]["value"].partition("\n")
    sample["conversations"][0]["value"] = diagnosis_question(record["scope"]) + separator + images_and_hints
    return sample


def export_image(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(target))
        return "hardlinked_images"
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS, errno.EMLINK):
            raise
        shutil.copy2(str(source), str(target))
        return "copied_images"


def configure_logging(output: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(str(output / "generation.log"), encoding="utf-8"),
    ):
        handler.setFormatter(TimestampFormatter() if isinstance(handler, logging.FileHandler) else ConsoleFormatter())
        LOGGER.addHandler(handler)


def source_manifest(source: Path) -> dict[str, Any]:
    for name in (CACHE_NAME, "manifest.json"):
        if not (source / name).is_file():
            raise FileNotFoundError("缺少源数据文件：" + str(source / name))
    manifest = read_json(source / "manifest.json")
    if manifest.get("status") not in ("completed", "completed_with_skips"):
        raise ValueError("源病例数据尚未生成完成，不能用于本次对照导出")
    if not isinstance(manifest.get("with_image_hints"), bool):
        raise ValueError("源manifest未明确记录with_image_hints，无法确认需要保留的输入形式")
    return manifest


def export_diagnosis_conversations(src_root: Path, dst_root: Path) -> dict[str, Any]:
    """Stream the source cache once; refuse implicit replacement of existing files."""
    if dst_root.is_symlink():
        raise ValueError("输出目录不能是符号链接")
    src_root, dst_root = src_root.resolve(), dst_root.resolve()
    validate_output(dst_root, [src_root])
    manifest = source_manifest(src_root)
    if dst_root.exists() and (not dst_root.is_dir() or any(dst_root.iterdir())):
        raise FileExistsError("目标必须为空；请通过脚本入口确认清空，或使用新目录：" + str(dst_root))
    dst_root.mkdir(parents=True, exist_ok=True)
    configure_logging(dst_root)
    include_hints = manifest["with_image_hints"]
    counts = Counter(cases=0, images=0, hardlinked_images=0, copied_images=0, skipped_excluded_keyword=0)
    by_scope = Counter()
    seen_cases = set()
    digest = hashlib.sha256()
    summary = {
        "mode": "diagnosis_only",
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "source": str(src_root),
        "output": str(dst_root),
        "with_image_hints": include_hints,
        "normal_file": "all_conversations.json",
        "dataset": "max_v1_case_diagnosis_only",
        "excluded_report_keywords": list(EXCLUDED_REPORT_KEYWORDS),
    }
    atomic_json(dst_root / "export_summary.json", summary)
    LOGGER.info("导出仅诊断结论：%s → %s；保留源逐图提示=%s", src_root, dst_root, include_hints)
    try:
        from tqdm import tqdm

        with (src_root / CACHE_NAME).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        summary["source_cache_sha256"] = digest.hexdigest()
        with ExitStack() as stack:
            conversations = stack.enter_context(atomic_array(dst_root / "all_conversations.json"))
            cache = stack.enter_context((dst_root / CACHE_NAME).open("w", encoding="utf-8"))
            records = tqdm(
                cache_records(src_root / CACHE_NAME),
                total=manifest.get("prepared", manifest.get("cases")),
                desc="仅诊断结论导出",
                unit="例",
                dynamic_ncols=True,
                mininterval=2,
            )
            for record in records:
                key = record["case_key"]
                if key in seen_cases:
                    raise ValueError("缓存包含重复病例：" + key)
                seen_cases.add(key)
                if cached_report_keyword_matches(record, src_root):
                    counts["skipped_excluded_keyword"] += 1
                    continue
                images = record["images"]
                if not images or len(images) != len(set(images)):
                    raise ValueError("病例无图片或包含重复图片：" + key)
                if include_hints:
                    record["image_hints"] = load_image_hints(record, src_root)
                sample = build_diagnosis_sample(record, include_image_hints=include_hints)
                reduced = {
                    name: record[name]
                    for name in ("case_key", "case_id", "scope", "source_dir", "images", "image_info", "image_hints")
                    if name in record
                }
                reduced.update(
                    schema_version=SCHEMA_VERSION,
                    task="diagnosis_only",
                    question=diagnosis_question(record["scope"]),
                    gt={"诊断结论": record["gt"]["诊断结论"]},
                )
                for relative in images:
                    if Path(relative).parent.parent.as_posix() != key:
                        raise ValueError("图片路径与病例不对应：" + relative)
                    source, target = output_file(src_root, relative), output_file(dst_root, relative)
                    if not source.is_file():
                        raise FileNotFoundError("源图片不存在：" + str(source))
                    counts[export_image(source, target)] += 1
                write_sample(conversations, sample, counts["cases"])
                cache.write(json.dumps(reduced, ensure_ascii=False, allow_nan=False) + "\n")
                counts["cases"] += 1
                counts["images"] += len(images)
                by_scope[record["scope"]] += 1
                if counts["cases"] % 500 == 0:
                    LOGGER.info("已导出%d例，%d张图片", counts["cases"], counts["images"])
            if not counts["cases"]:
                raise ValueError("没有可导出的病例")
        atomic_json(
            dst_root / "dataset_info.json",
            {
                "max_v1_case_diagnosis_only": {
                    "file_name": "all_conversations.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations", "images": "images"},
                    "tags": {
                        "role_tag": "from",
                        "content_tag": "value",
                        "user_tag": "human",
                        "assistant_tag": "gpt",
                    },
                }
            },
        )
        summary["status"] = "completed"
        LOGGER.info(
            "完成：%d例；胃镜%d例、肠镜%d例；%d张图片",
            counts["cases"],
            by_scope["gs"],
            by_scope["cs"],
            counts["images"],
        )
        return summary
    except BaseException:
        summary["status"] = "interrupted_or_failed"
        LOGGER.exception("导出未完成；没有修改源数据。目标保留中间文件以供排查")
        raise
    finally:
        summary.update(
            dict(counts), cases_by_scope=dict(by_scope), finished_at=datetime.now().astimezone().isoformat()
        )
        atomic_json(dst_root / "export_summary.json", summary)
        for handler in LOGGER.handlers[:]:
            handler.close()
            LOGGER.removeHandler(handler)


# 默认参数放在parser附近；本次只导出普通版，提示开关沿用源manifest。
DEFAULT_SOURCE_ROOT = Path("/media/maxlin/SATA/ReportData/Max_V1_case_dataV2")
DEFAULT_OUTPUT_ROOT = Path("/media/maxlin/SATA/ReportData/Max_V1_case_dataV2_diagonly")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="已完成的源病例数据集")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="仅诊断结论输出目录")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_root.expanduser().is_symlink():
        raise ValueError("输出目录不能是符号链接")
    source, output = args.dataset_root.expanduser().resolve(), args.output_root.expanduser().resolve()
    source_manifest(source)
    validate_output(output, [source])
    if not prepare_output_directory(output):
        return 0
    summary = export_diagnosis_conversations(source, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中止。", file=sys.stderr)
        raise SystemExit(130) from None
