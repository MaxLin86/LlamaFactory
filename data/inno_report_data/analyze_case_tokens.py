#!/usr/bin/env python3
"""Plot case token distributions with the current local model and image settings.

Run in the llama_factory environment::

    python data/inno_report_data/analyze_case_tokens.py

If prepared_cases.jsonl exists, use its actual selected images and crop sizes.
Otherwise read source reports one at a time and explicitly mark total lengths
as PRE-SELECTION ESTIMATES: up to 15/12 images, one 512x512 image budget each
with the current default YAML. Final selection may choose fewer images and
different shapes. The report/think response token counts are exact in both
modes. Nothing is tokenized with cutoff_len truncation.

The image geometry calculation follows this checkout's Qwen3VLPlugin and the
installed Transformers smart_resize, without decoding full-size image pixels.
It loads the tokenizer and processor only; it never loads model weights, runs
the selector, copies images, or modifies the source reports.

The output includes PNG histograms, per-case CSV and summary.json. For example,
to measure an existing small dataset instead::

    python data/inno_report_data/analyze_case_tokens.py \
      --dataset-root /tmp/case_dataset_test --output-dir /tmp/case_token_test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from data_generate_case_images import (
    CACHE_NAME,
    DEFAULT_CS_ROOT,
    DEFAULT_GS_ROOT,
    DEFAULT_MAX_IMAGES_CS,
    DEFAULT_MAX_IMAGES_GS,
    DEFAULT_OUTPUT_ROOT,
    build_sample,
    cache_records,
    cached_report_keyword_matches,
    discover_cases,
    normalize_report,
    read_json,
    report_keyword_matches,
    thinking_block,
)


class TokenCounter:
    """Use real chat templates/tokenizer and the Qwen3-VL image-grid geometry."""

    def __init__(self, config: dict[str, Any]):
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams import DataArguments

        self.model_path = config["model_name_or_path"]
        model_config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        if model_config.model_type != "qwen3_vl":
            raise ValueError("本统计器目前仅验证过qwen3_vl，不能套用到其他模型的图像Token几何")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.min_pixels = int(config.get("image_min_pixels", 32 * 32))
        self.max_pixels = int(config.get("image_max_pixels", 768 * 768))
        self.processor.image_min_pixels = self.min_pixels
        self.processor.image_max_pixels = self.max_pixels
        self.templates = {
            False: get_template_and_fix_tokenizer(self.tokenizer, DataArguments(template="qwen3_vl_nothink")),
            True: get_template_and_fix_tokenizer(
                self.tokenizer, DataArguments(template="qwen3_vl", enable_thinking=True)
            ),
        }
        image_processor = self.processor.image_processor
        self.patch_size = image_processor.patch_size
        self.merge_size = image_processor.merge_size
        self.factor = self.patch_size * self.merge_size
        self.reference_edge = math.isqrt(self.max_pixels)
        self.reference_tokens = self.image_tokens(self.reference_edge, self.reference_edge)

    def image_tokens(self, width: int, height: int) -> int:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        plugin = self.templates[False].mm_plugin
        width, height = plugin._get_qwen_video_size_after_regularization(
            width, height, self.max_pixels, self.min_pixels
        )
        size = self.processor.image_processor.size
        minimum = size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
        maximum = size["longest_edge"] if isinstance(size, dict) else size.longest_edge
        resized_height, resized_width = smart_resize(
            height=height, width=width, factor=self.factor, min_pixels=minimum, max_pixels=maximum
        )
        return (resized_height // self.factor) * (resized_width // self.factor)

    def count(self, sample: dict[str, Any], visual_tokens: list[int], thinking: bool) -> dict[str, int]:
        template = self.templates[thinking]
        plugin = template.mm_plugin
        # Count the exact text/chat template using one image token per image.
        # Repeating a registered special token N times then adds exactly N ids.
        stub = plugin.vision_bos_token + plugin.image_token + plugin.vision_eos_token
        messages = [
            {
                "role": "user" if item["from"] == "human" else "assistant",
                "content": item["value"].replace("<image>", stub),
            }
            for item in sample["conversations"]
        ]
        source, response = template.encode_oneturn(self.tokenizer, messages)
        overhead = len(source) - len(visual_tokens)
        eos = int(template.efficient_eos)
        return {
            "total": overhead + sum(visual_tokens) + len(response) + eos,
            "response": len(response) + eos,
            "text_and_special": overhead + len(response) + eos,
        }


def source_records(args: argparse.Namespace, exclusions: list[dict[str, str]]) -> Iterator[dict[str, Any]]:
    """Source-only preflight; selected image counts/shapes are not yet known."""
    for scope, source in (("gs", args.gs_root), ("cs", args.cs_root)):
        root = source.expanduser().resolve()
        for case in discover_cases(scope, root):
            matches = report_keyword_matches(case.directory / "reports")
            if matches:
                exclusions.append({"case": case.key, "reason": "excluded_report_keyword"})
                continue
            if not case.report_path.is_file():
                exclusions.append({"case": case.key, "reason": "missing_struct"})
                continue
            if not (case.directory / "images").is_dir():
                exclusions.append({"case": case.key, "reason": "missing_images_directory"})
                continue
            try:
                raw = read_json(case.report_path)
                gt = normalize_report(raw)
                think = case.think_path.read_text(encoding="utf-8-sig") if case.think_path.is_file() else ""
            except (OSError, ValueError, UnicodeError) as exc:
                # Use the same GT validation as full generation. A struct file
                # existing does not guarantee a nonempty usable training GT.
                exclusions.append({"case": case.key, "reason": "invalid_gt_or_text", "detail": str(exc)})
                continue
            try:
                thinking_block(think)
                think_error = None
            except ValueError as exc:
                think_error = str(exc)
            available = sum(
                path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                for path in (case.directory / "images").iterdir()
            )
            maximum = args.max_images_gs if scope == "gs" else args.max_images_cs
            count = min(available, maximum)
            if not count:
                exclusions.append({"case": case.key, "reason": "no_image_files"})
                continue
            yield {
                "case_key": case.key,
                "case_id": case.directory.name,
                "scope": scope,
                "images": [f"unselected_{i}" for i in range(count)],
                "gt": gt,
                "think": think,
                "think_error": think_error,
                "source_images": available,
            }


def distribution(values: list[int]) -> dict[str, Any]:
    import numpy as np

    if not values:
        return {"count": 0}
    data = np.asarray(values)
    return {
        "count": len(values),
        "min": int(data.min()),
        "max": int(data.max()),
        "mean": round(float(data.mean()), 2),
        "p50": round(float(np.percentile(data, 50)), 2),
        "p90": round(float(np.percentile(data, 90)), 2),
        "p95": round(float(np.percentile(data, 95)), 2),
        "p99": round(float(np.percentile(data, 99)), 2),
        "above_cutoff": {str(limit): int((data > limit).sum()) for limit in (2048, 4096, 8192, 16384)},
    }


def plot_histograms(
    series: dict[str, dict[str, list[int]]], output: Path, estimated: bool, config: dict[str, Any]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager

    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    labels = {"gs": "胃镜", "cs": "肠镜"}
    colors = {"gs": "#2563eb", "cs": "#e27718"}
    for field, filename, title in (
        ("total", "case_token_histogram.png", "病例完整序列 Token（图像 + 问题 + 回答 + 模板）"),
        ("response", "response_token_histogram.png", "病例回答 Token：报告 / 思考与报告（文本精确统计）"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for axis, variant in zip(axes, ("normal", "think")):
            all_values = [v for scope in labels for v in series[scope][f"{variant}_{field}"]]
            bins = np.linspace(0, max(all_values or [1]) * 1.02, 36)
            for scope in labels:
                values = series[scope][f"{variant}_{field}"]
                if values:
                    axis.hist(
                        values, bins=bins, alpha=0.6, color=colors[scope], label=f"{labels[scope]}：{len(values)}例"
                    )
            if field == "total":
                axis.axvline(config.get("cutoff_len", 2048), color="#b91c1c", linestyle="--", label="当前 cutoff_len")
            axis.set_title("普通报告版" if variant == "normal" else "Think + 报告版")
            axis.set_xlabel("Tokens / 病例（未截断）")
            axis.set_ylabel("病例数")
            axis.grid(axis="y", alpha=0.2)
            axis.legend()
        note = (
            "选图前预算估算；图像按配置最大像素的方形参考预算，非实际选图结果"
            if estimated and field == "total"
            else "精确统计；文本使用当前病例对话构造及本地模型 tokenizer"
        )
        figure.suptitle(title + "\n" + note, fontsize=13)
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
    if not estimated:
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for scope in labels:
            axis.hist(series[scope]["image"], bins=24, alpha=0.6, color=colors[scope], label=labels[scope])
        axis.set(title="单张入选图片视觉 Token（按实际裁图尺寸）", xlabel="视觉 Tokens / 图片", ylabel="图片数")
        axis.legend()
        figure.savefig(output / "image_token_histogram.png", dpi=160)
        plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from tqdm import tqdm

    config = OmegaConf.to_container(OmegaConf.load(args.train_config), resolve=True)
    counter = TokenCounter(config)
    root, output = args.dataset_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    estimated = not (root / CACHE_NAME).is_file()
    exclusions: list[dict[str, str]] = []
    records = source_records(args, exclusions) if estimated else cache_records(root / CACHE_NAME)
    series: dict[str, dict[str, list[int]]] = {"gs": defaultdict(list), "cs": defaultdict(list)}
    counts = Counter()
    thinking_issues = []
    columns = [
        "case_key",
        "scope",
        "measurement",
        "images",
        "visual_tokens",
        "normal_response_tokens",
        "normal_text_and_special_tokens",
        "normal_total_tokens",
        "think_response_tokens",
        "think_text_and_special_tokens",
        "think_total_tokens",
    ]
    with (output / "case_token_lengths.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in tqdm(records, desc="病例Token统计", unit="例", mininterval=2):
            if not estimated and cached_report_keyword_matches(record, root):
                exclusions.append({"case": record["case_key"], "reason": "excluded_report_keyword"})
                continue
            scope = record["scope"]
            if estimated:
                visual = [counter.reference_tokens] * len(record["images"])
            else:
                visual = [counter.image_tokens(*info["saved_size"]) for info in record["image_info"]]
                if len(visual) != len(record["images"]):
                    raise ValueError("缓存图片与尺寸数量不一致：" + record["case_key"])
                series[scope]["image"].extend(visual)
            normal_sample = build_sample(record)
            normal = counter.count(normal_sample, visual, False)
            row = {
                "case_key": record["case_key"],
                "scope": scope,
                "measurement": "preselection_estimate" if estimated else "selected_image_exact",
                "images": len(visual),
                "visual_tokens": sum(visual),
            }
            for metric, value in normal.items():
                row[f"normal_{metric}_tokens"] = value
                series[scope]["normal_" + metric].append(value)
            counts[scope] += 1
            counts["normal"] += 1
            thought_sample = None if record.get("think_error") else build_sample(record, include_thinking=True)
            if thought_sample:
                # Validate that only the thought was added, never a different
                # image list, user prompt or final GT JSON.
                same_input = (
                    normal_sample["images"] == thought_sample["images"]
                    and normal_sample["conversations"][0] == thought_sample["conversations"][0]
                )
                same_gt = (
                    json.loads(thought_sample["conversations"][1]["value"].split("</think>", 1)[1]) == record["gt"]
                )
                if not same_input or not same_gt:
                    raise ValueError("普通版/think版内容对应关系错误：" + record["case_key"])
                thought = counter.count(thought_sample, visual, True)
                for metric, value in thought.items():
                    row[f"think_{metric}_tokens"] = value
                    series[scope]["think_" + metric].append(value)
                counts["think"] += 1
            else:
                thinking_issues.append(
                    {"case": record["case_key"], "reason": record.get("think_error") or "missing_or_empty_think"}
                )
            writer.writerow(row)
    summary = {
        "measurement": "preselection_estimate" if estimated else "selected_image_exact",
        "dataset_root": str(root),
        "train_config": str(args.train_config.resolve()),
        "model": counter.model_path,
        "image_min_pixels": counter.min_pixels,
        "image_max_pixels": counter.max_pixels,
        "patch_size": counter.patch_size,
        "merge_size": counter.merge_size,
        "effective_pixels_per_token": counter.factor**2,
        "reference_square_edge": counter.reference_edge,
        "reference_image_tokens": counter.reference_tokens,
        "image_budget_assumption": "min(source image count, scope image limit); one reference square budget per image; not a strict upper bound"
        if estimated
        else None,
        "counts": dict(counts),
        "thinking_issues": thinking_issues,
        "source_exclusions": exclusions,
        "source_exclusion_counts": dict(Counter(item["reason"] for item in exclusions)),
        "content_relation": "Every thinking sample has the same images, question, and complete final GT as its normal counterpart; verified from current builder, not from ungenerated full JSON files.",
        "by_scope": {
            scope: {key: distribution(values) for key, values in metrics.items()} for scope, metrics in series.items()
        },
        "all": {
            key: distribution([value for scope in series for value in series[scope][key]])
            for key in ("normal_total", "think_total", "normal_response", "think_response", "image")
        },
    }
    plot_histograms(series, output, estimated, config)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "measurement": summary["measurement"],
                "counts": summary["counts"],
                "all": summary["all"],
                "output_dir": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "examples/train_lora/qwen3vl_lora_sft_inno.yaml"
DEFAULT_STATS_DIR = Path(__file__).with_name("case_token_stats")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--gs-root", type=Path, default=DEFAULT_GS_ROOT)
    parser.add_argument("--cs-root", type=Path, default=DEFAULT_CS_ROOT)
    parser.add_argument("--max-images-gs", type=int, default=DEFAULT_MAX_IMAGES_GS)
    parser.add_argument("--max-images-cs", type=int, default=DEFAULT_MAX_IMAGES_CS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
