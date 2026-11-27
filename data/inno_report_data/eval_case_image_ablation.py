#!/usr/bin/env python3
"""Compare real/gray images with/without auxiliary hints in 20 validation cases.

在 llama_factory 环境、项目根目录运行：

    CUDA_VISIBLE_DEVICES=0 python data/inno_report_data/eval_case_image_ablation.py

只准备或核查图片、不加载模型、不打开窗口：

    python data/inno_report_data/eval_case_image_ablation.py --prepare-only

复用原病例评测的模型配置和界面，不修改原评测脚本、训练配置或源数据。
首次运行准备固定的20例验证集病例（胃镜10例、肠镜10例），包含之前的4例。
以后复用已有测试包，不重新选例。旧4例测试包不变，新测试包独立保存。
每例依次推理A/B/D/E，右侧同时显示四组回答；左侧展示当前组的真实输入。
A/B保留相同的原始文字提示，分别使用原图、等尺寸灰图；不再进行C组换图。
D/E去掉全部逐图部位、疑似病变提示及其说明，只保留原始基础提问和<image>。
D复用A图片，E复用B图片，图片数量、顺序、分辨率不变，不增加或改写图片文件。
D/E在运行时构造，不增加图片副本；A独立复制原图，B新建等尺寸RGB灰图。
该测试包仅用于图像依赖性诊断，不能用于训练或估算总体准确率。
病例按现有GT覆盖不同任务，固定验证集行号和病例编号，不改选图结果或GT。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import eval_case_images as evaluation


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected_sources() -> dict[int, dict[str, Any]]:
    from datasets import load_from_disk
    from PIL import Image

    root = DATASET_ROOT.resolve()
    dataset = load_from_disk(str(TOKENIZED_ROOT / "validation"), keep_in_memory=False).select_columns(["images"])
    selected = {}
    for index, expected_key, focus in SELECTED_CASES:
        images = evaluation.image_paths(dataset[index]["images"], root)
        key = Path(images[0]).parent.parent.relative_to(root).as_posix()
        if key != expected_key:
            raise ValueError(f"验证集划分已变化：第{index}行应为{expected_key}，实际为{key}；停止，不猜测病例。")
        selected[index] = {
            "validation_index": index,
            "case_key": key,
            "focus": focus,
            "scope": key.split("/", 1)[0],
            "images": images,
        }
    by_key = {record["case_key"]: record for record in selected.values()}
    for sample in evaluation.iter_conversation_samples(root / "all_conversations.json"):
        first = Path(sample["images"][0])
        key = (first.relative_to(root) if first.is_absolute() else first).parent.parent.as_posix()
        if key not in by_key:
            continue
        record = by_key[key]
        if "prompt" in record or evaluation.image_paths(sample["images"], root) != record["images"]:
            raise ValueError(f"原始对话重复或图片顺序与验证集缓存不一致：{key}")
        messages = sample["conversations"]
        if len(messages) != 2 or [message["from"] for message in messages] != ["human", "gpt"]:
            raise ValueError(f"原始病例不是一问一答：{key}")
        record.update(prompt=messages[0]["value"], gt=messages[1]["value"])
    for record in selected.values():
        count = len(record["images"])
        if not count or record.get("prompt", "").count("<image>") != count:
            raise ValueError(f"缺少原始提问或图片占位符数量不一致：{record['case_key']}")
        record["sizes"] = []
        for value in record["images"]:
            with Image.open(value) as image:
                record["sizes"].append(image.size)
    return selected


def prepare_package() -> dict[str, Any]:
    from PIL import Image

    manifest_path = PACKAGE_ROOT / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if PACKAGE_ROOT.exists() and any(PACKAGE_ROOT.iterdir()):
        raise FileExistsError(f"测试包目录非空且缺少完成的manifest，不覆盖已有内容：{PACKAGE_ROOT}")
    selected = load_selected_sources()
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    groups = []
    for group_number, source in enumerate(selected.values(), 1):
        folder = Path(f"{group_number:02d}_{source['scope']}_{Path(source['case_key']).name}")
        variants = {condition: [] for condition in IMAGE_VARIANTS}
        for condition in IMAGE_VARIANTS:
            (PACKAGE_ROOT / folder / condition).mkdir(parents=True)
        originals = []
        for slot, value in enumerate(source["images"]):
            original_path = Path(value)
            size = source["sizes"][slot]
            a = folder / "A" / f"{slot + 1:03d}{original_path.suffix.lower()}"
            b = folder / "B" / f"{slot + 1:03d}.png"
            # 独立复制A；B写入新文件，绝不修改或重链接原图片。
            original_hash = file_hash(original_path)
            shutil.copy2(original_path, PACKAGE_ROOT / a)
            if file_hash(PACKAGE_ROOT / a) != original_hash:
                raise ValueError("原图复制校验失败")
            Image.new("RGB", size, color=(128, 128, 128)).save(PACKAGE_ROOT / b)
            for condition, path in zip(IMAGE_VARIANTS, (a, b)):
                variants[condition].append(path.as_posix())
            originals.append({"source": str(original_path), "sha256": original_hash, "size": size})
        groups.append(
            {
                "case_key": source["case_key"],
                "validation_index": source["validation_index"],
                "scope": source["scope"],
                "focus": source["focus"],
                "gt": source["gt"],
                "prompt": source["prompt"],
                "variants": variants,
                "originals": originals,
            }
        )
        print(f"[准备 {group_number}/{len(selected)}] {source['case_key']}：{len(originals)}张原图及灰图", flush=True)
    manifest = {
        "schema_version": 2,
        "purpose": "image_ablation_evaluation_only",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_root": str(DATASET_ROOT),
        "validation_cache": str(TOKENIZED_ROOT / "validation"),
        "groups": groups,
        # 测试包只登记实际生成的两套图片；D/E运行时复用A/B，不复制图片。
        "conditions": {condition: CONDITIONS[condition] for condition in IMAGE_VARIANTS},
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def prompt_without_hints(prompt: str, image_count: int) -> str:
    """Keep the original first-line question and ordered image placeholders only."""
    question = prompt.partition("\n")[0].rstrip("\r")
    if not question.strip() or "<image>" in question or image_count < 1 or prompt.count("<image>") != image_count:
        raise ValueError("无法安全去除辅助提示：基础提问必须独占首行，且图片占位符数量必须匹配")
    return question + "\n<image>" * image_count


def load_cases(manifest: dict[str, Any]) -> list[evaluation.EvalCase]:
    groups = manifest.get("groups", [])
    selection = [(group["validation_index"], group["case_key"], group["focus"]) for group in groups]
    if (
        manifest.get("schema_version") != 2
        or manifest.get("purpose") != "image_ablation_evaluation_only"
        or selection != list(SELECTED_CASES)
    ):
        raise ValueError("测试包与当前固定病例名单不一致；请检查PACKAGE_ROOT，不自动覆盖旧测试包")
    cases = []
    for number, group in enumerate(groups, 1):
        count = len(group["originals"])
        plain_prompt = prompt_without_hints(group["prompt"], count)
        reference = (
            f"原病例：{group['case_key']}（validation第{group['validation_index']}行，从0开始）\n重点观察：{group['focus']}\n{group['gt']}\n\n"
            "A/B：原始提问及全部辅助提示相同，仅图片不同；不再运行C组。\n"
            "D：A的原图，删除全部辅助提示；E：B的灰图，删除全部辅助提示。D/E基础提问与A相同，图片数量、顺序不变。\n"
            "B/E无有效视觉证据，不应把灰图视为阴性病例。原始GT仅供比较，不会发送给模型。"
        )
        for condition, label in CONDITIONS.items():
            image_variant = CONDITION_IMAGE_SOURCES[condition]
            paths = evaluation.image_paths(group["variants"][image_variant], PACKAGE_ROOT)
            if len(paths) != count or group["prompt"].count("<image>") != count:
                raise ValueError("测试包图片数或占位符数量不一致")
            cases.append(
                evaluation.EvalCase(
                    f"病例{number}/{len(groups)} {Path(group['case_key']).name} — {condition} {label}",
                    group["scope"],
                    paths,
                    reference,
                    source_prompt=plain_prompt if condition in ("D", "E") else group["prompt"],
                )
            )
    return cases


class AblationViewer(evaluation.CaseViewer):
    def render_answer(self) -> None:
        group_size = len(CONDITIONS)
        start = self.current // group_size * group_size
        sections = []
        for offset, (condition, label) in enumerate(CONDITIONS.items()):
            result = self.results.get(start + offset)
            if result is None:
                text = self.fatal_error or "等待后台推理……"
            elif "error" in result:
                text = "推理失败：\n" + result["error"]
            else:
                text = evaluation.format_answer(result["answer"])
                if result["finish_reason"] == "length":
                    text += "\n【达到输出长度上限，回答可能被截断】"
            sections.append(f"===== {condition}：{label} =====\n{text}")
        self.set_text(self.answer, "\n\n".join(sections))

    def poll(self) -> None:
        previous = set(self.results)
        super().poll()
        added = set(self.results) - previous
        # 即使正在查看A，只要同病例的其他组完成，也更新同列的对照回答。
        group_size = len(CONDITIONS)
        if (
            not self.closed
            and self.current not in added
            and any(index // group_size == self.current // group_size for index in added)
        ):
            self.render_answer()


# 固定病例和默认路径集中在入口附近；不要改动现有训练或推理配置。
DATASET_ROOT = Path("/media/maxlin/SATA/ReportData/Max_V1_case_dataV2_diagonly")
TOKENIZED_ROOT = Path(
    "/media/maxlin/SATA/ReportData/tokenized/max_case_v2_diagonly_qwen3vl2b_nothink_min256_max512_cut5120_val10pct_seed2026"
)
PACKAGE_ROOT = evaluation.PROJECT_ROOT / "outputs/case_image_ablation_20261127_20cases"
CONDITIONS = {
    "A": "原图",
    "B": "等尺寸灰图",
    "D": "原图，去掉辅助提示",
    "E": "等尺寸灰图，去掉辅助提示",
}
IMAGE_VARIANTS = ("A", "B")
CONDITION_IMAGE_SOURCES = {"A": "A", "B": "B", "D": "A", "E": "B"}
# 行号来自实际tokenized validation划分；focus只在界面显示，不进入模型提问。
SELECTED_CASES = (
    # 原4例保持顺序，便于与此前结果比较。
    (504, "gs/zsq/疣状隆起-糜烂_疣状、痘疹/608_3386", "胃溃疡"),
    (472, "gs/hrx/胃息肉_胃底腺/675_3304", "胃息肉"),
    (375, "cs/zsq/憩室_憩室/578_5396", "憩室、息肉"),
    (251, "cs/hrx/侧向发育型肿瘤_LST-NG/277_254", "直肠LST-NG"),
    # 新增8例胃镜，覆盖不同GT，不修改原始图片或报告。
    (7, "gs/zsq/食管静脉曲张_重度静脉/675_555", "食管静脉曲张"),
    (23, "gs/hrx/反流性食管炎_LA-B/675_3590", "反流性食管炎"),
    (50, "gs/zsq/霉菌性食管炎_霉菌/674_7280", "霉菌性食管炎"),
    (165, "gs/zsq/Barrett食管_岛型/594_1125", "Barrett食管"),
    (253, "gs/nwr/胃SMT_间质瘤/675_2592", "胃黏膜下隆起"),
    (6, "gs/nwr/萎缩性胃炎_O2或O-2/673_13544", "萎缩性胃炎"),
    (8, "gs/nwr/食管胃黏膜异位_异位/675_3516", "食管胃黏膜异位"),
    (197, "gs/hrx/食管裂孔疝_滑动型/583_3365", "食管裂孔疝、息肉"),
    # 新增8例肠镜；任务名以原始GT为准，不根据目录擅自改标签。
    (78, "cs/zsq/息肉_增生性/348_1569", "乙状结肠息肉"),
    (52, "cs/zsq/无敌锯齿状病变_SSL/491_25", "锯齿状病变、息肉"),
    (89, "cs/nwr/黏膜下肿瘤_囊肿/312_1683", "回盲部黏膜下隆起"),
    (92, "cs/nwr/山田分型_山田/578_8814", "山田I型息肉"),
    (13, "cs/nwr/术后_吻合口/578_2085", "吻合口术后、息肉"),
    (27, "cs/zsq/JNET_JNET2A、JNET 2A/467_1329", "GT为未见明显异常"),
    (16, "cs/nwr/炎症_回肠末端炎/173_4930", "回肠末端糜烂、息肉"),
    (60, "cs/zsq/JNET_JNET1、JNET 1/279_1479", "结肠溃疡、直肠息肉"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, default=evaluation.DEFAULT_CONFIG, help="沿用病例推理YAML中的模型和图像参数"
    )
    parser.add_argument("--prepare-only", action="store_true", help="只准备/核查测试包，不加载模型、不打开窗口")
    args = parser.parse_args()
    try:
        manifest = prepare_package()
        cases = load_cases(manifest)
        print(
            f"测试包：{PACKAGE_ROOT}\n{len(manifest['groups'])}个验证集病例，每例{'/'.join(CONDITIONS)}，共{len(cases)}次推理。",
            flush=True,
        )
        print("A/B保留原始提示；D原图去提示，E灰图去提示。D/E直接复用A/B图片，不运行C组。", flush=True)
        for group in manifest["groups"]:
            print(
                f"{group['case_key']}（{group['focus']}）",
                flush=True,
            )
        if args.prepare_only:
            return 0
        config_path = args.config.expanduser().resolve()
        config = evaluation.read_config(config_path)
        if config.get("do_sample", True):
            raise ValueError("图像对照测试要求推理YAML设置do_sample: false，以免采样差异混入结果。")
        print(
            f"推理配置：{config_path}\nLoRA：{config.get('adapter_name_or_path')}\nA/B文字相同；D/E文字相同，仅保留基础提问及图片占位符。",
            flush=True,
        )
        evaluation.ROUTE_LABELS["ablation"] = "验证集A/B/D/E图像及提示对照"
        AblationViewer(cases, config_path, "ablation").run()
    except KeyboardInterrupt:
        print("\n已结束测试。")
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    evaluation.mp.freeze_support()
    raise SystemExit(main())
