#!/usr/bin/env python3
"""Build a single-image VLM SFT dataset from ``rules.json``.

The source datasets are treated as read-only.  Images are copied (or processed
through a hook) into::

    /media/maxlin/SATA/ReportData/Max_V1_single/
        all_conversations.json
        prepared_samples.sqlite3
        dataset_info.json
        manifest.json
        previews/<save_name>/
            <index>_<image>.jpg
            <index>_input_<image>
            preview_conversations.json
        <save_name>/
            <generated images>

Each output sample uses LLaMA-Factory's ShareGPT multimodal format.  Exactly
one ``<image>`` placeholder is emitted for the one image in ``images``.

The implementation deliberately separates *format parsers* from *dataset
hooks*.  COCO/VOC/folder/INNO layout differences are discovered by inspecting
directory and JSON content.  Dataset-specific vocabulary conversion can be
registered in ``SPEC_FUNCS`` without duplicating a complete parser.

Run modes::

    # Rebuild everything and replace the output directory.
    python data/inno_report_data/data_generate_single_images.py --run-mode full

    # Add only datasets absent from an existing completed output.
    python data/inno_report_data/data_generate_single_images.py --run-mode add --datasets cs_qc_5cls

    # Rebuild conversations from cache without touching images.
    python data/inno_report_data/data_generate_single_images.py --run-mode adjust
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


LOGGER = logging.getLogger("vlm_dataset_builder")

DEFAULT_CONFIG = Path(__file__).with_name("rules.json")
DEFAULT_OUTPUT_ROOT = Path("/media/maxlin/SATA/ReportData/Max_V1_single")
LOG_FILE_NAME = "generation.log"
CACHE_FILE_NAME = "prepared_samples.sqlite3"
CACHE_SCHEMA_VERSION = "1"
CACHE_COMMIT_INTERVAL = 1_000
PROGRESS_LOG_INTERVAL = 100_000
DEFAULT_TRT_PYTHON = Path("/home/maxlin/anaconda3/envs/py36torch171/bin/python")
DEFAULT_TRT_LIBRARY_DIR = Path("/home/maxlin/TensorRT-8.2.0.6/lib")
DEFAULT_CUDA_LIBRARY_DIR = Path("/usr/local/cuda-11.1/lib64")
INNO_MODELS_DIR = Path(__file__).with_name("models")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PREVIEW_DISPLAY_MAX_EDGE = 1080
IGNORED_MASK_VALUES = {0, 255}
ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff"

# Former ``*.names`` assets are kept here so this generator has a single
# source of truth for model output order and display labels.
GQC_CLASSES = (
    "口部", "咽部", "食管入口", "食管中段", "食管下段", "胃食管结合部",
    "十二指肠降部", "十二指肠球部", "幽门", "胃窦", "胃窦大弯", "胃窦前壁",
    "胃窦小弯", "胃窦后壁", "胃角", "胃角前壁", "胃角后壁",
    "倒镜-胃体下部小弯", "倒镜-胃体中部小弯", "倒镜-贲门小弯", "倒镜-贲门大弯",
    "倒镜-贲门前壁", "倒镜-贲门后壁", "倒镜-胃底", "正镜-胃体下部大弯",
    "正镜-胃体下部前壁", "正镜-胃体下部小弯", "正镜-胃体下部后壁",
    "正镜-胃体中部大弯", "正镜-胃体中部前壁", "正镜-胃体中部小弯",
    "正镜-胃体中部后壁", "正镜-胃体上部大弯", "正镜-胃体上部前壁",
    "正镜-胃体上部小弯", "正镜-胃体上部后壁", "无效",
)
CQC_CLASSES = ("体外", "回盲部", "大肠", "小肠", "无效")
STATUS_CLASSES = (
    "WLI", "NBI", "RDI", "TXI", "BLI", "LCI",
    "无染色", "碘染", "靛胭脂", "其他染色",
    "无手术帽", "有手术帽", "无器械", "有器械",
    "非放大", "食管放大", "胃放大",
    "正常质量", "颜色异常", "低效图像",
    "体内", "正常体外", "样本体外",
)

TRT_MODEL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "gs_qc_fine": {
        "model": "gqc_v3_fine37_20230925.trt",
        "classes": GQC_CLASSES,
    },
    "es_qc": {
        "model": "EC01-02-QC01-1.0.0.1.trt",
        "classes": CQC_CLASSES,
    },
    "status_cls": {
        "model": "GI-QC06-1.1.1.2-cuda580.trt",
        "classes": STATUS_CLASSES,
    },
}

STATUS_STAGE_SPECS = (
    ("lightsource", 0, 5),
    ("staining", 6, 9),
    ("surgical_cap", 10, 11),
    ("instrument", 12, 13),
    ("magnification", 14, 16),
    ("quality", 17, 19),
    ("has_external", 20, 22),
)
STATUS_STAGE_DISPLAY_NAMES = {
    "lightsource": "光源",
    "staining": "染色",
    "surgical_cap": "手术帽",
    "instrument": "器械",
    "magnification": "放大",
    "quality": "图像质量",
    "has_external": "体内外",
    "lesion_morphology": "形态学",
    "gt_pathology": "病理结论",
}
STATUS_FIELD_ALIASES = {
    "morphology": "lesion_morphology",
    "pathology": "gt_pathology",
}
LOCATION_L2_ALIASES = {
    "antrum_of_stomach": "胃窦",
    "body_of_stomach": "胃体",
    "cardia": "贲门",
    "duodenum": "十二指肠",
    "esophagogastric_junction": "胃食管结合部",
    "fundus_of_stomach": "胃底",
    "gastric_angle": "胃角",
    "lower_esophagus": "食管下段",
    "midesophagus": "食管中段",
    "upper_esophagus": "食管入口",
    "unknown": "无法识别",
}
ANNOTATION_STATUS_ALIASES: dict[str, dict[str, str]] = {
    "lightsource": {
        "wli": "WLI", "nbi": "NBI", "rdi": "RDI", "txi": "TXI", "bli": "BLI", "lci": "LCI",
    },
    "magnification": {
        "non_mag": "非放大",
        "esophageal_mag": "食管放大",
        "gastric_mag": "胃放大",
        "completely_mag": "胃放大（完全）",
        "incompletely_mag": "胃放大（不完全）",
    },
    "quality": {
        "normal": "正常质量",
        "abnormal_coloration": "颜色异常",
        "abnormal_color": "颜色异常",
        "blurred": "低效图像",
        "inefficient_image": "低效图像",
    },
    "staining": {
        "nothing": "无染色",
        "iodination": "碘染",
        "indigo": "靛胭脂",
        "other": "其他染色",
    },
    "surgical_cap": {
        "nothing": "无手术帽",
        "having": "有手术帽",
    },
    "instrument": {
        "nothing": "无器械",
        "having": "有器械",
    },
    "has_external": {
        "nothing": "体内",
        "normal_external": "正常体外",
        "sample_external": "样本体外",
    },
    "lesion_morphology": {
        "nothing": "无",
        "yamada_type_i": "山田I型",
        "yamada_type_ii": "山田II型",
        "yamada_type_iii": "山田III型",
        "yamada_type_iv": "山田IV型",
    },
    "gt_pathology": {
        "img_other": "其他",
        "img_nice_i": "NICE-I型",
        "img_nice_ii": "NICE-II型",
        "img_nice_iii": "NICE-III型",
        "adenocarcinoma": "腺癌",
        "adenoma": "腺瘤",
        "aseptate_serrated": "无蒂锯齿状病变",
        "inflammatory": "炎性病变",
        "other": "其他",
    },
}
LOCATION_CONFIDENCE_THRESHOLD = 0.9
STATUS_CONFIDENCE_THRESHOLD = 0.9
LOW_CONFIDENCE_MAX_CLS_PATTERN = re.compile(r"max_cls=([^,，)）]+)")
CONFIDENCE_SUFFIX_PATTERN = re.compile(r"（置信度：[0-9.]+）$")
CHINESE_CHARACTER_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

STATUS_QUESTIONS = {
    "lightsource": "该图像是什么光源？",
    "staining": "该图像是否有染色？",
    "surgical_cap": "该图像是否有手术帽？",
    "instrument": "该图像中是否有器械？",
    "magnification": "该图像是否开启放大？",
    "quality": "该图像质量如何？",
    "has_external": "该图像在体内还是体外？",
    "lesion_morphology": "该病变的形态学表现是什么？",
    "gt_pathology": "该病变的病理结论是什么？",
}

# Some classification annotations contain only a grading code. The existing
# dataset description supplies the missing task-level meaning for the answer,
# while the question remains disease-agnostic and therefore leaks no label.
CADX_DESCRIPTION_SUBJECT_RULES = (
    ("反流性食管炎la分级", "反流性食管炎"),
    ("溃疡ahs分期", "消化性溃疡"),
    ("溃疡forrest分级", "消化性溃疡"),
    ("nice分型", "结直肠病变"),
)


class TimestampedMultilineFormatter(logging.Formatter):
    """Prefix every physical log line, including traceback continuation lines."""

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        lines = rendered.splitlines()
        if len(lines) <= 1:
            return rendered
        prefix = f"{self.formatTime(record, self.datefmt)} [{record.levelname}] "
        return "\n".join([lines[0], *[prefix + line for line in lines[1:]]])


def confirm_and_reset_output_root(output_root: Path) -> bool:
    """Confirm once, then remove the complete generated-output directory."""

    if output_root.is_symlink():
        raise ValueError(f"Refusing to recursively delete a symbolic-link output root: {output_root}")
    resolved = output_root.resolve(strict=False)
    protected = {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }
    workspace_root = Path.cwd().resolve()
    if resolved in protected or workspace_root in resolved.parents:
        raise ValueError(f"Refusing to recursively delete unsafe output root: {resolved}")
    if not resolved.exists():
        return True
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError(f"Existing output root must be a real directory: {resolved}")
    if not sys.stdin.isatty():
        raise SystemExit(f"Existing output directory requires an interactive overwrite confirmation: {resolved}")
    print("检测到目标输出文件夹已存在：")
    print(f"  {resolved}")
    answer = input("覆盖将先删除该文件夹内的全部内容，确认继续？[y/N]: ").strip().casefold()
    if answer not in {"y", "yes"}:
        return False
    shutil.rmtree(resolved)
    return True


def configure_logging(log_level: str, output_root: Path, *, append: bool = False) -> Path:
    """Log to the terminal and a UTF-8 file, optionally starting a new appended section."""

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / LOG_FILE_NAME
    if append and log_path.is_file() and log_path.stat().st_size:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n\n\n")
    formatter = TimestampedMultilineFormatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, log_level),
        handlers=[stream_handler, file_handler],
        force=True,
    )
    return log_path


@dataclass
class Box:
    """Internal box representation: absolute ``x_min, y_min, x_max, y_max``."""

    xyxy: tuple[float, float, float, float]
    label_id: str | None = None
    label: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageRecord:
    dataset_key: str
    source_path: Path
    source_root: Path
    root_index: int
    split: str
    label_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    annotation_path: Path | None = None
    mask_path: Path | None = None


@dataclass
class PreparedImage:
    """Result returned by the original-image crop hook.

    A future crop implementation may return multiple images.  ``boxes`` must
    be expressed relative to the corresponding *saved* image.
    """

    path: Path
    boxes: list[Box]
    metadata: dict[str, Any]
    width: int | None = None
    height: int | None = None


@dataclass
class BuildStats:
    discovered: int = 0
    written: int = 0
    skipped_missing_image: int = 0
    skipped_empty_dialogue: int = 0
    model_fields_omitted: int = 0


class PreviewSkipped(Exception):
    """Raised only to stop the remaining previews of the current dataset."""


SpecFunc = Callable[[ImageRecord, Dict[str, Any]], ImageRecord]
ModelRunner = Callable[[str, ImageRecord, Path, Dict[str, Any]], Any]
DatasetParser = Callable[
    [str, Dict[str, Any], Sequence[Path], "ImageResolver", int],
    List[ImageRecord],
]

SPEC_FUNCS: dict[str, SpecFunc] = {}
MODEL_RUNNERS: dict[str, ModelRunner] = {}
DATASET_PARSERS: dict[str, DatasetParser] = {}
_WARNED_HOOKS: set[str] = set()
_INNO_RUNTIME: "InnoAIRuntime | None" = None
_SKIP_MODEL_INFERENCE = False

MAP_TO_7PLUS1: dict[str, int] = {
    "0": 0, "1": 0,  # 口咽
    "2": 1, "3": 1, "4": 1, "5": 1,  # 食管及结合部
    "6": 2, "7": 2,  # 十二指肠
    "8": 3, "9": 3, "10": 3, "11": 3, "12": 3, "13": 3,  # 幽门及胃窦
    "14": 4, "15": 4, "16": 4,  # 胃角
    "17": 5, "18": 5, "19": 5, "20": 5, "21": 5, "22": 5, "23": 5,  # 倒镜胃体
    "24": 6, "25": 6, "26": 6, "27": 6, "28": 6, "29": 6, "30": 6,
    "31": 6, "32": 6, "33": 6, "34": 6, "35": 6,  # 正镜胃体
    "40": 7,  # 体外和无效
}
MAP_TO_7PLUS1_NAMES: dict[int, str] = {
    0: "口咽",
    1: "食管及结合部",
    2: "十二指肠",
    3: "幽门及胃窦",
    4: "胃角",
    5: "倒镜胃体",
    6: "正镜胃体",
    7: "体外和无效",
}


def register_spec_fun(name: str) -> Callable[[SpecFunc], SpecFunc]:
    """Register a dataset-specific record transformation."""

    def decorator(func: SpecFunc) -> SpecFunc:
        SPEC_FUNCS[normalize_hook_name(name)] = func
        return func

    return decorator


def register_model_runner(field_name: str) -> Callable[[ModelRunner], ModelRunner]:
    """Register a model inference function for ``location`` or ``status``."""

    def decorator(func: ModelRunner) -> ModelRunner:
        MODEL_RUNNERS[field_name] = func
        return func

    return decorator


def register_dataset_parser(dataset_key: str) -> Callable[[DatasetParser], DatasetParser]:
    """Register a full-parser override for a genuinely exceptional dataset.

    Most naming/layout differences should remain in the generic format
    parsers.  Use this only when content-based discovery cannot describe a
    dataset without dataset-specific assumptions.
    """

    def decorator(func: DatasetParser) -> DatasetParser:
        DATASET_PARSERS[dataset_key] = func
        return func

    return decorator


def normalize_hook_name(name: str) -> str:
    return str(name).translate({ord(char): None for char in ZERO_WIDTH_CHARS}).strip()


@register_spec_fun("map_to_7plus1")
def map_to_7plus1(record: ImageRecord, dataset_config: dict[str, Any]) -> ImageRecord:
    """Map GQC's 37 original IDs into the requested 7+1 IDs (0--7)."""

    raw_ids = [str(item) for item in record.label_ids]
    mapped = [MAP_TO_7PLUS1[item] for item in raw_ids if item in MAP_TO_7PLUS1]
    unknown = [item for item in raw_ids if item not in MAP_TO_7PLUS1]
    if unknown:
        LOGGER.warning("[%s] map_to_7plus1 has no mapping for labels: %s", record.dataset_key, unknown)
    if mapped:
        mapped = unique_preserve_order(mapped)
        # Keep original GT IDs and the configured cls label untouched.  The
        # mapped result is a separate debug-only field.
        mapped_names = [MAP_TO_7PLUS1_NAMES[item] for item in mapped]
        record.metadata["map_to_7plus1"] = mapped_names[0] if len(mapped_names) == 1 else mapped_names
        record.metadata["_map_to_7plus1_ids"] = mapped
        record.metadata["_map_to_7plus1_original_ids"] = raw_ids
    return record


@register_spec_fun("map_vocabulary")
def map_vocabulary(record: ImageRecord, dataset_config: dict[str, Any]) -> ImageRecord:
    """Normalize annotation status fields to the embedded Chinese vocabulary."""

    status_fields = dataset_config.get("info", {}).get("status", [])
    if not isinstance(status_fields, list):
        return record
    for raw_name in status_fields:
        name = str(raw_name)
        canonical_name = canonical_status_field_name(name)
        value = record.metadata.get(name)
        if not is_useful(value) and canonical_name != name:
            value = record.metadata.get(canonical_name)
        if not is_useful(value):
            continue
        record.metadata[canonical_name] = normalize_annotation_status_value(canonical_name, value)
        if canonical_name != name:
            record.metadata.pop(name, None)
    return record


def warn_unimplemented_hook(name: str) -> None:
    if name not in _WARNED_HOOKS:
        LOGGER.warning("Hook %s is a placeholder; see its function docstring before production use.", name)
        _WARNED_HOOKS.add(name)


# This worker source is intentionally Python 3.6-compatible.  TensorRT 8.2 is
# installed in a legacy environment, whereas this generator uses modern type
# syntax and cannot itself be parsed by that interpreter.  Launching the
# embedded worker with ``python -c`` keeps one maintained source file without
# coupling the main process to TensorRT/PyCUDA.
TRT_WORKER_SOURCE = r"""
from __future__ import print_function

import contextlib
import json
import os
import sys
import traceback

import cv2
import numpy as np

cuda = None
trt = None


MODEL_SPECS = json.loads(os.environ.pop("INNO_TRT_MODEL_SPECS"))
GPU_INDEX = int(os.environ.pop("INNO_TRT_GPU_INDEX", "0"))


class TensorRTInferCompat(object):
    def __init__(self, engine_path, gpu_index):
        self.cfx = cuda.Device(gpu_index).make_context()
        self.closed = False
        try:
            self.logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(self.logger, namespace="")
            with open(engine_path, "rb") as engine_file:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())
            if self.engine is None:
                raise RuntimeError("TensorRT could not deserialize engine: {}".format(engine_path))
            if not hasattr(self.engine, "num_bindings"):
                raise RuntimeError("TensorRT binding API is unavailable for this engine.")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT could not create an execution context.")
            self.inputs, self.outputs, self.allocations = [], [], []
            for index in range(self.engine.num_bindings):
                shape = tuple(self.engine.get_binding_shape(index))
                if any(size <= 0 for size in shape):
                    raise RuntimeError("Dynamic/invalid TensorRT binding shape: {}".format(shape))
                dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(index)))
                allocation = cuda.mem_alloc(int(np.prod(shape)) * dtype.itemsize)
                item = {"shape": shape, "dtype": dtype, "allocation": allocation}
                self.allocations.append(allocation)
                target = self.inputs if self.engine.binding_is_input(index) else self.outputs
                target.append(item)
            self.bindings = [int(allocation) for allocation in self.allocations]
            if not self.inputs or not self.outputs:
                raise RuntimeError("TensorRT engine has no input or output binding.")
        finally:
            self.cfx.pop()

    def infer(self, batch):
        if len(batch) != len(self.inputs):
            raise ValueError("Expected {} input tensors, got {}.".format(len(self.inputs), len(batch)))
        self.cfx.push()
        try:
            outputs = [np.empty(item["shape"], dtype=item["dtype"]) for item in self.outputs]
            for index, value in enumerate(batch):
                cuda.memcpy_htod(self.inputs[index]["allocation"], np.ascontiguousarray(value))
            if not self.context.execute_v2(self.bindings):
                raise RuntimeError("TensorRT execute_v2 returned False.")
            for index, value in enumerate(outputs):
                cuda.memcpy_dtoh(value, self.outputs[index]["allocation"])
            return outputs
        finally:
            self.cfx.pop()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.allocations, self.inputs, self.outputs, self.bindings = [], [], [], []
        self.context, self.engine = None, None
        self.cfx.detach()


def preprocess_image(image, config):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = int(config.get("img_size", 224))
    image = cv2.resize(image, (size, size)).astype(np.float32)
    mean = config.get("norm_mean")
    std = config.get("norm_std")
    if mean is None or std is None:
        result = image.transpose(2, 0, 1) / 255.0
    else:
        if config.get("norm_div_first", 1):
            image = image / 255.0
        mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        result = ((image - mean_array) / std_array).transpose(2, 0, 1)
    return result[np.newaxis, ...].astype(np.float32, copy=False)


def crop_invalid_region(image, ignore_square=True):
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
    mask = ((np.abs(channels[:, :, 0] - channels[:, :, 1]) > gap) |
            (np.abs(channels[:, :, 1] - channels[:, :, 2]) > gap) |
            (np.abs(channels[:, :, 0] - channels[:, :, 2]) > gap)).astype(np.uint8)
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
    return image[y:y + crop_height, x:x + crop_width], [x, y, crop_width, crop_height]


class Worker(object):
    def __init__(self, gpu_index):
        self.gpu_index = gpu_index
        self.models = {}

    def load_model(self, config_key):
        global cuda, trt
        if config_key in self.models:
            return self.models[config_key]
        if config_key not in MODEL_SPECS:
            raise KeyError("Unknown local model config: {}".format(config_key))
        config = MODEL_SPECS[config_key]
        if not os.path.isfile(config["model"]):
            raise IOError("Local model asset not found: {}".format(config["model"]))
        # Cropping is CPU-only. Import CUDA/TensorRT only for model inference.
        if cuda is None or trt is None:
            import pycuda.autoinit
            import pycuda.driver as cuda_module
            import tensorrt as trt_module
            cuda = cuda_module
            trt = trt_module
        print("Loading local TensorRT model {} from {}".format(config_key, config["model"]), file=sys.stderr)
        result = {
            "model": TensorRTInferCompat(config["model"], self.gpu_index),
            "config": config,
            "classes": config["classes"],
        }
        self.models[config_key] = result
        return result

    def infer(self, request):
        image = cv2.imread(request["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV cannot decode image: {}".format(request["image_path"]))
        model_info = self.load_model(request["config_key"])
        batch = preprocess_image(image, model_info["config"])
        probabilities = np.asarray(model_info["model"].infer(batch)[0], dtype=np.float64)
        if probabilities.ndim > 1:
            probabilities = probabilities[0]
        classes = model_info["classes"]
        if probabilities.size != len(classes):
            raise RuntimeError("{} returned {} scores for {} classes".format(
                request["config_key"], probabilities.size, len(classes)
            ))
        return {"probabilities": probabilities.tolist(), "classes": classes}

    def crop(self, request):
        image = cv2.imread(request["source_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV cannot decode image: {}".format(request["source_path"]))
        cropped, roi = crop_invalid_region(image, ignore_square=True)
        destination = request["destination"]
        directory = os.path.dirname(destination)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        if not cv2.imwrite(destination, cropped):
            raise IOError("OpenCV failed to write cropped image: {}".format(destination))
        return {"roi": roi}

    def dispatch(self, request):
        action = request.get("action")
        if action == "infer":
            return self.infer(request)
        if action == "crop":
            return self.crop(request)
        raise ValueError("Unknown worker action: {!r}".format(action))

    def close(self):
        for model_info in self.models.values():
            model_info["model"].close()
        self.models.clear()


def run_worker():
    protocol_stdout = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        worker = Worker(GPU_INDEX)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("action") == "close":
                    break
                with contextlib.redirect_stdout(sys.stderr):
                    result = worker.dispatch(request)
                response = {"ok": True}
                response.update(result)
            except Exception as error:
                response = {
                    "ok": False,
                    "error": "{}: {}".format(type(error).__name__, error),
                    "traceback": traceback.format_exc(),
                }
            protocol_stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            protocol_stdout.flush()
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            worker.close()


run_worker()
"""


def trt_model_specs() -> dict[str, dict[str, Any]]:
    """Build the JSON-serializable model configuration sent to the worker."""

    normalization = {
        "img_size": 224,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "norm_div_first": 1,
    }
    return {
        key: {
            **normalization,
            "model": str((INNO_MODELS_DIR / definition["model"]).resolve(strict=False)),
            "classes": list(definition["classes"]),
        }
        for key, definition in TRT_MODEL_DEFINITIONS.items()
    }


class InnoAIRuntime:
    """Lazy launcher for this directory's crop helper and TensorRT classifiers.

    The worker is self-contained so the generator never imports the former
    external utility repository.
    """

    def __init__(self, gpu_index: int = 0, trt_python: Path = DEFAULT_TRT_PYTHON) -> None:
        self.gpu_index = gpu_index
        self.trt_python = trt_python.resolve(strict=False)
        self._worker: subprocess.Popen[str] | None = None
        atexit.register(self.close)

    def process_classifier(self, image_path: Path, config_key: str) -> tuple[np.ndarray, list[str]]:
        response = self._worker_request(
            {"action": "infer", "config_key": config_key, "image_path": str(image_path)}
        )
        return np.asarray(response["probabilities"], dtype=np.float64), list(response["classes"])

    def crop_invalid_region(self, source_path: Path, destination: Path) -> Sequence[int] | None:
        response = self._worker_request(
            {"action": "crop", "source_path": str(source_path), "destination": str(destination)}
        )
        return response.get("roi")

    def _worker_request(self, request: dict[str, Any]) -> dict[str, Any]:
        worker = self._ensure_worker()
        assert worker.stdin is not None and worker.stdout is not None
        worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
        if not line:
            code = worker.poll()
            stderr = ""
            if worker.stderr is not None:
                stderr = worker.stderr.read().strip()
            detail = f"\nTensorRT worker stderr:\n{stderr}" if stderr else ""
            raise RuntimeError(f"TensorRT worker exited unexpectedly (exit code {code}).{detail}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                "TensorRT worker request failed: "
                + str(response.get("error", "unknown error"))
                + ("\n" + response["traceback"] if response.get("traceback") else "")
            )
        return response

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        if not self.trt_python.is_file():
            raise FileNotFoundError(f"TensorRT Python interpreter not found: {self.trt_python}")
        command = [str(self.trt_python), "-u", "-c", TRT_WORKER_SOURCE]
        LOGGER.info("Starting embedded TensorRT worker with %s", self.trt_python)
        worker_env = os.environ.copy()
        # VS Code/debugpy injects a sitecustomize module through PYTHONPATH so
        # child Python processes can be debugged too.  That bundled debugpy
        # currently uses syntax unsupported by the required Python 3.6 TRT
        # environment, so this isolated worker must not inherit the injection.
        worker_env.pop("PYTHONPATH", None)
        for name in list(worker_env):
            if name.startswith("PYDEVD_") or name.startswith("DEBUGPY_"):
                worker_env.pop(name, None)
        library_dirs = [str(DEFAULT_TRT_LIBRARY_DIR), str(DEFAULT_CUDA_LIBRARY_DIR)]
        inherited_library_path = worker_env.get("LD_LIBRARY_PATH", "")
        if inherited_library_path:
            library_dirs.append(inherited_library_path)
        worker_env["LD_LIBRARY_PATH"] = ":".join(
            unique_preserve_order(path for path in library_dirs if path)
        )
        worker_env["INNO_TRT_GPU_INDEX"] = str(self.gpu_index)
        worker_env["INNO_TRT_MODEL_SPECS"] = json.dumps(trt_model_specs(), ensure_ascii=False)
        self._worker = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        return self._worker

    def close(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None or worker.poll() is not None:
            return
        try:
            if worker.stdin is not None:
                worker.stdin.write('{"action":"close"}\n')
                worker.stdin.flush()
            worker.wait(timeout=5)
        except Exception:
            worker.terminate()


def configure_inno_runtime(gpu_index: int, trt_python: Path) -> None:
    global _INNO_RUNTIME
    _INNO_RUNTIME = InnoAIRuntime(gpu_index, trt_python)


def get_inno_runtime() -> InnoAIRuntime:
    global _INNO_RUNTIME
    if _INNO_RUNTIME is None:
        _INNO_RUNTIME = InnoAIRuntime()
    return _INNO_RUNTIME


def transform_boxes_after_crop(boxes: Sequence[Box], roi: Sequence[int] | None) -> list[Box]:
    if roi is None:
        return list(boxes)
    crop_x, crop_y, crop_width, crop_height = [int(value) for value in roi]
    transformed: list[Box] = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        clipped = (
            max(0.0, min(float(crop_width), x1 - crop_x)),
            max(0.0, min(float(crop_height), y1 - crop_y)),
            max(0.0, min(float(crop_width), x2 - crop_x)),
            max(0.0, min(float(crop_height), y2 - crop_y)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        transformed.append(Box(clipped, box.label_id, box.label, dict(box.raw)))
    return transformed


def crop_original_image(
    record: ImageRecord,
    destination: Path,
    dataset_config: dict[str, Any],
) -> list[PreparedImage]:
    """Crop an ``ori`` image with the local ``crop_invalid_region`` helper.

    The returned ROI uses ``[x, y, width, height]``.  GT boxes are shifted and
    clipped so they remain relative to the final saved image.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    roi = get_inno_runtime().crop_invalid_region(record.source_path, destination)
    metadata = dict(record.metadata)
    metadata["crop_roi_xywh"] = list(roi) if roi is not None else None
    metadata["_prepared_level"] = "ori"
    return [PreparedImage(destination, transform_boxes_after_crop(record.boxes, roi), metadata)]


def default_model_runner(
    field_name: str,
    record: ImageRecord,
    image_path: Path,
    dataset_config: dict[str, Any],
) -> None:
    """Placeholder for model-generated fields.

    Returning ``None`` intentionally omits the field instead of writing a fake
    answer into supervised training data.
    """

    warn_unimplemented_hook(f"model_run:{field_name}")
    return None


@register_model_runner("location")
def run_location_model(
    field_name: str,
    record: ImageRecord,
    image_path: Path,
    dataset_config: dict[str, Any],
) -> Any:
    """Run the GQC or CQC TensorRT classifier from ``demo_allInOne.py``."""

    op_type = infer_endoscope_type(record, dataset_config)
    model_key = "gs_qc_fine" if op_type == "G" else "es_qc"
    probabilities, classes = get_inno_runtime().process_classifier(image_path, model_key)
    index = int(np.argmax(probabilities))
    raw_label = classes[index]
    label = raw_label
    confidence = round(float(probabilities[index]), 4)
    if confidence < LOCATION_CONFIDENCE_THRESHOLD:
        value = low_confidence_message(LOCATION_CONFIDENCE_THRESHOLD, label, confidence)
    else:
        value = f"{label}（置信度：{confidence:.4f}）"
    detail = {
        "model": model_key,
        "scope": "GQC" if op_type == "G" else "CQC",
        "label": label,
        "raw_label": raw_label,
        "confidence": confidence,
        "threshold": LOCATION_CONFIDENCE_THRESHOLD,
        "accepted": confidence >= LOCATION_CONFIDENCE_THRESHOLD,
    }
    store_model_detail(record, field_name, value, detail)
    return value


@register_model_runner("status")
def run_status_model(
    field_name: str,
    record: ImageRecord,
    image_path: Path,
    dataset_config: dict[str, Any],
) -> Any:
    """Run GI-QC06 status inference with winner-takes-all per attribute group."""

    probabilities, classes = get_inno_runtime().process_classifier(image_path, "status_cls")
    if probabilities.size != 23 or len(classes) != 23:
        raise RuntimeError(
            f"status_cls returned {probabilities.size} scores for {len(classes)} classes; expected 23."
        )
    results: dict[str, dict[str, Any]] = {}
    for stage_name, start, end in STATUS_STAGE_SPECS:
        stage_scores = probabilities[start : end + 1]
        winner_offset = int(np.argmax(stage_scores))
        winner_index = start + winner_offset
        label = classes[winner_index]
        confidence = round(float(probabilities[winner_index]), 4)
        results[stage_name] = (
            {"label": label, "confidence": confidence}
            if confidence >= STATUS_CONFIDENCE_THRESHOLD
            else low_confidence_message(STATUS_CONFIDENCE_THRESHOLD, label, confidence)
        )
    store_model_detail(record, field_name, results, {"model": "status_cls", "stages": results})
    return results


def low_confidence_message(threshold: float, label: str, confidence: float) -> str:
    """Return the shared, user-visible low-confidence result wording."""

    return f"置信度过低（阈值={threshold:g}，max_cls={label}，max_conf={confidence:.4f}）"


def infer_endoscope_type(record: ImageRecord, dataset_config: dict[str, Any]) -> str:
    explicit = str(dataset_config.get("endoscope_type", "")).strip().upper()
    if explicit in {"G", "C"}:
        return explicit
    key = record.dataset_key.casefold()
    if key.startswith("gs_"):
        return "G"
    if key.startswith("cs_"):
        return "C"
    raise ValueError(
        f"[{record.dataset_key}] Cannot select GQC/CQC. Add endoscope_type: G or C to this dataset config."
    )


def model_details(record: ImageRecord) -> dict[str, Any]:
    value = record.metadata.setdefault("_model_run_details", {})
    if not isinstance(value, dict):
        value = {}
        record.metadata["_model_run_details"] = value
    return value


def model_cache(record: ImageRecord) -> dict[str, Any]:
    value = record.metadata.setdefault("_model_run_cache", {})
    if not isinstance(value, dict):
        value = {}
        record.metadata["_model_run_cache"] = value
    return value


def store_model_detail(record: ImageRecord, field_name: str, value: Any, detail: dict[str, Any]) -> None:
    model_cache(record)[field_name] = value
    model_details(record)[field_name] = detail


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def format_readable_vlm_sample(item: Any, max_line_width: int = 200) -> str:
    """Format one VLM sample compactly, expanding only unusually long items."""

    if not isinstance(item, dict) or set(item) != {"conversations", "images"}:
        return textwrap.indent(json.dumps(item, ensure_ascii=False, indent=2), "  ")
    conversations = item.get("conversations")
    images = item.get("images")
    if not isinstance(conversations, list) or not isinstance(images, list):
        return textwrap.indent(json.dumps(item, ensure_ascii=False, indent=2), "  ")

    lines = ["  {"]
    compact_images = json.dumps(images, ensure_ascii=False)
    if len(compact_images) + 14 <= max_line_width:
        lines.append(f'    "images": {compact_images},')
    else:
        lines.append('    "images": [')
        for index, image_path in enumerate(images):
            suffix = "," if index + 1 < len(images) else ""
            lines.append(f"      {json.dumps(image_path, ensure_ascii=False)}{suffix}")
        lines.append("    ],")

    lines.append('    "conversations": [')
    for index, message in enumerate(conversations):
        compact = json.dumps(message, ensure_ascii=False)
        suffix = "," if index + 1 < len(conversations) else ""
        if len(compact) + 6 <= max_line_width:
            lines.append(f"      {compact}{suffix}")
        else:
            expanded = textwrap.indent(json.dumps(message, ensure_ascii=False, indent=2), "      ")
            expanded_lines = expanded.splitlines()
            expanded_lines[-1] += suffix
            lines.extend(expanded_lines)
    lines.append("    ]")
    lines.append("  }")
    return "\n".join(lines)


class JsonArrayWriter:
    """Incrementally and atomically write a standard JSON array."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.file: Any = None
        self.count = 0

    def __enter__(self) -> "JsonArrayWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.temporary.open("w", encoding="utf-8")
        self.file.write("[\n")
        return self

    def write(self, item: Any) -> None:
        if self.file is None:
            raise RuntimeError("JsonArrayWriter must be used as a context manager.")
        if self.count:
            self.file.write(",\n")
        self.file.write(format_readable_vlm_sample(item))
        self.count += 1

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.file is not None:
            if exc_type is None:
                self.file.write("\n]\n")
            self.file.close()
        if exc_type is None:
            os.replace(self.temporary, self.path)
        elif self.temporary.exists():
            self.temporary.unlink()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class PreparedSampleCacheWriter:
    """Persist final-image annotations and model outputs for dialogue-only rebuilds."""

    def __init__(
        self,
        path: Path,
        output_root: Path,
        config_path: Path,
        *,
        append: bool = False,
    ) -> None:
        self.path = path
        self.output_root = output_root.resolve(strict=False)
        self.config_path = config_path.resolve(strict=False)
        self.append = append
        self.connection: sqlite3.Connection | None = None
        self.count = 0

    def __enter__(self) -> "PreparedSampleCacheWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        if self.append:
            schema_version = self.connection.execute(
                "SELECT value FROM cache_meta WHERE key = 'schema_version'"
            ).fetchone()
            complete = self.connection.execute(
                "SELECT value FROM cache_meta WHERE key = 'complete'"
            ).fetchone()
            if schema_version is None or str(schema_version[0]) != CACHE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported prepared cache schema in {self.path}; "
                    f"expected {CACHE_SCHEMA_VERSION!r}."
                )
            if complete is None or str(complete[0]) != "1":
                raise RuntimeError("Add mode requires a completed prepared sample cache.")
            row = self.connection.execute(
                "SELECT COALESCE(MAX(sample_order), 0) FROM samples"
            ).fetchone()
            self.count = int(row[0])
            self._set_meta("complete", "0")
            return self
        self.connection.executescript(
            """
            CREATE TABLE cache_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE samples (
                sample_order INTEGER PRIMARY KEY,
                dataset_key TEXT NOT NULL,
                record_index INTEGER NOT NULL,
                split TEXT NOT NULL,
                image_relpath TEXT NOT NULL UNIQUE,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                label_ids_json TEXT NOT NULL,
                labels_json TEXT NOT NULL,
                boxes_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                resolved_info_json TEXT NOT NULL
            );
            CREATE INDEX samples_dataset_order_idx
                ON samples(dataset_key, sample_order);
            """
        )
        self._set_meta("schema_version", CACHE_SCHEMA_VERSION)
        self._set_meta("complete", "0")
        self._set_meta("output_root", str(self.output_root))
        self._set_meta("config_path", str(self.config_path))
        self.connection.commit()
        return self

    def _set_meta(self, key: str, value: Any) -> None:
        if self.connection is None:
            raise RuntimeError("Prepared sample cache is not open.")
        self.connection.execute(
            "INSERT OR REPLACE INTO cache_meta(key, value) VALUES (?, ?)",
            (key, str(value)),
        )

    def write(
        self,
        record_index: int,
        record: ImageRecord,
        prepared: PreparedImage,
        resolved_info: dict[str, Any],
    ) -> None:
        if self.connection is None:
            raise RuntimeError("Prepared sample cache must be used as a context manager.")
        if prepared.width is None or prepared.height is None:
            raise ValueError(f"Final image size was not recorded for cache entry: {prepared.path}")
        try:
            image_relpath = prepared.path.resolve(strict=False).relative_to(self.output_root)
        except ValueError as error:
            raise ValueError(
                f"Prepared image must be inside output root: {prepared.path} vs {self.output_root}"
            ) from error
        boxes = [
            {
                "xyxy": list(box.xyxy),
                "label_id": box.label_id,
                "label": box.label,
                # Only lesion-level fields are needed when conversations are
                # rebuilt.  COCO ``raw`` annotations may contain very large
                # segmentation arrays and must not bloat the SQLite cache.
                "raw": {
                    key: box.raw[key]
                    for key in ("lesion_morphology", "morphology", "gt_pathology", "pathology")
                    if key in box.raw and is_useful(box.raw[key])
                },
            }
            for box in prepared.boxes
        ]
        self.count += 1
        self.connection.execute(
            """
            INSERT INTO samples(
                sample_order, dataset_key, record_index, split, image_relpath,
                width, height, label_ids_json, labels_json, boxes_json,
                metadata_json, resolved_info_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.count,
                record.dataset_key,
                record_index,
                record.split,
                image_relpath.as_posix(),
                prepared.width,
                prepared.height,
                compact_json(record.label_ids),
                compact_json(record.labels),
                compact_json(boxes),
                compact_json(prepared.metadata),
                compact_json(resolved_info),
            ),
        )
        if not self.append and self.count % CACHE_COMMIT_INTERVAL == 0:
            self.connection.commit()

    def mark_complete(self, conversation_count: int) -> None:
        self._set_meta("cached_samples", self.count)
        self._set_meta("conversation_samples", conversation_count)
        self._set_meta("complete", "1")
        if self.connection is not None:
            self.connection.commit()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.connection is not None:
            if exc_type is not None:
                if self.append:
                    self.connection.rollback()
                else:
                    self._set_meta("complete", "0")
                    self.connection.commit()
            else:
                self.connection.commit()
            self.connection.close()
            self.connection = None


class PreparedSampleCacheReader:
    """Read a completed cache without permitting writes during adjust mode."""

    def __init__(self, path: Path, output_root: Path) -> None:
        self.path = path
        self.output_root = output_root.resolve(strict=False)
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> "PreparedSampleCacheReader":
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Prepared sample cache does not exist: {self.path}. Run --run-mode full first."
            )
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only = ON")
        schema_version = self.meta("schema_version")
        if schema_version != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported prepared cache schema {schema_version!r}; expected {CACHE_SCHEMA_VERSION!r}."
            )
        if self.meta("complete") != "1":
            raise RuntimeError(
                "Prepared sample cache is incomplete. Adjust mode is available only after a successful full run."
            )
        return self

    def meta(self, key: str) -> str | None:
        if self.connection is None:
            raise RuntimeError("Prepared sample cache is not open.")
        row = self.connection.execute(
            "SELECT value FROM cache_meta WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row[0])

    def __len__(self) -> int:
        if self.connection is None:
            raise RuntimeError("Prepared sample cache is not open.")
        row = self.connection.execute("SELECT COUNT(*) FROM samples").fetchone()
        return int(row[0])

    def first_metadata(self, dataset_key: str) -> dict[str, Any] | None:
        if self.connection is None:
            raise RuntimeError("Prepared sample cache is not open.")
        row = self.connection.execute(
            "SELECT metadata_json FROM samples WHERE dataset_key = ? LIMIT 1",
            (dataset_key,),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def dataset_keys(self) -> set[str]:
        if self.connection is None:
            raise RuntimeError("Prepared sample cache is not open.")
        rows = self.connection.execute("SELECT DISTINCT dataset_key FROM samples")
        return {str(row[0]) for row in rows}

    def samples(self) -> Iterator[tuple[ImageRecord, PreparedImage, dict[str, Any]]]:
        if self.connection is None:
            raise RuntimeError("Prepared sample cache is not open.")
        query = """
            SELECT dataset_key, record_index, split, image_relpath, width, height,
                   label_ids_json, labels_json, boxes_json, metadata_json,
                   resolved_info_json
            FROM samples
            ORDER BY sample_order
        """
        for row in self.connection.execute(query):
            image_path = self.output_root / str(row["image_relpath"])
            metadata = json.loads(row["metadata_json"])
            boxes = [
                Box(
                    tuple(float(value) for value in item["xyxy"]),
                    item.get("label_id"),
                    item.get("label"),
                    item.get("raw") or {},
                )
                for item in json.loads(row["boxes_json"])
            ]
            record = ImageRecord(
                dataset_key=str(row["dataset_key"]),
                source_path=image_path,
                source_root=self.output_root,
                root_index=0,
                split=str(row["split"]),
                label_ids=[str(value) for value in json.loads(row["label_ids_json"])],
                labels=[str(value) for value in json.loads(row["labels_json"])],
                boxes=boxes,
                metadata=metadata,
            )
            prepared = PreparedImage(
                path=image_path,
                boxes=boxes,
                metadata=metadata,
                width=int(row["width"]),
                height=int(row["height"]),
            )
            resolved_info = json.loads(row["resolved_info_json"])
            yield record, prepared, resolved_info

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def is_useful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def to_string_list(value: Any, *, split_compound: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif split_compound and isinstance(value, str) and "_" in value:
        values = value.split("_")
    else:
        values = [value]
    return unique_preserve_order(str(item) for item in values if is_useful(item))


def iter_dirs(root: Path, max_depth: int) -> Iterator[Path]:
    """Walk directories without enumerating every image file recursively."""

    if not root.is_dir():
        return
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        yield current
        if depth >= max_depth:
            continue
        try:
            children = [entry for entry in os.scandir(current) if entry.is_dir(follow_symlinks=False)]
        except (OSError, PermissionError) as error:
            LOGGER.warning("Cannot scan directory %s: %s", current, error)
            continue
        stack.extend((Path(entry.path), depth + 1) for entry in reversed(children))


def discover_named_dirs(root: Path, names: Sequence[str], max_depth: int) -> list[Path]:
    wanted = {str(name).casefold() for name in names}
    result = [path for path in iter_dirs(root, max_depth) if path.name.casefold() in wanted]
    return unique_paths(result)


def discover_files(root: Path, suffixes: set[str], max_depth: int) -> list[Path]:
    files: list[Path] = []
    for directory in iter_dirs(root, max_depth):
        try:
            for entry in os.scandir(directory):
                if entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.casefold() in suffixes:
                    files.append(Path(entry.path))
        except (OSError, PermissionError):
            continue
    return unique_paths(files)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def image_files_under(directory: Path, recursive: bool = True) -> Iterator[Path]:
    if not directory.is_dir():
        return
    if recursive:
        for current, dirnames, filenames in os.walk(directory, followlinks=False):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in filenames:
                path = Path(current, filename)
                if path.suffix.casefold() in IMAGE_SUFFIXES:
                    yield path
    else:
        try:
            for entry in os.scandir(directory):
                path = Path(entry.path)
                if entry.is_file(follow_symlinks=False) and path.suffix.casefold() in IMAGE_SUFFIXES:
                    yield path
        except (OSError, PermissionError):
            return


class ImageResolver:
    """Resolve inconsistent image locations while indexing every source root once."""

    def __init__(self) -> None:
        self._indexes: dict[Path, dict[str, list[Path]]] = {}
        self._layout_cache: dict[tuple[tuple[str, ...], str], Path] = {}
        self._directory_entries: dict[Path, set[str] | None] = {}

    def _direct_file(
        self,
        directory: Path,
        relative: Path,
        indexed: bool,
        allow_alternate_suffix: bool,
    ) -> Path | None:
        """Resolve a direct child, optionally indexing a large directory once."""

        if not indexed or relative.parent != Path("."):
            candidate = directory / relative
            return candidate if candidate.is_file() else None
        directory = directory.resolve(strict=False)
        if directory not in self._directory_entries:
            try:
                self._directory_entries[directory] = {
                    entry.name
                    for entry in os.scandir(directory)
                    if entry.is_file(follow_symlinks=False)
                }
            except (OSError, PermissionError):
                self._directory_entries[directory] = None
        entries = self._directory_entries[directory]
        if entries is None:
            return None
        if relative.name in entries:
            return directory / relative
        if allow_alternate_suffix:
            for suffix in sorted(IMAGE_SUFFIXES):
                alternate = relative.with_suffix(suffix)
                if alternate.name in entries:
                    return directory / alternate
        return None

    def resolve(
        self,
        file_name: str,
        roots: Sequence[Path],
        split: str = "",
        explicit_paths: Sequence[Any] = (),
        layout_hints: Sequence[str] = (),
        preferred_subdirs: Sequence[str] = (),
        cache_layout: bool = True,
        allow_index_fallback: bool = True,
        allow_alternate_suffix: bool = False,
    ) -> Path | None:
        relative = Path(file_name)
        layout_key = (tuple(os.path.abspath(root) for root in roots), split)
        cached_base = self._layout_cache.get(layout_key) if cache_layout else None
        if cached_base is not None:
            # COCO images belonging to one annotation split share a layout.
            # Avoid an expensive network-filesystem stat for every image; the
            # final build validates each selected/copied file explicitly.
            return cached_base / relative

        direct_subdirs: list[Path | str] = []
        for preferred in preferred_subdirs:
            if not preferred:
                continue
            direct_subdirs.append(preferred)
            if split:
                direct_subdirs.append(Path(preferred) / split)
        direct_subdirs.extend(["", split])
        if split:
            direct_subdirs.extend(
                [
                    Path("images") / split,
                    Path("image") / split,
                    Path("JPEGImages") / split,
                    Path("images_crop") / split,
                ]
            )
        for hint in layout_hints:
            if hint:
                direct_subdirs.extend(
                    [hint, Path("images") / hint, Path("image") / hint, Path("JPEGImages") / hint]
                )
        direct_subdirs.extend(["images", "image", "JPEGImages", "images_crop"])
        for root in roots:
            for subdir in direct_subdirs:
                candidate = self._direct_file(
                    root / subdir,
                    relative,
                    indexed=not cache_layout,
                    allow_alternate_suffix=allow_alternate_suffix,
                )
                if candidate is not None:
                    if cache_layout:
                        self._layout_cache[layout_key] = root / subdir
                    return candidate

        # Absolute paths embedded in metadata are a fallback.  Prefer the
        # configured src_path roots so a stale path in JSON cannot silently
        # redirect a dataset to a different mirror.
        for raw_path in explicit_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path)
            if candidate.is_file():
                return candidate

        if not allow_index_fallback:
            return None

        basename = relative.name
        candidates: list[Path] = []
        for root in roots:
            candidates.extend(self._index(root).get(basename, []))
        if not candidates:
            return None
        if split:
            split_matches = [path for path in candidates if split.casefold() in {p.casefold() for p in path.parts}]
            if split_matches:
                candidates = split_matches
        preferred = [path for path in candidates if "images_crop" in {p.casefold() for p in path.parts}]
        return sorted(preferred or candidates, key=lambda item: (len(item.parts), str(item)))[0]

    def _index(self, root: Path) -> dict[str, list[Path]]:
        root = Path(os.path.abspath(root))
        if root not in self._indexes:
            LOGGER.info("Indexing images under %s ...", root)
            index: dict[str, list[Path]] = defaultdict(list)
            for path in image_files_under(root):
                folded_parts = {part.casefold() for part in path.parts}
                if folded_parts.intersection({"mask", "masks", "color", "segmentationclass"}):
                    continue
                index[path.name].append(path)
            self._indexes[root] = dict(index)
        return self._indexes[root]


def existing_roots(dataset_key: str, config: dict[str, Any]) -> list[Path]:
    raw_roots = config.get("src_path", [])
    if isinstance(raw_roots, str):
        raw_roots = [raw_roots]
    roots: list[Path] = []
    for raw_root in raw_roots:
        root = Path(raw_root)
        if root.is_dir():
            roots.append(root)
        else:
            LOGGER.warning("[%s] Source root does not exist and will be skipped: %s", dataset_key, root)
    return unique_paths(roots)


def infer_split(path: Path, configured_splits: Sequence[str]) -> str:
    configured = {str(item).casefold(): str(item) for item in configured_splits}

    # Annotation files are often stored under roots such as
    # ``GI-QC06-Train-D1-val`` or ``DataImages``.  A plain substring check is
    # ambiguous there: ``train`` appears in the former even though its split
    # is ``val``, and ``images`` appears inside the latter although it is the
    # test set.  Keep true directory/file-name matches as the strongest signal,
    # then choose the right-most complete token among the remaining path parts.
    for part in reversed(path.parts):
        folded = part.casefold()
        if folded in configured:
            return configured[folded]
        stem = Path(part).stem.casefold()
        if stem in configured:
            return configured[stem]

    matches: list[tuple[int, int, int, str]] = []
    for distance, part in enumerate(reversed(path.parts)):
        stem = Path(part).stem.casefold()
        for candidate, original in configured.items():
            if not candidate:
                continue
            # A configured split is a token, not an arbitrary suffix.  Thus
            # ``DataImages`` does not count as ``images``.
            token_matches = list(
                re.finditer(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", stem)
            )
            if token_matches:
                # Prefer a nearer component; for e.g. "...-train-...-val",
                # prefer the later token (val) within that component.
                matches.append((-distance, token_matches[-1].start(), len(candidate), original))

    if matches:
        return max(matches)[3]
    return "unspecified"


def config_label_maps(config: dict[str, Any]) -> tuple[dict[str, str], list[str], list[str]]:
    info = config.get("info", {})
    label_ids = [str(item) for item in info.get("label", [])]
    class_names = [str(item) for item in info.get("cls", [])]
    mapping = {label_id: class_names[index] for index, label_id in enumerate(label_ids) if index < len(class_names)}
    return mapping, label_ids, class_names


def map_label(label_id: Any, fallback: Any, config: dict[str, Any]) -> str:
    mapping, _, _ = config_label_maps(config)
    key = str(label_id)
    return mapping.get(key, str(fallback if is_useful(fallback) else key))


def parse_coco_dataset(
    dataset_key: str,
    config: dict[str, Any],
    roots: Sequence[Path],
    resolver: ImageResolver,
    search_depth: int,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    configured_splits = config.get("sub_folder", [])
    for root_index, root in enumerate(roots):
        json_files = [
            path
            for path in discover_files(root, {".json"}, search_depth)
            if "status" not in path.stem.casefold()
        ]
        for annotation_path in json_files:
            try:
                payload = json_load(annotation_path)
            except Exception as error:
                LOGGER.warning("[%s] Cannot read %s: %s", dataset_key, annotation_path, error)
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
                continue

            images = payload["images"]
            annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for annotation in payload.get("annotations", []) or []:
                if isinstance(annotation, dict):
                    annotations_by_image[annotation.get("image_id")].append(annotation)
            categories = {
                category.get("id"): category
                for category in (payload.get("categories", []) or [])
                if isinstance(category, dict)
            }
            split = infer_split(annotation_path, configured_splits)

            for image_info in images:
                if not isinstance(image_info, dict):
                    continue
                file_name = image_info.get("file_name") or image_info.get("filename") or image_info.get("name")
                if not isinstance(file_name, str):
                    continue
                explicit = preferred_explicit_image_paths(image_info, config.get("level", "crop"))
                image_path = resolver.resolve(
                    file_name,
                    [root],
                    split,
                    explicit,
                    layout_hints=[annotation_path.stem],
                )
                if image_path is None:
                    LOGGER.warning("[%s] Image not found for %s in %s", dataset_key, file_name, annotation_path)
                    continue

                boxes: list[Box] = []
                raw_label_ids: list[str] = []
                raw_labels: list[str] = []
                for annotation in annotations_by_image.get(image_info.get("id"), []):
                    category_id = annotation.get("category_id")
                    category = categories.get(category_id, {})
                    label = map_label(category_id, category.get("name"), config)
                    raw_label_ids.append(str(category_id))
                    raw_labels.append(label)
                    bbox = annotation.get("bbox")
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        x, y, width, height = (float(value) for value in bbox[:4])
                        boxes.append(Box((x, y, x + width, y + height), str(category_id), label, annotation))

                if not raw_label_ids:
                    image_label = first_present(image_info, "gt_cls", "class", "label", "category_id")
                    for label_id in to_string_list(image_label, split_compound=False):
                        raw_label_ids.append(label_id)
                        raw_labels.append(map_label(label_id, label_id, config))
                boxes.extend(boxes_from_image_metadata(image_info, raw_label_ids, raw_labels))

                records.append(
                    ImageRecord(
                        dataset_key=dataset_key,
                        source_path=image_path,
                        source_root=root,
                        root_index=root_index,
                        split=split,
                        label_ids=unique_preserve_order(raw_label_ids),
                        labels=unique_preserve_order(raw_labels),
                        boxes=deduplicate_boxes(boxes),
                        metadata=dict(image_info),
                        annotation_path=annotation_path,
                    )
                )
    # ``images`` is the authoritative sample list.  Missing annotations are
    # intentionally retained as negative samples, while source files absent
    # from ``images`` are never silently added to the training dataset.
    return deduplicate_records(records)


def preferred_explicit_image_paths(image_info: dict[str, Any], level: str) -> list[Any]:
    level = normalize_level(level)
    if level == "lesion":
        keys = ["image_path_lesion_crop", "lesion_path", "file_path", "image_path_origin"]
    elif level == "ori":
        keys = ["image_path_origin", "file_path", "image_path_lesion_crop"]
    else:
        keys = ["file_path", "image_path", "image_path_lesion_crop", "image_path_origin"]
    return [image_info.get(key) for key in keys]


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and is_useful(mapping[key]):
            return mapping[key]
    return None


def boxes_from_image_metadata(
    image_info: dict[str, Any],
    label_ids: Sequence[str],
    labels: Sequence[str],
) -> list[Box]:
    candidates = [image_info.get("lesion_pos"), image_info.get("bbox"), image_info.get("box")]
    raw_boxes = image_info.get("boxes")
    if isinstance(raw_boxes, list):
        if len(raw_boxes) >= 4 and not isinstance(raw_boxes[0], (list, tuple, dict)):
            candidates.append(raw_boxes)
        else:
            candidates.extend(raw_boxes)
    result: list[Box] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, dict):
            candidate = candidate.get("bbox") or candidate.get("box") or candidate.get("xyxy")
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in candidate[:4])
        except (TypeError, ValueError):
            continue
        # Image-level boxes in INNO metadata are normally already xyxy.
        if x2 <= x1 or y2 <= y1:
            continue
        label_id = label_ids[min(index, len(label_ids) - 1)] if label_ids else None
        label = labels[min(index, len(labels) - 1)] if labels else label_id
        # Retain the annotation row on the box.  INNO classification JSON is
        # lesion-oriented, so fields such as morphology belong to this exact
        # lesion rather than to the full image.  Keeping it here lets repeated
        # original images be merged without losing the box/label/attribute
        # relationship.
        raw = dict(image_info)
        raw["source"] = "image_metadata"
        result.append(Box((x1, y1, x2, y2), label_id, label, raw))
    return result


def deduplicate_boxes(boxes: Sequence[Box]) -> list[Box]:
    result: list[Box] = []
    seen: set[tuple[Any, ...]] = set()
    for box in boxes:
        key = (*[round(value, 4) for value in box.xyxy], box.label_id, box.label)
        if key not in seen:
            seen.add(key)
            result.append(box)
    return result


def parse_inno_dataset(
    dataset_key: str,
    config: dict[str, Any],
    roots: Sequence[Path],
    resolver: ImageResolver,
    search_depth: int,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    configured_splits = config.get("sub_folder", [])
    level = normalize_level(config.get("level", "lesion"))
    raw_original_roots = config.get("original_src_path", [])
    if isinstance(raw_original_roots, str):
        raw_original_roots = [raw_original_roots]
    original_roots = unique_paths(
        Path(path) for path in raw_original_roots if Path(path).is_dir()
    )
    missing_originals: list[tuple[Path, str]] = []
    for root_index, root in enumerate(roots):
        json_files = discover_files(root, {".json"}, search_depth)
        # ``*_ori.json`` describes the original full-frame image.  Processed
        # JSON describes crop/lesion images; use the matching form whenever
        # both are available in an INNO source directory.
        if level == "ori":
            ori_files = [path for path in json_files if "_ori" in path.stem.casefold()]
            if ori_files:
                json_files = ori_files
        else:
            processed_files = [path for path in json_files if "_ori" not in path.stem.casefold()]
            if processed_files:
                json_files = processed_files
        for annotation_path in json_files:
            try:
                payload = json_load(annotation_path)
            except Exception as error:
                LOGGER.warning("[%s] Cannot read %s: %s", dataset_key, annotation_path, error)
                continue
            if isinstance(payload, dict):
                images = payload.get("images")
            elif isinstance(payload, list):
                images = payload
            else:
                images = None
            if not isinstance(images, list):
                continue
            split = infer_split(annotation_path, configured_splits)
            if split == "unspecified" and len(configured_splits) == 1:
                # A single layout directory such as images_crop is not part of
                # the JSON file name, but it is still the intended split.
                split = configured_splits[0]
            for image_info in images:
                if not isinstance(image_info, dict):
                    continue
                if level == "ori":
                    file_name = (
                        image_info.get("file_name_ori")
                        or image_info.get("file_name_origin")
                        or image_info.get("file_name")
                        or Path(str(image_info.get("file_path", ""))).name
                    )
                else:
                    file_name = image_info.get("file_name") or Path(str(image_info.get("file_path", ""))).name
                if not file_name:
                    continue
                preferred_subdirs = (
                    ("image_ori", "images_ori", "original", "originals", "images", "image")
                    if level == "ori"
                    else ("images_crop", "crop", "images")
                )
                image_path = resolver.resolve(
                    str(file_name),
                    unique_paths([root, *original_roots]),
                    split,
                    preferred_explicit_image_paths(image_info, level),
                    preferred_subdirs=preferred_subdirs,
                    # Original-image sources can be spread across several
                    # roots.  A layout cached from the first hit must not make
                    # later missing files look valid.
                    cache_layout=level != "ori",
                    # A basename-only recursive fallback could accidentally
                    # select a lesion crop.  Original-image matching is exact.
                    allow_index_fallback=level != "ori",
                    # Some INNO JSON rows retain a historical .jpg name while
                    # the copied source is losslessly stored as .png.
                    allow_alternate_suffix=level == "ori",
                )
                if image_path is None:
                    if level == "ori":
                        if config.get("require_original_image"):
                            raise FileNotFoundError(
                                f"[{dataset_key}] Original image required by {annotation_path.name} "
                                f"was not found: {file_name}. Add its directory to original_src_path; "
                                "lesion crops will never be used as an implicit fallback."
                            )
                        missing_originals.append((annotation_path, str(file_name)))
                    else:
                        LOGGER.warning("[%s] INNO image not found: %s", dataset_key, file_name)
                    continue
                raw_ids = to_string_list(first_present(image_info, "gt_cls", "class", "label"))
                labels = [map_label(label_id, label_id, config) for label_id in raw_ids]
                boxes = boxes_from_image_metadata(image_info, raw_ids, labels)
                records.append(
                    ImageRecord(
                        dataset_key=dataset_key,
                        source_path=image_path,
                        source_root=root,
                        root_index=root_index,
                        split=split,
                        label_ids=raw_ids,
                        labels=unique_preserve_order(labels),
                        boxes=boxes,
                        metadata=dict(image_info),
                        annotation_path=annotation_path,
                    )
                )
    records = deduplicate_records(records)
    if config.get("merge_original_lesions"):
        for record in records:
            record.metadata["_merged_by_original"] = True
            record.metadata["_merged_lesion_count"] = len(record.boxes)
    if missing_originals:
        examples = ", ".join(
            f"{path.name}:{name}" for path, name in missing_originals[:5]
        )
        message = (
            f"[{dataset_key}] {len(missing_originals)} lesion annotations could not resolve "
            f"their original image. Examples: {examples}"
        )
        LOGGER.warning(message)
    return records


def parse_folder_dataset(
    dataset_key: str,
    config: dict[str, Any],
    roots: Sequence[Path],
    search_depth: int,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    subfolders = [str(item) for item in config.get("sub_folder", [])]
    _, configured_labels, _ = config_label_maps(config)
    configured_set = set(configured_labels)

    for root_index, root in enumerate(roots):
        split_dirs = discover_named_dirs(root, subfolders, search_depth)
        if root.name in subfolders and root not in split_dirs:
            split_dirs.insert(0, root)
        if not split_dirs:
            split_dirs = [root]
        for split_dir in split_dirs:
            split = infer_split(split_dir, subfolders)
            for image_path in image_files_under(split_dir):
                try:
                    relative_parts = image_path.relative_to(split_dir).parts[:-1]
                except ValueError:
                    relative_parts = image_path.parts[:-1]
                raw_label = next((part for part in reversed(relative_parts) if part in configured_set), None)
                if raw_label is None and relative_parts:
                    raw_label = relative_parts[-1]
                if raw_label is None:
                    LOGGER.warning("[%s] Cannot infer folder label for %s", dataset_key, image_path)
                    continue
                label = map_label(raw_label, raw_label, config)
                records.append(
                    ImageRecord(
                        dataset_key=dataset_key,
                        source_path=image_path,
                        source_root=root,
                        root_index=root_index,
                        split=split,
                        label_ids=[str(raw_label)],
                        labels=[label],
                        metadata={"gt_cls": raw_label},
                    )
                )
    return deduplicate_records(records)


def parse_voc_dataset(
    dataset_key: str,
    config: dict[str, Any],
    roots: Sequence[Path],
    resolver: ImageResolver,
    search_depth: int,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    splits = [str(item) for item in config.get("sub_folder", [])]
    for root_index, root in enumerate(roots):
        jpeg_roots = discover_named_dirs(root, ["JPEGImages", "images"], search_depth)
        xml_files = discover_files(root, {".xml"}, search_depth)
        xml_by_stem = {path.stem: path for path in xml_files}
        mask_roots = discover_named_dirs(root, ["mask", "masks", "SegmentationClass"], search_depth)
        mask_by_key: dict[tuple[str, str], Path] = {}
        for mask_root in mask_roots:
            for mask_path in image_files_under(mask_root):
                mask_by_key[(infer_split(mask_path, splits), mask_path.stem)] = mask_path
                mask_by_key.setdefault(("unspecified", mask_path.stem), mask_path)

        image_candidates: list[Path] = []
        for jpeg_root in jpeg_roots:
            image_candidates.extend(image_files_under(jpeg_root))
        if not image_candidates:
            for split_dir in discover_named_dirs(root, splits, search_depth):
                image_candidates.extend(image_files_under(split_dir))

        for image_path in unique_paths(image_candidates):
            split = infer_split(image_path, splits)
            xml_path = xml_by_stem.get(image_path.stem)
            mask_path = mask_by_key.get((split, image_path.stem)) or mask_by_key.get(("unspecified", image_path.stem))
            boxes: list[Box] = []
            label_ids: list[str] = []
            labels: list[str] = []
            metadata: dict[str, Any] = {}
            if xml_path:
                xml_boxes, xml_ids, xml_labels = parse_voc_xml(xml_path, config)
                boxes.extend(xml_boxes)
                label_ids.extend(xml_ids)
                labels.extend(xml_labels)
            elif mask_path:
                mask_boxes, mask_ids, mask_labels = parse_semantic_mask(mask_path, config)
                boxes.extend(mask_boxes)
                label_ids.extend(mask_ids)
                labels.extend(mask_labels)
                metadata["mask_path"] = str(mask_path)
            else:
                LOGGER.warning("[%s] No XML or mask found for %s", dataset_key, image_path)
                continue
            records.append(
                ImageRecord(
                    dataset_key=dataset_key,
                    source_path=image_path,
                    source_root=root,
                    root_index=root_index,
                    split=split,
                    label_ids=unique_preserve_order(label_ids),
                    labels=unique_preserve_order(labels),
                    boxes=boxes,
                    metadata=metadata,
                    annotation_path=xml_path,
                    mask_path=mask_path,
                )
            )
    return deduplicate_records(records)


def parse_voc_xml(path: Path, config: dict[str, Any]) -> tuple[list[Box], list[str], list[str]]:
    boxes: list[Box] = []
    label_ids: list[str] = []
    labels: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except Exception as error:
        LOGGER.warning("Cannot parse VOC XML %s: %s", path, error)
        return boxes, label_ids, labels
    for obj in root.findall("object"):
        raw_label = (obj.findtext("name") or "").strip()
        bndbox = obj.find("bndbox")
        if not raw_label or bndbox is None:
            continue
        try:
            xyxy = tuple(float(bndbox.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        label = map_label(raw_label, raw_label, config)
        label_ids.append(raw_label)
        labels.append(label)
        boxes.append(Box(xyxy, raw_label, label, {"source": str(path)}))
    return boxes, label_ids, labels


def parse_semantic_mask(path: Path, config: dict[str, Any]) -> tuple[list[Box], list[str], list[str]]:
    with Image.open(path) as mask_image:
        mask = np.asarray(mask_image)
    if mask.ndim == 3:
        # Color masks need an authoritative color mapping; do not invent one.
        LOGGER.warning("RGB semantic mask requires a spec_fun/color mapping and is skipped: %s", path)
        return [], [], []
    boxes: list[Box] = []
    label_ids: list[str] = []
    labels: list[str] = []
    for value in np.unique(mask):
        class_id = int(value)
        if class_id in IGNORED_MASK_VALUES:
            continue
        binary = mask == value
        components = connected_component_boxes(binary)
        label_id = str(class_id)
        label = map_label(label_id, label_id, config)
        label_ids.append(label_id)
        labels.append(label)
        for xyxy in components:
            boxes.append(Box(xyxy, label_id, label, {"mask_value": class_id}))
    return boxes, label_ids, labels


def connected_component_boxes(binary_mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    try:
        from scipy import ndimage

        components, count = ndimage.label(binary_mask)
        objects = ndimage.find_objects(components, max_label=count)
        boxes = []
        for slices in objects:
            if not slices:
                continue
            y_slice, x_slice = slices
            boxes.append((float(x_slice.start), float(y_slice.start), float(x_slice.stop), float(y_slice.stop)))
        return boxes
    except ImportError:
        ys, xs = np.where(binary_mask)
        if not len(xs):
            return []
        return [(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))]


def deduplicate_records(records: Sequence[ImageRecord]) -> list[ImageRecord]:
    """Merge repeated JSON descriptions of the same image without losing boxes."""

    by_key: dict[str, ImageRecord] = {}
    for record in records:
        key = os.path.abspath(record.source_path)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = record
            continue
        previous.label_ids = unique_preserve_order([*previous.label_ids, *record.label_ids])
        previous.labels = unique_preserve_order([*previous.labels, *record.labels])
        previous.boxes = deduplicate_boxes([*previous.boxes, *record.boxes])
        # Image-level metadata should occur once after lesion rows are merged.
        # Keep the first useful value for determinism; lesion-level values are
        # preserved independently in ``Box.raw`` above.
        for metadata_key, value in record.metadata.items():
            if metadata_key not in previous.metadata or not is_useful(previous.metadata[metadata_key]):
                previous.metadata[metadata_key] = value
    return list(by_key.values())


FORMAT_PARSERS: dict[str, Callable[..., list[ImageRecord]]] = {
    "coco": parse_coco_dataset,
    "inno": parse_inno_dataset,
    "folder": parse_folder_dataset,
    "voc": parse_voc_dataset,
}


def discover_dataset_records(
    dataset_key: str,
    config: dict[str, Any],
    resolver: ImageResolver,
    search_depth: int,
) -> list[ImageRecord]:
    roots = existing_roots(dataset_key, config)
    if not roots:
        return []
    custom_parser = DATASET_PARSERS.get(dataset_key)
    if custom_parser is not None:
        records = custom_parser(dataset_key, config, roots, resolver, search_depth)
    else:
        format_name = str(config.get("format", "")).casefold()
        parser = FORMAT_PARSERS.get(format_name)
        if parser is None:
            raise ValueError(f"[{dataset_key}] Unsupported format: {config.get('format')!r}")
        if format_name == "folder":
            records = parser(dataset_key, config, roots, search_depth)
        else:
            records = parser(dataset_key, config, roots, resolver, search_depth)
    records = deduplicate_records(records)
    hook_name = config.get("spec_fun")
    if hook_name:
        normalized = normalize_hook_name(hook_name)
        hook = SPEC_FUNCS.get(normalized)
        if hook is None:
            warn_unimplemented_hook(f"spec_fun:{normalized}")
        else:
            records = [hook(record, config) for record in records]
    return records


def configured_subfolders(config: dict[str, Any]) -> list[str]:
    """Return the configured sub-folder names in their user-defined order."""

    configured = config.get("sub_folder", [])
    if isinstance(configured, str):
        configured = [configured]
    return [str(item) for item in configured if str(item)]


def count_source_images_by_subfolder(dataset_key: str, config: dict[str, Any]) -> dict[str, int]:
    """Count physical source image files, independently of annotation validity."""

    configured_names = configured_subfolders(config)
    roots = existing_roots(dataset_key, config)
    counts: dict[str, int] = defaultdict(int)
    for root in roots:
        for name in configured_names:
            # Layouts vary: ``root/train`` (COCO) and
            # ``root/JPEGImages/train`` (VOC) are both common.
            candidates = [
                root if root.name.casefold() == name.casefold() else root / name,
                root / "images" / name,
                root / "image" / name,
                root / "JPEGImages" / name,
                root / "images_crop" / name,
            ]
            seen_directories: set[Path] = set()
            for directory in candidates:
                directory = directory.resolve(strict=False)
                if directory in seen_directories or not directory.is_dir():
                    continue
                seen_directories.add(directory)
                for path in image_files_under(directory):
                    parts = {part.casefold() for part in path.relative_to(directory).parts[:-1]}
                    if parts.intersection({"mask", "masks", "color", "segmentationclass"}):
                        continue
                    counts[name] += 1
    return counts


def log_subfolder_counts(dataset_key: str, config: dict[str, Any], records: Sequence[ImageRecord]) -> None:
    """Log physical image counts and usable annotated-record counts separately."""

    configured_names = configured_subfolders(config)
    source_counts = count_source_images_by_subfolder(dataset_key, config)
    if source_counts or configured_names:
        source_summary = ", ".join(
            f"{name}={source_counts.get(name, 0)}" for name in configured_names
        )
        LOGGER.info("[%s] Source image files by sub_folder: %s", dataset_key, source_summary)

    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.split or "(未识别子目录)"] += 1

    ordered_names = list(configured_names)
    ordered_names.extend(name for name in counts if name not in ordered_names)
    if not ordered_names:
        ordered_names = ["(未识别子目录)"]
    summary = ", ".join(f"{name}={counts.get(name, 0)}" for name in ordered_names)
    LOGGER.info("[%s] Usable annotated records by split: %s", dataset_key, summary)


def normalize_level(level: Any) -> str:
    normalized = str(level or "crop").casefold().strip()
    if normalized == "lession":
        normalized = "lesion"
    if normalized not in {"ori", "crop", "lesion"}:
        raise ValueError(f"Unsupported level: {level!r}; expected ori/crop/lesion")
    return normalized


def preview_category_value(record: ImageRecord, config: dict[str, Any]) -> Any:
    """Return exactly the value shown in the preview's ``类别`` field."""

    info = config.get("info", {})
    location_spec = info.get("location", info.get("localtion"))
    if location_spec == "cls":
        return record.label_ids[0] if record.label_ids else None
    if normalize_hook_name(config.get("spec_fun", "")) == "map_to_7plus1":
        return record.label_ids
    return record.labels or record.label_ids


def preview_dataset(
    dataset_key: str,
    save_name: str,
    config: dict[str, Any],
    records: Sequence[ImageRecord],
    preview_root: Path,
    count: int,
    mode: str,
    rng: random.Random,
) -> Path | None:
    if mode == "none" or not records:
        return None
    sample_indices = rng.sample(range(len(records)), k=min(count, len(records)))
    preview_dir = preview_root / save_name
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_json_path = preview_dir / "preview_conversations.json"
    preview_items: list[tuple[int, ImageRecord, PreparedImage, dict[str, Any]]] = []
    preview_stats = BuildStats()
    with tempfile.TemporaryDirectory(prefix=f"max_v1_preview_{save_name}_") as temporary:
        temporary_image_dir = Path(temporary) / "images"
        for record_offset in sample_indices:
            record_index = record_offset + 1
            record = records[record_offset]
            for prepared in prepare_record_image(
                record, config, temporary_image_dir, dry_run=False, record_index=record_index
            ):
                resolved_info = resolve_preview_info(record, prepared, config, preview_stats)
                preview_items.append((record_index, record, prepared, resolved_info))

        with JsonArrayWriter(preview_json_path) as preview_writer:
            for index, (record_index, record, prepared, resolved_info) in enumerate(preview_items, start=1):
                output_stem = Path(safe_output_name(record, record_index)).stem
                destination = preview_dir / f"{index:02d}_{output_stem}.jpg"
                input_suffix = prepared.path.suffix.casefold() or ".jpg"
                input_destination = preview_dir / f"{index:02d}_input_{output_stem}{input_suffix}"
                shutil.copy2(prepared.path, input_destination)
                sample = make_vlm_sample(
                    record,
                    prepared,
                    config,
                    preview_stats,
                    resolved_info=resolved_info,
                    image_path=input_destination,
                )
                if sample is not None:
                    preview_writer.write(sample)
                try:
                    show_single_preview(
                        dataset_key,
                        save_name,
                        config,
                        record,
                        prepared,
                        resolved_info,
                        destination,
                        index,
                        len(preview_items),
                        interactive=mode == "show",
                    )
                except PreviewSkipped:
                    LOGGER.info(
                        "[%s] Preview skipped at %d/%d; partial conversation preview saved to %s.",
                        dataset_key,
                        index,
                        len(preview_items),
                        preview_json_path,
                    )
                    return preview_dir
    LOGGER.info("[%s] %d detailed previews saved to %s", dataset_key, len(preview_items), preview_dir)
    LOGGER.info("[%s] Preview conversations saved to %s", dataset_key, preview_json_path)

    return preview_dir


def show_single_preview(
    dataset_key: str,
    save_name: str,
    config: dict[str, Any],
    record: ImageRecord,
    prepared: PreparedImage,
    resolved_info: dict[str, Any],
    destination: Path,
    index: int,
    total: int,
    interactive: bool,
) -> None:
    try:
        import matplotlib

        if interactive and str(matplotlib.get_backend()).casefold() == "agg":
            matplotlib.use("Qt5Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "An interactive Matplotlib GUI backend is required. Run the VS Code launch entry, "
            "which uses /home/maxlin/anaconda3/bin/python with Qt5Agg."
        ) from error

    try:
        with Image.open(prepared.path) as source:
            image = source.convert("RGB")
    except Exception as error:
        raise RuntimeError(f"Cannot open prepared preview image {prepared.path}: {error}") from error
    # Work on a display-only copy.  The prepared file remains untouched, and
    # small images keep their original display resolution.
    display_scale = min(1.0, PREVIEW_DISPLAY_MAX_EDGE / max(image.size))
    if display_scale < 1.0:
        display_size = tuple(max(1, round(value * display_scale)) for value in image.size)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize(display_size, resampling)

    draw = ImageDraw.Draw(image)
    font = load_preview_font(max(16, round(min(image.size) / 45)))
    line_width = max(2, round(min(image.size) / 250))
    for box in prepared.boxes:
        display_box = tuple(value * display_scale for value in box.xyxy)
        draw.rectangle(display_box, outline=(255, 40, 40), width=line_width)
        label = box.label or box.label_id
        if label:
            text_y = max(0.0, display_box[1] - getattr(font, "size", 16) - 2)
            draw.text((display_box[0] + 2, text_y), str(label), fill=(255, 40, 40), font=font)

    font_path = find_cjk_font_path()
    mpl_font = FontProperties(fname=str(font_path)) if font_path else None
    figure = plt.figure(figsize=(20, 11))
    grid = figure.add_gridspec(
        1, 2, width_ratios=(1.35, 1.15), left=0.025, right=0.985, top=0.9, bottom=0.1, wspace=0.015
    )
    image_axis = figure.add_subplot(grid[0, 0])
    info_axis = figure.add_subplot(grid[0, 1])
    # Keep both panels fixed.  The image is placed at native display size in
    # the center of a fixed square canvas instead of expanding to fill its
    # axes, so a 100px image remains visually small.
    canvas_edge = PREVIEW_DISPLAY_MAX_EDGE
    offset_x = (canvas_edge - image.width) / 2.0
    offset_y = (canvas_edge - image.height) / 2.0
    image_axis.imshow(
        image,
        interpolation="nearest",
        extent=(offset_x, offset_x + image.width, offset_y + image.height, offset_y),
    )
    image_axis.set_xlim(0, canvas_edge)
    image_axis.set_ylim(canvas_edge, 0)
    image_axis.set_aspect("equal", adjustable="box")
    image_axis.axis("off")
    image_axis.set_title("最终保存图像 + GT", fontproperties=mpl_font, fontsize=14)
    info_axis.axis("off")
    info_text = make_preview_info_text(dataset_key, save_name, config, record, prepared, resolved_info)
    # The right panel is intended for manual verification, so favor readable
    # type over fitting every detail into a small preview image.
    info_font_size = max(13.0, min(17.0, 680.0 / max(1, len(info_text.splitlines()))))
    info_axis.text(
        0,
        1,
        info_text,
        va="top",
        ha="left",
        fontsize=info_font_size,
        linespacing=1.25,
        fontproperties=mpl_font,
        transform=info_axis.transAxes,
    )
    title = f"{dataset_key}  预览 {index}/{total}"
    if interactive:
        title += "  |  空格/Enter/→/N：下一张，Q/Esc：取消（其他按键忽略）"
    figure.suptitle(title, fontproperties=mpl_font, fontsize=16)
    progress_axis = figure.add_axes((0.1, 0.035, 0.8, 0.025))
    progress_axis.barh([0], [index], color="#2b8cbe")
    progress_axis.set_xlim(0, max(1, total))
    progress_axis.set_yticks([])
    progress_axis.set_xticks(range(0, total + 1) if total <= 10 else [])
    progress_axis.set_title(f"{index}/{total}", fontproperties=mpl_font, fontsize=9, pad=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120, bbox_inches="tight")

    if not interactive:
        plt.close(figure)
        return
    # A preview is only a reference window.  Explicitly keep it non-modal and
    # remove the always-on-top flag so VS Code/Codex and other windows remain
    # usable while the image stays open.
    try:
        from matplotlib.backends.qt_compat import QtCore

        window = figure.canvas.manager.window
        window.setWindowModality(QtCore.Qt.NonModal)
        window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, False)
        window.show()
    except (AttributeError, ImportError):
        LOGGER.debug("Preview backend does not expose Qt window controls.")
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
        raise PreviewSkipped()


def make_preview_info_text(
    dataset_key: str,
    save_name: str,
    config: dict[str, Any],
    record: ImageRecord,
    prepared: PreparedImage,
    resolved_info: dict[str, Any],
) -> str:
    width, height = prepared_image_size(prepared, record.source_path)
    info = config.get("info", {})
    target_spec = info.get("target")
    # ``prepared.path`` is the exact image that will be copied to the output:
    # ori has already been cropped, whereas crop/lesion retain their input size.
    sections: list[tuple[str, Any]] = [
        ("info_config", info),
        ("保存图片分辨率", f"{width} × {height}"),
    ]
    if target_spec is None:
        # No target specification denotes an image-classification dataset.
        # Such images legitimately have no bbox, so never mark them negative.
        category = preview_category_value(record, config)
        sections.append(("类别", category if is_useful(category) else "未标注"))
    else:
        lesions: Any = "阴性"
        if prepared.boxes:
            lesions = []
            for index, box in enumerate(prepared.boxes):
                # Parsers normally put the label on each Box.  The fallback keeps
                # single-label datasets useful and preserves one-to-one display.
                label = box.label or box.label_id
                if label is None and index < len(record.labels):
                    label = record.labels[index]
                if label is None and len(record.labels) == 1:
                    label = record.labels[0]
                lesions.append(
                    {
                        "目标": label or "目标",
                        "bbox_norm_xyxy": normalized_box(box, width, height),
                    }
                )
        sections.append(("lesions", lesions))
    sections.append(("所处位置", resolved_info.get("location")))
    if "map_to_7plus1" in record.metadata:
        sections.append(("map_to_7plus1", record.metadata["map_to_7plus1"]))
    entries: list[str] = []
    for title, value in sections:
        if value is None or value == [] or value == {}:
            continue
        if title == "info_config":
            entries.append(f"{title}:\n" + format_preview_info_config(value))
            continue
        if title == "lesions":
            entries.append(format_preview_lesions(value))
            continue
        serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        wrapped = textwrap.wrap(str(serialized), width=46, replace_whitespace=False, break_long_words=True) or [""]
        entries.append("\n".join([f"{title}: {wrapped[0]}", *[f"    {line}" for line in wrapped[1:]]]))
    status_text = format_preview_status(resolved_info.get("status"))
    if status_text:
        entries.append(status_text)
    # Separate every field visually. Wrapped continuations stay attached to
    # their own field, while adjacent information items get one blank line.
    return "\n\n".join(entries)


def format_preview_info_config(value: Any) -> str:
    """Render only top-level config fields on separate lines.

    Lists remain compact on their owning field's line, matching the logical
    ``label`` / ``cls`` rows in rules.json instead of expanding each choice.
    """

    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return "\n".join(
        wrap_preview_field(str(key), json.dumps(item, ensure_ascii=False, default=str))
        for key, item in value.items()
    )


def format_preview_lesions(value: Any) -> str:
    """Show every target with its matching normalized bounding box."""

    if value == "阴性":
        return "目标: 阴性"
    if not isinstance(value, list):
        return f"目标: {value}"
    lines = ["目标:"]
    for index, lesion in enumerate(value, start=1):
        if not isinstance(lesion, dict):
            lines.append(f"  {index}. {lesion}")
            continue
        label = lesion.get("目标", "目标")
        box = lesion.get("bbox_norm_xyxy", [])
        lines.append(f"  {index}. {label}: {box}")
    return "\n".join(lines)


def wrap_preview_field(title: str, value: str, width: int = 50) -> str:
    """Wrap one top-level field, with continuation lines indented beneath it."""

    prefix = f"  {title}: "
    continuation = "    "
    wrapped = textwrap.wrap(
        value,
        width=max(12, width - len(prefix)),
        break_long_words=True,
        break_on_hyphens=True,
        replace_whitespace=False,
    ) or [""]
    return "\n".join([prefix + wrapped[0], *[continuation + line for line in wrapped[1:]]])


def format_preview_status(status: Any) -> str | None:
    """Display status as one major item with indented, line-by-line children."""

    if not isinstance(status, dict):
        return None
    items: list[str] = []
    handled_stages: set[str] = set()
    for stage_name, _, _ in STATUS_STAGE_SPECS:
        winner = status.get(stage_name)
        if winner is None:
            continue
        handled_stages.add(stage_name)
        if isinstance(winner, dict):
            label = winner.get("label")
            confidence = winner.get("confidence")
            if label is None or confidence is None:
                continue
            display_name = STATUS_STAGE_DISPLAY_NAMES.get(stage_name, stage_name)
            items.append(f"{display_name}：{label}（置信度：{float(confidence):.4f}）")
            continue
        if not is_useful(winner):
            continue
        display_name = STATUS_STAGE_DISPLAY_NAMES.get(stage_name, stage_name)
        items.append(f"{display_name}：{format_annotation_status_value(stage_name, winner)}")
    # Annotation-backed datasets can include additional status fields beyond
    # the seven model groups (for example lesion_morphology and gt_pathology).
    for stage_name, winner in status.items():
        if stage_name in handled_stages or not is_useful(winner):
            continue
        canonical_name = canonical_status_field_name(stage_name)
        display_name = STATUS_STAGE_DISPLAY_NAMES.get(canonical_name, canonical_name)
        if isinstance(winner, dict) and isinstance(winner.get("per_target"), list):
            target_values = []
            for item in winner["per_target"]:
                if not isinstance(item, dict) or not is_useful(item.get("value")):
                    continue
                target_values.append(
                    f"目标{item.get('target_index')}："
                    f"{format_annotation_status_value(canonical_name, item['value'])}"
                )
            if target_values:
                items.append(f"{display_name}：" + "；".join(target_values))
            continue
        items.append(f"{display_name}：{format_annotation_status_value(canonical_name, winner)}")
    return "状态:\n" + "\n".join(f"  {item}" for item in items) if items else None


def format_annotation_status_value(field_name: str, value: Any) -> str:
    """Normalize annotation status values to the local status vocabulary."""

    normalized = normalize_annotation_status_value(field_name, value)
    if isinstance(normalized, (list, tuple, set)):
        return "、".join(str(item) for item in normalized)
    return str(normalized)


def normalize_annotation_status_value(field_name: str, value: Any) -> Any:
    """Map an annotation value while preserving its scalar/list structure."""

    field_name = canonical_status_field_name(field_name)
    if isinstance(value, (list, tuple, set)):
        return [normalize_annotation_status_value(field_name, item) for item in value]
    if not isinstance(value, str):
        return value
    return ANNOTATION_STATUS_ALIASES.get(field_name, {}).get(value.casefold(), value)


def normalize_location_l2(value: Any) -> Any:
    """Map raw ``location_L2`` values to the GQC-aligned Chinese names."""

    if isinstance(value, (list, tuple, set)):
        return [normalize_location_l2(item) for item in value]
    if not isinstance(value, str):
        return value
    return LOCATION_L2_ALIASES.get(value.casefold(), value)


def canonical_status_field_name(field_name: str) -> str:
    """Resolve annotation aliases to their shared status field names."""

    return STATUS_FIELD_ALIASES.get(field_name, field_name)


def resolve_preview_info(
    record: ImageRecord,
    prepared: PreparedImage,
    config: dict[str, Any],
    stats: BuildStats,
) -> dict[str, Any]:
    """Resolve preview-only ``info`` values without constructing conversations."""

    info = config.get("info", {})
    result: dict[str, Any] = {}
    location_spec = info.get("location", info.get("localtion"))
    if location_spec is not None:
        result["location"] = resolve_info_field(
            "location", location_spec, record, prepared.path, config, stats
        )

    status_spec = info.get("status")
    if status_spec is not None:
        result["status"] = resolve_info_field(
            "status", status_spec, record, prepared.path, config, stats
        )
        if config.get("per_lesion_cadx") and isinstance(result["status"], dict):
            for raw_name in status_spec if isinstance(status_spec, list) else []:
                canonical_name = canonical_status_field_name(str(raw_name))
                if canonical_name not in {"lesion_morphology", "gt_pathology"}:
                    continue
                values = per_target_annotation_values(prepared.boxes, canonical_name)
                if values:
                    result["status"][canonical_name] = {"per_target": values}
                else:
                    result["status"].pop(canonical_name, None)
    return result


def load_preview_font(size: int) -> ImageFont.ImageFont:
    candidate = find_cjk_font_path()
    if candidate is not None:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            pass
    LOGGER.warning("No CJK font found; Chinese labels in preview may not render correctly.")
    return ImageFont.load_default()


def find_cjk_font_path() -> Path | None:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    return next((path for path in candidates if path.is_file()), None)


def safe_output_name(record: ImageRecord, record_index: int) -> str:
    split = "".join(char if char.isalnum() or char in "-_" else "_" for char in record.split)
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in record.source_path.stem)
    return f"{record_index}_{split}_{stem}{record.source_path.suffix.casefold()}"


def prepare_record_image(
    record: ImageRecord,
    config: dict[str, Any],
    image_dir: Path,
    dry_run: bool,
    record_index: int,
) -> list[PreparedImage]:
    destination = image_dir / safe_output_name(record, record_index)
    level = normalize_level(config.get("level", "crop"))
    if dry_run:
        metadata = dict(record.metadata)
        metadata["_prepared_level"] = level
        return [PreparedImage(destination, list(record.boxes), metadata)]
    image_dir.mkdir(parents=True, exist_ok=True)
    if level == "ori":
        return crop_original_image(record, destination, config)
    shutil.copy2(record.source_path, destination)
    metadata = dict(record.metadata)
    metadata["_prepared_level"] = level
    return [PreparedImage(destination, list(record.boxes), metadata)]


def image_size(path: Path, fallback: Path) -> tuple[int, int]:
    candidate = path if path.is_file() else fallback
    with Image.open(candidate) as image:
        return image.size


def prepared_image_size(prepared: PreparedImage, fallback: Path) -> tuple[int, int]:
    """Return and cache the final image size so adjust mode never reopens images."""

    if prepared.width is not None and prepared.height is not None:
        return prepared.width, prepared.height
    prepared.width, prepared.height = image_size(prepared.path, fallback)
    return prepared.width, prepared.height


def normalized_box(box: Box, width: int, height: int) -> list[float]:
    """Return [x_min, y_min, x_max, y_max], normalized to the final image."""

    x1, y1, x2, y2 = box.xyxy
    values = [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]
    return [round(value, 4) for value in values]


def aligned_location_mapping(record: ImageRecord, config: dict[str, Any], value: list[Any]) -> list[str] | None:
    info = config.get("info", {})
    label_ids = [str(item) for item in info.get("label", [])]
    if info.get("cls") is None and label_ids and len(value) == len(label_ids):
        mapping = {label_id: str(value[index]) for index, label_id in enumerate(label_ids)}
        return unique_preserve_order(mapping[item] for item in record.label_ids if item in mapping)
    return None


def metadata_fields(record: ImageRecord, field_names: Sequence[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    other_class = any(str(label).strip() == "其他" for label in record.labels)
    for raw_name in field_names:
        name = str(raw_name)
        canonical_name = canonical_status_field_name(name)
        if (
            record.metadata.get("_merged_by_original")
            and canonical_name in {"lesion_morphology", "gt_pathology"}
        ):
            values = per_target_annotation_values(record.boxes, canonical_name)
            if values:
                result[canonical_name] = {"per_target": values}
            continue
        if other_class and canonical_name in {"lesion_morphology", "gt_pathology"}:
            continue
        value = record.metadata.get(name)
        if not is_useful(value) and canonical_name != name:
            value = record.metadata.get(canonical_name)
        if not is_useful(value):
            for box in record.boxes:
                value = box.raw.get(name)
                if not is_useful(value) and canonical_name != name:
                    value = box.raw.get(canonical_name)
                if is_useful(value):
                    break
        if is_useful(value):
            result[canonical_name] = value
    return result


def per_target_annotation_values(
    boxes: Sequence[Box],
    field_name: str,
) -> list[dict[str, Any]]:
    """Return one lesion attribute per box while preserving target indices."""

    canonical_name = canonical_status_field_name(field_name)
    candidate_names = [canonical_name]
    candidate_names.extend(
        alias for alias, canonical in STATUS_FIELD_ALIASES.items() if canonical == canonical_name
    )
    values: list[dict[str, Any]] = []
    for index, box in enumerate(boxes, start=1):
        # Earlier dataset rules establish that the catch-all class has no
        # morphology/pathology annotation worth supervising.
        if str(box.label or "").strip() == "其他" and canonical_name in {
            "lesion_morphology",
            "gt_pathology",
        }:
            continue
        value = first_present(box.raw, *candidate_names)
        if not is_useful(value):
            continue
        values.append(
            {
                "target_index": index,
                "value": normalize_annotation_status_value(canonical_name, value),
            }
        )
    return values


def resolve_info_field(
    field_name: str,
    spec: Any,
    record: ImageRecord,
    prepared_path: Path,
    config: dict[str, Any],
    stats: BuildStats,
) -> Any:
    if isinstance(spec, str):
        if spec == "model_run":
            if _SKIP_MODEL_INFERENCE:
                stats.model_fields_omitted += 1
                return None
            cache = model_cache(record)
            if field_name in cache:
                return cache[field_name]
            runner = MODEL_RUNNERS.get(field_name, default_model_runner)
            value = runner(field_name, record, prepared_path, config)
            if not is_useful(value):
                stats.model_fields_omitted += 1
            return value
        if spec == "cls":
            if field_name == "location":
                labels = record.labels or record.label_ids
                return labels[0] if labels else None
            return record.labels
        # A string that exactly names an annotation field (e.g.
        # ``location_L2``) is a field reference.  Ordinary Chinese strings do
        # not match metadata keys and therefore remain fixed literal answers.
        if spec in record.metadata and is_useful(record.metadata[spec]):
            value = record.metadata[spec]
            if field_name == "location" and spec == "location_L2":
                return normalize_location_l2(value)
            return value
        return spec
    if isinstance(spec, list):
        if field_name == "location":
            mapped = aligned_location_mapping(record, config, spec)
            if mapped is not None:
                return mapped
        return metadata_fields(record, spec)
    return spec


def plain_answer(value: Any) -> str:
    """Render an answer without JSON quotes, brackets, or confidence suffixes."""

    if isinstance(value, (list, tuple, set)):
        answers: list[str] = []
        for item in value:
            answer = plain_answer(item)
            if is_useful(answer):
                answers.append(answer)
        answers = unique_preserve_order(answers)
        return "、".join(str(answer) for answer in answers)
    if isinstance(value, dict):
        if is_useful(value.get("label")):
            return str(value["label"])
        return "、".join(
            f"{key}：{plain_answer(item)}" for key, item in value.items() if is_useful(item)
        )
    return str(value).strip()


def semantic_prediction_answer(value: Any) -> str:
    """Convert debug model output into the supervised semantic answer."""

    if isinstance(value, (list, tuple, set)):
        answers: list[str] = []
        for item in value:
            answer = semantic_prediction_answer(item)
            if is_useful(answer):
                answers.append(answer)
        return "、".join(str(item) for item in unique_preserve_order(answers))
    if isinstance(value, dict):
        if is_useful(value.get("label")):
            return str(value["label"])
        return plain_answer(value)
    text = str(value).strip()
    low_confidence = LOW_CONFIDENCE_MAX_CLS_PATTERN.search(text)
    if low_confidence:
        return f"可能为{low_confidence.group(1).strip()}"
    return CONFIDENCE_SUFFIX_PATTERN.sub("", text).strip()


def category_contains_chinese(value: Any) -> bool:
    """Whether the displayed category contains a meaningful Chinese label."""

    return bool(CHINESE_CHARACTER_PATTERN.search(plain_answer(value)))


def cadx_answer_with_description(category_answer: str, config: dict[str, Any]) -> str:
    """Add a diagnosis subject when ``description`` says labels are only grades."""

    description = str(config.get("description", "")).casefold()
    for marker, subject in CADX_DESCRIPTION_SUBJECT_RULES:
        if marker not in description:
            continue
        if subject in category_answer:
            return category_answer
        return f"{subject}，{category_answer}"
    return category_answer


def status_question(stage_name: str) -> str:
    canonical_name = canonical_status_field_name(stage_name)
    if canonical_name in STATUS_QUESTIONS:
        return STATUS_QUESTIONS[canonical_name]
    display_name = STATUS_STAGE_DISPLAY_NAMES.get(canonical_name, canonical_name)
    return f"该图像的{display_name}是什么？"


def status_answer(stage_name: str, value: Any) -> str:
    canonical_name = canonical_status_field_name(stage_name)
    if isinstance(value, dict) or (
        isinstance(value, str) and LOW_CONFIDENCE_MAX_CLS_PATTERN.search(value)
    ):
        return semantic_prediction_answer(value)
    normalized = normalize_annotation_status_value(canonical_name, value)
    return semantic_prediction_answer(normalized)


def approximate_box_position(box: Sequence[float]) -> str:
    """Describe a normalized xyxy box by its center in a 3x3 image grid."""

    center_x = (float(box[0]) + float(box[2])) / 2
    center_y = (float(box[1]) + float(box[3])) / 2
    column = 0 if center_x < 1 / 3 else (2 if center_x > 2 / 3 else 1)
    row = 0 if center_y < 1 / 3 else (2 if center_y > 2 / 3 else 1)
    positions = (
        ("左上角", "上方", "右上角"),
        ("左侧", "图像中央", "右侧"),
        ("左下角", "下方", "右下角"),
    )
    return positions[row][column]


def box_display_label(box: Box, record: ImageRecord, index: int) -> str:
    label = box.label or box.label_id
    if label is None and index < len(record.labels):
        label = record.labels[index]
    if label is None and len(record.labels) == 1:
        label = record.labels[0]
    return str(label or "目标")


def per_target_answer(
    values: Sequence[dict[str, Any]],
    total_targets: int,
) -> str:
    """Render lesion values, omitting a redundant target number for one box."""

    useful_values = [item for item in values if is_useful(item.get("value"))]
    if total_targets == 1 and useful_values:
        return f"{plain_answer(useful_values[0]['value'])}。"
    return "\n".join(
        f"{item['target_index']}. 目标{item['target_index']}：{plain_answer(item['value'])}。"
        for item in useful_values
    )


def build_per_target_cadx_qa(
    record: ImageRecord,
    prepared: PreparedImage,
    config: dict[str, Any],
) -> tuple[str, str] | None:
    """Build one diagnosis line for every lesion on a merged full image."""

    values: list[dict[str, Any]] = []
    for index, box in enumerate(prepared.boxes, start=1):
        label = box_display_label(box, record, index - 1)
        # In the merged NICE/CADX datasets, the catch-all class denotes a
        # detector false positive.  It belongs in CADE as "疑似假阳", not
        # in a lesion diagnosis/grading answer.
        if label.strip() == "其他":
            continue
        if not category_contains_chinese(label):
            continue
        values.append(
            {
                "target_index": index,
                "value": cadx_answer_with_description(label, config),
            }
        )
    if not values:
        return None
    question = (
        "请进行CADX：鉴别图像中的病变，并给出诊断及相应分级。"
        if len(prepared.boxes) == 1
        else "请进行CADX：逐一鉴别图像中的目标，并给出诊断及相应分级。"
    )
    return question, per_target_answer(values, len(prepared.boxes))


def is_bowel_cleanliness_dataset(dataset_key: str) -> bool:
    tokens = re.split(r"[^a-z0-9]+", dataset_key.casefold())
    return "bc" in tokens


def build_cade_qa(
    record: ImageRecord,
    prepared: PreparedImage,
    target_spec: Any,
    config: dict[str, Any],
) -> tuple[str, str] | None:
    if target_spec is None:
        return None

    is_cleanliness = is_bowel_cleanliness_dataset(record.dataset_key)
    question = (
        "图像上是否有肠道清洁不到位的区域，例如粪渣、粪块、粪水、气泡等杂质？"
        if is_cleanliness
        else "请进行CADE：图像上是否有病变存在？若有，请指出病变位置。"
    )
    if not prepared.boxes:
        answer = (
            "未发现清洁不到位区域。"
            if is_cleanliness
            else "未发现病变。"
        )
        return question, answer

    width, height = prepared_image_size(prepared, record.source_path)
    details: list[str] = []
    per_lesion_cadx = bool(config.get("per_lesion_cadx"))
    single_target = len(prepared.boxes) == 1
    false_positive_flags: list[bool] = []
    for index, box in enumerate(prepared.boxes):
        coordinates = normalized_box(box, width, height)
        diagnosis_label = box_display_label(box, record, index)
        is_false_positive = per_lesion_cadx and diagnosis_label.strip() == "其他"
        false_positive_flags.append(is_false_positive)
        if per_lesion_cadx:
            if single_target:
                label = "疑似假阳" if is_false_positive else "病变"
            else:
                label = f"目标{index + 1}"
                if is_false_positive:
                    label += "（疑似假阳）"
        else:
            label = diagnosis_label
        detail_prefix = "" if single_target else f"{index + 1}. "
        details.append(
            f"{detail_prefix}{label}，大约在{approximate_box_position(coordinates)}，"
            f"{json.dumps(coordinates, ensure_ascii=False)}"
        )
    answer_lead = (
        "未发现明确病变，但有疑似假阳区域。"
        if per_lesion_cadx and all(false_positive_flags)
        else "有。"
    )
    answer = (
        answer_lead
        + "\n"
        + "\n".join(details)
        + "\n坐标：归一化 xyxy [x_min, y_min, x_max, y_max]（0-1）。"
    )
    return question, answer


def build_conversations(
    record: ImageRecord,
    prepared: PreparedImage,
    config: dict[str, Any],
    stats: BuildStats,
    resolved_info: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    info = config.get("info", {})
    qas: list[tuple[str, str]] = []
    lesion_qas: list[tuple[str, str]] = []
    per_lesion_cadx = bool(config.get("per_lesion_cadx"))

    resolved_info = resolved_info or {}

    dataset_key = record.dataset_key.casefold()
    if dataset_key.startswith("gs_"):
        qas.append(("这是肠镜图像还是胃镜图像？", "胃镜图像"))
    elif dataset_key.startswith("cs_"):
        qas.append(("这是肠镜图像还是胃镜图像？", "肠镜图像"))

    location_spec = info.get("location", info.get("localtion"))
    location_value = (
        resolved_info["location"]
        if "location" in resolved_info
        else resolve_info_field("location", location_spec, record, prepared.path, config, stats)
    )
    mapped_location = record.metadata.get("map_to_7plus1")
    if is_useful(mapped_location):
        qas.append(
            ("该图像所处的消化道大类位置是什么？", semantic_prediction_answer(mapped_location))
        )
    if is_useful(location_value):
        qas.append(("该图像所处的具体位置是什么？", semantic_prediction_answer(location_value)))

    status_spec = info.get("status")
    status_value = (
        resolved_info["status"]
        if "status" in resolved_info
        else resolve_info_field("status", status_spec, record, prepared.path, config, stats)
    )
    if isinstance(status_value, dict):
        allowed_status_names: set[str] | None = None
        if isinstance(status_spec, list):
            allowed_status_names = {
                canonical_status_field_name(str(name)) for name in status_spec
            }
        ordered_status_names = [stage_name for stage_name, _, _ in STATUS_STAGE_SPECS]
        ordered_status_names.extend(
            name for name in status_value if name not in ordered_status_names
        )
        handled_names: set[str] = set()
        for raw_name in ordered_status_names:
            if raw_name not in status_value or not is_useful(status_value[raw_name]):
                continue
            canonical_name = canonical_status_field_name(raw_name)
            if allowed_status_names is not None and canonical_name not in allowed_status_names:
                continue
            if canonical_name in handled_names:
                continue
            handled_names.add(canonical_name)
            if (
                per_lesion_cadx
                and canonical_name in {"lesion_morphology", "gt_pathology"}
            ):
                target_values = per_target_annotation_values(prepared.boxes, canonical_name)
                if target_values:
                    if len(prepared.boxes) == 1:
                        question = status_question(canonical_name)
                    else:
                        question = (
                            "图像中各目标的形态学表现是什么？"
                            if canonical_name == "lesion_morphology"
                            else "图像中各目标的病理结论是什么？"
                        )
                    lesion_qas.append(
                        (question, per_target_answer(target_values, len(prepared.boxes)))
                    )
                continue
            qas.append(
                (status_question(canonical_name), status_answer(canonical_name, status_value[raw_name]))
            )
    elif is_useful(status_value):
        qas.append(("该图像的状态是什么？", semantic_prediction_answer(status_value)))

    target_spec = info.get("target")
    if per_lesion_cadx:
        # CADE establishes target numbers and coordinates first.  Subsequent
        # lesion-level morphology/CADX answers can then refer to target1,
        # target2, ... without an arbitrary or ambiguous ordering.
        cade_qa = build_cade_qa(record, prepared, target_spec, config)
        if cade_qa is not None:
            qas.append(cade_qa)
        qas.extend(lesion_qas)
        cadx_qa = build_per_target_cadx_qa(record, prepared, config)
        if cadx_qa is not None:
            qas.append(cadx_qa)
    else:
        qas.extend(lesion_qas)

    category = preview_category_value(record, config) if target_spec is None else None
    category_answer = plain_answer(category) if is_useful(category) else ""
    location_answer = semantic_prediction_answer(location_value) if is_useful(location_value) else ""
    if (
        target_spec is None
        and category_answer
        and category_contains_chinese(category)
        and category_answer != location_answer
    ):
        question = "请进行CADX：鉴别图像中的病变，并给出诊断及相应分级。"
        qas.append((question, cadx_answer_with_description(category_answer, config)))

    if not per_lesion_cadx:
        cade_qa = build_cade_qa(record, prepared, target_spec, config)
        if cade_qa is not None:
            qas.append(cade_qa)

    conversations: list[dict[str, str]] = []
    for question, answer in qas:
        if not is_useful(answer):
            continue
        if not conversations:
            question = "<image>" + question
        conversations.append({"from": "human", "value": question})
        conversations.append({"from": "gpt", "value": answer})
    return conversations


def make_vlm_sample(
    record: ImageRecord,
    prepared: PreparedImage,
    config: dict[str, Any],
    stats: BuildStats,
    resolved_info: dict[str, Any] | None = None,
    image_path: Path | None = None,
) -> dict[str, Any] | None:
    conversations = build_conversations(record, prepared, config, stats, resolved_info)
    if not conversations:
        return None
    sample_image = image_path or prepared.path
    return {"images": [str(sample_image.resolve(strict=False))], "conversations": conversations}


def build_one_dataset(
    dataset_key: str,
    config: dict[str, Any],
    records: Sequence[ImageRecord],
    output_root: Path,
    dry_run: bool,
    limit: int | None,
    aggregate_writer: JsonArrayWriter | None,
    cache_writer: PreparedSampleCacheWriter | None,
) -> BuildStats:
    stats = BuildStats(discovered=len(records))
    save_name = str(config["save_name"])
    image_dir = output_root / save_name
    selected = records[:limit] if limit is not None else records
    with tqdm(
        selected,
        total=len(selected),
        desc=f"[{dataset_key}] 生成",
        unit="张",
        dynamic_ncols=True,
        leave=True,
    ) as progress:
        for record_index, record in enumerate(progress, start=1):
            if not record.source_path.is_file():
                stats.skipped_missing_image += 1
            else:
                for prepared in prepare_record_image(
                    record, config, image_dir, dry_run, record_index=record_index
                ):
                    resolved_info = resolve_preview_info(record, prepared, config, stats)
                    prepared.metadata.update(record.metadata)
                    if cache_writer is not None:
                        prepared_image_size(prepared, record.source_path)
                        cache_writer.write(record_index, record, prepared, resolved_info)
                    sample = make_vlm_sample(
                        record,
                        prepared,
                        config,
                        stats,
                        resolved_info=resolved_info,
                    )
                    if sample is None:
                        stats.skipped_empty_dialogue += 1
                        continue
                    if aggregate_writer is not None:
                        aggregate_writer.write(sample)
                    stats.written += 1
            progress.set_postfix(
                generated=stats.written,
                skipped=stats.skipped_missing_image + stats.skipped_empty_dialogue,
                refresh=False,
            )
            if record_index % PROGRESS_LOG_INTERVAL == 0 or record_index == len(selected):
                skipped = stats.skipped_missing_image + stats.skipped_empty_dialogue
                percentage = 100.0 * record_index / max(1, len(selected))
                LOGGER.info(
                    "[%s] Progress %d/%d (%.2f%%), generated=%d, skipped=%d",
                    dataset_key,
                    record_index,
                    len(selected),
                    percentage,
                    stats.written,
                    skipped,
                )
    return stats


def validate_config(config: dict[str, Any]) -> None:
    allowed_formats = set(FORMAT_PARSERS)
    seen_save_names: dict[str, str] = {}
    for dataset_key, dataset in config.items():
        missing = [key for key in ("save_name", "src_path", "format", "level", "info") if key not in dataset]
        if missing:
            raise ValueError(f"[{dataset_key}] Missing required config keys: {missing}")
        format_name = str(dataset["format"]).casefold()
        if format_name not in allowed_formats:
            raise ValueError(f"[{dataset_key}] Unsupported format {dataset['format']!r}")
        normalize_level(dataset["level"])
        save_name = str(dataset["save_name"])
        if save_name in seen_save_names:
            raise ValueError(
                f"[{dataset_key}] save_name duplicates {seen_save_names[save_name]!r}: {save_name!r}"
            )
        seen_save_names[save_name] = dataset_key
        info = dataset.get("info", {})
        labels = info.get("label", [])
        classes = info.get("cls", [])
        if labels and classes and len(labels) != len(classes):
            raise ValueError(f"[{dataset_key}] info.label and info.cls must have equal lengths")


def validate_output_is_separate(output_root: Path, config: dict[str, Any]) -> None:
    """Refuse any output/source overlap to preserve source datasets as read-only."""

    output = str(output_root.resolve(strict=False))
    for dataset in config.values():
        for root_field in ("src_path", "original_src_path"):
            raw_roots = dataset.get(root_field, [])
            if isinstance(raw_roots, str):
                raw_roots = [raw_roots]
            for raw_root in raw_roots:
                source = str(Path(raw_root).resolve(strict=False))
                try:
                    common = os.path.commonpath([output, source])
                except ValueError:
                    continue
                if common in {output, source}:
                    raise ValueError(
                        "Output root and source dataset must not overlap: "
                        f"output={output}, source={source}"
                    )


def select_dataset_keys(config: dict[str, Any], requested: Sequence[str] | None) -> list[str]:
    if not requested:
        return list(config)
    result: list[str] = []
    for item in requested:
        for key in item.split(","):
            key = key.strip()
            if key not in config:
                raise KeyError(f"Unknown dataset key: {key}. Available: {', '.join(config)}")
            result.append(key)
    return unique_preserve_order(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to rules.json.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Generated dataset root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--run-mode",
        choices=("full", "add", "adjust"),
        default="full",
        help=(
            "full: replace all output; add: append new --datasets to an existing completed output; "
            "adjust: rebuild only dialogues from the completed cache."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "Dataset keys to build; accepts spaces or comma-separated values. "
            "Required by add mode; full mode defaults to all."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional fixed seed for reproducible previews; omit it to sample different images each run.",
    )
    parser.add_argument("--preview-count", type=int, default=10, help="Random GT images shown per sub-dataset.")
    parser.add_argument(
        "--preview-mode",
        choices=("show", "save", "none"),
        default="show",
        help="show: interactively show/save each image; save: only save detailed images; none: disable.",
    )
    parser.add_argument("--yes", action="store_true", help="Continue without the per-dataset text confirmation.")
    parser.add_argument("--search-depth", type=int, default=6, help="Maximum metadata/sub-folder discovery depth.")
    parser.add_argument("--limit", type=int, help="Maximum records per dataset, useful for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Discover and validate without copying/writing output.")
    parser.add_argument("--gpu-index", type=int, default=0, help="CUDA device index used by TensorRTInfer.")
    parser.add_argument(
        "--trt-python",
        type=Path,
        default=DEFAULT_TRT_PYTHON,
        help=f"Python interpreter containing TensorRT/PyCUDA (default: {DEFAULT_TRT_PYTHON}).",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def adjust_conversations(config: dict[str, Any], config_path: Path, output_root: Path) -> int:
    """Atomically rebuild conversations from cache without touching prepared images."""

    global _SKIP_MODEL_INFERENCE
    _SKIP_MODEL_INFERENCE = True
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"Adjust mode requires an existing full-run output directory: {output_root}"
        )
    log_path = configure_logging("INFO", output_root, append=True)
    LOGGER.info("=" * 80)
    LOGGER.info("Conversation adjustment run started; log appended to %s", log_path)
    LOGGER.info("Config: %s; output root: %s", config_path.resolve(), output_root.resolve())
    cache_path = output_root / CACHE_FILE_NAME
    aggregate_path = output_root / "all_conversations.json"
    stats = BuildStats()
    dataset_counts: dict[str, int] = defaultdict(int)

    with PreparedSampleCacheReader(cache_path, output_root) as cache_reader:
        stats.discovered = len(cache_reader)
        for dataset_key, dataset_config in config.items():
            if not dataset_config.get("merge_original_lesions"):
                continue
            cached = cache_reader.first_metadata(dataset_key)
            if cached is not None and cached.get("_prepared_level") != "ori":
                raise RuntimeError(
                    f"Cached dataset {dataset_key!r} predates the merged-original-image pipeline. "
                    "Run --run-mode full before using adjust."
                )
        with JsonArrayWriter(aggregate_path) as aggregate_writer:
            with tqdm(
                cache_reader.samples(),
                total=len(cache_reader),
                desc="[adjust] 重建对话",
                unit="条",
                dynamic_ncols=True,
                leave=True,
            ) as progress:
                for record, prepared, resolved_info in progress:
                    if not prepared.path.is_file():
                        raise FileNotFoundError(
                            f"Cached prepared image is missing; adjust mode will not recreate it: {prepared.path}"
                        )
                    dataset_config = config.get(record.dataset_key)
                    if dataset_config is None:
                        raise KeyError(
                            f"Cached dataset {record.dataset_key!r} is absent from current rules.json."
                        )
                    if (
                        dataset_config.get("merge_original_lesions")
                        and prepared.metadata.get("_prepared_level") != "ori"
                    ):
                        raise RuntimeError(
                            f"Cached dataset {record.dataset_key!r} was not prepared with the current "
                            "merged-original-image pipeline. Run --run-mode full before using adjust."
                        )
                    sample = make_vlm_sample(
                        record,
                        prepared,
                        dataset_config,
                        stats,
                        resolved_info=resolved_info,
                    )
                    if sample is None:
                        stats.skipped_empty_dialogue += 1
                        continue
                    aggregate_writer.write(sample)
                    stats.written += 1
                    dataset_counts[record.dataset_key] += 1
                    progress.set_postfix(
                        generated=stats.written,
                        skipped=stats.skipped_empty_dialogue,
                        refresh=False,
                    )

    for dataset_key, count in dataset_counts.items():
        LOGGER.info("[%s] Rebuilt %d conversation samples from cache.", dataset_key, count)
    if stats.model_fields_omitted:
        LOGGER.warning(
            "Adjust mode omitted %d model-backed fields absent from the cache; no inference was run.",
            stats.model_fields_omitted,
        )
    LOGGER.info("Adjusted combined dataset written to %s", aggregate_path)
    LOGGER.info(
        "Conversation adjustment completed successfully; cached=%d, written=%d, skipped=%d",
        stats.discovered,
        stats.written,
        stats.skipped_empty_dialogue,
    )
    return 0


def main() -> int:
    global _SKIP_MODEL_INFERENCE
    args = parse_args()
    config = json_load(args.config)
    if not isinstance(config, dict):
        raise TypeError("Top-level rules.json value must be an object.")
    validate_config(config)
    if args.run_mode == "adjust":
        return adjust_conversations(config, args.config, args.output_root)
    if args.run_mode == "add" and not args.datasets:
        raise ValueError("Add mode requires --datasets; implicit addition of every dataset is disabled.")

    _SKIP_MODEL_INFERENCE = args.dry_run
    validate_output_is_separate(args.output_root, config)
    dataset_keys = select_dataset_keys(config, args.datasets)
    resolver = ImageResolver()
    add_mode = args.run_mode == "add"
    existing_conversation_samples = 0
    if add_mode:
        if not args.output_root.is_dir():
            raise FileNotFoundError(
                f"Add mode requires an existing full-run output directory: {args.output_root}"
            )
        cache_path = args.output_root / CACHE_FILE_NAME
        aggregate_path = args.output_root / "all_conversations.json"
        manifest_path = args.output_root / "manifest.json"
        for required_path in (cache_path, aggregate_path, manifest_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"Add mode requires existing output file: {required_path}")
        with PreparedSampleCacheReader(cache_path, args.output_root) as cache_reader:
            existing_dataset_keys = cache_reader.dataset_keys()
            duplicate_keys = [key for key in dataset_keys if key in existing_dataset_keys]
            if duplicate_keys:
                raise ValueError(
                    "Add mode refuses to duplicate datasets already in the cache: "
                    + ", ".join(duplicate_keys)
                )
            existing_conversation_samples = int(
                cache_reader.meta("conversation_samples") or len(cache_reader)
            )
        for dataset_key in dataset_keys:
            image_dir = args.output_root / str(config[dataset_key]["save_name"])
            if image_dir.exists() and (
                not image_dir.is_dir() or any(image_dir.iterdir())
            ):
                raise FileExistsError(
                    f"Add target directory already contains files but has no cache records: {image_dir}"
                )
        manifest = json_load(manifest_path)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), dict):
            raise TypeError(f"Existing manifest has an invalid structure: {manifest_path}")
        log_path = configure_logging(args.log_level, args.output_root, append=True)
        LOGGER.info("=" * 80)
        LOGGER.info("Add run started; existing output will be preserved; log file: %s", log_path)
    else:
        output_existed = args.output_root.exists()
        if not confirm_and_reset_output_root(args.output_root):
            print("已取消生成，现有输出文件夹未作任何修改。")
            return 0
        log_path = configure_logging(args.log_level, args.output_root)
        LOGGER.info("Generation run started; log file: %s", log_path)
        if output_existed:
            LOGGER.info("Previous output directory was fully removed before this run.")
        manifest = {
            "config": str(args.config.resolve()),
            "output_root": str(args.output_root.resolve(strict=False)),
            "prepared_cache": CACHE_FILE_NAME,
            "prepared_cache_schema": CACHE_SCHEMA_VERSION,
            "bbox_format": "[x_min, y_min, x_max, y_max], normalized to final saved image, 4 decimals",
            "datasets": {},
        }
    LOGGER.info("Config: %s; output root: %s", args.config.resolve(), args.output_root.resolve())
    configure_inno_runtime(args.gpu_index, args.trt_python)
    rng = random.Random(args.seed)
    output_dataset_info: dict[str, Any] = {
        "max_v1_single": {
            "file_name": "all_conversations.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
        }
    }

    discovered_datasets: list[tuple[str, dict[str, Any], list[ImageRecord]]] = []
    for dataset_key in dataset_keys:
        dataset_config = config[dataset_key]
        LOGGER.info("[%s] Discovering %s dataset ...", dataset_key, dataset_config["format"])
        records = discover_dataset_records(dataset_key, dataset_config, resolver, args.search_depth)
        LOGGER.info("[%s] Discovered %d usable source images.", dataset_key, len(records))
        log_subfolder_counts(dataset_key, dataset_config, records)
        if not records:
            manifest["datasets"][dataset_key] = {"warning": "no usable records discovered"}
            continue
        # Preview every dataset in discovery order. Final generation remains
        # deferred until all previews finish and the user confirms once.
        if not args.dry_run:
            preview_dataset(
                dataset_key,
                str(dataset_config["save_name"]),
                dataset_config,
                records,
                args.output_root / "previews",
                args.preview_count,
                args.preview_mode,
                rng,
            )
        discovered_datasets.append((dataset_key, dataset_config, records))

    if not args.dry_run and not args.yes:
        if not sys.stdin.isatty():
            raise RuntimeError("Final preview confirmation requires a TTY. Rerun with --yes to skip it.")
        answer = input("全部数据集预览已完成。确认标注及全部 info 正确并开始统一生成？[y/N]: ").strip().casefold()
        if answer not in {"y", "yes"}:
            raise RuntimeError("Generation cancelled after all dataset previews.")

    new_samples = 0
    aggregate_path = args.output_root / "all_conversations.json"
    aggregate_context = (
        contextlib.nullcontext(None)
        if args.dry_run or add_mode
        else JsonArrayWriter(aggregate_path)
    )
    cache_path = args.output_root / CACHE_FILE_NAME
    cache_context = (
        contextlib.nullcontext(None)
        if args.dry_run
        else PreparedSampleCacheWriter(
            cache_path,
            args.output_root,
            args.config,
            append=add_mode,
        )
    )
    # Keep the cache context outermost so a failure while atomically finalizing
    # all_conversations.json resets the cache's completion marker.
    with cache_context as cache_writer, aggregate_context as aggregate_writer:
        for dataset_key, dataset_config, records in discovered_datasets:
            stats = build_one_dataset(
                dataset_key,
                dataset_config,
                records,
                args.output_root,
                args.dry_run,
                args.limit,
                aggregate_writer,
                cache_writer,
            )
            new_samples += stats.written
            manifest["datasets"][dataset_key] = {
                "save_name": dataset_config["save_name"],
                "format": dataset_config["format"],
                "level": dataset_config["level"],
                **asdict(stats),
            }
            LOGGER.info("[%s] Generated %d VLM samples.", dataset_key, stats.written)
        if cache_writer is not None:
            cache_writer.mark_complete(existing_conversation_samples + new_samples)

    total_samples = existing_conversation_samples + new_samples
    manifest["total_samples"] = total_samples
    if not args.dry_run:
        atomic_json_dump(manifest, args.output_root / "manifest.json")
        if add_mode:
            LOGGER.info(
                "Added %d samples; rebuilding the combined conversation JSON from cache.",
                new_samples,
            )
            adjust_conversations(config, args.config, args.output_root)
            LOGGER.info("Add run completed successfully; total samples: %d", total_samples)
            return 0
        atomic_json_dump(output_dataset_info, args.output_root / "dataset_info.json")
        LOGGER.info("Combined dataset written to %s", aggregate_path)
        LOGGER.info("Prepared sample cache written to %s", cache_path)
    else:
        LOGGER.info("Dry run complete: %d new samples would be generated.", new_samples)
    LOGGER.info("Generation run completed successfully; total samples: %d", total_samples)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Generation interrupted by user.")
        raise SystemExit(130)
    except Exception:
        LOGGER.exception("Generation failed with an unhandled exception.")
        raise
