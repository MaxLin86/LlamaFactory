#!/usr/bin/env python3
"""Export completed HuggingFace Arrow caches as a reusable tokenized DatasetDict.

This utility is for the one-time case where LLaMA-Factory has already finished
tokenizing the full dataset before ``tokenized_path`` was configured. It does
not tokenize images again; it validates the completed train/validation Arrow
shards and writes the standard ``save_to_disk`` layout expected by
LLaMA-Factory.

Run with the LLaMA-Factory environment, for example::

    /home/maxlin/anaconda3/envs/llama_factory/bin/python \
      data/inno_report_data/export_tokenized_cache.py
"""

import argparse
import os
from pathlib import Path
from typing import List, Sequence

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


DEFAULT_CACHE_DIR = Path(
    "/media/maxlin/SATA/Huggingface/datasets/json/"
    "default-8d2a0e129072b577/0.0.0/"
    "f4e89e8750d5d5ffbef2c078bf0ddfedef29dc2faff52a6255cf513c05eb1092"
)
DEFAULT_OUTPUT_PATH = Path(
    "/media/maxlin/SATA/ReportData/tokenized/"
    "max_v1_qwen3vl2b_min256_max512_cut2048_val20k"
)
DEFAULT_TRAIN_PREFIX = "cache-882c82a29ccadb36"
DEFAULT_VALIDATION_PREFIX = "cache-9e8a4dd46de4e863"
EXPECTED_COLUMNS = ["input_ids", "attention_mask", "labels", "images", "videos", "audios"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert completed HF Arrow cache shards into a tokenized DatasetDict."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--train-prefix", default=DEFAULT_TRAIN_PREFIX)
    parser.add_argument("--validation-prefix", default=DEFAULT_VALIDATION_PREFIX)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--expected-train-size", type=int, default=853124)
    parser.add_argument("--expected-validation-size", type=int, default=20000)
    parser.add_argument(
        "--max-shard-size",
        default="1GB",
        help="Maximum size of each Arrow shard written by save_to_disk (default: 1GB).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report source shards without writing the output dataset.",
    )
    return parser.parse_args()


def discover_shards(cache_dir: Path, prefix: str) -> List[Path]:
    shards = sorted(cache_dir.glob("{}*.arrow".format(prefix)))
    if not shards:
        raise FileNotFoundError("No Arrow shards found for prefix {!r} under {}".format(prefix, cache_dir))

    return shards


def load_split(name: str, shards: Sequence[Path], expected_size: int) -> Dataset:
    datasets = []
    total = 0
    reference_columns = None

    print("[{}] source shards:".format(name))
    for shard in shards:
        dataset = Dataset.from_file(str(shard))
        columns = dataset.column_names
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(
                "Column mismatch in {}: expected {}, got {}".format(shard, reference_columns, columns)
            )

        if columns != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected columns in {}: expected {}, got {}".format(shard, EXPECTED_COLUMNS, columns)
            )

        datasets.append(dataset)
        total += len(dataset)
        print("  {}: {:,} examples ({:.2f} GiB)".format(shard.name, len(dataset), shard.stat().st_size / 1024**3))

    if total != expected_size:
        raise ValueError(
            "{} size mismatch: expected {:,}, found {:,}. Refusing to export.".format(
                name, expected_size, total
            )
        )

    merged = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    print("[{}] validated total: {:,} examples".format(name, len(merged)))
    return merged


def validate_output(path: Path, expected_train_size: int, expected_validation_size: int) -> None:
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected DatasetDict at {}, got {}".format(path, type(dataset).__name__))

    if set(dataset.keys()) != {"train", "validation"}:
        raise ValueError("Unexpected dataset splits at {}: {}".format(path, list(dataset.keys())))

    actual_train_size = len(dataset["train"])
    actual_validation_size = len(dataset["validation"])
    if actual_train_size != expected_train_size or actual_validation_size != expected_validation_size:
        raise ValueError(
            "Saved dataset size mismatch: train={:,}, validation={:,}".format(
                actual_train_size, actual_validation_size
            )
        )


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    output_path = args.output_path.resolve()
    staging_path = output_path.with_name(output_path.name + ".incomplete")

    if not cache_dir.is_dir():
        raise NotADirectoryError("HF cache directory does not exist: {}".format(cache_dir))

    train_shards = discover_shards(cache_dir, args.train_prefix)
    validation_shards = discover_shards(cache_dir, args.validation_prefix)
    train_dataset = load_split("train", train_shards, args.expected_train_size)
    validation_dataset = load_split("validation", validation_shards, args.expected_validation_size)

    print("Output path: {}".format(output_path))
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    if output_path.exists():
        raise FileExistsError(
            "Output path already exists; refusing to overwrite it: {}".format(output_path)
        )
    if staging_path.exists():
        raise FileExistsError(
            "Incomplete staging path already exists; inspect or remove it before retrying: {}".format(
                staging_path
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset})
    dataset_dict.save_to_disk(str(staging_path), max_shard_size=args.max_shard_size)
    validate_output(staging_path, args.expected_train_size, args.expected_validation_size)
    os.replace(str(staging_path), str(output_path))

    print("Export complete: {}".format(output_path))
    print("train={:,}, validation={:,}".format(args.expected_train_size, args.expected_validation_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
