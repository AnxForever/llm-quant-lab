"""量化精度影响：在同一份标注数据上比较 PyTorch FP32、ONNX FP32、ONNX INT8。

为什么必须做这一步：
- 只报"体积降到 25%"是不完整的；量化可能改变预测。
- 只看"准确率没掉"也不够；要同时看**判定翻转率**，因为翻转可能正负抵消。
- 所以这里同时报 accuracy、与基准的判定一致率、以及翻转条数。
"""

from __future__ import annotations

import csv as csvmod
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AccuracyConfig:
    """一次精度对比的参数。"""

    model_id: str = "AnxForever/chinese-ai-detector-bert"
    dataset_id: str = "AnxForever/chinese-ai-detection-dataset"
    split_file: str = "test.csv"
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 256
    batch_size: int = 16

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AccuracyReport:
    config: dict
    samples: int
    results: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"config": self.config, "samples": self.samples, "results": self.results, "notes": self.notes}


def load_labeled_rows(config: AccuracyConfig, cache_dir: Path | None = None) -> list[dict]:
    """读取标注数据；优先用本地缓存，缺失时从 Hugging Face 下载。"""
    if cache_dir is not None:
        local = cache_dir / config.split_file
        if local.is_file():
            return list(csvmod.DictReader(local.open(encoding="utf-8")))

    cached = sorted(
        Path.home().glob(
            f".cache/huggingface/hub/datasets--{config.dataset_id.replace('/', '--')}/snapshots/*/{config.split_file}"
        )
    )
    if cached:
        return list(csvmod.DictReader(cached[-1].open(encoding="utf-8")))

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(config.dataset_id, config.split_file, repo_type="dataset")
    return list(csvmod.DictReader(open(path, encoding="utf-8")))


def infer_torch(model_id: str, texts: list[str], config: AccuracyConfig) -> np.ndarray:
    """PyTorch FP32 预测，作为精度基准。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(texts), config.batch_size):
            batch = tokenizer(
                texts[start:start + config.batch_size],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=config.max_length,
            )
            predictions.append(model(**batch).logits.argmax(-1).numpy())
    return np.concatenate(predictions)


def infer_onnx(path: Path, texts: list[str], config: AccuracyConfig) -> np.ndarray:
    """ONNX Runtime 预测（CPU）。"""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    predictions = []
    for start in range(0, len(texts), config.batch_size):
        encoded = tokenizer(
            texts[start:start + config.batch_size],
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )
        logits = session.run(
            None,
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )[0]
        predictions.append(logits.argmax(-1))
    return np.concatenate(predictions)


def compare(
    config: AccuracyConfig,
    model_paths: dict[str, Path],
    out_path: Path | None = None,
    cache_dir: Path | None = None,
    torch_runner=None,
    onnx_runner=None,
) -> AccuracyReport:
    """在标注集上比较 FP32 基准与量化版本的精度和判定一致性。

    torch_runner(texts) / onnx_runner(path, texts) 可注入，用于测试或换后端；
    默认实现分别走 transformers 与 onnxruntime。
    """
    rows = load_labeled_rows(config, cache_dir)
    if not rows:
        raise ValueError("标注数据为空，无法计算精度")
    missing = {config.text_column, config.label_column} - set(rows[0])
    if missing:
        raise ValueError(f"标注数据缺少列：{sorted(missing)}")

    # 默认 runner 延迟到数据校验之后才构造，避免校验失败时还去下载模型。
    if torch_runner is None:
        torch_runner = lambda texts: infer_torch(config.model_id, texts, config)
    if onnx_runner is None:
        onnx_runner = lambda path, texts: infer_onnx(path, texts, config)

    texts = [row[config.text_column] for row in rows]
    labels = np.array([int(row[config.label_column]) for row in rows])

    reference = torch_runner(texts)
    report = AccuracyReport(config=config.as_dict(), samples=len(texts))
    report.notes.append("accuracy 是在本 split 上的实测值；agreement_with_torch 是与 PyTorch FP32 判定的逐条一致率。")
    report.notes.append("翻转条数（flips）说明量化改变了多少预测；翻转可能正负抵消，所以要与 accuracy 一起看。")

    for tag, path in model_paths.items():
        predictions = onnx_runner(path, texts)
        report.results[tag] = {
            "accuracy": round(float((predictions == labels).mean()), 4),
            "agreement_with_torch": round(float((predictions == reference).mean()), 4),
            "flips_vs_torch": int((predictions != reference).sum()),
            "errors": int((predictions != labels).sum()),
        }

    report.results["torch_fp32"] = {
        "accuracy": round(float((reference == labels).mean()), 4),
        "agreement_with_torch": 1.0,
        "flips_vs_torch": 0,
        "errors": int((reference != labels).sum()),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return report
