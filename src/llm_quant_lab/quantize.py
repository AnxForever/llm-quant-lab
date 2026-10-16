"""ONNX 导出 + 动态 INT8 量化。

为什么用动态量化（quantize_dynamic）：
- 不需要校准数据集，权重直接量化成 INT8，激活在运行时按需量化。
- 对 Transformer 类模型，权重占体积的绝大部分，所以体积收益最明显。
- 代价是激活量化在运行时做，延迟收益不如静态量化稳定（见 README 的失败案例）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MB = 2**20


@dataclass
class ExportConfig:
    """一次导出与量化的参数。"""

    model_id: str = "AnxForever/chinese-ai-detector-bert"
    max_length: int = 128
    opset: int = 17

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def export_onnx(config: ExportConfig, out_dir: Path) -> dict[str, Path]:
    """导出 FP32 ONNX，再生成 INT8 动态量化版本；返回两个模型路径。

    注意：这里用 dynamo=False 走 TorchScript 导出路径。新版 torch.export
    默认导出器在 BERT 上会触发 shape inference 冲突（见 README 失败案例 1）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_id).eval()

    dummy = tokenizer(["占位文本"], return_tensors="pt", padding="max_length",
                      truncation=True, max_length=config.max_length)

    fp32_path = out_dir / "model_fp32.onnx"
    torch.onnx.export(
        model,
        (dummy["input_ids"][:1], dummy["attention_mask"][:1]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        },
        opset_version=config.opset,
        dynamo=False,
    )

    int8_path = out_dir / "model_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    return {"fp32": fp32_path, "int8": int8_path}


def build_report(config: ExportConfig, paths: dict[str, Path]) -> dict:
    """体积对比报告（体积是确定性的，每次都一样）。"""
    fp32_mb = paths["fp32"].stat().st_size / MB
    int8_mb = paths["int8"].stat().st_size / MB
    return {
        "model_id": config.model_id,
        "max_length": config.max_length,
        "opset": config.opset,
        "sizes_mb": {"fp32": round(fp32_mb, 1), "int8": round(int8_mb, 1)},
        "size_ratio_int8_vs_fp32": round(int8_mb / fp32_mb, 3),
        "note": "动态 INT8 量化只量化权重，不需要校准数据；体积收益稳定，延迟收益随负载波动。",
    }


def write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
