"""多后端推理基准：同一 ONNX 模型在 CPU / CUDA 上的延迟与判定一致性。

关键设计：先建立参考预测（FP32 + CPU），再让每个后端和它比一致率。
只看延迟会漏掉"跑得快但答案变了"这种问题。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MB = 2**20

BASE_TEXTS = [
    "今天天气不错，我们下午去公园散步吧。",
    "基于深度学习的文本分类方法在多个基准数据集上取得了显著效果，本文提出一种改进框架。",
    "这家店的火锅很好吃，服务也特别周到，下次还会再来。",
    "综上所述，本研究通过多阶段数据治理与长度感知采样，有效缓解了类别不均衡问题。",
]


@dataclass
class BenchmarkConfig:
    """一次基准测试的参数。"""

    model_id: str = "AnxForever/chinese-ai-detector-bert"
    max_length: int = 128
    batch_size: int = 16
    warmup: int = 5
    repeats: int = 30
    providers: tuple[str, ...] = ("CPUExecutionProvider", "CUDAExecutionProvider")

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class BenchmarkReport:
    config: dict
    onnxruntime_version: str
    results: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "onnxruntime_version": self.onnxruntime_version,
            "results": self.results,
            "notes": self.notes,
        }


def repeat_to_batch(base: list[str], batch_size: int) -> list[str]:
    """把基准文本循环扩展到目标 batch；batch_size 小于基准条数时截断。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if not base:
        raise ValueError("基准文本不能为空")
    return (base * ((batch_size // len(base)) + 1))[:batch_size]


def make_texts(model_id: str, batch_size: int, max_length: int):
    """构造固定基准输入；重复同一组文本以达到目标 batch。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    texts = repeat_to_batch(BASE_TEXTS, batch_size)
    encoded = tokenizer(texts, return_tensors="np", padding="max_length",
                        truncation=True, max_length=max_length)
    return tokenizer, {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }


def load_cuda_dlls() -> str | None:
    """把 torch 自带的 CUDA 运行库加入 DLL 搜索路径。

    Windows 上 onnxruntime-gpu 需要 cublasLt/cudnn 等 DLL；torch 的 wheel 里已经带了
    一整套 CUDA 12 运行库，直接复用可以避免单独安装 CUDA Toolkit。
    """
    try:
        import torch
    except ImportError:
        return None
    lib_dir = Path(torch.__file__).parent / "lib"
    if not lib_dir.is_dir():
        return None
    os.add_dll_directory(str(lib_dir))
    os.environ["PATH"] = f"{lib_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(lib_dir)


def benchmark_provider(path: Path, provider: str, feeds: dict, config: BenchmarkConfig):
    """在指定 provider 上测延迟；provider 不可用时抛出异常由调用方记录。"""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.log_severity_level = 3
    # CPU 作为兜底，但请求的就是 CPU 时不要重复，否则 onnxruntime 会告警。
    providers = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), options, providers=providers)
    actual = session.get_providers()[0]

    for _ in range(config.warmup):
        output = session.run(None, feeds)[0]

    started = time.perf_counter()
    for _ in range(config.repeats):
        output = session.run(None, feeds)[0]
    latency_ms = (time.perf_counter() - started) / config.repeats * 1000

    return {
        "requested_provider": provider,
        "actual_provider": actual,
        # provider 缺失时 onnxruntime 会静默回落到 CPU；不标记的话，报告里
        # 标着 CUDA 的那行其实是 CPU 延迟，会得出错误结论。
        "fell_back": actual != provider,
        "latency_ms": round(latency_ms, 1),
        "predicted_labels": output.argmax(1).tolist(),
    }


def agreement(reference: list[int], candidate: list[int]) -> float:
    """逐条判定一致率；长度不同视为不一致，返回 0。"""
    if len(reference) != len(candidate):
        return 0.0
    if not reference:
        return 0.0
    return round(float(np.mean(np.array(reference) == np.array(candidate))), 4)


def run_benchmark(
    config: BenchmarkConfig,
    model_paths: dict[str, Path],
    out_path: Path | None = None,
    feeds: dict | None = None,
) -> BenchmarkReport:
    """跑完所有 (模型, provider) 组合，并计算与 FP32-CPU 基准的判定一致率。"""
    import onnxruntime as ort

    dll_dir = load_cuda_dlls()
    report = BenchmarkReport(
        config=config.as_dict(),
        onnxruntime_version=ort.__version__,
    )
    if dll_dir:
        report.notes.append(f"CUDA DLL 搜索路径复用 torch: {dll_dir}")

    if feeds is None:
        _, feeds = make_texts(config.model_id, config.batch_size, config.max_length)

    reference: list[int] | None = None
    for tag, path in model_paths.items():
        for provider in config.providers:
            key = f"{tag}_{provider.replace('ExecutionProvider', '')}"
            try:
                result = benchmark_provider(path, provider, feeds, config)
            except Exception as exc:  # provider 缺失或依赖不全
                report.results[key] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                continue

            if reference is None:
                reference = result["predicted_labels"]
                result["agreement_with_reference"] = 1.0
            else:
                result["agreement_with_reference"] = agreement(reference, result["predicted_labels"])
            report.results[key] = result

    if reference is not None:
        report.notes.append("agreement_with_reference 以第一个成功组合（FP32 + CPU）为基准。")

    fallbacks = [k for k, v in report.results.items() if v.get("fell_back")]
    if fallbacks:
        report.notes.append(
            f"以下组合请求的 provider 不可用、已回落到 CPU，其延迟不代表该加速器性能：{fallbacks}"
        )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return report
