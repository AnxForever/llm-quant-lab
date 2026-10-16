"""QLoRA 微调：4bit 量化基座 + LoRA 适配器，记录显存与 loss。

为什么这样写：
- 4bit NF4 + double quant 把基座压到约 1/4，让 8GB 消费级卡能训 1.5B 级模型。
- LoRA 只训练低秩增量，可训练参数占比通常 <1.5%，优化器状态因此很小。
- 梯度检查点用时间换显存，是长序列下能训起来的关键开关。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GB = 2**30


@dataclass
class FinetuneConfig:
    """一次 QLoRA 运行的全部可调参数。"""

    model_id: str = "Qwen/Qwen3-0.6B"
    max_length: int = 128
    steps: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-4
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    seed: int = 0

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class FinetuneReport:
    """一次 QLoRA 运行的观测结果。"""

    config: dict
    device: str
    load_seconds: float
    trainable_params: int
    total_params: int
    vram_after_load_gb: float
    peak_vram_gb: float
    steps: list[dict] = field(default_factory=list)
    total_seconds: float = 0.0

    @property
    def trainable_ratio(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "device": self.device,
            "load_seconds": round(self.load_seconds, 1),
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_ratio": round(self.trainable_ratio, 6),
            "vram_after_load_gb": round(self.vram_after_load_gb, 2),
            "peak_vram_gb": round(self.peak_vram_gb, 2),
            "steps": self.steps,
            "total_seconds": round(self.total_seconds, 1),
        }


def build_quantization_config() -> BitsAndBytesConfig:
    """4bit NF4 + double quant；计算仍用 bfloat16。"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qlora_model(model_id: str, config: FinetuneConfig):
    """加载 4bit 基座并挂上 LoRA 适配器，返回 (model, tokenizer, load_seconds, vram_after_load)。"""
    tokenizer = load_tokenizer(model_id)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=build_quantization_config(), device_map={"": 0}
    )
    load_seconds = time.perf_counter() - started
    vram_after_load = torch.cuda.max_memory_allocated() / GB

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.target_modules),
        ),
    )
    return model, tokenizer, load_seconds, vram_after_load


def encode_example(tokenizer, instruction: str, answer: str, max_length: int):
    """把一条指令样本编码成模型输入；padding 位置在 labels 里设为 -100，不计入 loss。"""
    text = f"### 指令\n{instruction}\n### 输出\n{answer}"
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return encoded, labels


def run_finetune(
    config: FinetuneConfig,
    samples: list[tuple[str, str]],
    out_path: Path | None = None,
) -> FinetuneReport:
    """跑若干步 QLoRA 训练并返回观测报告。

    samples 是 (指令, 期望输出) 列表；步数不足时会循环取用。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备；本实验用于验证消费级 GPU 上的 QLoRA 可行性")
    if not samples:
        raise ValueError("samples 不能为空")

    torch.manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer, load_seconds, vram_after_load = load_qlora_model(config.model_id, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    report = FinetuneReport(
        config=config.as_dict(),
        device=torch.cuda.get_device_name(0),
        load_seconds=load_seconds,
        trainable_params=trainable,
        total_params=total,
        vram_after_load_gb=vram_after_load,
        peak_vram_gb=vram_after_load,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    model.train()
    started = time.perf_counter()

    for step in range(config.steps):
        instruction, answer = samples[step % len(samples)]
        encoded, labels = encode_example(tokenizer, instruction, answer, config.max_length)
        batch = {k: v.cuda() for k, v in encoded.items()}
        labels = labels.cuda()

        step_started = time.perf_counter()
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        report.steps.append(
            {
                "step": step,
                "loss": round(loss.item(), 4),
                "step_seconds": round(time.perf_counter() - step_started, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
            }
        )

    report.total_seconds = time.perf_counter() - started
    report.peak_vram_gb = torch.cuda.max_memory_allocated() / GB

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return report
