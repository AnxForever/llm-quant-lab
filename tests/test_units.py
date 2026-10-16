"""纯逻辑测试：不加载模型、不联网、不需要 GPU。"""

from __future__ import annotations

import pytest

from llm_quant_lab.benchmark import BASE_TEXTS, agreement, repeat_to_batch
from llm_quant_lab.qlora import FinetuneConfig, FinetuneReport
from llm_quant_lab.quantize import ExportConfig


class TestRepeatToBatch:
    """基准输入扩展：batch 大小不整除时不能多也不能少。"""

    def test_exact_multiple(self):
        assert repeat_to_batch(["a", "b"], 4) == ["a", "b", "a", "b"]

    def test_remainder_is_truncated(self):
        assert repeat_to_batch(["a", "b", "c"], 5) == ["a", "b", "c", "a", "b"]

    def test_smaller_than_base_truncates(self):
        assert repeat_to_batch(["a", "b", "c"], 2) == ["a", "b"]

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_rejects_non_positive_batch(self, batch_size):
        with pytest.raises(ValueError, match="batch_size"):
            repeat_to_batch(["a"], batch_size)

    def test_rejects_empty_base(self):
        with pytest.raises(ValueError, match="基准文本"):
            repeat_to_batch([], 4)

    def test_base_texts_are_usable(self):
        assert repeat_to_batch(BASE_TEXTS, 16) == (BASE_TEXTS * 4)


class TestAgreement:
    """判定一致率：量化后预测翻转是核心观测指标。"""

    def test_identical_is_one(self):
        assert agreement([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0

    def test_partial_flip(self):
        assert agreement([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75

    def test_length_mismatch_is_zero_not_partial(self):
        """长度不同不能按位置硬比，否则会给出虚高的比例。"""
        assert agreement([0, 1, 1], [0, 1]) == 0.0

    def test_empty_is_zero(self):
        assert agreement([], []) == 0.0


class TestFinetuneConfig:
    """配置序列化：tuple 要转成 list 才能写进 JSON。"""

    def test_as_dict_is_json_serializable(self):
        import json

        config = FinetuneConfig()
        payload = config.as_dict()
        assert json.loads(json.dumps(payload))["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_defaults_are_consumer_hardware_friendly(self):
        config = FinetuneConfig()
        assert config.gradient_checkpointing is True
        assert config.lora_r == 16


class TestFinetuneReport:
    """报告比率计算。"""

    def test_trainable_ratio(self):
        report = FinetuneReport(
            config={}, device="test", load_seconds=1.0,
            trainable_params=4_587_520, total_params=380_436_480,
            vram_after_load_gb=0.87, peak_vram_gb=1.09,
        )
        assert report.as_dict()["trainable_ratio"] == pytest.approx(0.012059, abs=1e-6)

    def test_zero_total_does_not_divide_by_zero(self):
        report = FinetuneReport(
            config={}, device="test", load_seconds=1.0,
            trainable_params=0, total_params=0,
            vram_after_load_gb=0.0, peak_vram_gb=0.0,
        )
        assert report.trainable_ratio == 0.0


class TestExportConfig:
    def test_as_dict_roundtrip(self):
        config = ExportConfig(model_id="example/model", max_length=256)
        assert config.as_dict() == {"model_id": "example/model", "max_length": 256, "opset": 17}
