"""集成测试：用最小合成 ONNX 模型验证导出、量化、基准与精度比较的完整链路。

不下载真实模型、不需要 GPU；用 torch 现场构造一个极小网络导出成 ONNX，
这样测试是确定性的，也能在 CI 里跑。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")


class TinyClassifier(torch.nn.Module):
    """两层 MLP，接口模仿文本分类模型的 (input_ids, attention_mask) -> logits。"""

    def __init__(self, vocab: int = 32, hidden: int = 16, classes: int = 2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, classes)

    def forward(self, input_ids, attention_mask):
        hidden = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    """导出 FP32 与 INT8 两个极小 ONNX 模型。"""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir = tmp_path_factory.mktemp("tiny-onnx")
    torch.manual_seed(0)
    model = TinyClassifier().eval()

    input_ids = torch.randint(1, 32, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)

    fp32_path = out_dir / "tiny_fp32.onnx"
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )

    int8_path = out_dir / "tiny_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    return {"fp32": fp32_path, "int8": int8_path}


def make_feeds(batch: int = 4, sequence: int = 8) -> dict:
    rng = np.random.default_rng(0)
    return {
        "input_ids": rng.integers(1, 32, (batch, sequence)).astype(np.int64),
        "attention_mask": np.ones((batch, sequence), dtype=np.int64),
    }


class TestExportAndQuantize:
    def test_both_models_exist(self, tiny_onnx):
        assert tiny_onnx["fp32"].is_file()
        assert tiny_onnx["int8"].is_file()

    def test_int8_is_smaller(self, tiny_onnx):
        fp32_size = tiny_onnx["fp32"].stat().st_size
        int8_size = tiny_onnx["int8"].stat().st_size
        assert int8_size < fp32_size

    def test_both_models_run(self, tiny_onnx):
        feeds = make_feeds()
        for path in tiny_onnx.values():
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            logits = session.run(None, feeds)[0]
            assert logits.shape == (4, 2)


class TestBenchmark:
    """基准测试要真的跑到每个组合，并算出和参考的一致性。"""

    def test_runs_both_models_on_cpu(self, tiny_onnx):
        from llm_quant_lab.benchmark import BenchmarkConfig, run_benchmark

        config = BenchmarkConfig(
            batch_size=4, max_length=8, warmup=1, repeats=2,
            providers=("CPUExecutionProvider",),
        )
        report = run_benchmark(config, tiny_onnx, feeds=make_feeds())

        assert "fp32_CPU" in report.results
        assert "int8_CPU" in report.results
        for key in ("fp32_CPU", "int8_CPU"):
            assert "error" not in report.results[key]
            assert report.results[key]["actual_provider"] == "CPUExecutionProvider"
            assert report.results[key]["latency_ms"] >= 0

    def test_first_combination_is_the_reference(self, tiny_onnx):
        from llm_quant_lab.benchmark import BenchmarkConfig, run_benchmark

        config = BenchmarkConfig(batch_size=4, max_length=8, warmup=1, repeats=2,
                                 providers=("CPUExecutionProvider",))
        report = run_benchmark(config, tiny_onnx, feeds=make_feeds())
        assert report.results["fp32_CPU"]["agreement_with_reference"] == 1.0

    def test_provider_fallback_is_flagged(self, tiny_onnx):
        """请求 CUDA 但不可用时会静默回落 CPU；报告必须标出来，否则会误读延迟。"""
        from llm_quant_lab.benchmark import BenchmarkConfig, run_benchmark

        config = BenchmarkConfig(batch_size=4, max_length=8, warmup=1, repeats=1,
                                 providers=("CUDAExecutionProvider",))
        report = run_benchmark(config, tiny_onnx, feeds=make_feeds())
        entry = report.results["fp32_CUDA"]

        if entry.get("actual_provider") == "CPUExecutionProvider":
            assert entry["fell_back"] is True
            assert any("回落到 CPU" in note for note in report.notes)
        else:
            assert entry["fell_back"] is False


class TestAccuracyComparison:
    """精度比较：量化后 accuracy 与翻转率都要报，且翻转不能抵消 accuracy 的变化。"""

    @staticmethod
    def _write_labels(tmp_path, labels):
        """文件名必须等于 AccuracyConfig.split_file，否则会去下载真实数据集。"""
        csv_path = tmp_path / "test.csv"
        rows = ["text,label"] + [f"row-{i},{int(v)}" for i, v in enumerate(labels)]
        csv_path.write_text("\n".join(rows), encoding="utf-8")
        return csv_path

    def test_reports_accuracy_and_flips(self, tmp_path):
        from llm_quant_lab.accuracy import AccuracyConfig, compare

        labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        self._write_labels(tmp_path, labels)

        # 模拟：INT8 与 FP32 完全一致。
        config = AccuracyConfig(model_id="tiny", max_length=8, batch_size=8)
        report = compare(
            config,
            {"onnx_int8": tmp_path / "unused.onnx"},
            cache_dir=tmp_path,
            torch_runner=lambda texts: labels.copy(),
            onnx_runner=lambda path, texts: labels.copy(),
        )

        assert report.samples == 8
        assert report.results["torch_fp32"]["accuracy"] == 1.0
        assert report.results["torch_fp32"]["flips_vs_torch"] == 0
        assert report.results["onnx_int8"]["accuracy"] == 1.0
        assert report.results["onnx_int8"]["flips_vs_torch"] == 0
        assert report.results["onnx_int8"]["agreement_with_torch"] == 1.0

    def test_flips_are_reported_separately_from_accuracy(self, tmp_path):
        """量化翻转了两条但正负抵消：accuracy 不变，flips 必须暴露出来。"""
        from llm_quant_lab.accuracy import AccuracyConfig, compare

        reference = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        quantized = np.array([1, 1, 1, 0, 1, 0, 0, 0])  # 两条翻转，两条都从对变错/错变对
        self._write_labels(tmp_path, reference)

        config = AccuracyConfig(model_id="tiny", max_length=8, batch_size=8)
        report = compare(
            config,
            {"onnx_int8": tmp_path / "unused.onnx"},
            cache_dir=tmp_path,
            torch_runner=lambda texts: reference.copy(),
            onnx_runner=lambda path, texts: quantized.copy(),
        )

        int8 = report.results["onnx_int8"]
        assert int8["flips_vs_torch"] == 2
        assert int8["agreement_with_torch"] == 0.75
        assert int8["errors"] == 2

    def test_missing_column_is_rejected(self, tmp_path):
        from llm_quant_lab.accuracy import AccuracyConfig, compare

        (tmp_path / "test.csv").write_text("text\nhello\n", encoding="utf-8")
        config = AccuracyConfig(model_id="tiny", max_length=8, batch_size=8)
        with pytest.raises(ValueError, match="缺少列"):
            compare(config, {"onnx_fp32": tmp_path / "unused.onnx"}, cache_dir=tmp_path)

    def test_empty_dataset_is_rejected(self, tmp_path):
        from llm_quant_lab.accuracy import AccuracyConfig, compare

        (tmp_path / "test.csv").write_text("text,label\n", encoding="utf-8")
        config = AccuracyConfig(model_id="tiny", max_length=8, batch_size=8)
        with pytest.raises(ValueError, match="为空"):
            compare(config, {"onnx_fp32": tmp_path / "unused.onnx"}, cache_dir=tmp_path)

    def test_report_is_json_serializable(self, tmp_path):
        import json

        from llm_quant_lab.accuracy import AccuracyConfig, compare

        labels = np.array([0, 1, 1, 0])
        self._write_labels(tmp_path, labels)
        config = AccuracyConfig(model_id="tiny", max_length=8, batch_size=8)
        report = compare(
            config, {"onnx_int8": tmp_path / "unused.onnx"}, cache_dir=tmp_path,
            torch_runner=lambda texts: labels.copy(),
            onnx_runner=lambda path, texts: labels.copy(),
        )
        assert json.loads(json.dumps(report.as_dict(), ensure_ascii=False))

