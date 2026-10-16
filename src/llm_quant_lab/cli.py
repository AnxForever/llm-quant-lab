"""命令行入口：微调、导出量化、基准测试、精度对比。

用法示例：
    python -m llm_quant_lab qlora --model Qwen/Qwen3-0.6B --steps 3
    python -m llm_quant_lab quantize
    python -m llm_quant_lab benchmark
    python -m llm_quant_lab accuracy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
REPORTS = REPO_ROOT / "reports"

# 微调用的小样本：把商品标题转成结构化 JSON，和商品质检项目的任务同源。
SAMPLE_INSTRUCTIONS = [
    ("把商品标题转成 JSON：Apple iPhone 15 128G 黑色",
     '{"brand":"Apple","model":"iPhone 15","storage":"128G","color":"黑色"}'),
    ("把商品标题转成 JSON：小米 14 Pro 12+256 白色",
     '{"brand":"小米","model":"14 Pro","storage":"12+256","color":"白色"}'),
]


def cmd_qlora(args: argparse.Namespace) -> int:
    from llm_quant_lab.qlora import FinetuneConfig, run_finetune

    config = FinetuneConfig(
        model_id=args.model,
        max_length=args.max_length,
        steps=args.steps,
        learning_rate=args.lr,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    report = run_finetune(config, SAMPLE_INSTRUCTIONS, out_path=REPORTS / f"qlora-{args.tag}.json")
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    from llm_quant_lab.quantize import ExportConfig, build_report, export_onnx, write_report

    config = ExportConfig(model_id=args.model, max_length=args.max_length)
    paths = export_onnx(config, ARTIFACTS / args.tag)
    report = build_report(config, paths)
    write_report(report, REPORTS / f"quantize-{args.tag}.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from llm_quant_lab.benchmark import BenchmarkConfig, run_benchmark

    config = BenchmarkConfig(model_id=args.model, batch_size=args.batch_size, max_length=args.max_length)
    paths = {
        "fp32": ARTIFACTS / args.tag / "model_fp32.onnx",
        "int8": ARTIFACTS / args.tag / "model_int8.onnx",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"缺少 ONNX 产物，请先运行 quantize：{missing}")

    report = run_benchmark(config, paths, out_path=REPORTS / f"benchmark-{args.tag}.json")
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_accuracy(args: argparse.Namespace) -> int:
    from llm_quant_lab.accuracy import AccuracyConfig, compare

    config = AccuracyConfig(model_id=args.model, max_length=args.max_length)
    paths = {
        "onnx_fp32": ARTIFACTS / args.tag / "model_fp32.onnx",
        "onnx_int8": ARTIFACTS / args.tag / "model_int8.onnx",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"缺少 ONNX 产物，请先运行 quantize：{missing}")

    report = compare(config, paths, out_path=REPORTS / f"accuracy-{args.tag}.json")
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_quant_lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="AnxForever/chinese-ai-detector-bert", help="模型 ID")
    common.add_argument("--tag", default="bert", help="产物与报告的标签名")
    common.add_argument("--max-length", type=int, default=128, dest="max_length")

    p_qlora = sub.add_parser("qlora", help="跑 QLoRA 微调并记录显存/loss")
    p_qlora.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p_qlora.add_argument("--tag", default="qwen3-0.6b")
    p_qlora.add_argument("--max-length", type=int, default=128, dest="max_length")
    p_qlora.add_argument("--steps", type=int, default=3)
    p_qlora.add_argument("--lr", type=float, default=2e-4)
    p_qlora.add_argument("--no-gradient-checkpointing", action="store_true")
    p_qlora.set_defaults(func=cmd_qlora)

    p_quant = sub.add_parser("quantize", parents=[common], help="导出 ONNX 并做 INT8 动态量化")
    p_quant.set_defaults(func=cmd_quantize)

    p_bench = sub.add_parser("benchmark", parents=[common], help="比较 CPU/CUDA 后端的延迟与一致性")
    p_bench.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p_bench.set_defaults(func=cmd_benchmark)

    p_acc = sub.add_parser("accuracy", parents=[common], help="在标注集上比较 FP32 与 INT8 精度")
    p_acc.set_defaults(func=cmd_accuracy)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
