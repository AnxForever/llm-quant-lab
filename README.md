# llm-quant-lab

在**消费级单卡**（RTX 5060 Laptop 8GB）上做三件事，并给出可复现的数字：

1. **QLoRA 微调** —— 4bit 量化基座 + LoRA 适配器，记录显存与 loss
2. **ONNX INT8 量化** —— 导出与体积收益
3. **多后端推理** —— CPU / CUDA 延迟对比 + **量化精度影响**

这不是"跑通 demo"。每个结论都带可重跑的脚本和报告，包含失败案例。

## 为什么做这个

模型微调、量化、多硬件适配是 AI 工程岗的常见要求，但很少有人在**自己的一台机器**上把完整链路跑通并记录数据。这个仓库回答四个具体问题：

- 8GB 显存能训多大的模型？（实测：1.5B QLoRA 峰值 2.53 GB）
- 量化后体积降多少？（实测：390 MB → 98 MB，25%）
- 量化会让精度掉吗？（实测：6601 条上 0.9686 → 0.9689，判定一致率 99.21%）
- 换后端能快多少？（实测：CUDA 比 CPU 快 27.8×；但 INT8 在 CUDA 上反而变慢）

## 硬件与软件基线

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA RTX 5060 Laptop，8151 MiB，sm_120（Blackwell） |
| CPU / 内存 | Intel i9-14900HX / 15.7 GB |
| 驱动 | 610.62（CUDA UMD 13.3） |
| PyTorch | 2.11.0.dev+cu128 |
| bitsandbytes | 0.50.2 |
| ONNX Runtime | 1.28.0（CPU）/ 1.22.0（GPU） |

## 快速开始

```powershell
uv venv --python 3.12 --system-site-packages
uv pip install -e .

# 1. QLoRA 微调（记录显存和 loss）
python -m llm_quant_lab qlora --model Qwen/Qwen3-0.6B --steps 3

# 2. 导出 ONNX 并做 INT8 动态量化
python -m llm_quant_lab quantize --model AnxForever/chinese-ai-detector-bert

# 3. 多后端延迟与一致性
python -m llm_quant_lab benchmark --batch-size 16

# 4. 量化精度影响（在标注集上，约 20 分钟）
python -m llm_quant_lab accuracy
```

报告写到 `reports/`，模型产物写到 `artifacts/`。两者都不进版本库。

测试：

```powershell
python -m pytest -q
```

## 结果

### QLoRA 显存

| 模型 | seq_len | 梯度检查点 | 可训练参数占比 | 峰值显存 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 128 | 否 | 1.21% | **1.09 GB** |
| Qwen2.5-1.5B-Instruct | 512 | 是 | 0.49% | **2.53 GB** |

8GB 显存对 1.5B 级 QLoRA 有充足余量。**真正的瓶颈是 16 GB 内存**——1.5B 加载耗时约 28 分钟。

### 量化：体积与精度

| 指标 | 值 |
| --- | --- |
| 体积 | 390.3 MB → **98.3 MB**（25.2%） |
| 精度（公开 test.csv，6601 条） | 0.9686 → **0.9689** |
| 与 PyTorch FP32 判定一致率 | **99.21%** |
| 判定翻转条数 | 52 / 6601（0.79%） |

翻转 52 条但准确率没掉，说明翻转正负抵消。**只报 accuracy 会漏掉这件事，所以报告里同时给一致率和翻转数。**

> 体积是确定性的，每次复现都一样；CPU 延迟会波动（见失败案例 5）。

### 多后端

通过 `python -m llm_quant_lab benchmark --batch-size 16` 产出（`reports/benchmark-bert.json`）：

| 后端 | 实际 provider | 延迟 | 相对 CPU |
| --- | --- | --- | --- |
| FP32 CPU | CPUExecutionProvider | 898.0 ms | 1× |
| FP32 CUDA | **CUDAExecutionProvider** | **33.3 ms** | **27.0×** |
| INT8 CPU | CPUExecutionProvider | 397.0 ms | 2.26× |
| INT8 CUDA | CUDAExecutionProvider | 452.7 ms | 1.98× |

四种组合判定结果 100% 一致。

> 延迟是单次测量，绝对值随负载波动（见失败案例 4）；**27× 这个量级稳定**。

## 失败案例（比成功更有信息量）

### 1. 新版 torch.export 导出器在 BERT 上失败

PyTorch 2.9+ 默认走 `torch.export`，对 BERT 导出时报：

```
InferenceError: [ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (768) vs (2)
```

原因是新导出器生成的图在 shape inference 阶段冲突。**解法**：显式 `dynamo=False` 走 TorchScript 路径（见 `quantize.py`）。

### 2. INT8 在 CUDA 上比 FP32 CUDA 慢

452.7 ms vs 33.3 ms，慢了约 13.6 倍。原因是 `quantize_dynamic` 产生的 INT8 算子大多**没有 CUDA 实现，只能在 CPU 上执行**，GPU 端反而要等数据来回搬。

**教训**：量化 + GPU 不是无脑叠加。要按 (模型, 后端, 量化方式) 三元组实测，不能假设"量化一定更快"。

### 3. provider 会静默回落，延迟结论会错

第一次跑 benchmark 时，开发环境只装了 CPU 版 onnxruntime：请求 `CUDAExecutionProvider` 不报错，而是**静默回落到 CPU**，报告里"CUDA 809 ms"其实是 CPU 延迟，会直接得出"CUDA 比 CPU 慢"的错误结论。

**解法**：每条结果记录 `actual_provider` 和 `fell_back`，出现回落时在 `notes` 里点名（见 `benchmark.py`）。装了 `onnxruntime-gpu==1.22.0` 后 CUDA 才真正生效。

### 4. Windows 上 onnxruntime-gpu 缺 CUDA DLL

`onnxruntime-gpu` 1.31 需要 CUDA 13 运行库（`cublasLt64_13.dll`），本机装不上。**解法**：降到 1.22.0（CUDA 12），并复用 torch wheel 自带的 CUDA 12 DLL（见 `benchmark.py` 的 `load_cuda_dlls()`）。

### 5. 延迟数字会飘

同一脚本两次运行，batch 4 的 CPU 延迟分别是 630.4 ms 和 268.1 ms（差 2.3 倍），但**加速比稳定在 1.88–1.89×**。

**教训**：报性能数字优先报**比例和体积**，绝对延迟要说明是单次测量。

## 本机做不到的（诚实边界）

| 想做的事 | 状态 | 原因 |
| --- | --- | --- |
| vLLM / SGLang | ❌ | Windows 原生支持有限，主流路线需 WSL2；本机 WSL 无 torch/nvcc/docker CLI |
| TensorRT / TRT-LLM | ❌ | 需额外 CUDA Toolkit，本机无 `nvcc` |
| 7B 全参微调 | ❌ | 8 GB 显存不可能 |
| 7B QLoRA | ⚠️ 未验证 | 内存和加载时间会成为瓶颈 |

## 结构

| 文件 | 职责 |
| --- | --- |
| `src/llm_quant_lab/qlora.py` | 4bit 加载、LoRA 挂载、训练循环与显存记录 |
| `src/llm_quant_lab/quantize.py` | ONNX 导出 + 动态 INT8 量化 + 体积报告 |
| `src/llm_quant_lab/benchmark.py` | 多 provider 延迟与判定一致性 |
| `src/llm_quant_lab/accuracy.py` | 标注集上的精度影响与翻转率 |
| `src/llm_quant_lab/cli.py` | 命令行入口 |
| `tests/test_units.py` | 纯逻辑（无模型、无网络、无 GPU） |
| `tests/test_pipeline.py` | 用极小合成 ONNX 模型跑通全链路 |

## 模型

量化与精度实验用的是自己的本科毕设模型：
[AnxForever/chinese-ai-detector-bert](https://huggingface.co/AnxForever/chinese-ai-detector-bert)（基于 `bert-base-chinese` 的中文 AI 生成文本检测，二分类）。

> 精度口径：README 中的 0.9686/0.9689 是在数据集公开 `test.csv`（6601 条）上的实测值，**不等于**模型卡上写的 98.57%（910 条 independent 子集）。
