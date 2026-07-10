# Qwen3-ASR ONNX Demo

这个项目把本地 `Qwen3-ASR-0.6B` safetensors 模型导出成 ONNX Runtime 可运行的子图。默认只加载一个 decoder：

- `token_embed.onnx`
- `audio_encoder.onnx`
- `decoder_with_past.onnx`

`decoder_with_past.onnx` 在 prefill 时接收空 KV cache，后续复用同一个 session 做增量解码。运行时只驻留一份 decoder 权重，也避开了 TensorRT 不支持 `decoder_merged.onnx` 顶层 `If` 的问题。

## 1. 准备模型目录

把 Hugging Face / ModelScope 下载到的 Qwen3-ASR 模型文件放到项目根目录的 `Qwen3-ASR-0.6B`：

```text
Qwen3-ASR-0.6B/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors
  ...
```

导出脚本会读取 `model.safetensors`。运行 ONNX demo 时只读取 tokenizer / processor / config 等轻量文件，不再读取 safetensors。

## 2. 创建 Python 环境

### pip / venv

cmd：

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch onnx
```

### uv

cmd：

```bat
uv venv --python 3.12
.venv\Scripts\activate.bat
uv pip install -r requirements.txt
uv pip install torch onnx
```

如果要用 CUDA 导出，请按你的 CUDA 版本安装对应的 PyTorch；上面的 `torch` 命令只是最简单的默认安装方式。

## 3. 从 safetensors 导出 ONNX

推荐先导出 optimized 版本：

```bat
.venv\Scripts\python.exe export_qwen3_asr_onnx.py ^
  --model-dir .\Qwen3-ASR-0.6B ^
  --out-dir .\onnx_models ^
  --device cuda ^
  --dtype fp16
```

如果没有 CUDA，也可以用 CPU 导出，但会慢很多：

```bat
.venv\Scripts\python.exe export_qwen3_asr_onnx.py ^
  --model-dir .\Qwen3-ASR-0.6B ^
  --out-dir .\onnx_models ^
  --device cpu ^
  --dtype fp32
```

导出完成后，正式分发/运行需要这些文件：

```text
onnx_models/
  token_embed.onnx
  audio_encoder.onnx
  decoder_with_past.onnx
  decoder_with_past.onnx.data
  decoder_merged.onnx
  decoder_merged.onnx.data
```

如果优化成功，还会有：

```text
onnx_models/
  token_embed.optimized.onnx
  token_embed.optimized.onnx.data
  audio_encoder.optimized.onnx
  audio_encoder.optimized.onnx.data
  decoder_with_past.optimized.onnx
  decoder_with_past.optimized.onnx.data
  decoder_merged.optimized.onnx
  decoder_merged.optimized.onnx.data
```

`decoder_init.tmp.onnx` 只是导出中间文件，脚本会在合并后清理。加 `--keep-decoder-parts` 可额外保留它用于 split 对照。

## 4. 运行 ONNX Demo

使用普通 ONNX：

```bat
.venv\Scripts\python.exe onnx_asr_demo.py ^
  --model-dir .\Qwen3-ASR-0.6B ^
  --onnx-dir .\onnx_models ^
  --provider directml ^
  --audio .\your_audio.wav
```

使用 optimized ONNX：

```bat
.venv\Scripts\python.exe onnx_asr_demo.py ^
  --model-dir .\Qwen3-ASR-0.6B ^
  --onnx-dir .\onnx_models ^
  --optimized ^
  --provider directml ^
  --audio .\your_audio.wav
```

`--provider` 可选：

```text
auto
directml
cuda
tensorrt
openvino
cpu
```

Windows + `onnxruntime-directml` 环境通常用 `--provider directml`。
多 GPU 机器可用 `--directml-device-id` 选择 DXGI adapter；本机启用两块 GPU 时，`0` 是 Intel iGPU，`1` 是 RTX 4090 Laptop。
`--provider tensorrt` 需要 TensorRT 10 runtime。脚本会自动发现 pip 安装的 `tensorrt_libs`，并默认使用单 decoder、动态 shape profile 和磁盘 engine cache。可用 `--trt-max-seq-len` 设置 prompt 加生成长度的上限。
`--provider openvino` 需要 `onnxruntime-openvino` 和版本匹配的 `openvino` Python 包；脚本会自动把 `.venv\Lib\site-packages\openvino\libs` 加入 DLL 搜索路径，所以不需要手动把 DLL 复制到项目根目录。

## 5. 性能实测结论

以下结果均使用 `asr_en.wav`、`max-new-tokens=256`、warmup 1 次、benchmark 5 秒。不同 ONNX Runtime wheel 不能稳定混装，DirectML / CUDA / OpenVINO 结果来自切换到对应 wheel 后的测试。

PyTorch CUDA 对照：

```text
asr.py bf16 CUDA: 4.628x
asr.py fp16 CUDA: 8.213x
```

ONNX Runtime DirectML EP 实测，RTX 4090 Laptop GPU。DirectML `device_id=1`，已用 NVIDIA 显存活动确认：

| provider | decoder | optimized | KV cache | 吞吐 | decode decoder |
|---|---|---:|---|---:|---:|
| DirectML | merged | no | host NumPy | 4.101x | 71.38 ms/token |
| DirectML | merged | yes | host NumPy | 4.054x | 71.91 ms/token |
| DirectML | split | no | host NumPy | 4.458x | 65.22 ms/token |
| DirectML | split | yes | host NumPy | 4.340x | 66.97 ms/token |
| DirectML | single | no | host NumPy | 1.606x | 188.36 ms/token |
| DirectML | merged | no | DML I/O Binding | 4.940x | 59.27 ms/token |
| DirectML | split | no | DML I/O Binding | 5.018x | 58.47 ms/token |
| DirectML | single | no | DML I/O Binding | 1.657x | 182.25 ms/token |

RTX 测试期间 `nvidia-smi` 确认 Python 进程使用独显；新的 DML I/O Binding 路径让 KV cache 以 OrtValue 留在所选 DirectML adapter。

ONNX Runtime DirectML EP 实测，禁用 RTX 4090 后的 Intel iGPU：

环境为 `onnxruntime-directml 1.24.4`，DirectML `device_id=0`。ORT 可用 provider 为 `DmlExecutionProvider` 和 `CPUExecutionProvider`。运行时 NVIDIA 显存保持 0，确认使用 Intel iGPU。

| provider | decoder | optimized | KV cache | 吞吐 | decode decoder |
|---|---|---:|---|---:|---:|
| DirectML Intel iGPU | merged | no | host NumPy | 0.728x | 333.62 ms/token |
| DirectML Intel iGPU | merged | yes | host NumPy | 0.735x | 329.54 ms/token |
| DirectML Intel iGPU | split | no | host NumPy | 0.732x | 331.23 ms/token |
| DirectML Intel iGPU | split | yes | host NumPy | 0.727x | 333.99 ms/token |
| DirectML Intel iGPU | single | no | host NumPy | 0.616x | 420.77 ms/token |
| DirectML Intel iGPU | merged | no | DML I/O Binding | 0.784x | 309.58 ms/token |
| DirectML Intel iGPU | split | no | DML I/O Binding | 0.787x | 307.77 ms/token |
| DirectML Intel iGPU | single | no | DML I/O Binding | 0.637x | 403.43 ms/token |

ONNX Runtime CUDA EP 实测：

可用 provider：

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

`onnx_asr_demo.py --provider cuda` 会显式选择 `CUDAExecutionProvider`，不是 TensorRT。
TensorRT 使用单 `decoder_with_past` session。`token_embed` 和 `audio_encoder` 保持 CUDA EP，避免不必要的 engine 和数值偏差；decoder 使用 FP16 engine，并对 LayerNorm 启用 FP32 fallback。

| provider | decoder | optimized | KV cache | 吞吐 | decode decoder |
|---|---|---:|---|---:|---:|
| CUDA | merged | no | host NumPy | 12.753x | 22.36 ms/token |
| CUDA | merged | yes | host NumPy | 12.695x | 22.49 ms/token |
| CUDA | split | no | host NumPy | 13.531x | 20.95 ms/token |
| CUDA | split | yes | host NumPy | 13.580x | 20.90 ms/token |
| CUDA | single | no | CUDA I/O Binding | 15.459x | 17.50 ms/token |

TensorRT 10.16 + ORT 1.27 实测，条件同样为 `asr_en.wav`、`max-new-tokens=256`、warmup 1 次、benchmark 5 秒。profile 上限为 512，engine cache 命中：

| provider | decoder | optimized | KV cache | 吞吐 | decode decoder |
|---|---|---:|---|---:|---:|
| TensorRT decoder + CUDA support graphs | single | no | CUDA I/O Binding | 22.215x | 9.93 ms/token |

CUDA single 和 TensorRT single 都在每轮生成 47 token 后遇到 EOS，并使用相同的 GPU 常驻 KV 路径，因此可以直接比较。TensorRT 吞吐高约 44%，逐 token decoder 延迟低约 43%。生成的 FP16 decoder engine 约 1.20 GB。

ONNX Runtime OpenVINO EP 实测：

环境为 `onnxruntime-openvino 1.24.1` + `openvino 2025.4.1`。Intel iGPU 使用显式设备 `--openvino-device GPU.0`。单独 `uv pip install openvino` 在旧脚本里没用，是因为 Windows 不会自动把 `openvino\libs` 加入 DLL 搜索路径；当前脚本已在导入 `onnxruntime` 前处理这个路径。

| provider | decoder | optimized | 吞吐 | decode decoder |
|---|---|---:|---:|---:|
| OpenVINO GPU | merged | no | 3.628x | 57.74 ms/token |
| OpenVINO GPU | merged | yes | 3.406x | 61.68 ms/token |
| OpenVINO GPU | split | no | 3.907x | 51.59 ms/token |
| OpenVINO GPU | split | yes | 3.507x | 57.14 ms/token |
| OpenVINO Intel GPU.0 | single | no | 3.796x | 53.13 ms/token |

OpenVINO 能枚举 RTX 4090 为 `GPU.1`，但 single decoder 在 session 初始化时编译失败。Intel GPU plugin 无法为 `MatMul_9847` 的 FP16 到 FP32 reorder 选择 kernel，因此没有有效的 OpenVINO RTX benchmark。

结论：

- `asr.py` 默认 `--dtype bf16`，在这台机器上明显慢于 `--dtype fp16`。如果使用 PyTorch CUDA，优先试 `--dtype fp16`。
- 当前最快的统一 single 路径是 TensorRT + GPU 常驻 KV，约 `22.215x`；CUDA single 为 `15.459x`。
- DML I/O Binding 后，RTX 4090 的 DirectML merged/split 约 `4.94-5.02x`，Intel iGPU 约 `0.78-0.79x`；single 在两块 GPU 上都明显更慢。
- CUDA 和 RTX 4090 DirectML 下，未合并 decoder 的 split 版更快一些；Intel iGPU DirectML 下 merged/split 差别很小，主要已经被逐 token decoder 拖住。
- split 版速度更好，但会保留两份 decoder 权重：`decoder_init` 和 `decoder_with_past`；merged 版只保留一份 decoder 权重，更适合减小分发体积。
- ONNX 的主要瓶颈仍在逐 token decoder。CUDA EP 下 `decode_decoder` 约 `20.90-22.49 ms/token`；RTX 4090 DirectML 下约 `65.22-71.91 ms/token`；Intel iGPU DirectML 下约 `329.54-333.99 ms/token`。
- OpenVINO Intel GPU 后端可以正常运行，single 为 `3.796x`，明显快于 DirectML single 跑同一块 iGPU；OpenVINO RTX decoder 编译失败。
- TensorRT 必须使用 single decoder；merged `If` 图会被 TensorRT parser 拒绝。不要移除 `trt_layer_norm_fp32_fallback`，否则本机实测会产生重复乱码。
- GPU 常驻 KV 后，同条件 TensorRT single 为 `22.215x / 9.93 ms/token`，CUDA single 为 `15.459x / 17.50 ms/token`。

如需导出 split decoder 做对照：

```bat
.venv\Scripts\python.exe export_qwen3_asr_onnx.py ^
  --model-dir .\Qwen3-ASR-0.6B ^
  --out-dir .\onnx_models ^
  --device cuda ^
  --dtype fp16 ^
  --keep-decoder-parts
```

运行 split decoder benchmark：

```bat
.venv\Scripts\python.exe onnx_asr_demo.py ^
  --model-dir .\Qwen3-ASR-0.6B ^
  --onnx-dir .\onnx_models ^
  --provider cuda ^
  --decoder-layout split ^
  --benchmark ^
  --audio .\asr_en.wav
```

运行 OpenVINO GPU benchmark：

```bat
.venv\Scripts\python.exe onnx_asr_demo.py ^
  --model-dir .\Qwen3-ASR-0.6B ^
  --onnx-dir .\onnx_models ^
  --provider openvino ^
  --openvino-device GPU ^
  --decoder-layout split ^
  --benchmark ^
  --audio .\asr_en.wav
```

## 6. 常见问题

- 不要在同一个环境里同时安装 `onnxruntime`、`onnxruntime-gpu`、`onnxruntime-directml`，否则 `import onnxruntime` 最终加载哪个包可能不清楚。
- `decoder_merged.onnx.data` 必须和 `decoder_merged.onnx` 放在同一个目录。
- `--optimized` 运行时必须确保对应的 `.optimized.onnx.data` 也在同一目录。
- 导出阶段需要 `torch` 和 `onnx`；运行 demo 主要需要 `onnxruntime-directml` 或 `onnxruntime-gpu`、`numpy`、`qwen-asr`、`transformers`。
