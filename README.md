# Qwen3-ASR ONNX Demo

这个项目把本地 `Qwen3-ASR-0.6B` safetensors 模型导出成 ONNX Runtime 可运行的子图，并使用一个 merged decoder：

- `token_embed.onnx`
- `audio_encoder.onnx`
- `decoder_merged.onnx`

`decoder_merged.onnx` 内部用 `If(use_past)` 同时保留首轮 prefill 和后续 KV cache 解码路径，但只保存一份 decoder 权重，避免分发两份 decoder。

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
  decoder_merged.optimized.onnx
  decoder_merged.optimized.onnx.data
```

`decoder_init.tmp.onnx` 和 `decoder_with_past.tmp.onnx` 只是导出中间文件，脚本会在合并后清理。

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
`--provider tensorrt` 需要本机安装 TensorRT runtime，并确保 `nvinfer_10.dll` 等依赖在 `PATH` 中。
`--provider openvino` 需要 `onnxruntime-openvino` 和版本匹配的 `openvino` Python 包；脚本会自动把 `.venv\Lib\site-packages\openvino\libs` 加入 DLL 搜索路径，所以不需要手动把 DLL 复制到项目根目录。

## 5. 性能实测结论

以下结果均使用 `asr_en.wav`、`max-new-tokens=256`、warmup 1 次、benchmark 5 秒。不同 ONNX Runtime wheel 不能稳定混装，DirectML / CUDA / OpenVINO 结果来自切换到对应 wheel 后的测试。

PyTorch CUDA 对照：

```text
asr.py bf16 CUDA: 4.628x
asr.py fp16 CUDA: 8.213x
```

ONNX Runtime DirectML EP 实测，RTX 4090 Laptop GPU：

| provider | decoder | optimized | 吞吐 | decode decoder |
|---|---|---:|---:|---:|
| DirectML | merged | no | 4.101x | 71.38 ms/token |
| DirectML | merged | yes | 4.054x | 71.91 ms/token |
| DirectML | split | no | 4.458x | 65.22 ms/token |
| DirectML | split | yes | 4.340x | 66.97 ms/token |

ONNX Runtime DirectML EP 实测，禁用 RTX 4090 后的 Intel iGPU：

环境为 `onnxruntime-directml 1.24.4`，ORT 可用 provider 为 `DmlExecutionProvider` 和 `CPUExecutionProvider`。OpenVINO device 列表显示当前 GPU 为 `Intel(R) RaptorLake-S Mobile Graphics Controller (iGPU)`。

| provider | decoder | optimized | 吞吐 | decode decoder |
|---|---|---:|---:|---:|
| DirectML Intel iGPU | merged | no | 0.728x | 333.62 ms/token |
| DirectML Intel iGPU | merged | yes | 0.735x | 329.54 ms/token |
| DirectML Intel iGPU | split | no | 0.732x | 331.23 ms/token |
| DirectML Intel iGPU | split | yes | 0.727x | 333.99 ms/token |

ONNX Runtime CUDA EP 实测：

可用 provider：

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

`onnx_asr_demo.py --provider cuda` 会显式选择 `CUDAExecutionProvider`，不是 TensorRT。
当前机器尝试 `--provider tensorrt` 时，TensorRT EP 可以挂上，但首次构建/执行耗时很长，暂未得到可用 benchmark 结果；不要把 CUDA fallback 或中途停止的结果记作 TensorRT 成绩。

| provider | decoder | optimized | 吞吐 | decode decoder |
|---|---|---:|---:|---:|
| CUDA | merged | no | 12.753x | 22.36 ms/token |
| CUDA | merged | yes | 12.695x | 22.49 ms/token |
| CUDA | split | no | 13.531x | 20.95 ms/token |
| CUDA | split | yes | 13.580x | 20.90 ms/token |

ONNX Runtime OpenVINO EP 实测：

环境为 `onnxruntime-openvino 1.24.1` + `openvino 2025.4.1`，设备为 `--openvino-device GPU`，OpenVINO 识别到的 GPU 为 Intel iGPU。单独 `uv pip install openvino` 在旧脚本里没用，是因为 Windows 不会自动把 `openvino\libs` 加入 DLL 搜索路径；当前脚本已在导入 `onnxruntime` 前处理这个路径。

| provider | decoder | optimized | 吞吐 | decode decoder |
|---|---|---:|---:|---:|
| OpenVINO GPU | merged | no | 3.628x | 57.74 ms/token |
| OpenVINO GPU | merged | yes | 3.406x | 61.68 ms/token |
| OpenVINO GPU | split | no | 3.907x | 51.59 ms/token |
| OpenVINO GPU | split | yes | 3.507x | 57.14 ms/token |

结论：

- `asr.py` 默认 `--dtype bf16`，在这台机器上明显慢于 `--dtype fp16`。如果使用 PyTorch CUDA，优先试 `--dtype fp16`。
- ONNX CUDA EP 明显快于其他 ONNX EP。当前最快的是 `--provider cuda --decoder-layout split --optimized`，约 `13.58x`。
- RTX 4090 Laptop 上的 DirectML 能到 `4.05-4.46x`；禁用 RTX 4090 后，DirectML 跑 Intel iGPU 只有 `0.73x` 左右。
- CUDA 和 RTX 4090 DirectML 下，未合并 decoder 的 split 版更快一些；Intel iGPU DirectML 下 merged/split 差别很小，主要已经被逐 token decoder 拖住。
- split 版速度更好，但会保留两份 decoder 权重：`decoder_init` 和 `decoder_with_past`；merged 版只保留一份 decoder 权重，更适合减小分发体积。
- ONNX 的主要瓶颈仍在逐 token decoder。CUDA EP 下 `decode_decoder` 约 `20.90-22.49 ms/token`；RTX 4090 DirectML 下约 `65.22-71.91 ms/token`；Intel iGPU DirectML 下约 `329.54-333.99 ms/token`。
- OpenVINO GPU 后端现在可以正常运行；在 Intel iGPU 上总吞吐 `3.41-3.91x`，明显快于 DirectML 跑同一块 iGPU。本轮 OpenVINO 测试里未优化模型反而比 optimized ONNX 快。
- TensorRT EP 暂未得到有效成绩。

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
