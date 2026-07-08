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
cpu
```

Windows + `onnxruntime-directml` 环境通常用 `--provider directml`。

## 5. 常见问题

- 不要在同一个环境里同时安装 `onnxruntime`、`onnxruntime-gpu`、`onnxruntime-directml`，否则 `import onnxruntime` 最终加载哪个包可能不清楚。
- `decoder_merged.onnx.data` 必须和 `decoder_merged.onnx` 放在同一个目录。
- `--optimized` 运行时必须确保对应的 `.optimized.onnx.data` 也在同一目录。
- 导出阶段需要 `torch` 和 `onnx`；运行 demo 主要需要 `onnxruntime-directml`、`numpy`、`qwen-asr`、`transformers`。
