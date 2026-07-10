# Qwen3-ASR ONNX Runtime TensorRT 适配记录

本文记录 `onnx_asr_demo.py` 从 CUDA/DirectML 路径适配到 ONNX Runtime TensorRT Execution Provider 的原因、方案、失败实验和最终结果。重点不是罗列最终参数，而是说明为什么需要这些改动，以及遇到相似现象时应如何定位。

## 1. 目标与约束

原始现象是程序长时间没有转写输出，同时显存占用反复升降，看起来像模型在不断加载和卸载。

本次适配的目标：

- 确认是否存在 Python 代码反复创建 session 或加载权重的问题；
- TensorRT 下只保留一份 decoder 权重，避免 split 模式的双 decoder 显存占用；
- 让 prefill 和逐 token decode 共用一个 decoder session；
- 避免动态 KV cache 导致 TensorRT 反复构建 engine；
- 保持输出与 CUDA 路径一致；
- 缓存 TensorRT engine，避免每次启动完整重建。

验证环境：

```text
Windows
onnxruntime-gpu 1.27.0
TensorRT 10.16.1 / CUDA 12 runtime
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

## 2. 原始代码并没有反复加载模型

代码检查确认：

- `make_session()` 只在启动阶段创建 `ort.InferenceSession`；
- `load_graphs()` 创建 session 后由 `OnnxGraphs` 持续持有；
- `generate()` 循环里只有 `session.run()`，没有重新创建 session；
- benchmark 的多轮转写也复用同一组 session。

因此显存锯齿不是 Python 层显式卸载、重载模型。更可能的来源是：

- TensorRT 针对新动态 shape 构建或更新 engine/profile；
- ORT 在 TensorRT、CUDA fallback 之间进行图分区和内存搬运；
- 普通 `session.run()` 把 KV cache 输出复制到 CPU NumPy，下一轮又上传 GPU；
- TensorRT/ORT allocator 为不同长度的临时张量反复申请和释放显存。

需要注意，`session.get_providers()` 包含 TensorRT 只表示该 EP 注册到了 session，并不证明所有节点都实际由 TensorRT 执行。判断 fallback 应结合 ORT 日志、profiling 或实际 EP error，不能只看 provider 列表。

## 3. 为什么不用 merged decoder

原来的 `decoder_merged.onnx` 用顶层 ONNX `If(use_past)` 包含两个分支：

- `use_past=False`：prefill；
- `use_past=True`：带 KV cache 的增量解码。

这种结构在 CUDA EP 下可以运行，而且磁盘和显存中只需要一份去重后的 decoder 权重。但 TensorRT parser 会直接拒绝该图：

```text
INVALID_GRAPH
Nodes in a graph must be topologically sorted,
however input 'inputs_embeds' ... is not output of any previous nodes.
```

原因是 `If` 分支图通过 ONNX 外层作用域捕获输入和 initializer。这个结构符合 ORT/CUDA 的执行方式，但不被当前 TensorRT parser 接受。

最终策略是：TensorRT 请求 `merged` layout 时直接 fail fast，提示改用 `single`，避免让用户等待很久后才看到难以理解的 parser 错误。

## 4. 为什么不用 split decoder

split 模式会分别创建：

```text
decoder_init.onnx
decoder_with_past.onnx
```

两个 ONNX Runtime session 无法可靠共享 TensorRT engine 中的权重。即使两个 ONNX 文件引用相同 external data，TensorRT 仍会创建两套 engine 和执行上下文，decoder 权重会重复占用显存。

`trt_context_memory_sharing_enable` 也不能解决这个问题。它面向同一 session 内 TensorRT 子图的部分 context memory，不会让两个独立 engine 共享模型权重。

所以本次新增 `single` layout，并将其设为默认：

```text
prefill:
decoder_with_past(full_prompt, empty_KV)
    -> logits + full_KV

decode:
decoder_with_past(one_token, previous_KV)
    -> logits + extended_KV
```

`decoder_with_past` 的计算图本来就支持长度为 0 的 past cache。CUDA 回归验证了完整 prompt + 空 KV 的 prefill 语义正确。

这样做的结果：

- 不需要 `decoder_init` session；
- 不需要 TensorRT 不支持的 `If`；
- decoder 参数只驻留一份；
- prefill 和 decode 复用同一个 TensorRT engine/session。

## 5. 动态 shape profile 是适配核心

decoder 每生成一个 token，以下维度都会变化：

- `inputs_embeds` 的序列长度：prefill 为完整 prompt，decode 为 1；
- `position_ids` 的序列长度；
- `attention_mask` 的总长度；
- 28 层 `past_key_N` 和 `past_value_N` 的 `past_seq` 长度。

如果不提供显式 profile，TensorRT EP 可能在第一次看到 shape 时构建一个窄 profile，后续 KV 长度增长超出范围时更新 profile 或重建 engine。这正是“长时间无输出 + 显存反复升降”的主要嫌疑。

### 5.1 Profile 到底是什么

ONNX 用符号维度表示动态 shape，例如 `inputs_embeds` 是 `[1, seq, 1024]`。TensorRT 构建 engine 时不能只知道 `seq` 是动态的，还必须知道它允许在哪个范围内变化。Optimization profile 为每个动态输入给出三组 shape：

- `min`：运行时允许的最小 shape；
- `max`：运行时允许的最大 shape；
- `opt`：TensorRT 选择和计时 tactic 时重点优化的代表 shape。

`opt` 不是平均值，也不是运行上限。运行 shape 只要落在 `min` 和 `max` 之间就可以执行，但接近 `opt` 的 shape 通常更容易得到较好的 tactic。范围越宽，engine 需要覆盖的情况越多，构建时间、workspace 或最终性能都可能变差。

Profile 是 engine 构建契约的一部分。修改 min/opt/max、精度配置、模型或 TensorRT 版本后，应视为需要重新构建 engine，不能假设旧 cache 仍然有效。

### 5.2 每个 shape 和数字从哪里来

Qwen3-ASR-0.6B decoder 的输入可写成：

| 输入 | shape | 数字来源 |
|---|---|---|
| `inputs_embeds` | `[1, S, 1024]` | batch 固定为 1；`S` 是本轮输入 token 数；`1024` 是 config 的 `hidden_size` |
| `position_ids` | `[3, 1, S]` | `3` 是 Qwen3-ASR MRoPE 的三路 position ids；batch 为 1 |
| `attention_mask` | `[1, T]` | `T` 是 past 与本轮输入合并后的总长度 |
| `past_key_N` | `[1, 8, P, 128]` | `8` 是 `num_key_value_heads`；`128` 是 `head_dim`；`P` 是 past 长度 |
| `past_value_N` | `[1, 8, P, 128]` | 与 key 相同 |

模型 config 中的结构常量是：

```text
num_hidden_layers      = 28
num_key_value_heads    = 8
head_dim               = 128
hidden_size            = 1024
max_position_embeddings = 65536
```

因此代码会生成 `28 * 2 = 56` 个 past profile 条目，即每层一个 key 和一个 value。这些数字来自模型结构，不能为了性能随意修改。

prefill 和 decode 的序列关系分别是：

```text
prefill: P = 0, S = prompt_len, T = S
decode:  S = 1, T = P + 1
```

代码里其他数字的来源：

- `0`：single decoder prefill 使用真正的空 KV cache；
- `1`：固定 batch、单 token decode，以及动态维度的最小非空长度；
- `256`：人为选择的 opt 总长度，不是模型常量；代码取 `min(256, max_seq_len)`；
- `255`：opt decode 点满足 `P = T - 1`，所以 `255 = 256 - 1`；
- `512`：本轮实验显式传入的 `--trt-max-seq-len`，因为 prompt 约 211，`211 + 256 = 467 <= 512`；
- `1023`：max profile 合法点的总长度，`1023 = 512 + 511 = 2 * 512 - 1`；
- 默认 `1024`：运行脚本给出的通用 profile 上限，不是模型最大上下文；
- `65536`：模型理论最大位置长度。直接用它构建当前 profile 会显著扩大搜索范围，没有必要；
- 代码把上限至少钳制到 `2`，因为 decode profile 至少要表达 1 个 past token 加 1 个新 token。

以本次 `--trt-max-seq-len 512` 为例，single decoder 的实际 profile 是：

| profile 点 | `inputs_embeds` | `position_ids` | `attention_mask` | 每个 past key/value |
|---|---|---|---|---|
| min | `[1,1,1024]` | `[3,1,1]` | `[1,1]` | `[1,8,0,128]` |
| opt | `[1,1,1024]` | `[3,1,1]` | `[1,256]` | `[1,8,255,128]` |
| max | `[1,512,1024]` | `[3,1,512]` | `[1,1023]` | `[1,8,511,128]` |

这里的 max 是为了让 TensorRT 的矩形范围同时容纳两种极端：`S` 很大的 prefill 和 `P` 很大的 decode。正常 decode 的实际 `T` 仍按 `P + 1` 增长，本次目标总长度不超过 512；`attention_mask=1023` 只是为了让 profile 的 max 组合本身满足 `T=S+P`，不是建议运行 1023-token decode。

代码还为 `token_embed` 和 `decoder_init` 实现了简单 profile，供直接使用 TensorRT 或 split 对照时使用。当前推荐的 TensorRT single 路径会把 `token_embed`、`audio_encoder` 放在 CUDA，因此性能实验中的关键 profile 是 `decoder_with_past` 这一组。

### 5.3 失败方案：两个 optimization profile

第一版为同一个 decoder 配置了两个 profile：

- profile 0：完整 prompt + 空 KV；
- profile 1：单 token + 增长 KV。

prefill 成功，但第一次 decode 报错：

```text
TensorRT EP failed to call IExecutionContext::setInputShape()
for input 'past_key_0'

Set dimensions:      [1, 8, 211, 128]
Expected dimensions: [1, 8,   0, 128]
```

ORT TensorRT EP 没有在 prefill 和 decode 之间切换到第二个 profile，执行上下文仍使用把 `past_seq` 固定为 0 的 profile。

经验：不要假设 ORT 会根据每次输入 shape 自动选择同一 fused subgraph 的另一个 TensorRT optimization profile。至少在 ORT 1.27 + TensorRT 10.16 的组合上，这个方案不可用。

### 5.4 最终方案：一个矩形动态范围

最终只提供一组 min/opt/max shape，同时包含两种运行模式：

```text
min: prefill 最小点
  inputs_embeds seq = 1
  total_seq = 1
  past_seq = 0

opt: 常见 decode 点
  inputs_embeds seq = 1
  total_seq = 256
  past_seq = 255

max: 同时覆盖最大 prompt 和最大 cache
  inputs_embeds seq = max_seq_len
  total_seq = max_seq_len * 2 - 1
  past_seq = max_seq_len - 1
```

max 点之所以使用 `total_seq = input_seq + past_seq`，是因为 profile 的 min/opt/max 点本身也必须是合法的输入组合。TensorRT profile 是各动态维度的矩形范围，无法直接表达“prefill 时 past 必须为 0，decode 时 input 必须为 1”这种联合约束，只能用一个稍宽的范围覆盖两条路径。

`--trt-max-seq-len` 控制 profile 上限，默认 1024。上限越大，engine 构建时间、workspace 和 tactic 搜索成本越高；应按实际最长 prompt + generation 设置，不要无条件设置到模型理论最大位置长度。

## 6. Engine cache 与可观察性

TensorRT session 初始化可能在几十秒内没有任何输出。原程序在全部 session 创建完成后才打印 provider，很容易被误判为死锁。

适配后在每个 session 创建前后打印：

```text
Loading ONNX session: decoder_with_past.onnx (tensorrt)
Loaded ONNX session: decoder_with_past.onnx in 30.92s
```

并启用了：

```python
trt_engine_cache_enable = True
trt_engine_cache_path = onnx_models/trt_engine_cache
trt_timing_cache_enable = True
trt_timing_cache_path = onnx_models/trt_engine_cache
```

缓存前缀设为 `qwen3_asr_lnfp32`。精度策略、profile、模型或 TensorRT 版本变化时，应使用新的前缀或清理旧 cache。不要盲目复用旧 engine；engine 与 GPU 架构、TensorRT 版本、模型内容和构建参数相关。

本机结果：

```text
首次构建 decoder session: 约 31 秒
engine cache 命中后:       约 9 秒
最终 FP16 decoder engine:  约 1.20 GB
```

## 7. FP16 数值精度踩坑

### 7.1 强制 FP16 会运行成功但输出错误

最初设置：

```python
trt_fp16_enable = True
```

程序不报错，也能连续生成，但输出是重复乱码：

```text
Їв
Їе
Їе
...
```

这比显式异常更危险，因为从退出码和 provider 列表看运行完全“成功”。CUDA 对照输出是正常英文：

```text
language English<asr_text>Uh huh. Oh yeah yeah, he wasn't even that big
```

TensorRT 日志同时提示 FP16 LayerNorm/Reduce/Pow 存在溢出风险。ASR 使用 greedy decoding，logits 的小误差可能直接改变第一个 token，之后整条生成轨迹都会偏离。

### 7.2 完全关闭 FP16 正确但 engine 翻倍

将 `trt_fp16_enable` 关闭后，输出与 CUDA 一致，但 decoder engine 从约 1.20 GB 增长到约 2.39 GB。这虽然仍只有一个 decoder session，但不符合降低显存占用的目标。

### 7.3 最终精度配置

最终配置：

```python
trt_fp16_enable = True
trt_layer_norm_fp32_fallback = True
```

结果：

- 输出 token 序列与 CUDA 对照一致；
- decoder engine 保持约 1.20 GB；
- 没有 decode profile error 或运行时 CUDA fallback；
- 标准可比 benchmark 中 decode 为 28.91 ms/token。

经验：FP16 engine 能构建并运行不等于数值正确。移植生成模型时必须对比实际 token 序列，不能只验证 session 创建、退出码或张量 shape。

## 8. 为什么辅助图保留 CUDA EP

最终 TensorRT 模式只把 `decoder_with_past` 放到 TensorRT：

```text
token_embed       -> CUDA EP
audio_encoder     -> CUDA EP
decoder_with_past -> TensorRT EP, CUDA fallback, CPU fallback
```

原因：

- decoder 是权重和逐 token 计算的主要瓶颈，TensorRT 收益最大；
- 为 `token_embed` 和 `audio_encoder` 额外构建 engine 会增加启动时间和磁盘 cache；
- 辅助图不是逐 token decode 的主要瓶颈，把 TensorRT 精度策略限制在 decoder 更容易验证；
- 辅助图使用 CUDA 不会复制 decoder 权重，也不会破坏“单 decoder session”的显存目标。

这是一种有意的混合 EP 策略，不是意外 fallback。启动日志会明确显示每个 session 的 provider。

早期“全部子图走 TensorRT FP16”的实验确实产生乱码，但将两个辅助图改回 CUDA 后乱码仍然存在，最终通过 decoder 的 LayerNorm FP32 fallback 才解决。因此可以确认的数值问题在 decoder；没有单独证明 audio encoder 的 TensorRT 输出错误。

## 9. Windows TensorRT DLL 版本问题

ORT 1.27 的 TensorRT EP 在本机查找：

```text
nvinfer_10.dll
```

直接安装最新 `tensorrt` 可能得到 TensorRT 11，只提供 `nvinfer_11.dll`，此时 `TensorrtExecutionProvider` 虽然会出现在 available providers 中，但 provider DLL 实际加载失败。

可用安装方式：

```powershell
uv pip install --python .\.venv\Scripts\python.exe "tensorrt-cu12<11"
```

pip 包把 DLL 放在：

```text
.venv\Lib\site-packages\tensorrt_libs
```

Windows 不会自动搜索这个目录，因此 `add_local_dll_directories()` 增加了 `tensorrt_libs`。这样无需每次手动修改 `PATH`。

经验：`ort.get_available_providers()` 只能说明 wheel 编译时注册了哪些 EP，不能证明对应 provider 的外部 DLL 依赖已满足。

## 10. 导出脚本为什么必须保留 decoder_with_past

旧导出流程只把 `decoder_with_past` 当作生成 merged decoder 的临时图；未指定 `--keep-decoder-parts` 时会在合并后删除。

single layout 将它变成正式运行模型，因此导出脚本现在始终保留：

```text
decoder_with_past.onnx
decoder_with_past.onnx.data
```

`--keep-decoder-parts` 现在只控制是否额外保留 `decoder_init`，用于 split 性能对照。保留额外 ONNX 文件只增加磁盘占用；single 运行时不会加载 `decoder_init` 或 `decoder_merged`，因此不会增加显存。

## 11. 验证结果

正确性命令：

```powershell
.\.venv\Scripts\python.exe -u onnx_asr_demo.py `
  --model-dir .\Qwen3-ASR-0.6B `
  --onnx-dir .\onnx_models `
  --provider tensorrt `
  --decoder-layout single `
  --trt-max-seq-len 512 `
  --audio .\asr_en.wav `
  --max-new-tokens 256
```

TensorRT 与 CUDA 对照输出：

```text
Language: English
Text: Uh huh. Oh yeah yeah, he wasn't even that big
Raw: language English<asr_text>Uh huh. Oh yeah yeah, he wasn't even that big
```

可横向比较的 benchmark 命令：

```powershell
.\.venv\Scripts\python.exe -u onnx_asr_demo.py `
  --model-dir .\Qwen3-ASR-0.6B `
  --onnx-dir .\onnx_models `
  --provider tensorrt `
  --decoder-layout single `
  --trt-max-seq-len 512 `
  --audio .\asr_en.wav `
  --max-new-tokens 256 `
  --benchmark `
  --benchmark-seconds 5 `
  --warmup-runs 1
```

CUDA single 使用完全相同参数，只把 `--provider` 改为 `cuda`。两次运行都在每轮生成 47 token 后遇到 EOS；`256` 是相同的生成上限，不代表每轮必须生成满 256 token。

| provider | decoder | optimized | 吞吐 | decode decoder | runs | tokens/run |
|---|---|---:|---:|---:|---:|---:|
| CUDA | single | no | 13.095x | 21.63 ms/token | 5 | 47 |
| TensorRT decoder + CUDA support graphs | single | no | 9.028x | 28.91 ms/token | 4 | 47 |

这组数据可以直接比较，因为音频、模型、decoder graph、生成上限、EOS 行为、warmup 和计时规则均相同。当前 TensorRT single 比 CUDA single 慢：吞吐低约 31%，逐 token decoder 延迟高约 34%。因此本次适配的成果是 TensorRT 可运行、输出正确且只有一份 decoder 权重，而不是 TensorRT 性能已经优于 CUDA。

## 12. 尚未解决和后续优化

### 12.1 KV cache 仍经过 CPU

当前使用普通 `session.run()`：

```text
GPU present KV -> CPU NumPy -> GPU past KV
```

这不会复制模型权重，但会产生大量 PCIe 传输和临时显存分配。下一步可使用 ORT I/O Binding 或 CUDA `OrtValue`，只把 logits 传回 CPU，让 KV cache 常驻 GPU。

### 12.2 Int64 binding 警告

TensorRT 仍提示：

```text
Make sure input position_ids has Int64 binding.
Make sure input attention_mask has Int64 binding.
```

目前不是致命错误，正确性实验已通过。未来可评估导出 int32 输入，但必须检查 `Gather`、`Range`、shape 运算和位置编码相关算子的类型要求，不能只在 Python feed 端强制转换。

### 12.3 更长序列需要单独验证

目前验证了 prompt 211 token、生成上限 256、实际 47 token 遇到 EOS、profile 上限 512。使用更长音频、context 或更大 `max_new_tokens` 时，应确保实际总长度不超过 `--trt-max-seq-len`，并重新做正确性和显存峰值测试。

### 12.4 不要混淆磁盘 cache 与显存

实验过程中可能留下多个 `.engine` 文件。只有当前 cache 前缀和配置匹配的 engine 会被加载到显存；旧文件只占磁盘空间。清理 cache 后首次启动会重新构建，出现几十秒无输出和显存波动是正常现象。

## 13. 排障顺序建议

遇到 TensorRT “卡住”时按以下顺序检查：

1. 确认 `nvinfer_10.dll` 等依赖能被加载，而不只是 provider 名称可见。
2. 使用 `python -u` 并在每个 session 创建前后打印时间。
3. 单独创建各 ONNX session，定位是 parser、engine build 还是首次 inference。
4. 对控制流图优先检查 TensorRT parser 兼容性。
5. 对所有动态输入配置显式 min/opt/max profile，尤其是每层 KV cache。
6. 检查 profile 是否真正覆盖 prefill 和 decode，而不是只覆盖其中一条路径。
7. 观察日志中是否发生 EP error 后自动回退 CUDA。
8. 对比 CUDA 和 TensorRT 的实际 token 输出，确认数值正确性。
9. 正确性通过后再启用 engine/timing cache 并做 benchmark。
10. 最后再优化 I/O Binding、int32 输入和更细的 profile/bucket 策略。
