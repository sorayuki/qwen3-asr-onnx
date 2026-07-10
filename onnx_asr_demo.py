import argparse
import json
import os
import site
import time
from dataclasses import dataclass
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


def add_local_dll_directories():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    roots = [Path.cwd(), Path(__file__).resolve().parent]
    candidates = []
    for root in roots:
        candidates.extend(
            [
                root,
                root / "third_party",
            ]
        )
        third_party = root / "third_party"
        if third_party.is_dir():
            for child in third_party.iterdir():
                candidates.extend(
                    [
                        child,
                        child / "bin",
                        child / "lib",
                        child / "runtime" / "bin" / "intel64" / "Release",
                    ]
                )

    try:
        site_dirs = site.getsitepackages()
    except Exception:
        site_dirs = []
    try:
        site_dirs.append(site.getusersitepackages())
    except Exception:
        pass

    for site_dir in site_dirs:
        package_root = Path(site_dir)
        candidates.extend(
            [
                package_root / "openvino" / "libs",
                package_root / "openvino" / "lib",
                package_root / "tensorrt_libs",
            ]
        )

    seen = set()
    path_entries = []
    for directory in candidates:
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(resolved)))
        path_entries.append(str(resolved))

    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries + [os.environ.get("PATH", "")])


add_local_dll_directories()

import numpy as np
import onnxruntime as ort

from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from qwen_asr.core.transformers_backend.processing_qwen3_asr import _get_feat_extract_output_lengths
from qwen_asr.inference.utils import (
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    validate_language,
)


EOS_TOKEN_IDS = {151645, 151643}
AUDIO_TOKEN_ID = 151676


# 这个 demo 让 Python 负责 ONNX 不擅长的动态部分：
# prompt 构造、音频切块、音频 token 的 embedding 替换、生成循环和 KV cache 转发。
# 真正耗算力的张量计算交给 ONNX Runtime 执行。
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 的 ONNX Runtime 运行示例。")
    parser.add_argument(
        "--model-dir",
        default="./Qwen3-ASR-0.6B",
        help="包含 tokenizer/processor/config 等轻量文件的目录；不会读取 model.safetensors。",
    )
    parser.add_argument("--onnx-dir", default="./onnx_models")
    parser.add_argument(
        "--audio",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    )
    parser.add_argument("--language", default=None, help='可选的强制语言，例如 "English"。')
    parser.add_argument("--context", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--optimized", action="store_true", help="使用 *.optimized.onnx 模型。")
    parser.add_argument(
        "--decoder-layout",
        choices=["single", "merged", "split"],
        default="single",
        help=(
            "single 只加载 decoder_with_past 并用空 KV 完成 prefill；"
            "merged 使用 decoder_merged；split 加载两个 decoder 做性能对照。"
        ),
    )
    parser.add_argument("--benchmark", action="store_true", help="反复处理同一个音频，输出 warmup 后的吞吐。")
    parser.add_argument("--benchmark-seconds", type=float, default=15.0, help="benchmark 至少运行多少秒。")
    parser.add_argument("--warmup-runs", type=int, default=1, help="正式计时前完整跑几次预热。")
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "directml", "cuda", "tensorrt", "openvino"],
        default="auto",
        help="ONNX Runtime 执行后端偏好。",
    )
    parser.add_argument(
        "--openvino-device",
        default="GPU",
        help='OpenVINO device_type，例如 "GPU"、"GPU.0"、"CPU"。',
    )
    parser.add_argument(
        "--trt-max-seq-len",
        type=int,
        default=1024,
        help="TensorRT profile 覆盖的最大 prompt + 生成序列长度。",
    )
    return parser.parse_args()


@dataclass
class OnnxGraphs:
    token_embed: ort.InferenceSession
    audio_encoder: ort.InferenceSession
    # decoder_merged.onnx 内部用 If(use_past) 在 prefill 和 decode 两条快路径之间切换。
    # 运行时只创建这一个 decoder session，避免重复加载 decoder 权重。
    decoder: ort.InferenceSession | None
    decoder_init: ort.InferenceSession | None
    decoder_with_past: ort.InferenceSession | None
    decoder_layout: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int


@dataclass
class TimingStats:
    runs: int = 0
    total: float = 0.0
    prepare_inputs: float = 0.0
    audio_encoder: float = 0.0
    prefill_token_embed: float = 0.0
    merge_embeddings: float = 0.0
    position_ids: float = 0.0
    decoder_prefill: float = 0.0
    decode_token_embed: float = 0.0
    decode_decoder: float = 0.0
    decode_misc: float = 0.0
    batch_decode_parse: float = 0.0
    decode_tokens: int = 0


def add_elapsed(stats: TimingStats | None, name: str, start: float):
    if stats is not None:
        setattr(stats, name, getattr(stats, name) + time.perf_counter() - start)


def trt_shape_profile(
    model_stem: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    max_seq_len: int,
) -> dict[str, str]:
    max_seq_len = max(2, max_seq_len)
    opt_seq_len = min(256, max_seq_len)

    if model_stem == "token_embed":
        return {
            "trt_profile_min_shapes": "input_ids:1x1",
            "trt_profile_opt_shapes": f"input_ids:1x{opt_seq_len}",
            "trt_profile_max_shapes": f"input_ids:1x{max_seq_len}",
        }

    if model_stem == "decoder_init":
        return {
            "trt_profile_min_shapes": (
                f"inputs_embeds:1x1x{hidden_size},position_ids:3x1x1,attention_mask:1x1"
            ),
            "trt_profile_opt_shapes": (
                f"inputs_embeds:1x{opt_seq_len}x{hidden_size},"
                f"position_ids:3x1x{opt_seq_len},attention_mask:1x{opt_seq_len}"
            ),
            "trt_profile_max_shapes": (
                f"inputs_embeds:1x{max_seq_len}x{hidden_size},"
                f"position_ids:3x1x{max_seq_len},attention_mask:1x{max_seq_len}"
            ),
        }

    if model_stem != "decoder_with_past":
        return {}

    def decoder_shapes(input_len: int, total_len: int, past_len: int) -> str:
        shapes = [
            f"inputs_embeds:1x{input_len}x{hidden_size}",
            f"position_ids:3x1x{input_len}",
            f"attention_mask:1x{total_len}",
        ]
        for i in range(num_layers):
            shapes.append(f"past_key_{i}:1x{num_kv_heads}x{past_len}x{head_dim}")
            shapes.append(f"past_value_{i}:1x{num_kv_heads}x{past_len}x{head_dim}")
        return ",".join(shapes)

    # ORT's TensorRT EP uses one optimization profile per fused subgraph and
    # does not switch profiles between prefill and decode. Use one rectangular
    # range that contains both full-prompt/empty-KV and one-token/growing-KV.
    # The max point must itself be valid, hence total_len=input_len+past_len.
    decode_opt_past = min(255, max_seq_len - 1)
    return {
        "trt_profile_min_shapes": decoder_shapes(1, 1, 0),
        "trt_profile_opt_shapes": decoder_shapes(1, decode_opt_past + 1, decode_opt_past),
        "trt_profile_max_shapes": decoder_shapes(max_seq_len, max_seq_len * 2 - 1, max_seq_len - 1),
    }


def provider_list(
    provider: str,
    openvino_device: str = "GPU",
    *,
    model_stem: str = "",
    trt_cache_dir: Path | None = None,
    trt_max_seq_len: int = 1024,
    num_layers: int = 0,
    num_kv_heads: int = 0,
    head_dim: int = 0,
    hidden_size: int = 0,
):
    # auto 优先使用当前环境已注册的 CUDA，其次是 Windows 上的 DirectML，最后回退到 CPU。
    # TensorRT 首次构建 engine 可能很慢，且对动态图支持更挑剔，所以只在显式指定时启用。
    # 这里不会安装 provider，只从 ONNX Runtime 实际能看到的 provider 里选择。
    available = ort.get_available_providers()
    if provider == "openvino" and "OpenVINOExecutionProvider" in available:
        return [
            ("OpenVINOExecutionProvider", {"device_type": openvino_device}),
            "CPUExecutionProvider",
        ]
    if provider == "tensorrt" and "TensorrtExecutionProvider" in available:
        if trt_cache_dir is None:
            raise ValueError("trt_cache_dir is required for TensorRT")
        trt_cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(trt_cache_dir.resolve()),
            "trt_engine_cache_prefix": "qwen3_asr_lnfp32",
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(trt_cache_dir.resolve()),
            "trt_fp16_enable": True,
            "trt_layer_norm_fp32_fallback": True,
        }
        trt_options.update(
            trt_shape_profile(
                model_stem,
                num_layers,
                num_kv_heads,
                head_dim,
                hidden_size,
                trt_max_seq_len,
            )
        )
        return [
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if provider == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider == "directml" and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if provider == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session(
    path: Path,
    provider: str,
    openvino_device: str = "GPU",
    *,
    trt_cache_dir: Path | None = None,
    trt_max_seq_len: int = 1024,
    num_layers: int = 0,
    num_kv_heads: int = 0,
    head_dim: int = 0,
    hidden_size: int = 0,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # prompt 和 KV cache 都会使用动态长度。关闭 memory pattern 可以避免部分
    # execution provider 复用旧 shape 假设导致的问题。
    options.enable_mem_pattern = False
    model_stem = path.name.split(".", 1)[0]
    print(f"Loading ONNX session: {path.name} ({provider})", flush=True)
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=provider_list(
            provider,
            openvino_device,
            model_stem=model_stem,
            trt_cache_dir=trt_cache_dir,
            trt_max_seq_len=trt_max_seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
        ),
    )
    print(f"Loaded ONNX session: {path.name} in {time.perf_counter() - started:.2f}s", flush=True)
    if provider == "openvino" and "OpenVINOExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "OpenVINOExecutionProvider was requested but the session fell back to "
            f"{session.get_providers()}. Install OpenVINO runtime libraries and make sure "
            "openvino.dll, tbb12.dll, and related plugin DLLs are on PATH."
        )
    if provider == "tensorrt" and "TensorrtExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "TensorrtExecutionProvider was requested but the session fell back to "
            f"{session.get_providers()}. Install TensorRT runtime libraries and make sure "
            "nvinfer_10.dll is on PATH."
        )
    return session


def onnx_name(onnx_dir: Path, stem: str, optimized: bool) -> Path:
    suffix = ".optimized.onnx" if optimized else ".onnx"
    return onnx_dir / f"{stem}{suffix}"


def decoder_cache_config(model_dir: str) -> tuple[int, int, int, int]:
    # merged decoder 的公开输入包含 past_key/value。首轮 prefill 虽然不使用 past，
    # 但 ORT 仍要求 feed 所有公开输入，所以这里从轻量 config 读取空 KV 的 shape。
    with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as file:
        config = json.load(file)
    text_config = config["thinker_config"]["text_config"]
    return (
        int(text_config["num_hidden_layers"]),
        int(text_config["num_key_value_heads"]),
        int(text_config["head_dim"]),
        int(text_config["hidden_size"]),
    )


def load_graphs(
    model_dir: str,
    onnx_dir: str,
    optimized: bool,
    provider: str,
    decoder_layout: str,
    openvino_device: str = "GPU",
    trt_max_seq_len: int = 1024,
) -> OnnxGraphs:
    root = Path(onnx_dir)
    num_layers, num_kv_heads, head_dim, hidden_size = decoder_cache_config(model_dir)
    decoder = None
    decoder_init = None
    decoder_with_past = None
    trt_cache_dir = root / "trt_engine_cache"

    def load(stem: str, session_provider: str | None = None) -> ort.InferenceSession:
        return make_session(
            onnx_name(root, stem, optimized),
            session_provider or provider,
            openvino_device,
            trt_cache_dir=trt_cache_dir,
            trt_max_seq_len=trt_max_seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
        )

    if decoder_layout == "merged":
        if provider == "tensorrt":
            raise ValueError("TensorRT does not support decoder_merged.onnx; use --decoder-layout single")
        decoder = load("decoder_merged")
    elif decoder_layout == "split":
        decoder_init = load("decoder_init")
        decoder_with_past = load("decoder_with_past")
    else:
        decoder_with_past = load("decoder_with_past")
    return OnnxGraphs(
        # The supporting graphs are not decode bottlenecks. Keep them on CUDA
        # to avoid extra TensorRT engines and confine the mixed-precision
        # TensorRT policy to the memory-dominant decoder.
        token_embed=load("token_embed", "cuda" if provider == "tensorrt" else None),
        audio_encoder=load("audio_encoder", "cuda" if provider == "tensorrt" else None),
        decoder=decoder,
        decoder_init=decoder_init,
        decoder_with_past=decoder_with_past,
        decoder_layout=decoder_layout,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
    )


def build_prompt(processor: Qwen3ASRProcessor, context: str, language: str | None) -> str:
    # 复用 Qwen 官方 chat template，让 prompt 和 PyTorch 推理路径保持一致。
    messages = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if language:
        prompt += f"language {language}<asr_text>"
    return prompt


def prepare_inputs(processor: Qwen3ASRProcessor, prompt: str, wav: np.ndarray):
    # processor 会把一个音频标记展开成 N 个音频占位 token；
    # 这个 N 必须和 audio encoder 输出的 embedding 数量一致。
    inputs = processor(text=[prompt], audio=[wav], return_tensors="np", padding=True)
    return {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "input_features": inputs["input_features"].astype(np.float16),
        "feature_attention_mask": inputs["feature_attention_mask"].astype(np.int64),
    }


def audio_output_len(frame_len: int) -> int:
    return int(_get_feat_extract_output_lengths(np.array([frame_len], dtype=np.int64))[0])


def run_audio_encoder(graphs: OnnxGraphs, input_features: np.ndarray, feature_attention_mask: np.ndarray) -> np.ndarray:
    features = input_features[0]
    valid_frames = int(feature_attention_mask[0].sum())
    features = features[:, :valid_frames]

    # 导出的 audio_encoder 图只处理一个 100-frame mel chunk。
    # 更长的音频在这里切块；不足 100 帧的尾块补零后送入，再按真实长度裁掉多余输出。
    pieces = []
    for start in range(0, valid_frames, 100):
        chunk = features[:, start : start + 100]
        chunk_len = chunk.shape[1]
        if chunk_len < 100:
            padded = np.zeros((features.shape[0], 100), dtype=np.float16)
            padded[:, :chunk_len] = chunk
            chunk = padded

        encoded = graphs.audio_encoder.run(None, {"input_features": chunk.astype(np.float16)})[0]
        pieces.append(encoded[: audio_output_len(chunk_len)])

    return np.concatenate(pieces, axis=0).astype(np.float16)


def token_embed(graphs: OnnxGraphs, input_ids: np.ndarray) -> np.ndarray:
    return graphs.token_embed.run(None, {"input_ids": input_ids.astype(np.int64)})[0].astype(np.float16)


def merge_audio_embeddings(input_ids: np.ndarray, inputs_embeds: np.ndarray, audio_embeddings: np.ndarray) -> np.ndarray:
    # decoder 接收的是 embedding。文本 token embedding 来自 token_embed；
    # 音频占位 token 对应的位置要替换成 audio_encoder 的输出。
    audio_positions = np.argwhere(input_ids[0] == AUDIO_TOKEN_ID).reshape(-1)
    if len(audio_positions) != audio_embeddings.shape[0]:
        raise ValueError(
            f"audio token count ({len(audio_positions)}) != audio embedding count ({audio_embeddings.shape[0]})"
        )
    merged = inputs_embeds.copy()
    merged[0, audio_positions, :] = audio_embeddings
    return merged


def position_ids_from_attention(attention_mask: np.ndarray) -> np.ndarray:
    # 对齐 Qwen3ASRPreTrainedModelForConditionalGeneration.get_rope_index()。
    # Qwen3-ASR 的 MRoPE 接口需要 3 路相同的 position ids。
    pos = attention_mask.astype(np.float32).cumsum(axis=-1) - 1
    pos[attention_mask == 0] = 1
    return np.broadcast_to(pos[None, :, :], (3, pos.shape[0], pos.shape[1])).astype(np.int64)


def output_map(session: ort.InferenceSession, values: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {out.name: value for out, value in zip(session.get_outputs(), values)}


def add_empty_past_feeds(graphs: OnnxGraphs, feeds: dict[str, np.ndarray]):
    # use_past=False 会走 If 的 decoder_init 分支，这些空 past 不参与计算；
    # 它们只是为了满足 merged graph 的统一输入签名。
    for i in range(graphs.num_layers):
        feeds[f"past_key_{i}"] = np.empty((1, graphs.num_kv_heads, 0, graphs.head_dim), dtype=np.float16)
        feeds[f"past_value_{i}"] = np.empty((1, graphs.num_kv_heads, 0, graphs.head_dim), dtype=np.float16)


def greedy_next_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits[0, -1, :]))


def load_runtime(
    model_dir: str,
    onnx_dir: str,
    optimized: bool,
    provider: str,
    decoder_layout: str,
    openvino_device: str = "GPU",
    trt_max_seq_len: int = 1024,
):
    # 模型加载和 session 创建不计入 benchmark；吞吐只统计真正开始送入音频后的耗时。
    processor = Qwen3ASRProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    graphs = load_graphs(
        model_dir,
        onnx_dir,
        optimized=optimized,
        provider=provider,
        decoder_layout=decoder_layout,
        openvino_device=openvino_device,
        trt_max_seq_len=trt_max_seq_len,
    )
    return processor, graphs


def audio_duration_seconds(processor: Qwen3ASRProcessor, wav: np.ndarray) -> float:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sampling_rate = getattr(feature_extractor, "sampling_rate", 16000)
    return float(len(wav)) / float(sampling_rate)


def print_runtime_providers(graphs: OnnxGraphs):
    print(f"Token embed providers: {graphs.token_embed.get_providers()}")
    print(f"Audio encoder providers: {graphs.audio_encoder.get_providers()}")
    if graphs.decoder_layout == "merged":
        if graphs.decoder is None:
            raise RuntimeError("merged decoder session is not loaded")
        print(f"Decoder mode: merged If decoder")
        print(f"Decoder providers: {graphs.decoder.get_providers()}")
    elif graphs.decoder_layout == "split":
        if graphs.decoder_init is None or graphs.decoder_with_past is None:
            raise RuntimeError("split decoder sessions are not loaded")
        print("Decoder mode: split decoder_init + decoder_with_past")
        print(f"Decoder init providers: {graphs.decoder_init.get_providers()}")
        print(f"Decoder step providers: {graphs.decoder_with_past.get_providers()}")
    else:
        if graphs.decoder_with_past is None:
            raise RuntimeError("single decoder session is not loaded")
        print("Decoder mode: single decoder_with_past")
        print(f"Decoder providers: {graphs.decoder_with_past.get_providers()}")


def run_transcription(
    processor: Qwen3ASRProcessor,
    graphs: OnnxGraphs,
    wav: np.ndarray,
    context: str,
    language: str | None,
    max_new_tokens: int,
    stats: TimingStats | None = None,
):
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    prompt = build_prompt(processor, context=context, language=language)
    inputs = prepare_inputs(processor, prompt, wav)
    add_elapsed(stats, "prepare_inputs", stage_start)

    stage_start = time.perf_counter()
    audio_embeddings = run_audio_encoder(graphs, inputs["input_features"], inputs["feature_attention_mask"])
    add_elapsed(stats, "audio_encoder", stage_start)

    stage_start = time.perf_counter()
    text_embeddings = token_embed(graphs, inputs["input_ids"])
    add_elapsed(stats, "prefill_token_embed", stage_start)

    stage_start = time.perf_counter()
    merged_embeddings = merge_audio_embeddings(inputs["input_ids"], text_embeddings, audio_embeddings)
    add_elapsed(stats, "merge_embeddings", stage_start)

    generated_ids = generate(
        graphs,
        input_ids=inputs["input_ids"],
        inputs_embeds=merged_embeddings,
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        stats=stats,
    )
    stage_start = time.perf_counter()
    raw = processor.batch_decode(
        [generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = parse_asr_output(raw, user_language=language)
    add_elapsed(stats, "batch_decode_parse", stage_start)

    if stats is not None:
        stats.runs += 1
        stats.total += time.perf_counter() - total_start
    return parsed, raw


def generate(
    graphs: OnnxGraphs,
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    attention_mask: np.ndarray,
    max_new_tokens: int,
    stats: TimingStats | None = None,
) -> list[int]:
    stage_start = time.perf_counter()
    position_ids = position_ids_from_attention(attention_mask)
    add_elapsed(stats, "position_ids", stage_start)

    feeds = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
    if graphs.decoder_layout == "merged":
        # 首轮 prefill：use_past=False 让 merged decoder 走原 decoder_init 快路径。
        # 这避免了 single-decoder 空 cache prefill 的慢路径，同时仍只加载一份 decoder 权重。
        feeds["use_past"] = np.array(False, dtype=np.bool_)
        add_empty_past_feeds(graphs, feeds)
        prefill_session = graphs.decoder
    elif graphs.decoder_layout == "single":
        add_empty_past_feeds(graphs, feeds)
        prefill_session = graphs.decoder_with_past
    else:
        prefill_session = graphs.decoder_init
    if prefill_session is None:
        raise RuntimeError("prefill decoder session is not loaded")

    stage_start = time.perf_counter()
    init_outputs = prefill_session.run(None, feeds)
    add_elapsed(stats, "decoder_prefill", stage_start)

    stage_start = time.perf_counter()
    current = output_map(prefill_session, init_outputs)
    next_id = greedy_next_token(current["logits"])
    generated = []
    add_elapsed(stats, "decode_misc", stage_start)

    for _ in range(max_new_tokens):
        if next_id in EOS_TOKEN_IDS:
            break
        generated.append(next_id)
        if stats is not None:
            stats.decode_tokens += 1

        next_input_ids = np.array([[next_id]], dtype=np.int64)
        stage_start = time.perf_counter()
        next_embeds = token_embed(graphs, next_input_ids)
        add_elapsed(stats, "decode_token_embed", stage_start)

        stage_start = time.perf_counter()
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        step_pos = np.full((3, 1, 1), attention_mask.shape[1] - 1, dtype=np.int64)

        # 后续 token：use_past=True 走原 decoder_with_past 快路径，并把上一轮
        # present_key/value 改名成下一轮的 past_key/value。
        feeds = {
            "inputs_embeds": next_embeds,
            "position_ids": step_pos,
            "attention_mask": attention_mask,
        }
        if graphs.decoder_layout == "merged":
            feeds["use_past"] = np.array(True, dtype=np.bool_)
            step_session = graphs.decoder
        else:
            step_session = graphs.decoder_with_past
        if step_session is None:
            raise RuntimeError("step decoder session is not loaded")

        # ONNX 里没有 Transformers 的 DynamicCache 对象，所以每层输出的
        # present_key/value 都要在宿主程序里改名后传回下一轮的 past_key/value。
        for name, value in current.items():
            if name.startswith("present_key_"):
                feeds[name.replace("present_", "past_")] = value
            elif name.startswith("present_value_"):
                feeds[name.replace("present_", "past_")] = value
        add_elapsed(stats, "decode_misc", stage_start)

        stage_start = time.perf_counter()
        step_outputs = step_session.run(None, feeds)
        add_elapsed(stats, "decode_decoder", stage_start)

        stage_start = time.perf_counter()
        current = output_map(step_session, step_outputs)
        next_id = greedy_next_token(current["logits"])
        add_elapsed(stats, "decode_misc", stage_start)

    return generated


def transcribe_onnx(
    model_dir: str,
    onnx_dir: str,
    audio,
    context: str = "",
    language: str | None = None,
    max_new_tokens: int = 256,
    optimized: bool = False,
    provider: str = "auto",
    decoder_layout: str = "single",
    openvino_device: str = "GPU",
    trt_max_seq_len: int = 1024,
):
    if language:
        language = normalize_language_name(language)
        validate_language(language)

    # model_dir 只用于读取 tokenizer/processor/config 等小文件。
    # 这个 ONNX Runtime demo 不会读取 PyTorch 的 model.safetensors。
    processor, graphs = load_runtime(
        model_dir,
        onnx_dir,
        optimized=optimized,
        provider=provider,
        decoder_layout=decoder_layout,
        openvino_device=openvino_device,
        trt_max_seq_len=trt_max_seq_len,
    )
    print(f"ORT providers: {ort.get_available_providers()}")
    print_runtime_providers(graphs)
    wav = normalize_audios(audio)[0]

    (parsed_language, text), raw = run_transcription(
        processor,
        graphs,
        wav,
        context=context,
        language=language,
        max_new_tokens=max_new_tokens,
    )
    return parsed_language, text, raw


def benchmark_onnx(
    model_dir: str,
    onnx_dir: str,
    audio,
    context: str = "",
    language: str | None = None,
    max_new_tokens: int = 256,
    optimized: bool = False,
    provider: str = "auto",
    decoder_layout: str = "single",
    openvino_device: str = "GPU",
    trt_max_seq_len: int = 1024,
    benchmark_seconds: float = 15.0,
    warmup_runs: int = 1,
):
    if language:
        language = normalize_language_name(language)
        validate_language(language)

    processor, graphs = load_runtime(
        model_dir,
        onnx_dir,
        optimized=optimized,
        provider=provider,
        decoder_layout=decoder_layout,
        openvino_device=openvino_device,
        trt_max_seq_len=trt_max_seq_len,
    )
    print(f"ORT providers: {ort.get_available_providers()}")
    print_runtime_providers(graphs)

    wav = normalize_audios(audio)[0]
    audio_seconds = audio_duration_seconds(processor, wav)
    if audio_seconds <= 0:
        raise ValueError("audio duration must be greater than zero")

    warmup_runs = max(0, warmup_runs)
    benchmark_seconds = max(0.0, benchmark_seconds)

    print(f"Audio duration: {audio_seconds:.3f} s")
    print(f"Warmup runs: {warmup_runs}")
    for _ in range(warmup_runs):
        run_transcription(
            processor,
            graphs,
            wav,
            context=context,
            language=language,
            max_new_tokens=max_new_tokens,
        )

    # 从这里开始才计入吞吐：模型加载、音频读取/normalize、warmup 都已完成。
    runs = 0
    processed_audio_seconds = 0.0
    timing_stats = TimingStats()
    start = time.perf_counter()
    elapsed = 0.0
    while elapsed < benchmark_seconds or runs == 0:
        run_transcription(
            processor,
            graphs,
            wav,
            context=context,
            language=language,
            max_new_tokens=max_new_tokens,
            stats=timing_stats,
        )
        runs += 1
        processed_audio_seconds += audio_seconds
        elapsed = time.perf_counter() - start

    audio_seconds_per_second = processed_audio_seconds / elapsed
    print(f"Benchmark wall time: {elapsed:.3f} s")
    print(f"Benchmark runs: {runs}")
    print(f"Processed audio: {processed_audio_seconds:.3f} s")
    print(f"Audio seconds / second: {audio_seconds_per_second:.3f}x")
    print_timing_breakdown(timing_stats)


def print_timing_breakdown(stats: TimingStats):
    if stats.runs <= 0:
        return

    rows = [
        ("prepare_inputs", stats.prepare_inputs),
        ("audio_encoder", stats.audio_encoder),
        ("prefill_token_embed", stats.prefill_token_embed),
        ("merge_embeddings", stats.merge_embeddings),
        ("position_ids", stats.position_ids),
        ("decoder_prefill", stats.decoder_prefill),
        ("decode_token_embed", stats.decode_token_embed),
        ("decode_decoder", stats.decode_decoder),
        ("decode_misc", stats.decode_misc),
        ("batch_decode_parse", stats.batch_decode_parse),
    ]
    print("Timing breakdown after warmup:")
    print(f"  total measured pipeline: {stats.total:.3f} s")
    for name, seconds in rows:
        percent = seconds / stats.total * 100.0 if stats.total > 0 else 0.0
        per_run = seconds / stats.runs
        print(f"  {name}: {seconds:.3f} s total, {per_run:.3f} s/run, {percent:.1f}%")
    if stats.decode_tokens:
        print(f"  decode tokens: {stats.decode_tokens} total, {stats.decode_tokens / stats.runs:.1f}/run")
        print(f"  decode decoder: {stats.decode_decoder / stats.decode_tokens * 1000.0:.2f} ms/token")


def main():
    args = parse_args()
    if args.benchmark:
        benchmark_onnx(
            model_dir=args.model_dir,
            onnx_dir=args.onnx_dir,
            audio=args.audio,
            context=args.context,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            optimized=args.optimized,
            provider=args.provider,
            decoder_layout=args.decoder_layout,
            openvino_device=args.openvino_device,
            trt_max_seq_len=args.trt_max_seq_len,
            benchmark_seconds=args.benchmark_seconds,
            warmup_runs=args.warmup_runs,
        )
        return

    language, text, raw = transcribe_onnx(
        model_dir=args.model_dir,
        onnx_dir=args.onnx_dir,
        audio=args.audio,
        context=args.context,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        optimized=args.optimized,
        provider=args.provider,
        decoder_layout=args.decoder_layout,
        openvino_device=args.openvino_device,
        trt_max_seq_len=args.trt_max_seq_len,
    )
    print(f"Language: {language}")
    print(f"Text: {text}")
    print(f"Raw: {raw}")


if __name__ == "__main__":
    main()
