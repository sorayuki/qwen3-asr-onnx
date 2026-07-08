import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

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
    parser.add_argument("--benchmark", action="store_true", help="反复处理同一个音频，输出 warmup 后的吞吐。")
    parser.add_argument("--benchmark-seconds", type=float, default=15.0, help="benchmark 至少运行多少秒。")
    parser.add_argument("--warmup-runs", type=int, default=1, help="正式计时前完整跑几次预热。")
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "directml", "cuda"],
        default="auto",
        help="ONNX Runtime 执行后端偏好。",
    )
    return parser.parse_args()


@dataclass
class OnnxGraphs:
    token_embed: ort.InferenceSession
    audio_encoder: ort.InferenceSession
    # decoder_merged.onnx 内部用 If(use_past) 在 prefill 和 decode 两条快路径之间切换。
    # 运行时只创建这一个 decoder session，避免重复加载 decoder 权重。
    decoder: ort.InferenceSession
    num_layers: int
    num_kv_heads: int
    head_dim: int


def provider_list(provider: str):
    # 优先使用当前环境已注册的 CUDA，其次是 Windows 上的 DirectML，最后回退到 CPU。
    # 这里不会安装 provider，只从 ONNX Runtime 实际能看到的 provider 里选择。
    available = ort.get_available_providers()
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


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # prompt 和 KV cache 都会使用动态长度。关闭 memory pattern 可以避免部分
    # execution provider 复用旧 shape 假设导致的问题。
    options.enable_mem_pattern = False
    return ort.InferenceSession(str(path), sess_options=options, providers=provider_list(provider))


def onnx_name(onnx_dir: Path, stem: str, optimized: bool) -> Path:
    suffix = ".optimized.onnx" if optimized else ".onnx"
    return onnx_dir / f"{stem}{suffix}"


def decoder_cache_config(model_dir: str) -> tuple[int, int, int]:
    # merged decoder 的公开输入包含 past_key/value。首轮 prefill 虽然不使用 past，
    # 但 ORT 仍要求 feed 所有公开输入，所以这里从轻量 config 读取空 KV 的 shape。
    with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as file:
        config = json.load(file)
    text_config = config["thinker_config"]["text_config"]
    return (
        int(text_config["num_hidden_layers"]),
        int(text_config["num_key_value_heads"]),
        int(text_config["head_dim"]),
    )


def load_graphs(model_dir: str, onnx_dir: str, optimized: bool, provider: str) -> OnnxGraphs:
    root = Path(onnx_dir)
    num_layers, num_kv_heads, head_dim = decoder_cache_config(model_dir)
    return OnnxGraphs(
        token_embed=make_session(onnx_name(root, "token_embed", optimized), provider),
        audio_encoder=make_session(onnx_name(root, "audio_encoder", optimized), provider),
        decoder=make_session(onnx_name(root, "decoder_merged", optimized), provider),
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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


def load_runtime(model_dir: str, onnx_dir: str, optimized: bool, provider: str):
    # 模型加载和 session 创建不计入 benchmark；吞吐只统计真正开始送入音频后的耗时。
    processor = Qwen3ASRProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    graphs = load_graphs(model_dir, onnx_dir, optimized=optimized, provider=provider)
    return processor, graphs


def audio_duration_seconds(processor: Qwen3ASRProcessor, wav: np.ndarray) -> float:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sampling_rate = getattr(feature_extractor, "sampling_rate", 16000)
    return float(len(wav)) / float(sampling_rate)


def run_transcription(
    processor: Qwen3ASRProcessor,
    graphs: OnnxGraphs,
    wav: np.ndarray,
    context: str,
    language: str | None,
    max_new_tokens: int,
):
    prompt = build_prompt(processor, context=context, language=language)
    inputs = prepare_inputs(processor, prompt, wav)
    audio_embeddings = run_audio_encoder(graphs, inputs["input_features"], inputs["feature_attention_mask"])
    text_embeddings = token_embed(graphs, inputs["input_ids"])
    merged_embeddings = merge_audio_embeddings(inputs["input_ids"], text_embeddings, audio_embeddings)

    generated_ids = generate(
        graphs,
        input_ids=inputs["input_ids"],
        inputs_embeds=merged_embeddings,
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
    )
    raw = processor.batch_decode(
        [generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_asr_output(raw, user_language=language), raw


def generate(
    graphs: OnnxGraphs,
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    attention_mask: np.ndarray,
    max_new_tokens: int,
) -> list[int]:
    position_ids = position_ids_from_attention(attention_mask)
    # 首轮 prefill：use_past=False 让 merged decoder 走原 decoder_init 快路径。
    # 这避免了 single-decoder 空 cache prefill 的慢路径，同时仍只加载一份 decoder 权重。
    feeds = {
        "use_past": np.array(False, dtype=np.bool_),
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
    add_empty_past_feeds(graphs, feeds)
    init_outputs = graphs.decoder.run(None, feeds)
    current = output_map(graphs.decoder, init_outputs)
    next_id = greedy_next_token(current["logits"])
    generated = []

    for _ in range(max_new_tokens):
        if next_id in EOS_TOKEN_IDS:
            break
        generated.append(next_id)

        next_input_ids = np.array([[next_id]], dtype=np.int64)
        next_embeds = token_embed(graphs, next_input_ids)
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        step_pos = np.full((3, 1, 1), attention_mask.shape[1] - 1, dtype=np.int64)

        # 后续 token：use_past=True 走原 decoder_with_past 快路径，并把上一轮
        # present_key/value 改名成下一轮的 past_key/value。
        feeds = {
            "use_past": np.array(True, dtype=np.bool_),
            "inputs_embeds": next_embeds,
            "position_ids": step_pos,
            "attention_mask": attention_mask,
        }
        # ONNX 里没有 Transformers 的 DynamicCache 对象，所以每层输出的
        # present_key/value 都要在宿主程序里改名后传回下一轮的 past_key/value。
        for name, value in current.items():
            if name.startswith("present_key_"):
                feeds[name.replace("present_", "past_")] = value
            elif name.startswith("present_value_"):
                feeds[name.replace("present_", "past_")] = value

        step_outputs = graphs.decoder.run(None, feeds)
        current = output_map(graphs.decoder, step_outputs)
        next_id = greedy_next_token(current["logits"])

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
):
    if language:
        language = normalize_language_name(language)
        validate_language(language)

    # model_dir 只用于读取 tokenizer/processor/config 等小文件。
    # 这个 ONNX Runtime demo 不会读取 PyTorch 的 model.safetensors。
    processor, graphs = load_runtime(model_dir, onnx_dir, optimized=optimized, provider=provider)
    print(f"ORT providers: {ort.get_available_providers()}")
    print(f"Using providers: {graphs.decoder.get_providers()}")
    print("Decoder mode: merged If decoder")
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
    benchmark_seconds: float = 15.0,
    warmup_runs: int = 1,
):
    if language:
        language = normalize_language_name(language)
        validate_language(language)

    processor, graphs = load_runtime(model_dir, onnx_dir, optimized=optimized, provider=provider)
    print(f"ORT providers: {ort.get_available_providers()}")
    print(f"Using providers: {graphs.decoder.get_providers()}")
    print("Decoder mode: merged If decoder")

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
        )
        runs += 1
        processed_audio_seconds += audio_seconds
        elapsed = time.perf_counter() - start

    audio_seconds_per_second = processed_audio_seconds / elapsed
    print(f"Benchmark wall time: {elapsed:.3f} s")
    print(f"Benchmark runs: {runs}")
    print(f"Processed audio: {processed_audio_seconds:.3f} s")
    print(f"Audio seconds / second: {audio_seconds_per_second:.3f}x")


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
    )
    print(f"Language: {language}")
    print(f"Text: {text}")
    print(f"Raw: {raw}")


if __name__ == "__main__":
    main()
