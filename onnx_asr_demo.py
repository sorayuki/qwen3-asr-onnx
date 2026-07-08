import argparse
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
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "directml", "cuda"],
        default="auto",
        help="ONNX Runtime 执行后端偏好。",
    )
    parser.add_argument(
        "--single-decoder",
        action="store_true",
        help="首轮也用 decoder_with_past 加空 past，不加载 decoder_init。",
    )
    return parser.parse_args()


@dataclass
class OnnxGraphs:
    token_embed: ort.InferenceSession
    audio_encoder: ort.InferenceSession
    decoder_init: ort.InferenceSession | None
    decoder_with_past: ort.InferenceSession


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


def load_graphs(onnx_dir: str, optimized: bool, provider: str, single_decoder: bool) -> OnnxGraphs:
    root = Path(onnx_dir)
    # single-decoder 模式下，首轮 prefill 也调用 decoder_with_past，
    # 只需要传入长度为 0 的 past_key/value。这样可以少加载一整份 decoder 权重，
    # 代价是部分 provider 上首轮会更慢。
    return OnnxGraphs(
        token_embed=make_session(onnx_name(root, "token_embed", optimized), provider),
        audio_encoder=make_session(onnx_name(root, "audio_encoder", optimized), provider),
        decoder_init=None
        if single_decoder
        else make_session(onnx_name(root, "decoder_init", optimized), provider),
        decoder_with_past=make_session(onnx_name(root, "decoder_with_past", optimized), provider),
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


def greedy_next_token(logits: np.ndarray) -> int:
    return int(np.argmax(logits[0, -1, :]))


def generate(
    graphs: OnnxGraphs,
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    attention_mask: np.ndarray,
    max_new_tokens: int,
) -> list[int]:
    position_ids = position_ids_from_attention(attention_mask)
    if graphs.decoder_init is None:
        # 用空 cache 让 decoder_with_past 执行首轮 prefill。
        # 这个导出的图已经验证可以接受 [batch, kv_heads, 0, head_dim]。
        feeds = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
        for i in range(28):
            feeds[f"past_key_{i}"] = np.empty((1, 8, 0, 128), dtype=np.float16)
            feeds[f"past_value_{i}"] = np.empty((1, 8, 0, 128), dtype=np.float16)
        init_outputs = graphs.decoder_with_past.run(None, feeds)
        current = output_map(graphs.decoder_with_past, init_outputs)
    else:
        init_outputs = graphs.decoder_init.run(
            None,
            {
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            },
        )
        current = output_map(graphs.decoder_init, init_outputs)
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

        feeds = {
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

        step_outputs = graphs.decoder_with_past.run(None, feeds)
        current = output_map(graphs.decoder_with_past, step_outputs)
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
    single_decoder: bool = False,
):
    if language:
        language = normalize_language_name(language)
        validate_language(language)

    # model_dir 只用于读取 tokenizer/processor/config 等小文件。
    # 这个 ONNX Runtime demo 不会读取 PyTorch 的 model.safetensors。
    processor = Qwen3ASRProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    graphs = load_graphs(onnx_dir, optimized=optimized, provider=provider, single_decoder=single_decoder)
    print(f"ORT providers: {ort.get_available_providers()}")
    print(f"Using providers: {graphs.decoder_with_past.get_providers()}")
    print(f"Decoder mode: {'single decoder_with_past' if single_decoder else 'decoder_init + decoder_with_past'}")
    wav = normalize_audios(audio)[0]

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
    parsed_language, text = parse_asr_output(raw, user_language=language)
    return parsed_language, text, raw


def main():
    args = parse_args()
    language, text, raw = transcribe_onnx(
        model_dir=args.model_dir,
        onnx_dir=args.onnx_dir,
        audio=args.audio,
        context=args.context,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        optimized=args.optimized,
        provider=args.provider,
        single_decoder=args.single_decoder,
    )
    print(f"Language: {language}")
    print(f"Text: {text}")
    print(f"Raw: {raw}")


if __name__ == "__main__":
    main()
