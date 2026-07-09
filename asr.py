import argparse
import time

import torch

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audios


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ASR PyTorch/backend 运行示例。")
    parser.add_argument("--model-dir", default="./Qwen3-ASR-0.6B")
    parser.add_argument(
        "--audio",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    )
    parser.add_argument("--language", default=None, help='可选的强制语言，例如 "English"。')
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--benchmark", action="store_true", help="反复处理同一个音频，输出 warmup 后的吞吐。")
    parser.add_argument("--benchmark-seconds", type=float, default=15.0, help="benchmark 至少运行多少秒。")
    parser.add_argument("--warmup-runs", type=int, default=1, help="正式计时前完整跑几次预热。")
    return parser.parse_args()


def dtype_from_name(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_model(args):
    print("Loading model...")
    started = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        args.model_dir,
        dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        # attn_implementation="flash_attention_2",
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Model loaded ({time.perf_counter() - started:.1f}s)")
    return model


def normalize_audio_once(audio):
    # normalize_audios 会把输入统一成 mono 16k float32；benchmark 不把这一步计入吞吐。
    wav = normalize_audios(audio)[0]
    return wav, float(len(wav)) / float(SAMPLE_RATE)


def transcribe_once(model: Qwen3ASRModel, wav, language: str | None):
    return model.transcribe(audio=(wav, SAMPLE_RATE), language=language)


def run_once(args):
    model = load_model(args)
    wav, _ = normalize_audio_once(args.audio)

    print("Transcribing...")
    started = time.perf_counter()
    results = transcribe_once(model, wav, args.language)
    print(f"Done ({time.perf_counter() - started:.1f}s)")

    print(f"Language: {results[0].language}")
    print(f"Text: {results[0].text}")


def run_benchmark(args):
    model = load_model(args)
    wav, audio_seconds = normalize_audio_once(args.audio)
    if audio_seconds <= 0:
        raise ValueError("audio duration must be greater than zero")

    warmup_runs = max(0, args.warmup_runs)
    benchmark_seconds = max(0.0, args.benchmark_seconds)

    print(f"Audio duration: {audio_seconds:.3f} s")
    print(f"Warmup runs: {warmup_runs}")
    for _ in range(warmup_runs):
        transcribe_once(model, wav, args.language)

    # 从这里开始才计入吞吐：模型加载、音频读取/normalize、warmup 都已完成。
    runs = 0
    processed_audio_seconds = 0.0
    started = time.perf_counter()
    elapsed = 0.0
    while elapsed < benchmark_seconds or runs == 0:
        transcribe_once(model, wav, args.language)
        runs += 1
        processed_audio_seconds += audio_seconds
        elapsed = time.perf_counter() - started

    audio_seconds_per_second = processed_audio_seconds / elapsed
    print(f"Benchmark wall time: {elapsed:.3f} s")
    print(f"Benchmark runs: {runs}")
    print(f"Processed audio: {processed_audio_seconds:.3f} s")
    print(f"Audio seconds / second: {audio_seconds_per_second:.3f}x")


def main():
    args = parse_args()
    if args.benchmark:
        run_benchmark(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
