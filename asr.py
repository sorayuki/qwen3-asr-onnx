import torch
import time
from qwen_asr import Qwen3ASRModel

print("Loading model...")
t0 = time.time()
model = Qwen3ASRModel.from_pretrained(
    "./Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)
print(f"Model loaded ({time.time() - t0:.1f}s)")

print("Transcribing...")
t0 = time.time()
results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    language=None, # set "English" to force the language
)
print(f"Done ({time.time() - t0:.1f}s)")

print(f"Language: {results[0].language}")
print(f"Text: {results[0].text}")