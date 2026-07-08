import argparse
from pathlib import Path

import onnx
import torch
from torch import nn

from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import apply_rotary_pos_emb, repeat_kv


# Optimum 的通用导出器不认识 qwen3_asr 这个自定义架构。
# 这个脚本直接从 PyTorch 模块导出部署用子图，并把 ONNX 不友好的部分
# 例如 DynamicCache 和 Transformers 的 causal-mask helper，替换成普通 tensor 输入/输出。
def parse_args():
    parser = argparse.ArgumentParser(description="把 Qwen3-ASR 0.6B 的子图导出为 ONNX。")
    parser.add_argument("--model-dir", default="./Qwen3-ASR-0.6B")
    parser.add_argument("--out-dir", default="./onnx_models")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--past-len", type=int, default=8)
    parser.add_argument("--skip-optimizer", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str):
    return torch.float16 if name == "fp16" else torch.float32


def save_external_onnx(model: nn.Module, args, path: Path, input_names, output_names, opset, dynamic_axes=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(path))


def try_optimize(path: Path, num_heads: int, hidden_size: int, use_gpu: bool):
    try:
        from onnxruntime.transformers.optimizer import optimize_model

        # ORT 优化器可能会在 .onnx 旁边生成外部数据文件。
        # 拷贝优化版模型时需要把 .onnx 和 .onnx.data 放在一起。
        optimized_path = path.with_name(path.stem + ".optimized.onnx")
        opt_model = optimize_model(
            str(path),
            model_type="gpt2",
            num_heads=num_heads,
            hidden_size=hidden_size,
            opt_level=1,
            use_gpu=use_gpu,
        )
        opt_model.save_model_to_file(str(optimized_path), use_external_data_format=True)
        onnx.checker.check_model(str(optimized_path))
        return optimized_path
    except Exception as exc:
        print(f"[warn] optimizer skipped for {path.name}: {exc}")
        return None


class TokenEmbeddingWrapper(nn.Module):
    # 单独导出成一个图，方便 demo 先合并文本/音频 embedding，
    # 再把合并后的 inputs_embeds 传给 decoder_init 或 decoder_with_past。
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class AudioEncoderWrapper(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.audio_tower = audio_tower
        # qwen_asr 本地 modeling 文件里已经有一个专门的 ONNX 导出分支，
        # 用来绕开 Python list 操作和不规则 chunk。这个分支只接收一个 100-frame mel chunk。
        self.audio_tower._onnx_export = True

    def forward(self, input_features):
        # fast path 当前只支持一个 100-frame chunk。
        feature_lens = torch.tensor([100], dtype=torch.long, device=input_features.device)
        return self.audio_tower(input_features, feature_lens=feature_lens).last_hidden_state


class Qwen3ASRDecoderBase(nn.Module):
    # decoder 导出共用实现。这里有意复刻 Qwen3-ASR 的层内计算，
    # 但 mask 和 KV cache 都只使用普通 tensor，避免导出 Python 对象。
    def __init__(self, thinker):
        super().__init__()
        self.text = thinker.model
        self.lm_head = thinker.lm_head
        self.layers = self.text.layers
        self.rotary_emb = self.text.rotary_emb
        self.norm = self.text.norm
        self.num_layers = len(self.layers)

    def _causal_mask(self, inputs_embeds, attention_mask, kv_length):
        # 避开 transformers.masking_utils.create_causal_mask：
        # 当前环境里那条路径会用 functorch/vmap，ONNX trace 会失败。
        batch, q_length = inputs_embeds.shape[:2]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        min_value = torch.finfo(dtype).min

        q_pos = torch.arange(q_length, device=device).view(q_length, 1)
        k_pos = torch.arange(kv_length, device=device).view(1, kv_length)
        past_length = kv_length - q_length
        allowed = k_pos <= (past_length + q_pos)
        mask = torch.zeros((q_length, kv_length), dtype=dtype, device=device)
        mask = mask.masked_fill(~allowed, min_value)
        mask = mask.view(1, 1, q_length, kv_length).expand(batch, 1, q_length, kv_length)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :kv_length] == 0
            mask = mask.masked_fill(key_mask, min_value)
        return mask

    def _attention(self, attn, hidden_states, position_embeddings, attention_mask, past_key=None, past_value=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key is not None:
            # ONNX 只能看到普通 tensor cache，看不到 Transformers 的 DynamicCache。
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        key_for_attn = repeat_kv(key_states, attn.num_key_value_groups)
        value_for_attn = repeat_kv(value_states, attn.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_for_attn.transpose(2, 3)) * attn.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_for_attn.shape[-2]]
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return attn.o_proj(attn_output), key_states, value_states

    def _layer(self, layer, hidden_states, position_embeddings, attention_mask, past_key=None, past_value=None):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, present_key, present_value = self._attention(
            layer.self_attn, hidden_states, position_embeddings, attention_mask, past_key, past_value
        )
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        return hidden_states, present_key, present_value


class DecoderInitWrapper(Qwen3ASRDecoderBase):
    # prefill 图：输入完整 prompt/audio embedding 序列，
    # 输出 logits，以及每个 decoder layer 的 present_key/value。
    def forward(self, inputs_embeds, position_ids, attention_mask):
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self._causal_mask(inputs_embeds, attention_mask, inputs_embeds.shape[1])

        presents = []
        for layer in self.layers:
            hidden_states, present_key, present_value = self._layer(
                layer, hidden_states, position_embeddings, causal_mask
            )
            presents.extend([present_key, present_value])

        logits = self.lm_head(self.norm(hidden_states))
        return (logits, *presents)


class DecoderWithPastWrapper(Qwen3ASRDecoderBase):
    # 单步解码图：输入新 token embedding 和 past_key/value。
    # 当 past 的序列长度为 0 时，也可以兼任首轮 prefill。
    def forward(self, inputs_embeds, position_ids, attention_mask, *past_key_values):
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = self._causal_mask(inputs_embeds, attention_mask, attention_mask.shape[1])

        presents = []
        for layer_idx, layer in enumerate(self.layers):
            past_key = past_key_values[layer_idx * 2]
            past_value = past_key_values[layer_idx * 2 + 1]
            hidden_states, present_key, present_value = self._layer(
                layer, hidden_states, position_embeddings, causal_mask, past_key, past_value
            )
            presents.extend([present_key, present_value])

        logits = self.lm_head(self.norm(hidden_states))
        return (logits, *presents)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device)

    print(f"[load] {args.model_dir} on {device} as {dtype}")
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_dir,
        dtype=dtype,
        attn_implementation="eager",
        device_map=None,
    ).eval().to(device)
    thinker = model.thinker
    num_layers = len(thinker.model.layers)
    num_kv_heads = thinker.model.config.num_key_value_heads
    head_dim = thinker.model.config.head_dim
    hidden_size = thinker.model.config.hidden_size

    exported = []

    print("[export] token_embed.onnx")
    token_ids = torch.ones((1, args.prefill_len), dtype=torch.long, device=device)
    save_external_onnx(
        TokenEmbeddingWrapper(thinker.model.embed_tokens).eval(),
        (token_ids,),
        out_dir / "token_embed.onnx",
        ["input_ids"],
        ["inputs_embeds"],
        args.opset,
        {"input_ids": {1: "seq"}, "inputs_embeds": {1: "seq"}},
    )
    exported.append(out_dir / "token_embed.onnx")

    print("[export] audio_encoder.onnx")
    input_features = torch.randn((128, 100), dtype=dtype, device=device)
    save_external_onnx(
        AudioEncoderWrapper(thinker.audio_tower).eval(),
        (input_features,),
        out_dir / "audio_encoder.onnx",
        ["input_features"],
        ["audio_embeddings"],
        args.opset,
    )
    exported.append(out_dir / "audio_encoder.onnx")

    print("[export] decoder_init.onnx")
    inputs_embeds = torch.randn((1, args.prefill_len, hidden_size), dtype=dtype, device=device)
    position_ids = torch.arange(args.prefill_len, dtype=torch.long, device=device).view(1, 1, -1).expand(3, 1, -1)
    attention_mask = torch.ones((1, args.prefill_len), dtype=torch.long, device=device)
    kv_names = [name for i in range(num_layers) for name in (f"present_key_{i}", f"present_value_{i}")]
    dynamic = {
        "inputs_embeds": {1: "seq"},
        "position_ids": {2: "seq"},
        "attention_mask": {1: "seq"},
        "logits": {1: "seq"},
    }
    for name in kv_names:
        dynamic[name] = {2: "seq"}
    save_external_onnx(
        DecoderInitWrapper(thinker).eval(),
        (inputs_embeds, position_ids, attention_mask),
        out_dir / "decoder_init.onnx",
        ["inputs_embeds", "position_ids", "attention_mask"],
        ["logits", *kv_names],
        args.opset,
        dynamic,
    )
    exported.append(out_dir / "decoder_init.onnx")

    print("[export] decoder_with_past.onnx")
    step_embeds = torch.randn((1, 1, hidden_size), dtype=dtype, device=device)
    step_pos = torch.full((3, 1, 1), args.past_len, dtype=torch.long, device=device)
    step_mask = torch.ones((1, args.past_len + 1), dtype=torch.long, device=device)
    past = []
    for _ in range(num_layers):
        past.append(torch.randn((1, num_kv_heads, args.past_len, head_dim), dtype=dtype, device=device))
        past.append(torch.randn((1, num_kv_heads, args.past_len, head_dim), dtype=dtype, device=device))
    past_names = [name for i in range(num_layers) for name in (f"past_key_{i}", f"past_value_{i}")]
    kv_dynamic = {
        "inputs_embeds": {1: "step"},
        "position_ids": {2: "step"},
        "attention_mask": {1: "total_seq"},
        "logits": {1: "step"},
    }
    for name in past_names:
        kv_dynamic[name] = {2: "past_seq"}
    for name in kv_names:
        kv_dynamic[name] = {2: "total_seq"}
    save_external_onnx(
        DecoderWithPastWrapper(thinker).eval(),
        (step_embeds, step_pos, step_mask, *past),
        out_dir / "decoder_with_past.onnx",
        ["inputs_embeds", "position_ids", "attention_mask", *past_names],
        ["logits", *kv_names],
        args.opset,
        kv_dynamic,
    )
    exported.append(out_dir / "decoder_with_past.onnx")

    if not args.skip_optimizer:
        for path in exported:
            print(f"[optimize] {path.name}")
            try_optimize(
                path,
                num_heads=thinker.model.config.num_attention_heads,
                hidden_size=hidden_size,
                use_gpu=device.type == "cuda",
            )

    print("[done]")
    for path in sorted(out_dir.glob("*.onnx")):
        print(f"{path} {path.stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
