import argparse
import hashlib
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper
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


def tensor_digest(tensor: TensorProto) -> str:
    # 用完整 tensor 内容做去重签名，但忽略名字和 external_data 位置。
    # 两个导出的 decoder 可能把同一份权重命名成不同的 onnx::MatMul_*，
    # 或写到不同外部文件；只要数值内容一致，就应该共享同一份 initializer。
    comparable = TensorProto()
    comparable.CopyFrom(tensor)
    comparable.name = ""
    comparable.doc_string = ""
    del comparable.external_data[:]
    h = hashlib.sha256()
    h.update(comparable.SerializeToString())
    return h.hexdigest()


def external_data_entries(tensor: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def tensors_share_storage(left: TensorProto, right: TensorProto) -> bool:
    # 同名 initializer 只有在确实指向同一份外部数据或内嵌内容一致时才能复用；
    # 否则给后来的 initializer 改名，避免误把不同常量合并。
    if left.data_type != right.data_type or tuple(left.dims) != tuple(right.dims):
        return False

    left_ext = external_data_entries(left)
    right_ext = external_data_entries(right)
    if left_ext or right_ext:
        keys = ("location", "offset", "length")
        return all(left_ext.get(key, "0" if key == "offset" else None) == right_ext.get(key, "0" if key == "offset" else None) for key in keys)

    return tensor_digest(left) == tensor_digest(right)


def graph_inputs_without_initializers(model: onnx.ModelProto) -> list:
    # torch.onnx 可能把 initializer 也列进 graph input。merged graph 的公开输入
    # 只保留真正需要 runtime feed 的值，权重统一放到外层 initializer。
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return [value for value in model.graph.input if value.name not in initializer_names]


def replace_value_names(graph: onnx.GraphProto, replacements: dict[str, str]):
    # 分支图里所有引用旧 initializer 名称的位置都要一起替换，
    # 否则节点会继续寻找已经被去重丢弃的权重名。
    def replace_name(name: str) -> str:
        return replacements.get(name, name)

    for value in graph.input:
        value.name = replace_name(value.name)
    for value in graph.output:
        value.name = replace_name(value.name)
    for value in graph.value_info:
        value.name = replace_name(value.name)
    for initializer in graph.initializer:
        initializer.name = replace_name(initializer.name)
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name:
                node.input[idx] = replace_name(name)
        for idx, name in enumerate(node.output):
            if name:
                node.output[idx] = replace_name(name)


def merge_decoder_if(init_path: Path, with_past_path: Path, merged_path: Path):
    # 构造一个单 session decoder：
    #   use_past=False 走 decoder_init 首轮 prefill 快路径；
    #   use_past=True 走 decoder_with_past 增量解码快路径。
    #
    # ONNX If 子图可以引用外层作用域里的 initializer，所以两个分支不需要
    # 各自携带一份 decoder 权重。这里把权重提升到外层 graph，并按内容去重。
    init_model = onnx.load(str(init_path), load_external_data=True)
    with_past_model = onnx.load(str(with_past_path), load_external_data=True)

    output_names = [output.name for output in init_model.graph.output]
    with_past_outputs = [output.name for output in with_past_model.graph.output]
    if output_names != with_past_outputs:
        raise ValueError("decoder init and decoder-with-past outputs must have identical names")

    shared_initializers = []
    initializer_by_name = {}
    initializer_by_digest = {}
    graph_replacements = []

    for model in (init_model, with_past_model):
        replacements = {}
        for initializer in model.graph.initializer:
            # 优先按内容去重，解决 torch.onnx 为相同权重生成不同名字的问题。
            digest = tensor_digest(initializer)
            existing_by_digest = initializer_by_digest.get(digest)
            if existing_by_digest is not None:
                if initializer.name != existing_by_digest.name:
                    replacements[initializer.name] = existing_by_digest.name
                continue

            existing_by_name = initializer_by_name.get(initializer.name)
            if existing_by_name is None:
                initializer_by_name[initializer.name] = initializer
                initializer_by_digest[digest] = initializer
                shared_initializers.append(initializer)
                continue
            if not tensors_share_storage(existing_by_name, initializer):
                # 名字相同但内容不同，保守地重命名后都保留。
                new_name = f"{initializer.name}_{tensor_digest(initializer)[:12]}"
                replacements[initializer.name] = new_name
                copied = TensorProto()
                copied.CopyFrom(initializer)
                copied.name = new_name
                initializer_by_name[new_name] = copied
                initializer_by_digest[digest] = copied
                shared_initializers.append(copied)
        graph_replacements.append(replacements)

    init_graph = onnx.GraphProto()
    init_graph.CopyFrom(init_model.graph)
    with_past_graph = onnx.GraphProto()
    with_past_graph.CopyFrom(with_past_model.graph)

    for graph, replacements in ((init_graph, graph_replacements[0]), (with_past_graph, graph_replacements[1])):
        replace_value_names(graph, replacements)
        # 分支 graph 不再声明 inputs/initializers：真实输入和共享权重都来自外层 graph。
        # 这样 ORT 只创建一个 decoder session，权重也只保存/加载一份。
        del graph.input[:]
        del graph.initializer[:]
        del graph.sparse_initializer[:]

    # merged graph 的公开输入是两个分支所需输入的并集：
    # init 分支需要 inputs_embeds/position_ids/attention_mask；
    # with-past 分支还需要 past_key/value。
    outer_inputs = [
        helper.make_tensor_value_info("use_past", TensorProto.BOOL, []),
        *graph_inputs_without_initializers(init_model),
    ]
    init_input_names = {value.name for value in outer_inputs}
    for value in graph_inputs_without_initializers(with_past_model):
        if value.name not in init_input_names:
            outer_inputs.append(value)
            init_input_names.add(value.name)

    if_node = helper.make_node(
        "If",
        inputs=["use_past"],
        outputs=output_names,
        name="decoder_merged_if",
        then_branch=with_past_graph,
        else_branch=init_graph,
    )

    # If 的输出名保持为原 decoder 输出名，运行脚本不需要区分分支输出。
    outer_graph = helper.make_graph(
        [if_node],
        "decoder_merged",
        outer_inputs,
        init_model.graph.output,
        initializer=shared_initializers,
    )
    merged = helper.make_model(
        outer_graph,
        opset_imports=init_model.opset_import,
        producer_name="qwen3_asr_merged_decoder_export",
        ir_version=max(init_model.ir_version, with_past_model.ir_version),
    )

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    # onnx.save_model 不会可靠截断旧的 external data 文件，先删掉可避免
    # 之前生成过的大文件尾部残留，导致磁盘大小看起来还是两份权重。
    for candidate in (merged_path, merged_path.with_name(merged_path.name + ".data")):
        if candidate.exists():
            candidate.unlink()
    onnx.save_model(
        merged,
        str(merged_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=merged_path.name + ".data",
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.checker.check_model(str(merged_path))


def remove_onnx_file_family(path: Path):
    # 清理导出中间图及其 external data。最终分发只保留 merged decoder。
    for candidate in (path, path.with_name(path.name + ".data")):
        if candidate.exists():
            candidate.unlink()


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
    # 再把合并后的 inputs_embeds 传给 merged decoder。
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

    print("[export] decoder_init.tmp.onnx")
    inputs_embeds = torch.randn((1, args.prefill_len, hidden_size), dtype=dtype, device=device)
    position_ids = torch.arange(args.prefill_len, dtype=torch.long, device=device).view(1, 1, -1).expand(3, 1, -1)
    attention_mask = torch.ones((1, args.prefill_len), dtype=torch.long, device=device)
    kv_names = [name for i in range(num_layers) for name in (f"present_key_{i}", f"present_value_{i}")]
    decoder_init_path = out_dir / "decoder_init.tmp.onnx"
    decoder_with_past_path = out_dir / "decoder_with_past.tmp.onnx"
    decoder_merged_path = out_dir / "decoder_merged.onnx"
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
        decoder_init_path,
        ["inputs_embeds", "position_ids", "attention_mask"],
        ["logits", *kv_names],
        args.opset,
        dynamic,
    )

    print("[export] decoder_with_past.tmp.onnx")
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
        decoder_with_past_path,
        ["inputs_embeds", "position_ids", "attention_mask", *past_names],
        ["logits", *kv_names],
        args.opset,
        kv_dynamic,
    )

    print("[merge] decoder_merged.onnx")
    merge_decoder_if(decoder_init_path, decoder_with_past_path, decoder_merged_path)
    exported.append(decoder_merged_path)

    if not args.skip_optimizer:
        for path in exported[:2]:
            print(f"[optimize] {path.name}")
            try_optimize(
                path,
                num_heads=thinker.model.config.num_attention_heads,
                hidden_size=hidden_size,
                use_gpu=device.type == "cuda",
            )

        print(f"[optimize] {decoder_init_path.name}")
        optimized_init_path = try_optimize(
            decoder_init_path,
            num_heads=thinker.model.config.num_attention_heads,
            hidden_size=hidden_size,
            use_gpu=device.type == "cuda",
        )
        print(f"[optimize] {decoder_with_past_path.name}")
        optimized_with_past_path = try_optimize(
            decoder_with_past_path,
            num_heads=thinker.model.config.num_attention_heads,
            hidden_size=hidden_size,
            use_gpu=device.type == "cuda",
        )
        if optimized_init_path is not None and optimized_with_past_path is not None:
            print("[merge] decoder_merged.optimized.onnx")
            merge_decoder_if(
                optimized_init_path,
                optimized_with_past_path,
                decoder_merged_path.with_name("decoder_merged.optimized.onnx"),
            )
            remove_onnx_file_family(optimized_init_path)
            remove_onnx_file_family(optimized_with_past_path)

    remove_onnx_file_family(decoder_init_path)
    remove_onnx_file_family(decoder_with_past_path)

    print("[done]")
    for path in sorted(path for path in out_dir.glob("*.onnx") if ".tmp" not in path.name):
        print(f"{path} {path.stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
