import math
import time
from typing import Tuple

import torch
import transformers
from torch import Tensor


def predict(model: transformers.PreTrainedModel, input_ids: Tensor) -> Tensor:
    p = {
        k.replace("model.", "").replace("layers.", "").replace(".weight", ""): v
        for k, v in model.named_parameters()
    }
    c = model.config
    (dtype,) = {x.dtype for x in p.values()}

    def rms_norm(x: Tensor, w: Tensor) -> Tensor:
        return w * x / torch.sqrt((x**2).mean(-1, keepdim=True) + c.rms_norm_eps)

    def rotary_cos_sin(n: int) -> Tuple[Tensor, Tensor]:
        freq = c.rope_theta ** -(
            torch.arange(0, c.head_dim, 2, dtype=torch.float) / c.head_dim
        )
        s = c.rope_scaling
        z = (
            s["original_max_position_embeddings"] * freq / (2 * math.pi)
            - s["low_freq_factor"]
        ) / (s["high_freq_factor"] - s["low_freq_factor"])
        freq *= torch.lerp(
            torch.tensor(1 / s["factor"]), torch.tensor(1.0), z.clip(0, 1)
        )
        angle = torch.arange(n)[:, None] * freq
        return angle.cos().to(dtype), angle.sin().to(dtype)

    def rotate(z: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        zx, zy = z.unflatten(-1, (2, -1)).movedim(-2, 0)
        return torch.cat([zx * cos - zy * sin, zy * cos + zx * sin], -1)

    def self_attn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # b=batch, t=target, s=source, n=kv-heads, m=q-heads-per-kv, d=head-dim
        a = torch.einsum("btnmd, bsnd -> bnmts", q, k) / math.sqrt(q.shape[-1])
        a += torch.full(a.shape[-2:], -torch.inf, dtype=a.dtype).tril_(-1).T
        a = a.softmax(dim=-1)
        return torch.einsum("bnmts, bsnd -> btnmd", a, v)

    def attn(x: Tensor, layer: int, cos: Tensor, sin: Tensor) -> Tensor:
        z = rms_norm(x, p[f"{layer}.input_layernorm"])
        q = (z @ p[f"{layer}.self_attn.q_proj"].T).unflatten(
            -1, (c.num_key_value_heads, -1, c.head_dim)
        )
        k = (z @ p[f"{layer}.self_attn.k_proj"].T).unflatten(
            -1, (c.num_key_value_heads, c.head_dim)
        )
        v = (z @ p[f"{layer}.self_attn.v_proj"].T).unflatten(
            -1, (c.num_key_value_heads, c.head_dim)
        )
        q = rotate(q, cos[None, :, None, None, :], sin[None, :, None, None, :])
        k = rotate(k, cos[None, :, None, :], sin[None, :, None, :])
        mix = self_attn(q, k, v)
        return mix.flatten(-3) @ p[f"{layer}.self_attn.o_proj"].T

    def silu(x: Tensor) -> Tensor:
        return x / (1 + torch.exp(-x))

    def mlp(x: Tensor, layer: int) -> Tensor:
        z = rms_norm(x, p[f"{layer}.post_attention_layernorm"])
        gate = silu(z @ p[f"{layer}.mlp.gate_proj"].T)
        up = z @ p[f"{layer}.mlp.up_proj"].T
        return (up * gate) @ p[f"{layer}.mlp.down_proj"].T

    cos, sin = rotary_cos_sin(input_ids.shape[1])
    hidden = p["embed_tokens"][input_ids]
    for layer in range(c.num_hidden_layers):
        hidden += attn(hidden, layer, cos, sin)
        hidden += mlp(hidden, layer)
    hidden = rms_norm(hidden, p["norm"])
    return hidden @ p["embed_tokens"].T


def main() -> None:
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    prompt = "I am"
    dtype = torch.float32

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    )
    input_ids = torch.tensor(tokenizer(prompt).input_ids)[None]

    t0 = time.time()
    hf_logits = model(input_ids).logits
    duration_hf = time.time() - t0

    t0 = time.time()
    logits = predict(model, input_ids)
    duration = time.time() - t0

    print(logits[:, -1].argmax(), f"in {duration:.3f} s (HF: {duration_hf:.3f} s)")
    torch.testing.assert_close(logits.float(), hf_logits)


if __name__ == "__main__":
    main()
