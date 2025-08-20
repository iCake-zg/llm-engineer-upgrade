import torch, math

@torch.no_grad()
def rope(x: torch.Tensor, pos_ids: torch.Tensor, dim: int, base: float = 10000.):
    """
    x: [B, H, L, D]  或 [B, L, D]
    pos_ids: [L]  每个 token 的真实位置
    dim: 需要做 RoPE 的维度（一般是 head_dim）
    return: 旋转后的张量，形状同 x
    """
    assert x.size(-1) == dim
    half = dim // 2
    # 1. 生成频率 inv_freq: [half]
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    print("inv_freq:", inv_freq.size())   # [half]
    # 2. 角度 theta = pos * inv_freq, 形状 [L, half]
    theta = pos_ids[:, None].float() * inv_freq[None, :]

    print("theta:", theta.size())         # [L, half]
    cos, sin = torch.cos(theta), torch.sin(theta)          # [L, half]
    # 3. 把 x 拆成 x1, x2
    x1, x2 = x[..., :half], x[..., half:]
    print("x1:", x1.size())               # [B, H, L, half]
    print("x2:", x2.size())               # [B, H, L, half]
    # 为了能广播，增加维度
    # 4. 旋转
    rotated = torch.cat([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)
    return rotated.type_as(x)

# ----------------- 3 行测试 -----------------
if __name__ == "__main__":
    x = torch.randn(1, 8, 5, 128)     # [B, H, L, D]
    pos = torch.arange(5)
    out = rope(x, pos, 128)
    print(out.shape)                  # torch.Size([1, 8, 5, 128])