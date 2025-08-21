


from torch import nn

class MuiltiHeadAttention(nn.Module):

    def __init__(self,d_model, num_heads):
        super(MuiltiHeadAttention, self).__init__()

        assert d_model % num_heads == 0,"d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads


        ## 线性变化层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = scores.softmax(dim=-1)
        return attn @ v , attn

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性变化并分头
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性层
        output = self.wo(attn_output)

        return output, attn_weights



