


from torch import nn




class FlashAttention(nn.module): 

    """
    Flash Attention implementation 
    """

    # def __init__(self, d_model,num_heads):
    #     super(FlashAttention, self).__init__()

    #     assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    #     self.d_model = d_model
    #     self.num_heads = num_heads
    #     self.d_k = d_model // num_heads

    #     ## 线性变化层
    #     self.wq = nn.Linear(d_model, d_model)
    #     self.wk = nn.Linear(d_model, d_model)
    #     self.wv = nn.Linear(d_model, d_model)
    #     self.wo = nn.Linear(d_model, d_model)



