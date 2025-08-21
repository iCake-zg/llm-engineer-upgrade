

#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


namespace naive_attention {

    void attention_cpu(
        const float* Q,
        const float* K,
        const float* V,
        float* O,
        int batch_size,
        int num_heads,
        int seq_length,
        int head_dim,
        bool causal = false)
    }












}