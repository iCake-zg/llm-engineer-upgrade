

#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


namespace naive_attention {

void attention_cpu(
    // Function to compute attention scores and output
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim,
    bool causal = false)

{
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int stride_b = heads * seq_length * head_dim;

}
}

















