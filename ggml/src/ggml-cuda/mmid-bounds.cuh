#pragma once

#include <cstdint>

// ids_src1/ids_dst of MUL_MAT_ID and expert_bounds[n_experts + 1], see mmid-bounds.cu
void ggml_cuda_launch_mm_ids_helper_bounds(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream);
