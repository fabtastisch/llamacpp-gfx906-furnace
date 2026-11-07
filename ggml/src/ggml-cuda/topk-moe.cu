#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    int  n_groups;
    int  n_group_used;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : log1pf(expf(vals[i]))) : -INFINITY;
    }
}

// lane exchange for the argmax butterfly. On GCN the __shfl_xor lowering is an LDS bpermute per
// step, 20 dependent round trips per selection round; DPP/swizzle moves carry the same values
template <int off>
static __device__ __forceinline__ int topk_moe_xfer(const int x) {
#if defined(GGML_USE_HIP) && defined(GCN)
    return ggml_gcn_xfer_xor_i32<off>(x);
#else
    return __shfl_xor_sync(0xFFFFFFFF, x, off, WARP_SIZE);
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

template <int off>
static __device__ __forceinline__ float topk_moe_xfer(const float x) {
    return __int_as_float(topk_moe_xfer<off>(__float_as_int(x)));
}

template <int off>
static __device__ __forceinline__ void topk_moe_argmax_step(float & max_val, int & max_expert) {
    const float val    = topk_moe_xfer<off>(max_val);
    const int   expert = topk_moe_xfer<off>(max_expert);
    if (val > max_val || (val == max_val && expert < max_expert)) {
        max_val    = val;
        max_expert = expert;
    }
}
#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
// Set -inf for experts outside the top n_group_used groups. Group score is the sum of its top-2 experts (DeepSeek V3).
template <int n_experts, int experts_per_thread>
__device__ void group_mask_warp_inplace(float (&vals)[experts_per_thread], const int n_groups, const int n_group_used, const int lane) {
    const int n_per_group = n_experts / n_groups;

    float score = -INFINITY; // lane g holds the score of group g
    for (int g = 0; g < n_groups; g++) {
        float top1 = -INFINITY;
        float top2 = -INFINITY;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if ((i * WARP_SIZE + lane) / n_per_group == g) {
                top2 = fmaxf(top2, fminf(top1, vals[i]));
                top1 = fmaxf(top1, vals[i]);
            }
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            const float o1 = __shfl_xor_sync(0xFFFFFFFF, top1, offset, WARP_SIZE);
            const float o2 = __shfl_xor_sync(0xFFFFFFFF, top2, offset, WARP_SIZE);
            top2 = fmaxf(fminf(top1, o1), fmaxf(top2, o2));
            top1 = fmaxf(top1, o1);
        }
        if (lane == g) {
            score = top1 + top2;
        }
    }

    // rank of each group, lower index wins ties
    int rank = 0;
    for (int g = 0; g < n_groups; g++) {
        const float other = __shfl_sync(0xFFFFFFFF, score, g, WARP_SIZE);
        rank += other > score || (other == score && g < lane);
    }
    const auto keep = __ballot_sync(0xFFFFFFFFULL, lane < n_groups && rank < n_group_used);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int g = (i * WARP_SIZE + lane) / n_per_group;
        if (g >= n_groups || !((keep >> g) & 1)) {
            vals[i] = -INFINITY;
        }
    }
}
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
            if (__isnanf(selection_wt[i / WARP_SIZE])) {
                selection_wt[i / WARP_SIZE] = -FLT_MAX;
            }
        }
    }

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
    // without bias, masking wt is safe: the matcher ensures the kept groups hold all used experts
    if (config.n_groups > 0) {
        if constexpr (has_bias) {
            group_mask_warp_inplace<n_experts>(selection_wt, config.n_groups, config.n_group_used, threadIdx.x);
        } else {
            group_mask_warp_inplace<n_experts>(wt, config.n_groups, config.n_group_used, threadIdx.x);
        }
    }
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    static_assert(experts_per_thread <= 32, "taken mask holds one bit per slot");
    uint32_t taken = 0;

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
            // taken experts are skipped through a bit mask: storing -inf at a data-dependent
            // register index is a waterfall loop on GCN
            if (taken & 1u) {
                max_val = -INFINITY;
            }
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                const float v = (taken >> i) & 1u ? -INFINITY : wt[i];
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && v > max_val) {
                    max_val    = v;
                    max_expert = expert;
                }
            }

            static_assert(WARP_SIZE == 32, "butterfly below is written for 32 lanes");
            topk_moe_argmax_step<16>(max_val, max_expert);
            topk_moe_argmax_step< 8>(max_val, max_expert);
            topk_moe_argmax_step< 4>(max_val, max_expert);
            topk_moe_argmax_step< 2>(max_val, max_expert);
            topk_moe_argmax_step< 1>(max_val, max_expert);

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                taken |= 1u << (max_expert / WARP_SIZE);
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

// Wide-router rank kernel (GCN; from reinstinct's moe_topk, crossport L6). One row per 256-thread
// block. Softmax is monotonic and the renormalisation over the selected k cancels its denominator,
// so the kernel ranks the raw logits and exponentiates only the k winners:
//   w_k = exp(l_k - l_max) / sum_{selected} exp(l_j - l_max)
// Each (logit, id) pair is one ordered 64-bit key (larger logit, then lower id), so a rank is
// n_experts single compares, read two keys per 16-byte LDS load with eight loads in flight. The
// serial argmax above costs n_expert_used dependent butterflies (17 us for 10 of 512 on gfx906).
// Used when: no bias, plain softmax, and either weights are renormalised (with_norm; the clamp
// never binds since the top k hold at least k/n of the mass) or the delayed softmax over the
// selected logits is wanted. Outputs match the serial kernel up to summation order.
static __device__ __forceinline__ uint32_t topk_moe_ordered_bits(const float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int n_experts>
__launch_bounds__(n_experts)
__global__ void topk_moe_rank_cuda(const float * logits, float * weights, int32_t * ids, const int n_rows,
        const int n_expert_used, const float scale_val) {
    // rank select: expert i goes to slot rank(i) = #{j : key_j > key_i or (key_j == key_i and j < i)}.
    // One expert per thread, so the n^2 compares spread over n/64 waves (the 256-thread, two experts
    // per thread version was issue bound on one or two waves: 41 us at 512 experts vs 20 us for the
    // serial argmax). Keys are 32-bit ordered floats read from LDS as wave-uniform (broadcast) uint4,
    // and the tie term only exists for the two chunks that overlap this wave's own 64 experts: chunks
    // below use >=, chunks above use >.
    static_assert(n_experts == 256 || n_experts == 512, "one expert per thread, 64-expert waves");
    constexpr int n_chunks = n_experts / 32;
    __shared__ __attribute__((aligned(16))) uint32_t keys[n_experts];
    __shared__ float chosen[64];
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const int t = threadIdx.x;
    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;

    float v = logits[t];
    if (isnan(v)) {
        v = -INFINITY;
    }
    const uint32_t k = topk_moe_ordered_bits(v);
    keys[t] = k;
    __syncthreads();

#if defined(GGML_USE_HIP)
    const int wave = __builtin_amdgcn_readfirstlane(t >> 6);
#else
    const int wave = t >> 6;
#endif
    const uint4 * k4 = reinterpret_cast<const uint4 *>(keys);
    int rank = 0;
    // chunks entirely below this wave's experts: every equal key has the lower index
    for (int c = 0; c < 2 * wave; c++) {
        uint4 q[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            q[u] = k4[c * 8 + u];
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            rank += (q[u].x >= k) + (q[u].y >= k) + (q[u].z >= k) + (q[u].w >= k);
        }
    }
    // the two chunks holding this wave's own experts
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
        const int c = 2 * wave + cc;
        uint4 q[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            q[u] = k4[c * 8 + u];
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            const int jb = c * 32 + 4 * u;
            rank += (q[u].x > k) + ((q[u].x == k) & (jb + 0 < t));
            rank += (q[u].y > k) + ((q[u].y == k) & (jb + 1 < t));
            rank += (q[u].z > k) + ((q[u].z == k) & (jb + 2 < t));
            rank += (q[u].w > k) + ((q[u].w == k) & (jb + 3 < t));
        }
    }
    // chunks entirely above: equal keys there rank below us
    for (int c = 2 * wave + 2; c < n_chunks; c++) {
        uint4 q[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            q[u] = k4[c * 8 + u];
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            rank += (q[u].x > k) + (q[u].y > k) + (q[u].z > k) + (q[u].w > k);
        }
    }

    if (rank < n_expert_used) {
        ids[rank]    = t;
        chosen[rank] = v;
    }
    __syncthreads();

    // softmax over the k winners (== full softmax renormalised over the top k), one wave
    if (t < 64) {
        const float x = t < n_expert_used ? chosen[t] : -INFINITY;
        const float m = warp_reduce_max(x);
        const float e = t < n_expert_used ? (x == m ? 1.0f : expf(x - m)) : 0.0f; // all -inf rows stay finite
        const float s = warp_reduce_sum(e);
        if (t < n_expert_used) {
            weights[t] = e / s * scale_val;
        }
    }
}

static bool topk_moe_rank_applicable(ggml_backend_cuda_context & ctx, const int n_expert, const int n_expert_used,
        const float clamp_val, const topk_moe_config config, const bool has_bias) {
#if defined(GGML_USE_HIP)
    static const bool disabled = getenv("GGML_CUDA_NO_TOPK_RANK") != nullptr;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (disabled || !GGML_CUDA_CC_IS_GCN(cc) || has_bias || config.use_sigmoid || config.use_sqrt_softplus ||
            n_expert_used > 64 || (n_expert != 256 && n_expert != 512)) {
        return false;
    }
    if (config.with_norm) {
        // the top k of n hold at least k/n of the softmax mass, so the clamp cannot bind
        return clamp_val <= (float) n_expert_used / (float) n_expert;
    }
    return config.delayed_softmax;
#else
    GGML_UNUSED_VARS(ctx, n_expert, n_expert_used, clamp_val, config, has_bias);
    return false;
#endif
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    if (topk_moe_rank_applicable(ctx, n_expert, n_expert_used, clamp_val, config, has_bias)) {
        if (n_expert == 512) {
            topk_moe_rank_cuda<512><<<n_rows, 512, 0, stream>>>(logits, weights, ids, n_rows, n_expert_used, scale_val);
        } else {
            topk_moe_rank_cuda<256><<<n_rows, 256, 0, stream>>>(logits, weights, ids, n_rows, n_expert_used, scale_val);
        }
        return;
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_ASSERT(config.n_groups == 0 && "expert groups fusion is not supported on this HIP version");
#endif

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 384: // DeepSeek V4.1
            ggml_cuda_kernel_launch(topk_moe_cuda<384, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 384 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
