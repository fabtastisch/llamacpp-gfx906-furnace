#include "qsa-mask.cuh"

// GGML_OP_QSA_MASK: the attention mask of a qwen4exp block-sparse (QSA) layer from the indexer's block scores.
// For query row i a block takes part when the KQ mask keeps one of its cells. A block scored +inf (the incomplete
// tail) is always kept; of the others the n_top highest scored are kept (ties in ascending block index), and the
// mask keeps its value at the cells of the kept blocks and is -inf elsewhere:
//   1. k_qsa_mask_vis:    which blocks have a visible cell (one pass over the cells)
//   2. k_qsa_mask_select: a radix select over the competing blocks finds the threshold key T, then marks the kept
//                         blocks, the ones at T in ascending index (one workgroup per row, passes over the blocks only)
//   3. k_qsa_mask_write:  the mask

#define QSA_MASK_NT    256
#define QSA_MASK_CHUNK 4096   // cells per workgroup in the cell passes: 16 rounds of 256

static __device__ __forceinline__ uint32_t qsa_mask_key(const float x) {
    const uint32_t u = __float_as_uint(x + 0.0f);   // -0 -> +0, so the two tie
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // larger float -> larger key
}

template <typename T>
static __device__ __forceinline__ bool qsa_mask_keeps(const T m) {
    return (float) m != -INFINITY;
}

// exclusive scan over the workgroup; *total gets the sum
static __device__ int qsa_mask_scan(const int v, int * sh, int * total) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int off = 1; off < QSA_MASK_NT; off *= 2) {
        const int t = tid >= off ? sh[tid - off] : 0;
        __syncthreads();
        sh[tid] += t;
        __syncthreads();
    }
    const int incl = sh[tid];
    *total = sh[QSA_MASK_NT - 1];
    __syncthreads();
    return incl - v;
}

struct qsa_mask_args {
    const float   * score;    // row i: score + (i % n_tps)*s_t + (i / n_tps)*s_s
    const int32_t * cell_blk; // stream s: cell_blk + s*cb_s
    const char    * mask;     // row i: mask + (i % n_tps)*m_t + (i / n_tps)*m_s (bytes)
    char          * dst;      // as mask, dst_t / dst_s
    int64_t s_t, s_s, cb_s, m_t, m_s, d_t, d_s;
    int     n_kv;
    int     n_blocks;
    int     n_tps;
    int     n_top;
    int     row0;             // first row of this launch
};

template <typename T>
static __device__ __forceinline__ const T * qsa_mask_row(const qsa_mask_args & a, const int i) {
    return (const T *) (a.mask + (int64_t) (i % a.n_tps)*a.m_t + (int64_t) (i / a.n_tps)*a.m_s);
}

// 1. w[row][b] = 1 when the mask keeps a cell of block b (every writer stores the same value)
template <typename T>
static __global__ void __launch_bounds__(QSA_MASK_NT) k_qsa_mask_vis(const qsa_mask_args a, uint32_t * __restrict__ w) {
    const int ir = blockIdx.y;
    const int i  = a.row0 + ir;
    const T       * m  = qsa_mask_row<T>(a, i);
    const int32_t * cb = a.cell_blk + (int64_t) (i / a.n_tps)*a.cb_s;
    uint32_t      * wr = w + (int64_t) ir*a.n_blocks;

    const int j0 = blockIdx.x*QSA_MASK_CHUNK;
    const int j1 = min(a.n_kv, j0 + QSA_MASK_CHUNK);
    for (int j = j0 + threadIdx.x; j < j1; j += QSA_MASK_NT) {
        if (qsa_mask_keeps(m[j])) {
            wr[cb[j]] = 1u;
        }
    }
}

// a block competes for the n_top places when it has a visible cell and is not the always-kept tail
static __device__ __forceinline__ bool qsa_mask_competes(const uint32_t w, const float s) {
    return w != 0 && s != INFINITY;
}

// one radix pass over the digit [shift, shift + BITS) of the keys matching the resolved prefix: the histogram, then
// the bin holding the need-th highest competing block (searching from the top). Resolves BITS more bits of T
template <int BITS>
static __device__ void qsa_mask_pass(const float * __restrict__ srow, const uint32_t * __restrict__ wr, const int n_blocks,
        const int shift, uint32_t * hist, int * sh, uint32_t * prefix, uint32_t * pmask, int * need, int * n_bin) {
    constexpr int NBINS = 1 << BITS;
    constexpr int PER   = NBINS / QSA_MASK_NT;
    static_assert(NBINS % QSA_MASK_NT == 0, "bins must split evenly over the threads");
    const int tid = threadIdx.x;

    for (int k = tid; k < NBINS; k += QSA_MASK_NT) {
        hist[k] = 0;
    }
    __syncthreads();
    const uint32_t pf = *prefix;
    const uint32_t pm = *pmask;
    for (int b = tid; b < n_blocks; b += QSA_MASK_NT) {
        const float s = srow[b];
        if (!qsa_mask_competes(wr[b], s)) {
            continue;
        }
        const uint32_t key = qsa_mask_key(s);
        if ((key & pm) == pf) {
            atomicAdd(&hist[(key >> shift) & (NBINS - 1)], 1u);
        }
    }
    __syncthreads();

    // thread t owns bins [NBINS - PER*(t+1), NBINS - PER*t), scanned from the top
    const int top = NBINS - PER*tid - 1;
    int own = 0;
    for (int k = 0; k < PER; ++k) {
        own += (int) hist[top - k];
    }
    int tot;
    const int above = qsa_mask_scan(own, sh, &tot);
    const int nd = *need;
    if (above < nd && above + own >= nd) {
        int acc = above;
        int bin = top - PER + 1;
        for (int k = 0; k < PER; ++k) {
            if (acc + (int) hist[top - k] >= nd) {
                bin = top - k;
                break;
            }
            acc += (int) hist[top - k];
        }
        *prefix = pf | ((uint32_t) bin << shift);
        *pmask  = pm | ((uint32_t) (NBINS - 1) << shift);
        *need   = nd - acc;
        *n_bin  = (int) hist[bin];
    }
    __syncthreads();
}

// the blocks at key T in the current round of 256 blocks: this lane's flag, the round's lanes before it, the round's
// total (a wave-ordered prefix, so ranks follow the block index)
static __device__ __forceinline__ int qsa_mask_round_prefix(const bool f, int * wsum, int * total) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    constexpr int NW = QSA_MASK_NT / WS;
    const int lane   = threadIdx.x % WS;
    const int wid    = threadIdx.x / WS;
#ifdef GGML_USE_HIP
    const uint64_t bal  = __ballot(f);
    const int      in_w = __popcll(bal & ((uint64_t(1) << lane) - 1));
    const int      n_w  = __popcll(bal);
#else
    const uint32_t bal  = __ballot_sync(0xffffffff, f);
    const int      in_w = __popc(bal & ((1u << lane) - 1));
    const int      n_w  = __popc(bal);
#endif
    if (lane == 0) {
        wsum[wid] = n_w;
    }
    __syncthreads();
    int before = 0;
    int all    = 0;
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        before += k < wid ? wsum[k] : 0;
        all    += wsum[k];
    }
    __syncthreads();
    *total = all;
    return before + in_w;
}

// 2. per row: the threshold key, then w[row][b] = 1 for the kept blocks. all[ir] = 1: every visible block is kept and
// w is left as it is
static __global__ void __launch_bounds__(QSA_MASK_NT) k_qsa_mask_select(const qsa_mask_args a, uint32_t * __restrict__ w,
        int * __restrict__ all) {
    const int ir  = blockIdx.x;
    const int i   = a.row0 + ir;
    const int tid = threadIdx.x;
    const float * srow = a.score + (int64_t) (i % a.n_tps)*a.s_t + (int64_t) (i / a.n_tps)*a.s_s;
    uint32_t    * wr   = w + (int64_t) ir*a.n_blocks;

    __shared__ uint32_t hist[4096];
    __shared__ int      sh[QSA_MASK_NT];
    __shared__ int      wsum[QSA_MASK_NT / ggml_cuda_get_physical_warp_size()];
    __shared__ uint32_t prefix, pmask;
    __shared__ int      need, n_bin;

    int part = 0;
    for (int b = tid; b < a.n_blocks; b += QSA_MASK_NT) {
        part += qsa_mask_competes(wr[b], srow[b]);
    }
    int total;
    qsa_mask_scan(part, sh, &total);

    if (total <= a.n_top) {
        if (tid == 0) {
            all[ir] = 1;
        }
        return;
    }

    if (tid == 0) {
        prefix = 0;
        pmask  = 0;
        need   = a.n_top;
        n_bin  = 0;
    }
    __syncthreads();

    // n_top == 0 keeps only the tail: T above every key
    if (a.n_top > 0) {
        qsa_mask_pass<12>(srow, wr, a.n_blocks, 20, hist, sh, &prefix, &pmask, &need, &n_bin);
        qsa_mask_pass<12>(srow, wr, a.n_blocks,  8, hist, sh, &prefix, &pmask, &need, &n_bin);
        qsa_mask_pass< 8>(srow, wr, a.n_blocks,  0, hist, sh, &prefix, &pmask, &need, &n_bin);
    }

    const bool     none  = a.n_top == 0;
    const uint32_t T     = prefix;
    const int      nd    = need;
    const bool     all_T = nd >= n_bin;

    // every thread runs the same number of rounds, so the round prefix's barriers line up
    int taken = 0;   // blocks at T in earlier rounds
    for (int base = 0; base < a.n_blocks; base += QSA_MASK_NT) {
        const int   b   = base + tid;
        const bool  in  = b < a.n_blocks;
        const uint32_t wb = in ? wr[b] : 0u;
        const float s   = in ? srow[b] : 0.0f;
        const bool  cmp = qsa_mask_competes(wb, s);
        const uint32_t key = cmp ? qsa_mask_key(s) : 0u;
        const bool  eq  = cmp && !none && key == T;
        bool keep = (wb != 0 && !cmp) || (cmp && !none && key > T);
        if (!all_T) {
            int n_round;
            const int r = qsa_mask_round_prefix(eq, wsum, &n_round);
            keep = keep || (eq && taken + r < nd);
            taken += n_round;
        } else {
            keep = keep || eq;
        }
        if (in) {
            wr[b] = keep ? 1u : 0u;
        }
    }
    if (tid == 0) {
        all[ir] = 0;
    }
}

// 3. the mask
template <typename T>
static __global__ void __launch_bounds__(QSA_MASK_NT) k_qsa_mask_write(const qsa_mask_args a, const uint32_t * __restrict__ w,
        const int * __restrict__ all) {
    const int ir = blockIdx.y;
    const int i  = a.row0 + ir;
    const T        * m  = qsa_mask_row<T>(a, i);
    const int32_t  * cb = a.cell_blk + (int64_t) (i / a.n_tps)*a.cb_s;
    const uint32_t * wr = w + (int64_t) ir*a.n_blocks;
    T              * d  = (T *) (a.dst + (int64_t) (i % a.n_tps)*a.d_t + (int64_t) (i / a.n_tps)*a.d_s);

    const int j0 = blockIdx.x*QSA_MASK_CHUNK;
    const int j1 = min(a.n_kv, j0 + QSA_MASK_CHUNK);
    const T drop = (T) -INFINITY;

    if (all[ir]) {
        for (int j = j0 + threadIdx.x; j < j1; j += QSA_MASK_NT) {
            d[j] = m[j];
        }
        return;
    }

    for (int j = j0 + threadIdx.x; j < j1; j += QSA_MASK_NT) {
        const T mj = m[j];
        d[j] = qsa_mask_keeps(mj) && wr[cb[j]] ? mj : drop;
    }
}

template <typename T>
static void qsa_mask_launch(ggml_backend_cuda_context & ctx, qsa_mask_args a, const int n_rows) {
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    const int n_cc = (a.n_kv + QSA_MASK_CHUNK - 1)/QSA_MASK_CHUNK;

    // rows per launch: the block flags of a launch stay within 32 MiB
    const int64_t per_row = (int64_t) a.n_blocks*sizeof(uint32_t);
    const int rc = (int) std::max<int64_t>(1, std::min<int64_t>(n_rows, (int64_t(32) << 20)/std::max<int64_t>(1, per_row)));

    ggml_cuda_pool_alloc<uint32_t> w  (pool, (size_t) rc*a.n_blocks);
    ggml_cuda_pool_alloc<int>      all(pool, (size_t) rc);

    for (int r0 = 0; r0 < n_rows; r0 += rc) {
        const int nr = std::min(rc, n_rows - r0);
        a.row0 = r0;
        CUDA_CHECK(cudaMemsetAsync(w.get(), 0, (size_t) nr*a.n_blocks*sizeof(uint32_t), stream));
        k_qsa_mask_vis<T>   <<<dim3(n_cc, nr, 1), QSA_MASK_NT, 0, stream>>>(a, w.get());
        k_qsa_mask_select   <<<nr,                QSA_MASK_NT, 0, stream>>>(a, w.get(), all.get());
        k_qsa_mask_write<T> <<<dim3(n_cc, nr, 1), QSA_MASK_NT, 0, stream>>>(a, w.get(), all.get());
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_qsa_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score    = dst->src[0];
    const ggml_tensor * cell_blk = dst->src[1];
    const ggml_tensor * mask     = dst->src[2];

    GGML_ASSERT(score->type == GGML_TYPE_F32 && cell_blk->type == GGML_TYPE_I32 && mask->type == dst->type);
    GGML_ASSERT(score->nb[0] == sizeof(float) && cell_blk->nb[0] == sizeof(int32_t) && mask->nb[0] == ggml_type_size(mask->type));

    qsa_mask_args a;
    a.score    = (const float *) score->data;
    a.cell_blk = (const int32_t *) cell_blk->data;
    a.mask     = (const char *) mask->data;
    a.dst      = (char *) dst->data;
    a.s_t      = score->nb[1]/sizeof(float);
    a.s_s      = score->nb[2]/sizeof(float);
    a.cb_s     = cell_blk->nb[1]/sizeof(int32_t);
    a.m_t      = mask->nb[1];
    a.m_s      = mask->nb[3];
    a.d_t      = dst->nb[1];
    a.d_s      = dst->nb[3];
    a.n_kv     = (int) mask->ne[0];
    a.n_blocks = (int) score->ne[0];
    a.n_tps    = (int) mask->ne[1];
    a.n_top    = ggml_get_op_params_i32(dst, 0);
    a.row0     = 0;

    const int n_rows = (int) (mask->ne[1]*mask->ne[3]);

    if (dst->type == GGML_TYPE_F16) {
        qsa_mask_launch<half>(ctx, a, n_rows);
    } else {
        qsa_mask_launch<float>(ctx, a, n_rows);
    }
}
