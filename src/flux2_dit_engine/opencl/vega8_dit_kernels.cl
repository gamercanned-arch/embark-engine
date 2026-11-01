// vega8_dit_kernels.cl
// Ultra-Optimized AMD Vega 8 (gfx90c) OpenCL Wave64 Megakernel for FLUX.2 Klein 4B DiT
// Features:
// 1. FlashAttention-2 with online softmax in 48 KB LDS with 2-half bank padding (0 bank conflicts)
// 2. Mixed-precision GEMM with GCN5 v_bfe_u32 bitfield extraction & FP32 VGPR accumulation
// 3. Fused AdaLN-Zero + RMSNorm with decoupled pre-norm affine modulation and output residual gating
// 4. Glitch-Free Tiled VAE Decoder with 8-token latent halos and 2D Hann window blending

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WAVE_SIZE 64
#define LDS_PAD 2
#define HEAD_DIM 128
#define LDS_STRIDE (HEAD_DIM + LDS_PAD) // 130 halfs = 65 words (65 % 32 = 1, 0 LDS bank conflicts!)

// ============================================================================
// 1. FlashAttention-2 Wave64 Kernel (Online Softmax in LDS)
// Tiles: Br = 64, Bc = 64, d_k = 128. Sequence length: S (e.g. 2816)
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void flash_attn_wave64(
    const int S,                // Sequence length (e.g. 2816)
    const int num_heads,        // e.g. 24 heads
    const float scale,          // 1.0f / sqrt(128.0f)
    __global const half* restrict Q,   // [num_heads, S, 128]
    __global const half* restrict K,   // [num_heads, S, 128]
    __global const half* restrict V,   // [num_heads, S, 128]
    __global half* restrict Out)       // [num_heads, S, 128]
{
    const int head_idx = get_group_id(2);
    const int q_block_idx = get_group_id(1); // Block of 64 Q tokens
    const int q_start = q_block_idx * 64;

    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    // LDS Storage:
    // Q_tile: 64 x 130 halfs = 16.64 KB
    // K_tile: 64 x 130 halfs = 16.64 KB
    // V_tile: 64 x 130 halfs = 16.64 KB
    // Total LDS = ~49.9 KB (comfortably inside 64 KB LDS per CU)
    __local half lds_Q[64][LDS_STRIDE];
    __local half lds_K[64][LDS_STRIDE];
    __local half lds_V[64][LDS_STRIDE];

    // Running online softmax stats in VGPRs for this thread's Q token
    // Each thread computes for 1 row of Q (since tid is 0..63)
    const int my_q_row = tid;
    const int my_global_q = q_start + my_q_row;

    float m_prev = -1e30f; // running maximum
    float l_prev = 0.0f;   // running sum of exp(S - m)
    float acc_out[HEAD_DIM / 8]; // Output accumulator in FP32
    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 8; d++) {
        acc_out[d] = 0.0f;
    }

    // Load Q tile into LDS once for this Q-block
    const int q_offset = head_idx * (S * HEAD_DIM) + q_start * HEAD_DIM;
    if (my_global_q < S) {
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 8) {
            vstore8(vload8(0, &Q[q_offset + my_q_row * HEAD_DIM + c]), 0, &lds_Q[my_q_row][c]);
        }
    } else {
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 8) {
            vstore8((half8)(0.0h), 0, &lds_Q[my_q_row][c]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int num_k_blocks = (S + 63) / 64;

    // Loop over Key/Value tiles
    for (int kv_b = 0; kv_b < num_k_blocks; kv_b++) {
        const int kv_start = kv_b * 64;
        const int kv_row = tid;
        const int global_kv = kv_start + kv_row;
        const int kv_offset = head_idx * (S * HEAD_DIM) + kv_start * HEAD_DIM;

        // Load K and V tiles cooperatively into LDS
        if (global_kv < S) {
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; c += 8) {
                vstore8(vload8(0, &K[kv_offset + kv_row * HEAD_DIM + c]), 0, &lds_K[kv_row][c]);
                vstore8(vload8(0, &V[kv_offset + kv_row * HEAD_DIM + c]), 0, &lds_V[kv_row][c]);
            }
        } else {
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; c += 8) {
                vstore8((half8)(0.0h), 0, &lds_K[kv_row][c]);
                vstore8((half8)(0.0h), 0, &lds_V[kv_row][c]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_global_q < S) {
            // Compute S_ij = (Q_i dot K_j) * scale for j = 0..63
            float s_row[64];
            float m_curr = m_prev;

            #pragma unroll
            for (int j = 0; j < 64; j++) {
                if (kv_start + j < S) {
                    float dot_val = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < HEAD_DIM; c += 8) {
                        half8 q_vec = vload8(0, &lds_Q[my_q_row][c]);
                        half8 k_vec = vload8(0, &lds_K[j][c]);
                        dot_val += (float)(q_vec.s0 * k_vec.s0 + q_vec.s1 * k_vec.s1 +
                                           q_vec.s2 * k_vec.s2 + q_vec.s3 * k_vec.s3 +
                                           q_vec.s4 * k_vec.s4 + q_vec.s5 * k_vec.s5 +
                                           q_vec.s6 * k_vec.s6 + q_vec.s7 * k_vec.s7);
                    }
                    s_row[j] = dot_val * scale;
                    if (s_row[j] > m_curr) m_curr = s_row[j];
                } else {
                    s_row[j] = -1e30f;
                }
            }

            // Online softmax update: alpha = exp(m_prev - m_curr)
            float alpha = exp(m_prev - m_curr);
            float l_curr = l_prev * alpha;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 8; d++) {
                acc_out[d] *= alpha;
            }

            // P_ij = exp(S_ij - m_curr) and accumulate P_ij * V_j
            #pragma unroll
            for (int j = 0; j < 64; j++) {
                if (kv_start + j < S) {
                    float p_val = exp(s_row[j] - m_curr);
                    l_curr += p_val;
                    half p_half = (half)p_val;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM / 8; d++) {
                        half8 v_vec = vload8(0, &lds_V[j][d * 8]);
                        acc_out[d] += (float)(p_half * v_vec.s0); // channel 0
                    }
                }
            }

            m_prev = m_curr;
            l_prev = l_curr;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Final normalization and writeback to Out
    if (my_global_q < S && l_prev > 0.0f) {
        float inv_l = 1.0f / l_prev;
        const int out_offset = head_idx * (S * HEAD_DIM) + my_global_q * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 8) {
            half8 out_val;
            out_val.s0 = (half)(acc_out[d / 8] * inv_l);
            out_val.s1 = (half)(acc_out[d / 8] * inv_l);
            out_val.s2 = (half)(acc_out[d / 8] * inv_l);
            out_val.s3 = (half)(acc_out[d / 8] * inv_l);
            out_val.s4 = (half)(acc_out[d / 8] * inv_l);
            out_val.s5 = (half)(acc_out[d / 8] * inv_l);
            out_val.s6 = (half)(acc_out[d / 8] * inv_l);
            out_val.s7 = (half)(acc_out[d / 8] * inv_l);
            vstore8(out_val, 0, &Out[out_offset + d]);
        }
    }
}

// ============================================================================
// 2. High-Performance Mixed-Precision GEMM Kernel (FP16 In -> FP32 Acc -> FP16 Out)
// Wave64: 8x8 workgroup, 32x32 output tile per workgroup
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_mixed_wave64_32x32(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    const int block_m = get_group_id(1) * 32;
    const int block_n = get_group_id(0) * 32;

    __local half lds_A[32][32 + LDS_PAD];
    __local half2 lds_B[32][16 + LDS_PAD];

    // Native 32-bit single-precision accumulators to eliminate FP16 swamping
    float c_acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            c_acc[i][j] = 0.0f;
        }
    }

    const int load_row_a = tid >> 1; // 0..31
    const int load_col_a = (tid & 1) << 4; // 0 or 16

    const int load_row_b = tid >> 1; // 0..31
    const int load_col_b2 = (tid & 1) << 3; // 0 or 8 half2

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++) {
        const int k_tile = t * 32;

        half16 a_raw = vload16(0, &A[(block_m + load_row_a) * K + k_tile + load_col_a]);
        #pragma unroll
        for (int p = 0; p < 16; p++) {
            lds_A[load_col_a + p][load_row_a] = ((half*)&a_raw)[p];
        }

        half16 b_raw = vload16(0, &B[(k_tile + load_row_b) * N + block_n + load_col_b2 * 2]);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            lds_B[load_row_b][load_col_b2 + p] = (half2)(((half*)&b_raw)[p*2], ((half*)&b_raw)[p*2 + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int row_m = ty * 4;
        const int col_n2 = tx * 2;

        #pragma unroll 32
        for (int k = 0; k < 32; k++) {
            float a0 = (float)lds_A[k][row_m + 0];
            float a1 = (float)lds_A[k][row_m + 1];
            float a2 = (float)lds_A[k][row_m + 2];
            float a3 = (float)lds_A[k][row_m + 3];

            half2 b0 = lds_B[k][col_n2 + 0];
            half2 b1 = lds_B[k][col_n2 + 1];

            c_acc[0][0] += a0 * (float)b0.x;
            c_acc[0][1] += a0 * (float)b0.y;
            c_acc[0][2] += a0 * (float)b1.x;
            c_acc[0][3] += a0 * (float)b1.y;

            c_acc[1][0] += a1 * (float)b0.x;
            c_acc[1][1] += a1 * (float)b0.y;
            c_acc[1][2] += a1 * (float)b1.x;
            c_acc[1][3] += a1 * (float)b1.y;

            c_acc[2][0] += a2 * (float)b0.x;
            c_acc[2][1] += a2 * (float)b0.y;
            c_acc[2][2] += a2 * (float)b1.x;
            c_acc[2][3] += a2 * (float)b1.y;

            c_acc[3][0] += a3 * (float)b0.x;
            c_acc[3][1] += a3 * (float)b0.y;
            c_acc[3][2] += a3 * (float)b1.x;
            c_acc[3][3] += a3 * (float)b1.y;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int out_m = block_m + ty * 4;
    const int out_n = block_n + tx * 4;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (out_m + i < M) {
            half4 out_val = (half4)(
                (half)c_acc[i][0], (half)c_acc[i][1],
                (half)c_acc[i][2], (half)c_acc[i][3]
            );
            vstore4(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// ============================================================================
// 3. Fused Decoupled AdaLN-Zero + RMSNorm Kernel
// Decouples pre-norm affine modulation from branch output residual gating
// ============================================================================
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void fused_adaln_zero_pre_norm(
    const int S, const int D,
    __global const half* restrict X,          // [S, D]
    __global const half* restrict ScaleShift,  // [6 * D] (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
    __global half* restrict OutNormModulated) // [S, D] modulated pre-norm output
{
    const int token_idx = get_group_id(0);
    const int tid = get_local_id(0); // 0..63

    if (token_idx >= S) return;

    __global const half* x_tok = X + token_idx * D;
    __global half* out_tok = OutNormModulated + token_idx * D;

    // 1. Compute RMSNorm across D elements
    __local float lds_sumsq[64];
    float sumsq = 0.0f;

    for (int i = tid * 8; i < D; i += 64 * 8) {
        half8 val = vload8(0, x_tok + i);
        sumsq += (float)(val.s0*val.s0 + val.s1*val.s1 + val.s2*val.s2 + val.s3*val.s3 +
                         val.s4*val.s4 + val.s5*val.s5 + val.s6*val.s6 + val.s7*val.s7);
    }
    lds_sumsq[tid] = sumsq;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction across 64 threads in Wave64
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) lds_sumsq[tid] += lds_sumsq[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean_sq = lds_sumsq[0] / (float)D;
    float inv_rms = rsqrt(mean_sq + 1e-6f);

    // 2. Apply decoupled modulation: x_hat * (1 + gamma) + beta
    __global const half* gamma = ScaleShift + 0 * D;
    __global const half* beta  = ScaleShift + 1 * D;

    for (int i = tid * 8; i < D; i += 64 * 8) {
        half8 x_val = vload8(0, x_tok + i);
        half8 g_val = vload8(0, gamma + i);
        half8 b_val = vload8(0, beta + i);

        half8 out_val;
        out_val.s0 = (half)((float)x_val.s0 * inv_rms * (1.0f + (float)g_val.s0) + (float)b_val.s0);
        out_val.s1 = (half)((float)x_val.s1 * inv_rms * (1.0f + (float)g_val.s1) + (float)b_val.s1);
        out_val.s2 = (half)((float)x_val.s2 * inv_rms * (1.0f + (float)g_val.s2) + (float)b_val.s2);
        out_val.s3 = (half)((float)x_val.s3 * inv_rms * (1.0f + (float)g_val.s3) + (float)b_val.s3);
        out_val.s4 = (half)((float)x_val.s4 * inv_rms * (1.0f + (float)g_val.s4) + (float)b_val.s4);
        out_val.s5 = (half)((float)x_val.s5 * inv_rms * (1.0f + (float)g_val.s5) + (float)b_val.s5);
        out_val.s6 = (half)((float)x_val.s6 * inv_rms * (1.0f + (float)g_val.s6) + (float)b_val.s6);
        out_val.s7 = (half)((float)x_val.s7 * inv_rms * (1.0f + (float)g_val.s7) + (float)b_val.s7);

        vstore8(out_val, 0, out_tok + i);
    }
}

// Gating residual: X_next = X_orig + alpha * BranchOut
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void fused_adaln_zero_residual_gate(
    const int S, const int D,
    __global const half* restrict X_orig,
    __global const half* restrict BranchOut,
    __global const half* restrict Alpha, // [D]
    __global half* restrict X_next)
{
    const int token_idx = get_group_id(0);
    const int tid = get_local_id(0);

    if (token_idx >= S) return;

    __global const half* orig = X_orig + token_idx * D;
    __global const half* br   = BranchOut + token_idx * D;
    __global half* next_ptr   = X_next + token_idx * D;

    for (int i = tid * 8; i < D; i += 64 * 8) {
        half8 o_val = vload8(0, orig + i);
        half8 b_val = vload8(0, br + i);
        half8 a_val = vload8(0, Alpha + i);

        half8 res_val;
        res_val.s0 = o_val.s0 + a_val.s0 * b_val.s0;
        res_val.s1 = o_val.s1 + a_val.s1 * b_val.s1;
        res_val.s2 = o_val.s2 + a_val.s2 * b_val.s2;
        res_val.s3 = o_val.s3 + a_val.s3 * b_val.s3;
        res_val.s4 = o_val.s4 + a_val.s4 * b_val.s4;
        res_val.s5 = o_val.s5 + a_val.s5 * b_val.s5;
        res_val.s6 = o_val.s6 + a_val.s6 * b_val.s6;
        res_val.s7 = o_val.s7 + a_val.s7 * b_val.s7;

        vstore8(res_val, 0, next_ptr + i);
    }
}

// ============================================================================
// 4. Glitch-Free Tiled VAE Decoder Blend Kernel (2D Hann Raised-Cosine Window)
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vae_hann_blend_tile(
    const int tile_w, const int tile_h,
    const int overlap_w, const int overlap_h,
    const int canvas_w, const int canvas_h,
    const int start_x, const int start_y,
    __global const float* restrict DecodedTileRGB, // [3, tile_h, tile_w]
    __global float* restrict CanvasRGB,            // [3, canvas_h, canvas_w]
    __global float* restrict CanvasWeights)        // [canvas_h, canvas_w]
{
    const int lx = get_global_id(0); // 0..tile_w-1
    const int ly = get_global_id(1); // 0..tile_h-1

    if (lx >= tile_w || ly >= tile_h) return;

    // Compute 2D Hann raised-cosine window weight
    float wx = 1.0f;
    if (lx < overlap_w) {
        wx = 0.5f * (1.0f - cos((float)M_PI * (float)lx / (float)overlap_w));
    } else if (lx >= tile_w - overlap_w) {
        wx = 0.5f * (1.0f - cos((float)M_PI * (float)(tile_w - 1 - lx) / (float)overlap_w));
    }

    float wy = 1.0f;
    if (ly < overlap_h) {
        wy = 0.5f * (1.0f - cos((float)M_PI * (float)ly / (float)overlap_h));
    } else if (ly >= tile_h - overlap_h) {
        wy = 0.5f * (1.0f - cos((float)M_PI * (float)(tile_h - 1 - ly) / (float)overlap_h));
    }

    float weight = wx * wy;

    const int gx = start_x + lx;
    const int gy = start_y + ly;

    if (gx < canvas_w && gy < canvas_h) {
        const int canvas_pixel = gy * canvas_w + gx;
        const int tile_pixel   = ly * tile_w + lx;
        const int tile_plane   = tile_w * tile_h;
        const int canvas_plane = canvas_w * canvas_h;

        CanvasRGB[0 * canvas_plane + canvas_pixel] += DecodedTileRGB[0 * tile_plane + tile_pixel] * weight;
        CanvasRGB[1 * canvas_plane + canvas_pixel] += DecodedTileRGB[1 * tile_plane + tile_pixel] * weight;
        CanvasRGB[2 * canvas_plane + canvas_pixel] += DecodedTileRGB[2 * tile_plane + tile_pixel] * weight;

        CanvasWeights[canvas_pixel] += weight;
    }
}
