// vega8_lfm2_gemm.cl
// High-Performance OpenCL Kernels for AMD Vega 8 (gfx90c, GCN 5.0)
// Tailored for LFM 2.5 350M (d_model=1024, d_ff=4608, 16 blocks, shortconv l_cache=3)
// Features:
// 1. AMD Wave64 (8x8 = 64 work items) 1:1 hardware Wavefront mapping.
// 2. LDS Tiling with bank-conflict-free stride (LDS_PAD = 1).
// 3. Global memory prefetch double-buffering in VGPRs.
// 4. DDR4 64-byte cache line burst loads (vload8 / vload16).
// 5. 8x8 VGPR micro-tiling yielding 64 FMAs per 16 LDS loads (4:1 arithmetic intensity).
// 6. Zero-copy UMA buffer mapping (CL_MEM_USE_HOST_PTR).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// ============================================================================
// Kernel 1: Vega 8 FP16 GEMM 64x64 Output Tile (Rapid Packed Math Peak: 4.1 TFLOPS)
// Matrix A: [M, K], Matrix B: [K, N], Matrix C: [M, N]
// Used for High-Throughput Prefill GEMM
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_lfm2_gemm_fp16_64x64_8x8(
    const int M,
    const int N,
    const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // LDS tiles: 64 rows x 32 cols of half (padded to 33 to prevent 32-bank conflicts)
    __local half lds_A[64][32 + LDS_PAD];
    __local half lds_B[32][64 + LDS_PAD];

    // 8x8 VGPR accumulators
    half c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c[i][j] = 0.0h;
        }
    }

    const int load_row_a = tid; // 0..63
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 1;        // 0..31
    const int load_col_b = (tid & 1) << 5;   // 0 or 32
    const int global_col_b = block_n + load_col_b;

    half16 pre_a[2];
    half16 pre_b[2];

    const int num_tiles = K / 32;

    // Initial load: Tile 0 into registers
    __global const half* ptr_a = A + global_row_a * K;
    __global const half* ptr_b = B + load_row_b * N + global_col_b;

    pre_a[0] = vload16(0, ptr_a);
    pre_a[1] = vload16(0, ptr_a + 16);
    pre_b[0] = vload16(0, ptr_b);
    pre_b[1] = vload16(0, ptr_b + 16);

    // Store tile 0 into LDS
    vstore16(pre_a[0], 0, &lds_A[load_row_a][0]);
    vstore16(pre_a[1], 0, &lds_A[load_row_a][16]);
    vstore16(pre_b[0], 0, &lds_B[load_row_b][load_col_b]);
    vstore16(pre_b[1], 0, &lds_B[load_row_b][load_col_b + 16]);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Double-buffered K-loop
    for (int t = 0; t < num_tiles; t++)
    {
        const int next_k = (t + 1) * 32;

        if (t + 1 < num_tiles)
        {
            __global const half* next_ptr_a = A + global_row_a * K + next_k;
            __global const half* next_ptr_b = B + (next_k + load_row_b) * N + global_col_b;

            pre_a[0] = vload16(0, next_ptr_a);
            pre_a[1] = vload16(0, next_ptr_a + 16);
            pre_b[0] = vload16(0, next_ptr_b);
            pre_b[1] = vload16(0, next_ptr_b + 16);
        }

        #pragma unroll 4
        for (int k = 0; k < 32; k++)
        {
            half a_vals[8];
            half b_vals[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a_vals[i] = lds_A[ty * 8 + i][k];
            }

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                b_vals[j] = lds_B[k][tx * 8 + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    c[i][j] = fma(a_vals[i], b_vals[j], c[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (t + 1 < num_tiles)
        {
            vstore16(pre_a[0], 0, &lds_A[load_row_a][0]);
            vstore16(pre_a[1], 0, &lds_A[load_row_a][16]);
            vstore16(pre_b[0], 0, &lds_B[load_row_b][load_col_b]);
            vstore16(pre_b[1], 0, &lds_B[load_row_b][load_col_b + 16]);

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Write out 8x8 result tile
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int row = block_m + ty * 8 + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int col = block_n + tx * 8 + j;
                if (col < N) {
                    C[row * N + col] = c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Kernel 2: Vega 8 FP32 GEMM 64x64 Output Tile (Peak: 2.05 TFLOPS)
// Matrix A: [M, K], Matrix B: [K, N], Matrix C: [M, N]
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_lfm2_gemm_fp32_64x64_8x8(
    const int M,
    const int N,
    const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    __local float lds_A[64][16 + LDS_PAD];
    __local float lds_B[16][64 + LDS_PAD];

    float c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c[i][j] = 0.0f;
        }
    }

    const int load_row_a = tid; // 0..63
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 2;        // 0..15
    const int load_col_b = (tid & 3) << 4;   // 0, 16, 32, 48
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 16;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_offset = t * 16;

        // Load A into LDS
        if (global_row_a < M) {
            float8 val_a0 = vload8(0, A + global_row_a * K + k_offset);
            float8 val_a1 = vload8(0, A + global_row_a * K + k_offset + 8);
            vstore8(val_a0, 0, &lds_A[load_row_a][0]);
            vstore8(val_a1, 0, &lds_A[load_row_a][8]);
        } else {
            #pragma unroll
            for (int k = 0; k < 16; k++) lds_A[load_row_a][k] = 0.0f;
        }

        // Load B into LDS
        if (global_col_b < N && (k_offset + load_row_b) < K) {
            float8 val_b0 = vload8(0, B + (k_offset + load_row_b) * N + global_col_b);
            float8 val_b1 = vload8(0, B + (k_offset + load_row_b) * N + global_col_b + 8);
            vstore8(val_b0, 0, &lds_B[load_row_b][load_col_b]);
            vstore8(val_b1, 0, &lds_B[load_row_b][load_col_b + 8]);
        } else {
            #pragma unroll
            for (int k = 0; k < 16; k++) lds_B[load_row_b][load_col_b + k] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll 4
        for (int k = 0; k < 16; k++)
        {
            float a_vals[8];
            float b_vals[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) a_vals[i] = lds_A[ty * 8 + i][k];
            #pragma unroll
            for (int j = 0; j < 8; j++) b_vals[j] = lds_B[k][tx * 8 + j];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    c[i][j] = fma(a_vals[i], b_vals[j], c[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write out results
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int row = block_m + ty * 8 + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int col = block_n + tx * 8 + j;
                if (col < N) {
                    C[row * N + col] = c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Kernel 3: Vega 8 ShortConv Prefill Kernel (K=3, d_model=1024)
// High-throughput 1D causal convolution across M tokens x 1024 channels
// ============================================================================
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void vega8_lfm2_shortconv_prefill(
    const int M,
    const int C,
    __global const float* restrict bx,       // [M, C]
    __global const float* restrict kernel_w, // [3, C]
    __global float* restrict conv_out)       // [M, C]
{
    const int ch = get_global_id(0); // 0..1023
    if (ch >= C) return;

    const float w0 = kernel_w[0 * C + ch];
    const float w1 = kernel_w[1 * C + ch];
    const float w2 = kernel_w[2 * C + ch];

    // Causal convolution with zero initial state for prefill
    float s0 = 0.0f; // t-2
    float s1 = 0.0f; // t-1

    for (int t = 0; t < M; t++) {
        float in_val = bx[t * C + ch];
        float out_val = w0 * s0 + w1 * s1 + w2 * in_val;

        // Circular update
        s0 = s1;
        s1 = in_val;

        conv_out[t * C + ch] = out_val;
    }
}
