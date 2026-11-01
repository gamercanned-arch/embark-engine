// vega8_gemma4_e4b_gemm.cl
// High-Performance OpenCL GEMM for AMD Vega 8 (gfx90c, GCN 5.0)
// Tailored for Gemma 4 E4B Prefill Matrices (d_model=2560, d_ff=10240, heads=8/2)
// Microarchitectural optimizations:
// 1. AMD Wave64 (8x8 = 64 threads) - 1:1 hardware Wavefront mapping.
// 2. LDS Tiling with bank-conflict-free stride (LDS_PAD = 1).
// 3. Global memory prefetch double-buffering in VGPRs.
// 4. DDR4 64-byte cache line burst loads (vload8 / vload16).
// 5. 8x8 VGPR micro-tiling yielding 64 FMAs per 16 LDS loads (4:1 arithmetic intensity).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// ============================================================================
// Kernel 1: Vega 8 FP16 GEMM 64x64 Output Tile (GCN Rapid Packed Math Peak: 4.1 TFLOPS)
// Matrix A: [M, K], Matrix B: [K, N], Matrix C: [M, N]
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_gemma4_e4b_gemm_fp16_64x64_8x8(
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

    // Write back 8x8 output tile
    const int out_row_base = block_m + ty * 8;
    const int out_col_base = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int out_r = out_row_base + i;
        if (out_r < M) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_c = out_col_base + j;
                if (out_c < N) {
                    C[out_r * N + out_c] = c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Kernel 2: Vega 8 FP32 GEMM 64x64 Output Tile (Fallback for high-precision projection)
// Matrix A: [M, K], Matrix B: [K, N], Matrix C: [M, N]
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_gemma4_e4b_gemm_fp32_64x64_8x8(
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

    // LDS tiles: 64 rows x 16 cols of float
    __local float lds_A[64][16 + LDS_PAD];
    __local float lds_B[16][64 + LDS_PAD];

    // 8x8 VGPR accumulators
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

    float16 pre_a;
    float16 pre_b;

    const int num_tiles = K / 16;

    // Initial load: Tile 0 into registers
    __global const float* ptr_a = A + global_row_a * K;
    __global const float* ptr_b = B + load_row_b * N + global_col_b;

    pre_a = vload16(0, ptr_a);
    pre_b = vload16(0, ptr_b);

    // Store tile 0 into LDS
    vstore16(pre_a, 0, &lds_A[load_row_a][0]);
    vstore16(pre_b, 0, &lds_B[load_row_b][load_col_b]);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Double-buffered K-loop
    for (int t = 0; t < num_tiles; t++)
    {
        const int next_k = (t + 1) * 16;

        if (t + 1 < num_tiles)
        {
            __global const float* next_ptr_a = A + global_row_a * K + next_k;
            __global const float* next_ptr_b = B + (next_k + load_row_b) * N + global_col_b;

            pre_a = vload16(0, next_ptr_a);
            pre_b = vload16(0, next_ptr_b);
        }

        #pragma unroll 4
        for (int k = 0; k < 16; k++)
        {
            float a_vals[8];
            float b_vals[8];

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
            vstore16(pre_a, 0, &lds_A[load_row_a][0]);
            vstore16(pre_b, 0, &lds_B[load_row_b][load_col_b]);

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Write back 8x8 output tile
    const int out_row_base = block_m + ty * 8;
    const int out_col_base = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int out_r = out_row_base + i;
        if (out_r < M) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_c = out_col_base + j;
                if (out_c < N) {
                    C[out_r * N + out_c] = c[i][j];
                }
            }
        }
    }
}
