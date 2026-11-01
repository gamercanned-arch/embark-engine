// vega8_gemm.cl
// High-Performance OpenCL GEMM for AMD Vega 8 (gfx90c, GCN 5.0)
// Architectural optimizations:
// 1. AMD Wave64 (64 threads per workgroup) - 1:1 mapping with hardware Wavefront.
// 2. LDS Tiling with bank-conflict-free stride (padding +1 to avoid 32-bank conflicts).
// 3. Global memory prefetch double-buffering into VGPRs.
// 4. DDR4 cache line alignment (64-byte burst vector loads: vload4 / vload8 / vload16).
// 5. Register micro-tiling (4x4 and 8x8 in VGPRs) maximizing arithmetic intensity.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// ============================================================================
// Kernel 1: FP32 GEMM 32x32 Output Tile with 4x4 Register Micro-Tile
// Local work size: (8, 8, 1) = 64 threads (Wave64)
// Tile M = 32, Tile N = 32, Tile K = 32
// Matrix A: [M, K], Matrix B: [K, N], Matrix C: [M, N]
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_gemm_fp32_32x32_4x4(
    const int M,
    const int N,
    const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C)
{
    // Workgroup and thread coordinates
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    const int block_m = get_group_id(1) * 32;
    const int block_n = get_group_id(0) * 32;

    // LDS tiles: 32x32 floats, padded to 33 to prevent 32-bank conflicts
    __local float lds_A[32][32 + LDS_PAD];
    __local float lds_B[32][32 + LDS_PAD];

    // Register accumulators: 4x4 micro-tile in VGPRs
    float c[4][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f}
    };

    // Each thread loads 16 floats (64 bytes = 1 DDR4 cache line) per tile for A and B
    // tid in 0..63 -> row = tid / 2 (0..31), col = (tid % 2) * 16 (0 or 16)
    const int load_row_a = tid >> 1;
    const int load_col_a = (tid & 1) << 4;
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 1;
    const int load_col_b = (tid & 1) << 4;
    const int global_col_b = block_n + load_col_b;

    // Prefetch registers for double buffering
    float8 pre_a0, pre_a1;
    float8 pre_b0, pre_b1;

    // Initial load: Tile 0 into registers
    const int num_tiles = K / 32;
    
    // Global addresses for tile 0
    __global const float* ptr_a = A + global_row_a * K + load_col_a;
    __global const float* ptr_b = B + load_row_b * N + global_col_b;

    pre_a0 = vload8(0, ptr_a);
    pre_a1 = vload8(0, ptr_a + 8);
    pre_b0 = vload8(0, ptr_b);
    pre_b1 = vload8(0, ptr_b + 8);

    // Store tile 0 into LDS
    vstore8(pre_a0, 0, &lds_A[load_row_a][load_col_a]);
    vstore8(pre_a1, 0, &lds_A[load_row_a][load_col_a + 8]);
    vstore8(pre_b0, 0, &lds_B[load_row_b][load_col_b]);
    vstore8(pre_b1, 0, &lds_B[load_row_b][load_col_b + 8]);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop with double buffering
    for (int t = 0; t < num_tiles; t++)
    {
        const int next_k = (t + 1) * 32;

        // Prefetch next tile into VGPRs while computing current LDS tile
        if (t + 1 < num_tiles)
        {
            __global const float* next_ptr_a = A + global_row_a * K + next_k + load_col_a;
            __global const float* next_ptr_b = B + (next_k + load_row_b) * N + global_col_b;

            pre_a0 = vload8(0, next_ptr_a);
            pre_a1 = vload8(0, next_ptr_a + 8);
            pre_b0 = vload8(0, next_ptr_b);
            pre_b1 = vload8(0, next_ptr_b + 8);
        }

        // Compute 32 outer products from LDS
        const int thread_m = ty * 4;
        const int thread_n = tx * 4;

        #pragma unroll 8
        for (int k = 0; k < 32; k++)
        {
            float a0 = lds_A[thread_m + 0][k];
            float a1 = lds_A[thread_m + 1][k];
            float a2 = lds_A[thread_m + 2][k];
            float a3 = lds_A[thread_m + 3][k];

            float b0 = lds_B[k][thread_n + 0];
            float b1 = lds_B[k][thread_n + 1];
            float b2 = lds_B[k][thread_n + 2];
            float b3 = lds_B[k][thread_n + 3];

            c[0][0] = fma(a0, b0, c[0][0]);
            c[0][1] = fma(a0, b1, c[0][1]);
            c[0][2] = fma(a0, b2, c[0][2]);
            c[0][3] = fma(a0, b3, c[0][3]);

            c[1][0] = fma(a1, b0, c[1][0]);
            c[1][1] = fma(a1, b1, c[1][1]);
            c[1][2] = fma(a1, b2, c[1][2]);
            c[1][3] = fma(a1, b3, c[1][3]);

            c[2][0] = fma(a2, b0, c[2][0]);
            c[2][1] = fma(a2, b1, c[2][1]);
            c[2][2] = fma(a2, b2, c[2][2]);
            c[2][3] = fma(a2, b3, c[2][3]);

            c[3][0] = fma(a3, b0, c[3][0]);
            c[3][1] = fma(a3, b1, c[3][1]);
            c[3][2] = fma(a3, b2, c[3][2]);
            c[3][3] = fma(a3, b3, c[3][3]);
        }

        // Sync before overwriting LDS with next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        if (t + 1 < num_tiles)
        {
            vstore8(pre_a0, 0, &lds_A[load_row_a][load_col_a]);
            vstore8(pre_a1, 0, &lds_A[load_row_a][load_col_a + 8]);
            vstore8(pre_b0, 0, &lds_B[load_row_b][load_col_b]);
            vstore8(pre_b1, 0, &lds_B[load_row_b][load_col_b + 8]);

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Write back 4x4 micro-tile to global C
    const int out_m = block_m + ty * 4;
    const int out_n = block_n + tx * 4;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if (out_m + i < M)
        {
            float4 out_val = (float4)(c[i][0], c[i][1], c[i][2], c[i][3]);
            vstore4(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// ============================================================================
// Kernel 2: FP32 GEMM 64x64 Output Tile with 8x8 Register Micro-Tile
// Local work size: (8, 8, 1) = 64 threads (Wave64)
// Tile M = 64, Tile N = 64, Tile K = 16
// Arithmetic intensity: 64 FMAs per 16 LDS loads (4:1 ratio)
// Matrix A: [M, K], Matrix B: [K, N], Matrix C: [M, N]
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_gemm_fp32_64x64_8x8(
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

    // LDS tiles:
    // A: 64 rows x 16 cols. Pad column stride to 17.
    // B: 16 rows x 64 cols. Pad column stride to 65.
    __local float lds_A[64][16 + LDS_PAD];
    __local float lds_B[16][64 + LDS_PAD];

    // 8x8 micro-tile in VGPRs (64 registers)
    float c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c[i][j] = 0.0f;
        }
    }

    // Tile A load mapping: 64 rows x 16 cols = 1024 floats
    // 64 threads -> each loads 16 floats (one row of 16 floats)
    const int load_row_a = tid; // 0..63
    const int global_row_a = block_m + load_row_a;

    // Tile B load mapping: 16 rows x 64 cols = 1024 floats
    // 64 threads -> each loads 16 floats:
    // row = tid / 4 (0..15), col = (tid % 4) * 16 (0, 16, 32, 48)
    const int load_row_b = tid >> 2;
    const int load_col_b = (tid & 3) << 4;
    const int global_col_b = block_n + load_col_b;

    // Prefetch registers
    float8 pre_a0, pre_a1;
    float8 pre_b0, pre_b1;

    const int num_tiles = K / 16;

    // Initial load: Tile 0
    __global const float* ptr_a = A + global_row_a * K;
    __global const float* ptr_b = B + load_row_b * N + global_col_b;

    pre_a0 = vload8(0, ptr_a);
    pre_a1 = vload8(0, ptr_a + 8);
    pre_b0 = vload8(0, ptr_b);
    pre_b1 = vload8(0, ptr_b + 8);

    vstore8(pre_a0, 0, &lds_A[load_row_a][0]);
    vstore8(pre_a1, 0, &lds_A[load_row_a][8]);
    vstore8(pre_b0, 0, &lds_B[load_row_b][load_col_b]);
    vstore8(pre_b1, 0, &lds_B[load_row_b][load_col_b + 8]);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop over K
    for (int t = 0; t < num_tiles; t++)
    {
        const int next_k = (t + 1) * 16;

        // Prefetch next tile into VGPRs
        if (t + 1 < num_tiles)
        {
            __global const float* next_ptr_a = A + global_row_a * K + next_k;
            __global const float* next_ptr_b = B + (next_k + load_row_b) * N + global_col_b;

            pre_a0 = vload8(0, next_ptr_a);
            pre_a1 = vload8(0, next_ptr_a + 8);
            pre_b0 = vload8(0, next_ptr_b);
            pre_b1 = vload8(0, next_ptr_b + 8);
        }

        // Inner loop: 16 steps of outer products
        const int thread_m = ty * 8;
        const int thread_n = tx * 8;

        #pragma unroll 16
        for (int k = 0; k < 16; k++)
        {
            float a[8];
            a[0] = lds_A[thread_m + 0][k];
            a[1] = lds_A[thread_m + 1][k];
            a[2] = lds_A[thread_m + 2][k];
            a[3] = lds_A[thread_m + 3][k];
            a[4] = lds_A[thread_m + 4][k];
            a[5] = lds_A[thread_m + 5][k];
            a[6] = lds_A[thread_m + 6][k];
            a[7] = lds_A[thread_m + 7][k];

            float b[8];
            b[0] = lds_B[k][thread_n + 0];
            b[1] = lds_B[k][thread_n + 1];
            b[2] = lds_B[k][thread_n + 2];
            b[3] = lds_B[k][thread_n + 3];
            b[4] = lds_B[k][thread_n + 4];
            b[5] = lds_B[k][thread_n + 5];
            b[6] = lds_B[k][thread_n + 6];
            b[7] = lds_B[k][thread_n + 7];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    c[i][j] = fma(a[i], b[j], c[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (t + 1 < num_tiles)
        {
            vstore8(pre_a0, 0, &lds_A[load_row_a][0]);
            vstore8(pre_a1, 0, &lds_A[load_row_a][8]);
            vstore8(pre_b0, 0, &lds_B[load_row_b][load_col_b]);
            vstore8(pre_b1, 0, &lds_B[load_row_b][load_col_b + 8]);

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Write back 8x8 micro-tile to global memory C
    const int out_m = block_m + ty * 8;
    const int out_n = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        if (out_m + i < M)
        {
            float8 out_val = (float8)(
                c[i][0], c[i][1], c[i][2], c[i][3],
                c[i][4], c[i][5], c[i][6], c[i][7]
            );
            vstore8(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// ============================================================================
// Kernel 3: FP32 GEMM 64x64 Output Tile, K_TILE=32, 8x8 Register Micro-Tile
// Increased K-depth for higher arithmetic intensity (2048 FMAs per tile)
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_gemm_fp32_64x64_k32_8x8(
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

    // LDS tiles:
    // A: 64 rows x 32 cols -> 64 x 33 = 2112 floats (8.4 KB)
    // B: 32 rows x 64 cols -> 32 x 65 = 2080 floats (8.3 KB)
    __local float lds_A[64][32 + LDS_PAD];
    __local float lds_B[32][64 + LDS_PAD];

    float c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c[i][j] = 0.0f;
        }
    }

    // 64 threads loading 2048 floats: each thread loads 32 floats (4 x float8)
    // For A: 64 rows x 32 cols -> thread 'tid' loads 1 row of 32 cols!
    const int load_row_a = tid; // 0..63
    const int global_row_a = block_m + load_row_a;

    // For B: 32 rows x 64 cols -> each loads half a row (32 floats):
    // row = tid / 2 (0..31), col = (tid % 2) * 32 (0 or 32)
    const int load_row_b = tid >> 1;
    const int load_col_b = (tid & 1) << 5;
    const int global_col_b = block_n + load_col_b;

    float8 pre_a[4];
    float8 pre_b[4];

    const int num_tiles = K / 32;

    // Tile 0 load
    __global const float* ptr_a = A + global_row_a * K;
    __global const float* ptr_b = B + load_row_b * N + load_col_b;

    #pragma unroll
    for (int p = 0; p < 4; p++) {
        pre_a[p] = vload8(0, ptr_a + p * 8);
        pre_b[p] = vload8(0, ptr_b + p * 8);
    }

    #pragma unroll
    for (int p = 0; p < 4; p++) {
        vstore8(pre_a[p], 0, &lds_A[load_row_a][p * 8]);
        vstore8(pre_b[p], 0, &lds_B[load_row_b][load_col_b + p * 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < num_tiles; t++)
    {
        const int next_k = (t + 1) * 32;

        if (t + 1 < num_tiles)
        {
            __global const float* next_ptr_a = A + global_row_a * K + next_k;
            __global const float* next_ptr_b = B + (next_k + load_row_b) * N + load_col_b;

            #pragma unroll
            for (int p = 0; p < 4; p++) {
                pre_a[p] = vload8(0, next_ptr_a + p * 8);
                pre_b[p] = vload8(0, next_ptr_b + p * 8);
            }
        }

        const int thread_m = ty * 8;
        const int thread_n = tx * 8;

        #pragma unroll 32
        for (int k = 0; k < 32; k++)
        {
            float a[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a[i] = lds_A[thread_m + i][k];
            }

            float b[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                b[j] = lds_B[k][thread_n + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    c[i][j] = fma(a[i], b[j], c[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (t + 1 < num_tiles)
        {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                vstore8(pre_a[p], 0, &lds_A[load_row_a][p * 8]);
                vstore8(pre_b[p], 0, &lds_B[load_row_b][load_col_b + p * 8]);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    const int out_m = block_m + ty * 8;
    const int out_n = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        if (out_m + i < M)
        {
            float8 out_val = (float8)(
                c[i][0], c[i][1], c[i][2], c[i][3],
                c[i][4], c[i][5], c[i][6], c[i][7]
            );
            vstore8(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// ============================================================================
// Kernel 4: Transposed B (C = A * B^T) for LLM Linear Layer Weights
// Weights B are [N, K] (contiguous along K)
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_gemm_fp32_64x64_nt(
    const int M,
    const int N,
    const int K,
    __global const float* restrict A,
    __global const float* restrict B_T, // shape [N, K]
    __global float* restrict C)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * 8 + tx;

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // LDS tiles:
    // lds_A: [64][17]
    // lds_B: stored as [64][17] (since B_T has row length K, we load 64 rows of N x 16 cols of K)
    __local float lds_A[64][16 + LDS_PAD];
    __local float lds_B[64][16 + LDS_PAD];

    float c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c[i][j] = 0.0f;
        }
    }

    const int load_row = tid; // 0..63
    const int global_row_a = block_m + load_row;
    const int global_row_b = block_n + load_row; // row of B_T corresponds to N!

    float8 pre_a0, pre_a1;
    float8 pre_b0, pre_b1;

    const int num_tiles = K / 16;

    __global const float* ptr_a = A + global_row_a * K;
    __global const float* ptr_b = B_T + global_row_b * K;

    pre_a0 = vload8(0, ptr_a);
    pre_a1 = vload8(0, ptr_a + 8);
    pre_b0 = vload8(0, ptr_b);
    pre_b1 = vload8(0, ptr_b + 8);

    vstore8(pre_a0, 0, &lds_A[load_row][0]);
    vstore8(pre_a1, 0, &lds_A[load_row][8]);
    vstore8(pre_b0, 0, &lds_B[load_row][0]);
    vstore8(pre_b1, 0, &lds_B[load_row][8]);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < num_tiles; t++)
    {
        const int next_k = (t + 1) * 16;

        if (t + 1 < num_tiles)
        {
            __global const float* next_ptr_a = A + global_row_a * K + next_k;
            __global const float* next_ptr_b = B_T + global_row_b * K + next_k;

            pre_a0 = vload8(0, next_ptr_a);
            pre_a1 = vload8(0, next_ptr_a + 8);
            pre_b0 = vload8(0, next_ptr_b);
            pre_b1 = vload8(0, next_ptr_b + 8);
        }

        const int thread_m = ty * 8;
        const int thread_n = tx * 8;

        #pragma unroll 16
        for (int k = 0; k < 16; k++)
        {
            float a[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a[i] = lds_A[thread_m + i][k];
            }

            float b[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                b[j] = lds_B[thread_n + j][k]; // contiguous along K!
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    c[i][j] = fma(a[i], b[j], c[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (t + 1 < num_tiles)
        {
            vstore8(pre_a0, 0, &lds_A[load_row][0]);
            vstore8(pre_a1, 0, &lds_A[load_row][8]);
            vstore8(pre_b0, 0, &lds_B[load_row][0]);
            vstore8(pre_b1, 0, &lds_B[load_row][8]);

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    const int out_m = block_m + ty * 8;
    const int out_n = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        if (out_m + i < M)
        {
            float8 out_val = (float8)(
                c[i][0], c[i][1], c[i][2], c[i][3],
                c[i][4], c[i][5], c[i][6], c[i][7]
            );
            vstore8(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// ============================================================================
// Kernel 5: Half Precision (FP16) GEMM 64x64 Output Tile with 8x8 Micro-Tile
// Uses GCN Rapid Packed Math FP16 operations
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void vega8_gemm_fp16_64x64_8x8(
    const int M,
    const int N,
    const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * 8 + tx;

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // LDS tiles: 64 rows x 32 cols of half
    __local half lds_A[64][32 + LDS_PAD];
    __local half lds_B[32][64 + LDS_PAD];

    half c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c[i][j] = 0.0h;
        }
    }

    const int load_row_a = tid;
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 1;
    const int load_col_b = (tid & 1) << 5;
    const int global_col_b = block_n + load_col_b;

    half16 pre_a[2];
    half16 pre_b[2];

    const int num_tiles = K / 32;

    __global const half* ptr_a = A + global_row_a * K;
    __global const half* ptr_b = B + load_row_b * N + load_col_b;

    pre_a[0] = vload16(0, ptr_a);
    pre_a[1] = vload16(0, ptr_a + 16);
    pre_b[0] = vload16(0, ptr_b);
    pre_b[1] = vload16(0, ptr_b + 16);

    vstore16(pre_a[0], 0, &lds_A[load_row_a][0]);
    vstore16(pre_a[1], 0, &lds_A[load_row_a][16]);
    vstore16(pre_b[0], 0, &lds_B[load_row_b][load_col_b]);
    vstore16(pre_b[1], 0, &lds_B[load_row_b][load_col_b + 16]);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < num_tiles; t++)
    {
        const int next_k = (t + 1) * 32;

        if (t + 1 < num_tiles)
        {
            __global const half* next_ptr_a = A + global_row_a * K + next_k;
            __global const half* next_ptr_b = B + (next_k + load_row_b) * N + load_col_b;

            pre_a[0] = vload16(0, next_ptr_a);
            pre_a[1] = vload16(0, next_ptr_a + 16);
            pre_b[0] = vload16(0, next_ptr_b);
            pre_b[1] = vload16(0, next_ptr_b + 16);
        }

        const int thread_m = ty * 8;
        const int thread_n = tx * 8;

        #pragma unroll 32
        for (int k = 0; k < 32; k++)
        {
            half a[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a[i] = lds_A[thread_m + i][k];
            }

            half b[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                b[j] = lds_B[k][thread_n + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    c[i][j] = fma(a[i], b[j], c[i][j]);
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

    const int out_m = block_m + ty * 8;
    const int out_n = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        if (out_m + i < M)
        {
            half8 out_val = (half8)(
                c[i][0], c[i][1], c[i][2], c[i][3],
                c[i][4], c[i][5], c[i][6], c[i][7]
            );
            vstore8(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// ============================================================================
// Baseline Stock GEMM (Generic OpenCL Implementation without Wave64/LDS tuning)
// Represents typical stock 0.27 TFLOPs baseline for direct comparison
// ============================================================================
__kernel void gemm_stock_baseline(
    const int M,
    const int N,
    const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C)
{
    const int row = get_global_id(1);
    const int col = get_global_id(0);

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
