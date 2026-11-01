// vega8_groot_gemm.cl
// AMD Radeon Vega 8 (gfx90c, GCN 5.0) Rapid Packed Math (FP16) Wave64 LDS GEMM Megakernel
// Tailored for Groot 19B-A3B Prefill & Vision Projector Inference

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// Wave64 Tile configuration:
// 64 work-items per workgroup (8x8)
// Output tile: 32x32 per workgroup
// Each work-item computes a 4x4 sub-tile (using 4 x half2 registers)
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_half2_wave64_lds(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wavefront Wave64)

    const int block_m = get_group_id(1) * 32;
    const int block_n = get_group_id(0) * 32;

    __local half lds_A_T[32][32 + LDS_PAD];
    __local half2 lds_B2[32][16 + LDS_PAD];

    half2 c[4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[i][0] = (half2)(0.0h, 0.0h);
        c[i][1] = (half2)(0.0h, 0.0h);
    }

    const int load_row_a = tid >> 1; // 0..31
    const int load_col_a = (tid & 1) << 4; // 0 or 16

    const int load_row_b = tid >> 1; // 0..31
    const int load_col_b2 = (tid & 1) << 3; // 0 or 8 half2 (16 halfs)

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 32;

        // Vectorized 16-half loads into LDS with transpose for A
        int a_idx = (block_m + load_row_a) * K + k_tile + load_col_a;
        half16 a_raw = vload16(0, &A[a_idx]);
        #pragma unroll
        for (int p = 0; p < 16; p++) {
            lds_A_T[load_col_a + p][load_row_a] = ((half*)&a_raw)[p];
        }

        // Vectorized loads into LDS for B
        int b_idx = (k_tile + load_row_b) * N + block_n + load_col_b2 * 2;
        half16 b_raw = vload16(0, &B[b_idx]);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            lds_B2[load_row_b][load_col_b2 + p] = (half2)(((half*)&b_raw)[p*2], ((half*)&b_raw)[p*2 + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int row_m = ty * 4;
        const int col_n2 = tx * 2;

        // Rapid Packed Math FP16 dual-issue FMA loop
        #pragma unroll 32
        for (int k = 0; k < 32; k++)
        {
            half a0 = lds_A_T[k][row_m + 0];
            half a1 = lds_A_T[k][row_m + 1];
            half a2 = lds_A_T[k][row_m + 2];
            half a3 = lds_A_T[k][row_m + 3];

            half2 b0 = lds_B2[k][col_n2 + 0];
            half2 b1 = lds_B2[k][col_n2 + 1];

            c[0][0] = fma((half2)(a0, a0), b0, c[0][0]);
            c[0][1] = fma((half2)(a0, a0), b1, c[0][1]);

            c[1][0] = fma((half2)(a1, a1), b0, c[1][0]);
            c[1][1] = fma((half2)(a1, a1), b1, c[1][1]);

            c[2][0] = fma((half2)(a2, a2), b0, c[2][0]);
            c[2][1] = fma((half2)(a2, a2), b1, c[2][1]);

            c[3][0] = fma((half2)(a3, a3), b0, c[3][0]);
            c[3][1] = fma((half2)(a3, a3), b1, c[3][1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int out_m = block_m + ty * 4;
    const int out_n = block_n + tx * 4;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if (out_m + i < M && out_n < N) {
            half4 out_val = (half4)(
                c[i][0].x, c[i][0].y,
                c[i][1].x, c[i][1].y
            );
            vstore4(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// Vision Projector FP16 GEMM with direct bias addition & SiLU activation
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_vision_fp16(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global const float* restrict bias,
    __global half* restrict C,
    const int apply_silu)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * 8 + tx;

    const int block_m = get_group_id(1) * 32;
    const int block_n = get_group_id(0) * 32;

    __local half lds_A_T[32][32 + LDS_PAD];
    __local half2 lds_B2[32][16 + LDS_PAD];

    half2 c[4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[i][0] = (half2)(0.0h, 0.0h);
        c[i][1] = (half2)(0.0h, 0.0h);
    }

    const int load_row_a = tid >> 1;
    const int load_col_a = (tid & 1) << 4;

    const int load_row_b = tid >> 1;
    const int load_col_b2 = (tid & 1) << 3;

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 32;

        int a_idx = (block_m + load_row_a) * K + k_tile + load_col_a;
        half16 a_raw = vload16(0, &A[a_idx]);
        #pragma unroll
        for (int p = 0; p < 16; p++) {
            lds_A_T[load_col_a + p][load_row_a] = ((half*)&a_raw)[p];
        }

        int b_idx = (k_tile + load_row_b) * N + block_n + load_col_b2 * 2;
        half16 b_raw = vload16(0, &B[b_idx]);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            lds_B2[load_row_b][load_col_b2 + p] = (half2)(((half*)&b_raw)[p*2], ((half*)&b_raw)[p*2 + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int row_m = ty * 4;
        const int col_n2 = tx * 2;

        #pragma unroll 32
        for (int k = 0; k < 32; k++)
        {
            half a0 = lds_A_T[k][row_m + 0];
            half a1 = lds_A_T[k][row_m + 1];
            half a2 = lds_A_T[k][row_m + 2];
            half a3 = lds_A_T[k][row_m + 3];

            half2 b0 = lds_B2[k][col_n2 + 0];
            half2 b1 = lds_B2[k][col_n2 + 1];

            c[0][0] = fma((half2)(a0, a0), b0, c[0][0]);
            c[0][1] = fma((half2)(a0, a0), b1, c[0][1]);

            c[1][0] = fma((half2)(a1, a1), b0, c[1][0]);
            c[1][1] = fma((half2)(a1, a1), b1, c[1][1]);

            c[2][0] = fma((half2)(a2, a2), b0, c[2][0]);
            c[2][1] = fma((half2)(a2, a2), b1, c[2][1]);

            c[3][0] = fma((half2)(a3, a3), b0, c[3][0]);
            c[3][1] = fma((half2)(a3, a3), b1, c[3][1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int out_m = block_m + ty * 4;
    const int out_n = block_n + tx * 4;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if (out_m + i < M && out_n < N) {
            half4 out_val = (half4)(c[i][0].x, c[i][0].y, c[i][1].x, c[i][1].y);
            if (bias) {
                float4 b = vload4(0, &bias[out_n]);
                out_val += (half4)((half)b.x, (half)b.y, (half)b.z, (half)b.w);
            }
            if (apply_silu) {
                // silu(x) = x / (1 + exp(-x))
                float4 f = (float4)(out_val.x, out_val.y, out_val.z, out_val.w);
                float4 sig = (float4)1.0f / ((float4)1.0f + exp(-f));
                float4 res = f * sig;
                out_val = (half4)((half)res.x, (half)res.y, (half)res.z, (half)res.w);
            }
            vstore4(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}
