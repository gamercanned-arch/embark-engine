// flux2_rope_lut.h
// Precomputed Static 704 KiB 3D RoPE Table for FLUX.2 Klein 4B DiT
// Text: L=512, (i, 0, 0)
// Latents: L=2304 (48x48), (0, h, w)
// Head dimension: 128 (32 text freq channels + 48 height channels + 48 width channels)

#pragma once

#include <vector>
#include <cmath>
#include <cstdint>

namespace flux2 {

constexpr int HEAD_DIM = 128;
constexpr int TEXT_TOKENS = 512;
constexpr int IMAGE_TOKENS = 2304; // 48 x 48
constexpr int TOTAL_TOKENS = TEXT_TOKENS + IMAGE_TOKENS; // 2816
constexpr float ROPE_THETA = 10000.0f;

// Precomputed FP16 cos/sin table: TOTAL_TOKENS x HEAD_DIM x 2 (cos and sin)
// 2816 x 128 x 2 bytes = 704 KiB
struct RoPE3DTable {
    // Flattened buffer: [TOTAL_TOKENS, HEAD_DIM] for cos, followed by [TOTAL_TOKENS, HEAD_DIM] for sin
    std::vector<uint16_t> cos_table_f16;
    std::vector<uint16_t> sin_table_f16;

    static inline uint16_t float_to_fp16(float f) {
        uint32_t x;
        memcpy(&x, &f, sizeof(float));
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (x >> 13) & 0x3FF;

        if (exp <= 0) {
            return (uint16_t)sign;
        } else if (exp >= 31) {
            return (uint16_t)(sign | 0x7C00);
        }
        return (uint16_t)(sign | (exp << 10) | mant);
    }

    void init() {
        cos_table_f16.resize(TOTAL_TOKENS * HEAD_DIM);
        sin_table_f16.resize(TOTAL_TOKENS * HEAD_DIM);

        // Precompute frequency scales
        // Text: 32 channels -> 16 frequency pairs
        std::vector<float> text_inv_freq(16);
        for (int i = 0; i < 16; i++) {
            text_inv_freq[i] = 1.0f / std::pow(ROPE_THETA, (float)(i * 2) / 32.0f);
        }

        // Height & Width: 48 channels each -> 24 frequency pairs each
        std::vector<float> spatial_inv_freq(24);
        for (int i = 0; i < 24; i++) {
            spatial_inv_freq[i] = 1.0f / std::pow(ROPE_THETA, (float)(i * 2) / 48.0f);
        }

        // 1. Text Tokens [0..511]: position i, coords (i, 0, 0)
        for (int i = 0; i < TEXT_TOKENS; i++) {
            int offset = i * HEAD_DIM;
            // First 32 dims: text frequency pairs
            for (int p = 0; p < 16; p++) {
                float theta = (float)i * text_inv_freq[p];
                float c = std::cos(theta);
                float s = std::sin(theta);
                cos_table_f16[offset + p * 2 + 0] = float_to_fp16(c);
                cos_table_f16[offset + p * 2 + 1] = float_to_fp16(c);
                sin_table_f16[offset + p * 2 + 0] = float_to_fp16(s);
                sin_table_f16[offset + p * 2 + 1] = float_to_fp16(s);
            }
            // Next 96 dims: spatial coords are 0 -> theta = 0 -> cos=1, sin=0
            for (int d = 32; d < HEAD_DIM; d++) {
                cos_table_f16[offset + d] = float_to_fp16(1.0f);
                sin_table_f16[offset + d] = float_to_fp16(0.0f);
            }
        }

        // 2. Image Tokens [512..2815]: position (h, w), coords (0, h, w)
        for (int img_idx = 0; img_idx < IMAGE_TOKENS; img_idx++) {
            int token_idx = TEXT_TOKENS + img_idx;
            int offset = token_idx * HEAD_DIM;
            int h = img_idx / 48; // row 0..47
            int w = img_idx % 48; // col 0..47

            // First 32 dims: text coord is 0 -> theta = 0 -> cos=1, sin=0
            for (int d = 0; d < 32; d++) {
                cos_table_f16[offset + d] = float_to_fp16(1.0f);
                sin_table_f16[offset + d] = float_to_fp16(0.0f);
            }

            // Next 48 dims (32..79): height frequency pairs
            for (int p = 0; p < 24; p++) {
                float theta = (float)h * spatial_inv_freq[p];
                float c = std::cos(theta);
                float s = std::sin(theta);
                cos_table_f16[offset + 32 + p * 2 + 0] = float_to_fp16(c);
                cos_table_f16[offset + 32 + p * 2 + 1] = float_to_fp16(c);
                sin_table_f16[offset + 32 + p * 2 + 0] = float_to_fp16(s);
                sin_table_f16[offset + 32 + p * 2 + 1] = float_to_fp16(s);
            }

            // Final 48 dims (80..127): width frequency pairs
            for (int p = 0; p < 24; p++) {
                float theta = (float)w * spatial_inv_freq[p];
                float c = std::cos(theta);
                float s = std::sin(theta);
                cos_table_f16[offset + 80 + p * 2 + 0] = float_to_fp16(c);
                cos_table_f16[offset + 80 + p * 2 + 1] = float_to_fp16(c);
                sin_table_f16[offset + 80 + p * 2 + 0] = float_to_fp16(s);
                sin_table_f16[offset + 80 + p * 2 + 1] = float_to_fp16(s);
            }
        }
    }
};

} // namespace flux2
