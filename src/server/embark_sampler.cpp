#include "embark_sampler.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace embark {

Sampler::Sampler(uint32_t seed) : m_rng(seed) {}

int32_t Sampler::argmax(const float* logits, size_t vocab_size) {
    if (vocab_size == 0) return 0;

    __m256 max_vals = _mm256_loadu_ps(logits);
    __m256i max_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i idx_step = _mm256_set1_epi32(8);
    __m256i curr_indices = max_indices;

    size_t i = 8;
    for (; i + 8 <= vocab_size; i += 8) {
        curr_indices = _mm256_add_epi32(curr_indices, idx_step);
        __m256 v = _mm256_loadu_ps(logits + i);
        __m256 mask = _mm256_cmp_ps(v, max_vals, _CMP_GT_OQ);
        max_vals = _mm256_blendv_ps(max_vals, v, mask);
        max_indices = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(max_indices),
            _mm256_castsi256_ps(curr_indices),
            mask
        ));
    }

    alignas(32) float vals[8];
    alignas(32) int32_t idxs[8];
    _mm256_storeu_ps(vals, max_vals);
    _mm256_storeu_si256((__m256i*)idxs, max_indices);

    float best_val = vals[0];
    int32_t best_idx = idxs[0];
    for (int k = 1; k < 8; ++k) {
        if (vals[k] > best_val) {
            best_val = vals[k];
            best_idx = idxs[k];
        }
    }

    for (; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = (int32_t)i;
        }
    }

    return best_idx;
}

int32_t Sampler::sample(float* logits, size_t vocab_size, const std::vector<int32_t>& history, const SamplingParams& params) {
    if (vocab_size == 0) return 0;

    // Repetition penalty
    if (params.repetition_penalty != 1.0f && !history.empty()) {
        for (int32_t tok : history) {
            if (tok >= 0 && (size_t)tok < vocab_size) {
                if (logits[tok] < 0.0f) {
                    logits[tok] *= params.repetition_penalty;
                } else {
                    logits[tok] /= params.repetition_penalty;
                }
            }
        }
    }

    // Greedy decoding
    if (params.temperature <= 0.001f) {
        return argmax(logits, vocab_size);
    }

    // Temperature scaling
    float inv_temp = 1.0f / params.temperature;
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }

    // Find max for numerical stability
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Softmax
    double sum_exp = 0.0;
    std::vector<std::pair<float, int32_t>> probs;
    probs.reserve(vocab_size);

    for (size_t i = 0; i < vocab_size; ++i) {
        float p = std::exp(logits[i] - max_logit);
        probs.push_back({p, (int32_t)i});
        sum_exp += p;
    }

    float inv_sum = (float)(1.0 / sum_exp);
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i].first *= inv_sum;
    }

    // Top-K
    if (params.top_k > 0 && params.top_k < (int)vocab_size) {
        std::partial_sort(probs.begin(), probs.begin() + params.top_k, probs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        probs.resize(params.top_k);
    } else {
        std::sort(probs.begin(), probs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // Top-P (nucleus)
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        float cum_p = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum_p += probs[i].first;
            cutoff = i + 1;
            if (cum_p >= params.top_p) break;
        }
        probs.resize(cutoff);
    }

    // Re-normalize remaining candidates
    float sub_sum = 0.0f;
    for (const auto& p : probs) sub_sum += p.first;
    float r = std::generate_canonical<float, 10>(m_rng) * sub_sum;

    float acc = 0.0f;
    for (const auto& p : probs) {
        acc += p.first;
        if (acc >= r) {
            return p.second;
        }
    }

    return probs[0].second;
}

} // namespace embark
