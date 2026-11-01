#pragma once

#include <vector>
#include <cstdint>
#include <random>

namespace embark {

struct SamplingParams {
    float temperature = 0.7f;
    float top_p = 0.8f;
    int top_k = 40;
    float repetition_penalty = 1.05f;
    int max_tokens = 2048;
};

class Sampler {
public:
    Sampler(uint32_t seed = 42);

    // Samples next token ID from raw logits
    int32_t sample(float* logits, size_t vocab_size, const std::vector<int32_t>& history, const SamplingParams& params);

    // Fast AVX2 Argmax (greedy decoding)
    int32_t argmax(const float* logits, size_t vocab_size);

private:
    std::mt19937 m_rng;
};

} // namespace embark
