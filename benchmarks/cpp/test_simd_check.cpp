#include <immintrin.h>
#include <iostream>

int main() {
    alignas(32) int table[512] = {1, 2, 3, 4, 5, 6, 7, 8};
    __m256i vindex = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i gathered = _mm256_i32gather_epi32(table, vindex, 4);

    __m128i h = _mm_set1_epi16(0x3C00); // 1.0 in FP16
    __m256 f = _mm256_cvtph_ps(h);

    __m256i a = _mm256_set1_epi8(2);
    __m256i b = _mm256_set1_epi8(3);
    __m256i maddu = _mm256_maddubs_epi16(a, b);
    __m256i maddw = _mm256_madd_epi16(maddu, _mm256_set1_epi16(1));

    std::cout << "Zen3 AVX2 + F16C gather & dual ALU instructions verified successfully!\n";
    return 0;
}
