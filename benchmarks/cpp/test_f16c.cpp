#include <immintrin.h>
#include <iostream>

int main() {
    __m128i h = _mm_setzero_si128();
    __m256 f = _mm256_cvtph_ps(h);
    float buf[8];
    _mm256_storeu_ps(buf, f);
    std::cout << "F16C working, val = " << buf[0] << std::endl;
    return 0;
}
