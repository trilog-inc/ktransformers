#ifndef CPUINFER_OPERATOR_MXFP4_UE8M0_H
#define CPUINFER_OPERATOR_MXFP4_UE8M0_H

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mxfp4 {

inline uint32_t ue8m0_fp32_bits(uint8_t exponent) { return static_cast<uint32_t>(exponent) << 23; }

inline float ue8m0_to_fp32(uint8_t exponent) {
  const uint32_t bits = ue8m0_fp32_bits(exponent);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

#if defined(__AVX512F__)
inline __m512 ue8m0_to_fp32x16(uint8_t exponent) {
  return _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int>(ue8m0_fp32_bits(exponent))));
}
#endif

inline size_t native_buffer_required_size(size_t n, size_t k, size_t group_size) {
  return n * k / 2 + n * (k / group_size);
}

}  // namespace mxfp4

#endif  // CPUINFER_OPERATOR_MXFP4_UE8M0_H
