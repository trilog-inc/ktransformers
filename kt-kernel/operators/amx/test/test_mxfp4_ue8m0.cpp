// Native UE8M0 storage/decoding parity for the MXFP4 CPU backends.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "../fp4-moe.hpp"

static uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

int main() {
  constexpr int n = 32;
  constexpr int k = 32;
  constexpr int group_size = 32;
  const size_t native_size = mxfp4::native_buffer_required_size(n, k, group_size);
  const size_t legacy_size = static_cast<size_t>(n) * k / 2 + static_cast<size_t>(n) * sizeof(float);
  assert(native_size == static_cast<size_t>(n) * k / 2 + static_cast<size_t>(n));
  assert(legacy_size == static_cast<size_t>(n) * k / 2 + static_cast<size_t>(n) * sizeof(float));

  constexpr size_t pro_hidden = 7'168;
  constexpr size_t pro_intermediate = 3'072;
  const size_t pro_expert_bytes =
      2 * mxfp4::native_buffer_required_size(pro_intermediate, pro_hidden, 32) +
      mxfp4::native_buffer_required_size(pro_hidden, pro_intermediate, 32);
  const size_t pro_bank_bytes = pro_expert_bytes * 61 * 384;
  assert(pro_expert_bytes == 35'094'528);
  assert(pro_bank_bytes == 822'054'223'872ULL);

  for (int exponent = 0; exponent <= 255; ++exponent) {
    const uint32_t expected = static_cast<uint32_t>(exponent) << 23;

    const float scalar = mxfp4::ue8m0_to_fp32(static_cast<uint8_t>(exponent));
    assert(bits(scalar) == expected);

    const __m512 vector = mxfp4::ue8m0_to_fp32x16(static_cast<uint8_t>(exponent));
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, vector);
    for (float lane : lanes) assert(bits(lane) == expected);
  }

#if defined(__GNUC__) || defined(__clang__)
  if (!__builtin_cpu_supports("avx512bw")) {
    std::cout << "MXFP4 UE8M0 scalar/vector parity passed; kernel parity skipped (AVX-512BW unavailable)\n";
    return 0;
  }
#endif

  using NativeKernel = amx::GemmKernel224MXFP4SmallKGroup;
  using LegacyKernel = amx::GemmKernel224FP4FP32ScaleSmallKGroup;
  auto allocate_aligned = [](size_t size) {
    void* pointer = std::aligned_alloc(64, (size + 63) & ~size_t{63});
    assert(pointer != nullptr);
    return pointer;
  };

  void* native_a_mem = allocate_aligned(NativeKernel::BufferA::required_size(1, k));
  void* native_b_mem = allocate_aligned(NativeKernel::BufferB::required_size(n, k, group_size));
  void* native_c_mem = allocate_aligned(NativeKernel::BufferC::required_size(1, n));
  void* legacy_a_mem = allocate_aligned(LegacyKernel::BufferA::required_size(1, k));
  void* legacy_b_mem = allocate_aligned(LegacyKernel::BufferB::required_size(n, k, group_size, false));
  void* legacy_c_mem = allocate_aligned(LegacyKernel::BufferC::required_size(1, n));

  NativeKernel::BufferA native_a(1, k, native_a_mem);
  NativeKernel::BufferB native_b(n, k, group_size, native_b_mem);
  NativeKernel::BufferC native_c(1, n, native_c_mem);
  LegacyKernel::BufferA legacy_a(1, k, legacy_a_mem);
  LegacyKernel::BufferB legacy_b(n, k, group_size, legacy_b_mem, false);
  LegacyKernel::BufferC legacy_c(1, n, legacy_c_mem);

  ggml_bf16_t one_bf16;
  one_bf16.bits = 0x3F80;
  std::vector<ggml_bf16_t> activations(k, one_bf16);
  std::vector<uint8_t> packed_weights(static_cast<size_t>(n) * k / 2, 0x22);
  native_a.from_mat(1, activations.data(), 0, 1);
  legacy_a.from_mat(1, activations.data(), 0, 1);
  native_b.from_raw_mat(packed_weights.data(), 0, 1);
  legacy_b.from_raw_mat(packed_weights.data(), 0, 1);

  // Exercise the actual AMX/AVX-512 MXFP4 kernel for every scale encoding,
  // comparing native one-byte scales with the legacy resident FP32 path.
  for (int exponent = 0; exponent <= 255; ++exponent) {
    const float decoded = mxfp4::ue8m0_to_fp32(static_cast<uint8_t>(exponent));
    std::fill(native_b.d_ue8m0, native_b.d_ue8m0 + n, static_cast<uint8_t>(exponent));
    std::fill(legacy_b.d, legacy_b.d + n, decoded);
    std::memset(native_c_mem, 0, NativeKernel::BufferC::required_size(1, n));
    std::memset(legacy_c_mem, 0, LegacyKernel::BufferC::required_size(1, n));

    NativeKernel::fp4_mat_vec_kgroup(1, n, k, group_size, &native_a, &native_b, &native_c, 0, 1);
    LegacyKernel::fp4_mat_vec_kgroup(1, n, k, group_size, &legacy_a, &legacy_b, &legacy_c, 0, 1);
    for (int row = 0; row < n; ++row) assert(bits(native_c.c[row]) == bits(legacy_c.c[row]));
  }

  std::free(native_a_mem);
  std::free(native_b_mem);
  std::free(native_c_mem);
  std::free(legacy_a_mem);
  std::free(legacy_b_mem);
  std::free(legacy_c_mem);

  std::cout << "MXFP4 UE8M0 scalar, vector, and kernel parity passed for all 256 exponent encodings\n";
  return 0;
}
