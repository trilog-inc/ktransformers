// Native UE8M0 end-to-end parity for the AVX2 MXFP4 fallback kernel.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "../../avx2/mxfp4-moe.hpp"

static uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

int main() {
  using NativeKernel = avx2::GemmKernelAVX2MXFP4;
  using LegacyKernel = avx2::GemmKernelAVX2FP4FP32Scale;
  constexpr int m = 1;
  constexpr int n = 8;
  constexpr int k = 32;
  constexpr int group_size = 32;

  auto allocate_aligned = [](size_t size) {
    void* pointer = std::aligned_alloc(64, (size + 63) & ~size_t{63});
    assert(pointer != nullptr);
    return pointer;
  };

  void* native_a_mem = allocate_aligned(NativeKernel::BufferA::required_size(m, k, group_size));
  void* native_b_mem = allocate_aligned(NativeKernel::BufferB::required_size(n, k, group_size));
  void* native_c_mem = allocate_aligned(NativeKernel::BufferC::required_size(m, n));
  void* legacy_a_mem = allocate_aligned(LegacyKernel::BufferA::required_size(m, k, group_size));
  void* legacy_b_mem = allocate_aligned(LegacyKernel::BufferB::required_size(n, k, group_size));
  void* legacy_c_mem = allocate_aligned(LegacyKernel::BufferC::required_size(m, n));

  NativeKernel::BufferA native_a(m, k, group_size, native_a_mem);
  NativeKernel::BufferB native_b(n, k, group_size, native_b_mem);
  NativeKernel::BufferC native_c(m, n, native_c_mem);
  LegacyKernel::BufferA legacy_a(m, k, group_size, legacy_a_mem);
  LegacyKernel::BufferB legacy_b(n, k, group_size, legacy_b_mem);
  LegacyKernel::BufferC legacy_c(m, n, legacy_c_mem);

  ggml_bf16_t one_bf16;
  one_bf16.bits = 0x3F80;
  std::vector<ggml_bf16_t> activations(k, one_bf16);
  std::vector<uint8_t> packed_weights(static_cast<size_t>(n) * k / 2, 0x22);
  native_a.from_mat(m, activations.data(), 0, 1);
  legacy_a.from_mat(m, activations.data(), 0, 1);
  native_b.from_raw_mat(packed_weights.data(), 0, 1);
  legacy_b.from_raw_mat(packed_weights.data(), 0, 1);

  for (int exponent = 0; exponent <= 255; ++exponent) {
    const float decoded = mxfp4::ue8m0_to_fp32(static_cast<uint8_t>(exponent));
    std::fill(native_b.d_ue8m0, native_b.d_ue8m0 + n, static_cast<uint8_t>(exponent));
    std::fill(legacy_b.d, legacy_b.d + n, decoded);
    std::memset(native_c_mem, 0, NativeKernel::BufferC::required_size(m, n));
    std::memset(legacy_c_mem, 0, LegacyKernel::BufferC::required_size(m, n));

    avx2::gemm_mxfp4<NativeKernel>(m, n, k, native_a, native_b, native_c, 0, 1);
    avx2::gemm_mxfp4<LegacyKernel>(m, n, k, legacy_a, legacy_b, legacy_c, 0, 1);
    for (int row = 0; row < n; ++row) assert(bits(native_c.data[row]) == bits(legacy_c.data[row]));
  }

  std::free(native_a_mem);
  std::free(native_b_mem);
  std::free(native_c_mem);
  std::free(legacy_a_mem);
  std::free(legacy_b_mem);
  std::free(legacy_c_mem);

  std::cout << "AVX2 MXFP4 UE8M0 kernel parity passed for all 256 exponent encodings\n";
  return 0;
}
