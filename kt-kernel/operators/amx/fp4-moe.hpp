/**
 * @Description  : MXFP4 MoE operator — FP4 E2M1 weights × BF16 activations
 * @Author       : oql, Codex and Claude
 * @Date         : 2026-04-20
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Based on k2-moe.hpp (RAWINT4). Key differences from RAWINT4:
 *   Weight:   FP4 E2M1 (nibble-packed, same layout) → PSHUFB lookup → BF16
 *   Act:      BF16 direct (BufferABF16Impl, no online INT8 quantization)
 *   Dot prod: _mm512_dpbf16_ps (BF16×BF16→FP32) instead of _mm512_dpbssd_epi32
 *   Scale:    native E4M3 group-16 or FP32 group-32 weight scale
 **/
#ifndef CPUINFER_OPERATOR_AMX_FP4_MOE_H
#define CPUINFER_OPERATOR_AMX_FP4_MOE_H

#include <cstdlib>
#include <cstring>

#include "la/amx_raw_buffers.hpp"  // BufferABF16Impl
#include "moe_base.hpp"

namespace amx {

inline bool use_nvfp4_blocked_layout() {
#if defined(__AVX512BF16__)
  static const bool enabled = [] {
    const char* value = std::getenv("KT_NVFP4_BLOCKED_LAYOUT");
    if (value == nullptr || *value == '\0') return true;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "false") != 0;
  }();
  return enabled;
#else
  return false;
#endif
}

inline int nvfp4_decode_tile_batch() {
#if defined(__AVX512BF16__)
  static const int tile_batch = [] {
    const char* value = std::getenv("KT_NVFP4_DECODE_TILE_BATCH");
    return value != nullptr && std::strcmp(value, "2") == 0 ? 2 : 1;
  }();
  return tile_batch;
#else
  return 1;
#endif
}

inline int nvfp4_prefetch_groups() {
#if defined(__AVX512BF16__)
  static const int distance = [] {
    const char* value = std::getenv("KT_NVFP4_PREFETCH_GROUPS");
    if (value == nullptr || *value == '\0') return 0;
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 0;
  }();
  return distance;
#else
  return 0;
#endif
}

// Group-32 MXFP4 keeps the historical row-major representation. Native
// group-16 NVFP4 is reordered into 16-output tiles so one DPBF16 vector
// produces 16 output channels and their E4M3 scales can be applied together.
// The packed weights and one-byte scales retain the checkpoint's 0.5625-byte
// per-parameter footprint.
template <typename K>
struct BufferBMXFP4KGroupImpl {
  using dt = typename K::dt;
  dt* b;
  float* d;
  float tensor_scale = 1.0f;
  int n, k, k_group_size, k_group_count;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int NVFP4_N_TILE = 16;
  static constexpr int NVFP4_K_GROUP = 16;

  static size_t required_size(int n, int k, int k_group_size) {
    const size_t weight_bytes = (size_t)n * k / 2;
    const size_t scale_count = (size_t)n * (k / k_group_size);
    const size_t scale_bytes =
        scale_count * (k_group_size == NVFP4_K_GROUP ? sizeof(uint8_t) : sizeof(float));
    return (weight_bytes + scale_bytes + 63) & ~size_t{63};
  }

  BufferBMXFP4KGroupImpl(int n, int k, int k_group_size, void* ptr)
      : n(n), k(k), k_group_size(k_group_size), k_group_count(k / k_group_size) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    if (n % N_STEP || k % K::K_STEP || k % k_group_size) {
      throw std::runtime_error("MXFP4 buffer dimensions are not aligned");
    }
    b = reinterpret_cast<dt*>(ptr);
    d = reinterpret_cast<float*>(offset_pointer(b, (size_t)n * k / 2));
  }

  bool blocked_nvfp4() const {
    return k_group_size == NVFP4_K_GROUP && use_nvfp4_blocked_layout();
  }

  static std::pair<int, int> split_rows(int n, int ith, int nth) {
    const int block_count = (n + N_BLOCK - 1) / N_BLOCK;
    const int blocks_per_thread = (block_count + nth - 1) / nth;
    const int start = std::min(n, ith * blocks_per_thread * N_BLOCK);
    const int end = std::min(n, start + blocks_per_thread * N_BLOCK);
    return {start, end};
  }

  size_t nvfp4_weight_tile_offset(int n_begin, int k_group_begin) const {
    const int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    const int n_tile = (n_begin - n_block_begin) / NVFP4_N_TILE;
    const int k_group = k_group_begin / NVFP4_K_GROUP;
    const size_t n_tile_bytes = (size_t)NVFP4_N_TILE * k / 2;
    const size_t k_group_bytes = (size_t)NVFP4_N_TILE * NVFP4_K_GROUP / 2;
    return (size_t)n_block_begin * k / 2 + (size_t)n_tile * n_tile_bytes +
           (size_t)k_group * k_group_bytes;
  }

  size_t nvfp4_scale_tile_offset(int n_begin, int k_group_begin) const {
    const int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    const int n_tile = (n_begin - n_block_begin) / NVFP4_N_TILE;
    const int k_group = k_group_begin / NVFP4_K_GROUP;
    return (size_t)n_block_begin * k_group_count +
           (size_t)n_tile * NVFP4_N_TILE * k_group_count +
           (size_t)k_group * NVFP4_N_TILE;
  }

  const uint8_t* get_nvfp4_weight_tile(int n_begin, int k_group_begin) const {
    return reinterpret_cast<const uint8_t*>(b) +
           nvfp4_weight_tile_offset(n_begin, k_group_begin);
  }

  uint8_t* get_nvfp4_weight_tile(int n_begin, int k_group_begin) {
    return reinterpret_cast<uint8_t*>(b) +
           nvfp4_weight_tile_offset(n_begin, k_group_begin);
  }

  const uint8_t* get_nvfp4_scale_tile(int n_begin, int k_group_begin) const {
    return reinterpret_cast<const uint8_t*>(d) +
           nvfp4_scale_tile_offset(n_begin, k_group_begin);
  }

  uint8_t* get_nvfp4_scale_tile(int n_begin, int k_group_begin) {
    return reinterpret_cast<uint8_t*>(d) +
           nvfp4_scale_tile_offset(n_begin, k_group_begin);
  }

  // Transpose 16 output rows x 8 packed K-pair bytes into eight contiguous
  // 16-byte vectors. This is the exact layout consumed by the decode kernel.
  static void transpose_nvfp4_weight_tile(const uint8_t* src,
                                           size_t row_stride, uint8_t* dst) {
    __m128i rows[16];
    for (int row = 0; row < 16; ++row) {
      rows[row] = _mm_loadl_epi64(
          reinterpret_cast<const __m128i*>(src + (size_t)row * row_stride));
    }

    const __m128i a0 = _mm_unpacklo_epi8(rows[0], rows[1]);
    const __m128i a1 = _mm_unpacklo_epi8(rows[2], rows[3]);
    const __m128i a2 = _mm_unpacklo_epi8(rows[4], rows[5]);
    const __m128i a3 = _mm_unpacklo_epi8(rows[6], rows[7]);
    const __m128i a4 = _mm_unpacklo_epi8(rows[8], rows[9]);
    const __m128i a5 = _mm_unpacklo_epi8(rows[10], rows[11]);
    const __m128i a6 = _mm_unpacklo_epi8(rows[12], rows[13]);
    const __m128i a7 = _mm_unpacklo_epi8(rows[14], rows[15]);

    const __m128i b0 = _mm_unpacklo_epi16(a0, a1);
    const __m128i b1 = _mm_unpackhi_epi16(a0, a1);
    const __m128i b2 = _mm_unpacklo_epi16(a2, a3);
    const __m128i b3 = _mm_unpackhi_epi16(a2, a3);
    const __m128i b4 = _mm_unpacklo_epi16(a4, a5);
    const __m128i b5 = _mm_unpackhi_epi16(a4, a5);
    const __m128i b6 = _mm_unpacklo_epi16(a6, a7);
    const __m128i b7 = _mm_unpackhi_epi16(a6, a7);

    const __m128i c0 = _mm_unpacklo_epi32(b0, b2);
    const __m128i c1 = _mm_unpackhi_epi32(b0, b2);
    const __m128i c2 = _mm_unpacklo_epi32(b1, b3);
    const __m128i c3 = _mm_unpackhi_epi32(b1, b3);
    const __m128i c4 = _mm_unpacklo_epi32(b4, b6);
    const __m128i c5 = _mm_unpackhi_epi32(b4, b6);
    const __m128i c6 = _mm_unpacklo_epi32(b5, b7);
    const __m128i c7 = _mm_unpackhi_epi32(b5, b7);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 0 * 16),
                     _mm_unpacklo_epi64(c0, c4));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 1 * 16),
                     _mm_unpackhi_epi64(c0, c4));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2 * 16),
                     _mm_unpacklo_epi64(c1, c5));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 3 * 16),
                     _mm_unpackhi_epi64(c1, c5));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 4 * 16),
                     _mm_unpacklo_epi64(c2, c6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 5 * 16),
                     _mm_unpackhi_epi64(c2, c6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 6 * 16),
                     _mm_unpacklo_epi64(c3, c7));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 7 * 16),
                     _mm_unpackhi_epi64(c3, c7));
  }

  void from_nvfp4_rows(const uint8_t* weights, size_t weight_row_stride,
                       const uint8_t* scales, size_t scale_row_stride,
                       int ith, int nth) {
    if (k_group_size != NVFP4_K_GROUP) {
      throw std::runtime_error("from_nvfp4_rows requires group_size=16");
    }
    auto [n_start, n_end] = split_rows(n, ith, nth);
    if (n_start >= n_end) return;

    if (!blocked_nvfp4()) {
      const size_t row_bytes = (size_t)k / 2;
      for (int row = n_start; row < n_end; ++row) {
        std::memcpy(reinterpret_cast<uint8_t*>(b) + (size_t)row * row_bytes,
                    weights + (size_t)row * weight_row_stride, row_bytes);
        if (scales != nullptr) {
          std::memcpy(reinterpret_cast<uint8_t*>(d) + (size_t)row * k_group_count,
                      scales + (size_t)row * scale_row_stride, k_group_count);
        }
      }
      return;
    }

    for (int n_begin = n_start; n_begin < n_end; n_begin += NVFP4_N_TILE) {
      for (int k_begin = 0; k_begin < k; k_begin += NVFP4_K_GROUP) {
        uint8_t* weight_tile = get_nvfp4_weight_tile(n_begin, k_begin);
        transpose_nvfp4_weight_tile(
            weights + (size_t)n_begin * weight_row_stride + k_begin / 2,
            weight_row_stride, weight_tile);
        if (scales != nullptr) {
          uint8_t* scale_tile = get_nvfp4_scale_tile(n_begin, k_begin);
          const int group = k_begin / NVFP4_K_GROUP;
          for (int lane = 0; lane < NVFP4_N_TILE; ++lane) {
            scale_tile[lane] =
                scales[(size_t)(n_begin + lane) * scale_row_stride + group];
          }
        }
      }
    }
  }

  void from_raw_mat(const uint8_t* weights, int ith, int nth) {
    if (k_group_size == NVFP4_K_GROUP) {
      from_nvfp4_rows(weights, (size_t)k / 2, nullptr, 0, ith, nth);
      return;
    }
    auto [n_start, n_end] = split_rows(n, ith, nth);
    const size_t row_bytes = (size_t)k / 2;
    std::memcpy(reinterpret_cast<uint8_t*>(b) + (size_t)n_start * row_bytes,
                weights + (size_t)n_start * row_bytes,
                (size_t)(n_end - n_start) * row_bytes);
  }

  dt* get_submat(int, int k_, int n_begin, int k_begin) {
    return reinterpret_cast<dt*>(reinterpret_cast<uint8_t*>(b) +
                                 (size_t)n_begin * k_ / 2 + k_begin / 2);
  }

  float* get_scale(int, int n_begin, int k_, int k_begin) {
    return d + (size_t)n_begin * (k_ / k_group_size) + k_begin / k_group_size;
  }

  uint8_t* get_native_scale(int, int n_begin, int k_, int k_begin) {
    return reinterpret_cast<uint8_t*>(d) +
           (size_t)n_begin * (k_ / k_group_size) + k_begin / k_group_size;
  }

  const uint8_t* get_native_scale(int, int n_begin, int k_, int k_begin) const {
    return reinterpret_cast<const uint8_t*>(d) +
           (size_t)n_begin * (k_ / k_group_size) + k_begin / k_group_size;
  }
};

// ============================================================================
// MXFP4 kernel: FP4 E2M1 weights × BF16 activations → FP32 output (AVX512)
// ============================================================================
struct GemmKernel224MXFP4SmallKGroup {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 0.5;

  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = 32;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "MXFP4_KGROUP"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static void config() {}

  // FP4 E2M1 → BF16 lookup table. The second half is duplicated because
  // VPERMW indexes all 32 lanes of a ZMM register.
  // E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
  alignas(64) static constexpr uint16_t fp4_bf16[32] = {
      0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
      0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
      0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
      0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};

  // Convert 16 packed FP4 bytes (32 values = 1 k_group) → 32 BF16 values (__m512i)
  // Output column order: [BF16(lo[0]),BF16(hi[0]), ..., BF16(lo[15]),BF16(hi[15])]
  __attribute__((always_inline)) static inline __m512i mxfp4_to_bf16_32(__m128i packed) {
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(packed, lo_mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

    // Build the 32 nibble indexes in matrix-column order, widen them, then
    // perform the complete E2M1-to-BF16 conversion with one VPERMW.
    const __m128i idx_lo = _mm_unpacklo_epi8(lo, hi);
    const __m128i idx_hi = _mm_unpackhi_epi8(lo, hi);
    const __m256i idx8 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(idx_lo), idx_hi, 1);
    const __m512i idx16 = _mm512_cvtepu8_epi16(idx8);
    const __m512i lut = _mm512_load_si512(fp4_bf16);
    return _mm512_permutexvar_epi16(idx16, lut);
  }

  struct ActivationBF16 {
    __m512bh a;
#if !defined(__AVX512BF16__)
    __m512 a_even;
    __m512 a_odd;
    inline static const __m512i odd_mask = _mm512_set1_epi32(0xFFFF0000);
#endif

    __attribute__((always_inline)) ActivationBF16(__m512bh a_) : a(a_) {
#if !defined(__AVX512BF16__)
      a_even = _mm512_castsi512_ps(_mm512_slli_epi32((__m512i)a_, 16));
      a_odd = _mm512_castsi512_ps(_mm512_and_si512((__m512i)a_, odd_mask));
#endif
    }
  };

  struct DequantizedWeight {
#if defined(__AVX512BF16__)
    __m512bh d;
#else
    __m512 w_even;
    __m512 w_odd;
    inline static const __m128i lo_mask = _mm_set1_epi8(0x0F);
    inline static const __m512 lut = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f,
                                                    -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
#endif

    __attribute__((always_inline)) DequantizedWeight(__m128i w) {
#if defined(__AVX512BF16__)
      d = (__m512bh)mxfp4_to_bf16_32(w);
#else
      __m128i lo = _mm_and_si128(w, lo_mask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(w, 4), lo_mask);

      __m512i lo_32 = _mm512_cvtepu8_epi32(lo);
      __m512i hi_32 = _mm512_cvtepu8_epi32(hi);

      w_even = _mm512_permutexvar_ps(lo_32, lut);
      w_odd = _mm512_permutexvar_ps(hi_32, lut);
#endif
    }
  };

  __attribute__((always_inline)) static inline __m512 mxfp4_dot_bf16(const DequantizedWeight& w,
                                                                     const ActivationBF16& act) {
#if defined(__AVX512BF16__)
    return _mm512_dpbf16_ps(_mm512_setzero_ps(), act.a, w.d);
#else
    __m512 dot = _mm512_mul_ps(act.a_odd, w.w_odd);
    return _mm512_fmadd_ps(act.a_even, w.w_even, dot);
#endif
  }

  template <int GROUP_SIZE>
  __attribute__((always_inline)) static inline __m512 load_group_scales(const void* scale_data, int vector_idx) {
    static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 32);
    if constexpr (GROUP_SIZE == 32) {
      const float* scales = static_cast<const float*>(scale_data);
      return _mm512_set1_ps(scales[vector_idx]);
    } else {
      const uint8_t* scales = static_cast<const uint8_t*>(scale_data);
      const auto& lut = e4m3fn_lut();
      const __m512 first = _mm512_set1_ps(lut[scales[vector_idx * 2]]);
      const __m512 second = _mm512_set1_ps(lut[scales[vector_idx * 2 + 1]]);
      // DPBF16 produces 16 FP32 lanes, each reducing two adjacent BF16
      // products. Lanes 0..7 belong to the first NVFP4 group and lanes 8..15
      // to the second.
      return _mm512_mask_blend_ps(0xFF00, first, second);
    }
  }

  // Buffers
  using BufferA = BufferABF16Impl<GemmKernel224MXFP4SmallKGroup>;        // raw BF16, no quant
  using BufferB = BufferBMXFP4KGroupImpl<GemmKernel224MXFP4SmallKGroup>;
  using BufferC = BufferCReduceImpl<GemmKernel224MXFP4SmallKGroup>;      // FP32 reduce

#if defined(__AVX512BF16__)
  // Convert 16 native E4M3FN scale bytes directly to FP32. Normal values map
  // to IEEE exponents with one shift/add; only the eight subnormal magnitudes
  // need a small vector lookup. Weight scales are kept byte-packed in RAM.
  __attribute__((always_inline)) static inline __m512 e4m3fnx16_to_fp32(
      const uint8_t* packed) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m512i raw = _mm512_cvtepu8_epi32(bytes);
    const __m512i magnitude = _mm512_and_si512(raw, _mm512_set1_epi32(0x7F));
    const __m512i sign = _mm512_slli_epi32(
        _mm512_and_si512(raw, _mm512_set1_epi32(0x80)), 24);

    const __m512i normal_bits = _mm512_add_epi32(
        _mm512_slli_epi32(magnitude, 20), _mm512_set1_epi32(0x3C000000));
    const __m512 subnormal_lut = _mm512_setr_ps(
        0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
        0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    const __m512 subnormal = _mm512_permutexvar_ps(magnitude, subnormal_lut);
    const __mmask16 is_subnormal = _mm512_cmplt_epi32_mask(
        magnitude, _mm512_set1_epi32(8));
    const __m512 magnitude_value = _mm512_mask_blend_ps(
        is_subnormal, _mm512_castsi512_ps(normal_bits), subnormal);
    return _mm512_castsi512_ps(
        _mm512_or_si512(_mm512_castps_si512(magnitude_value), sign));
  }

  __attribute__((always_inline)) static inline __m512bh broadcast_bf16_pair(
      const ggml_bf16_t* values) {
    uint32_t pair;
    std::memcpy(&pair, values, sizeof(pair));
    return (__m512bh)_mm512_set1_epi32(pair);
  }

  // Decode-oriented kernel. B is arranged as eight packed K-pairs across 16
  // output rows. DPBF16 therefore accumulates 16 independent output channels
  // and one scale vector handles the complete group-16 tile.
  static void fp4_mat_vec_nvfp4_blocked(int m, int n, int k, BufferA* ba,
                                        BufferB* bb, BufferC* bc, int ith,
                                        int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    if (k > K_BLOCK) {
      throw std::runtime_error("Blocked NVFP4 kernel requires k <= K_BLOCK");
    }

    const __m512 tensor_scale = _mm512_set1_ps(bb->tensor_scale);
    for (int m_idx = 0; m_idx < m; ++m_idx) {
      const ggml_bf16_t* a_row = ba->get_submat(m, k, m_idx, 0);
      for (int n_pos = n_start; n_pos < n_end; n_pos += 16) {
        __m512 total = _mm512_setzero_ps();

        for (int k_begin = 0; k_begin < k; k_begin += 16) {
          const uint8_t* weights = bb->get_nvfp4_weight_tile(n_pos, k_begin);
          __m512 partial0 = _mm512_setzero_ps();
          __m512 partial1 = _mm512_setzero_ps();
          __m512 partial2 = _mm512_setzero_ps();
          __m512 partial3 = _mm512_setzero_ps();

#define NVFP4_DP_PAIR(PAIR, ACC)                                                   \
  do {                                                                             \
    const DequantizedWeight weight(                                                \
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + (PAIR) * 16))); \
    const ActivationBF16 activation(                                               \
        broadcast_bf16_pair(a_row + k_begin + (PAIR) * 2));                        \
    (ACC) = _mm512_dpbf16_ps((ACC), activation.a, weight.d);                       \
  } while (0)
          NVFP4_DP_PAIR(0, partial0);
          NVFP4_DP_PAIR(1, partial1);
          NVFP4_DP_PAIR(2, partial2);
          NVFP4_DP_PAIR(3, partial3);
          NVFP4_DP_PAIR(4, partial0);
          NVFP4_DP_PAIR(5, partial1);
          NVFP4_DP_PAIR(6, partial2);
          NVFP4_DP_PAIR(7, partial3);
#undef NVFP4_DP_PAIR

          const __m512 group_sum = _mm512_add_ps(
              _mm512_add_ps(partial0, partial1),
              _mm512_add_ps(partial2, partial3));
          const __m512 scales = e4m3fnx16_to_fp32(
              bb->get_nvfp4_scale_tile(n_pos, k_begin));
          total = _mm512_fmadd_ps(group_sum, scales, total);
        }

        float* output = bc->get_submat(m, n, m_idx, n_pos);
        _mm512_storeu_ps(output, _mm512_mul_ps(total, tensor_scale));
      }
    }
  }

  // Process two adjacent output tiles together. Both tiles use the same
  // activation pair, so its load and broadcast are paid once for 32 outputs.
  // Optional software prefetching is kept runtime-tunable because the useful
  // distance depends strongly on the CPU and memory topology.
  static void fp4_mat_vec_nvfp4_blocked_two_tiles(
      int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
      int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    if (k > K_BLOCK) {
      throw std::runtime_error("Blocked NVFP4 kernel requires k <= K_BLOCK");
    }
    assert((n_end - n_start) % 32 == 0);

    constexpr size_t WEIGHT_GROUP_BYTES =
        BufferB::NVFP4_N_TILE * BufferB::NVFP4_K_GROUP / 2;
    constexpr size_t SCALE_GROUP_BYTES = BufferB::NVFP4_N_TILE;
    const int group_count = k / BufferB::NVFP4_K_GROUP;
    const int prefetch_groups = nvfp4_prefetch_groups();
    const __m512 tensor_scale = _mm512_set1_ps(bb->tensor_scale);

    for (int m_idx = 0; m_idx < m; ++m_idx) {
      const ggml_bf16_t* a_row = ba->get_submat(m, k, m_idx, 0);
      for (int n_pos = n_start; n_pos < n_end; n_pos += 32) {
        const uint8_t* weight_base0 = bb->get_nvfp4_weight_tile(n_pos, 0);
        const uint8_t* weight_base1 =
            bb->get_nvfp4_weight_tile(n_pos + 16, 0);
        const uint8_t* scale_base0 = bb->get_nvfp4_scale_tile(n_pos, 0);
        const uint8_t* scale_base1 =
            bb->get_nvfp4_scale_tile(n_pos + 16, 0);
        __m512 total0 = _mm512_setzero_ps();
        __m512 total1 = _mm512_setzero_ps();

        for (int group = 0; group < group_count; ++group) {
          const size_t weight_offset = (size_t)group * WEIGHT_GROUP_BYTES;
          const size_t scale_offset = (size_t)group * SCALE_GROUP_BYTES;
          const uint8_t* weights0 = weight_base0 + weight_offset;
          const uint8_t* weights1 = weight_base1 + weight_offset;

          const int future_group = group + prefetch_groups;
          if (prefetch_groups > 0 && future_group < group_count) {
            const size_t future_weight_offset =
                (size_t)future_group * WEIGHT_GROUP_BYTES;
            const size_t future_scale_offset =
                (size_t)future_group * SCALE_GROUP_BYTES;
            const uint8_t* future_weights0 =
                weight_base0 + future_weight_offset;
            const uint8_t* future_weights1 =
                weight_base1 + future_weight_offset;
            _mm_prefetch(reinterpret_cast<const char*>(future_weights0),
                         _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(future_weights0 + 64),
                         _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(future_weights1),
                         _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(future_weights1 + 64),
                         _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(scale_base0 +
                                                       future_scale_offset),
                         _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(scale_base1 +
                                                       future_scale_offset),
                         _MM_HINT_T0);
          }

          __m512 partial00 = _mm512_setzero_ps();
          __m512 partial01 = _mm512_setzero_ps();
          __m512 partial02 = _mm512_setzero_ps();
          __m512 partial03 = _mm512_setzero_ps();
          __m512 partial10 = _mm512_setzero_ps();
          __m512 partial11 = _mm512_setzero_ps();
          __m512 partial12 = _mm512_setzero_ps();
          __m512 partial13 = _mm512_setzero_ps();

#define NVFP4_DP_PAIR_TWO_TILES(PAIR, ACC0, ACC1)                              \
  do {                                                                          \
    const ActivationBF16 activation(broadcast_bf16_pair(                       \
        a_row + group * BufferB::NVFP4_K_GROUP + (PAIR) * 2));                 \
    const DequantizedWeight weight0(_mm_loadu_si128(                           \
        reinterpret_cast<const __m128i*>(weights0 + (PAIR) * 16)));            \
    const DequantizedWeight weight1(_mm_loadu_si128(                           \
        reinterpret_cast<const __m128i*>(weights1 + (PAIR) * 16)));            \
    (ACC0) = _mm512_dpbf16_ps((ACC0), activation.a, weight0.d);                \
    (ACC1) = _mm512_dpbf16_ps((ACC1), activation.a, weight1.d);                \
  } while (0)
          NVFP4_DP_PAIR_TWO_TILES(0, partial00, partial10);
          NVFP4_DP_PAIR_TWO_TILES(1, partial01, partial11);
          NVFP4_DP_PAIR_TWO_TILES(2, partial02, partial12);
          NVFP4_DP_PAIR_TWO_TILES(3, partial03, partial13);
          NVFP4_DP_PAIR_TWO_TILES(4, partial00, partial10);
          NVFP4_DP_PAIR_TWO_TILES(5, partial01, partial11);
          NVFP4_DP_PAIR_TWO_TILES(6, partial02, partial12);
          NVFP4_DP_PAIR_TWO_TILES(7, partial03, partial13);
#undef NVFP4_DP_PAIR_TWO_TILES

          const __m512 group_sum0 = _mm512_add_ps(
              _mm512_add_ps(partial00, partial01),
              _mm512_add_ps(partial02, partial03));
          const __m512 group_sum1 = _mm512_add_ps(
              _mm512_add_ps(partial10, partial11),
              _mm512_add_ps(partial12, partial13));
          const __m512 scales0 =
              e4m3fnx16_to_fp32(scale_base0 + scale_offset);
          const __m512 scales1 =
              e4m3fnx16_to_fp32(scale_base1 + scale_offset);
          total0 = _mm512_fmadd_ps(group_sum0, scales0, total0);
          total1 = _mm512_fmadd_ps(group_sum1, scales1, total1);
        }

        float* output = bc->get_submat(m, n, m_idx, n_pos);
        _mm512_storeu_ps(output, _mm512_mul_ps(total0, tensor_scale));
        _mm512_storeu_ps(output + 16,
                         _mm512_mul_ps(total1, tensor_scale));
      }
    }
  }
#endif

  // 4 个 zmm 的 horizontal reduce → 4 个连续 fp32。
  // 4 次 reduce_add_ps 之间无依赖，编译器/CPU 可并行调度。
  template <int GROUP_SIZE>
  __attribute__((always_inline)) static inline float finalize(float value, const BufferB* bb) {
    if constexpr (GROUP_SIZE == 16) return value * bb->tensor_scale;
    return value;
  }

  template <int GROUP_SIZE>
  __attribute__((always_inline)) static inline void reduce4(__m512 s0, __m512 s1, __m512 s2, __m512 s3, float* dst,
                                                             const BufferB* bb) {
    dst[0] = finalize<GROUP_SIZE>(_mm512_reduce_add_ps(s0), bb);
    dst[1] = finalize<GROUP_SIZE>(_mm512_reduce_add_ps(s1), bb);
    dst[2] = finalize<GROUP_SIZE>(_mm512_reduce_add_ps(s2), bb);
    dst[3] = finalize<GROUP_SIZE>(_mm512_reduce_add_ps(s3), bb);
  }

  // mat-vec: M 个独立 token，N 维 4 行一组累加，摊销 horizontal reduce。
  template <int GROUP_SIZE>
  static void fp4_mat_vec_kgroup_impl(int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;

    for (int m_idx = 0; m_idx < m; m_idx++) {
      float* c_row = bc->get_submat(m, n, m_idx, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, m_idx, 0);

      int n_pos = n_start;
      // 主循环: N 维 4 行一组
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const void* s0 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 0, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 0, k, 0));
        const void* s1 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 1, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 1, k, 0));
        const void* s2 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 2, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 2, k, 0));
        const void* s3 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 3, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 3, k, 0));

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          acc0 = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s0, g), mxfp4_dot_bf16(d0, a), acc0);
          acc1 = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s1, g), mxfp4_dot_bf16(d1, a), acc1);
          acc2 = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s2, g), mxfp4_dot_bf16(d2, a), acc2);
          acc3 = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s3, g), mxfp4_dot_bf16(d3, a), acc3);
        }
        reduce4<GROUP_SIZE>(acc0, acc1, acc2, acc3, c_row + (n_pos - n_start), bb);
      }
      // N 尾巴: N % 4 != 0 时单行 fallback
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const void* s = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos, k, 0))
                                         : static_cast<const void*>(bb->get_scale(n, n_pos, k, 0));
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s, g), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = finalize<GROUP_SIZE>(_mm512_reduce_add_ps(acc), bb);
      }
    }
  }

  static void fp4_mat_vec_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
#if defined(__AVX512BF16__)
    if (k_group_size == 16 && bb->blocked_nvfp4()) {
      if (nvfp4_decode_tile_batch() == 2) {
        fp4_mat_vec_nvfp4_blocked_two_tiles(m, n, k, ba, bb, bc, ith,
                                            nth);
      } else {
        fp4_mat_vec_nvfp4_blocked(m, n, k, ba, bb, bc, ith, nth);
      }
      return;
    }
#endif
    if (k_group_size == 16) {
      fp4_mat_vec_kgroup_impl<16>(m, n, k, ba, bb, bc, ith, nth);
    } else if (k_group_size == 32) {
      fp4_mat_vec_kgroup_impl<32>(m, n, k, ba, bb, bc, ith, nth);
    } else {
      throw std::runtime_error("FP4 AMX supports only group sizes 16 and 32");
    }
  }

  // mat-mat: 4×4 register tile (M_TILE=4, N_TILE=4 → 16 累加器)。
  // 每 K-group 解码 4 行 N 一次, 被 4 个 token 共享 → PSHUFB 解码开销 / 4。
  // M / N 尾巴回退到 mat-vec 单 token 内层 (V4 chunked-prefill 16/32/64 整数倍, 极少触发)。
  template <int GROUP_SIZE>
  static void fp4_mat_mat_kgroup_impl(int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;
    constexpr int MB = 4;
    constexpr int NB = 4;

    int m_pos = 0;
    for (; m_pos + MB <= m; m_pos += MB) {
      __m512bh* a_rows[MB] = {
          (__m512bh*)ba->get_submat(m, k, m_pos + 0, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 1, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 2, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 3, 0),
      };

      int n_pos = n_start;
      for (; n_pos + NB <= n_end; n_pos += NB) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const void* s0 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 0, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 0, k, 0));
        const void* s1 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 1, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 1, k, 0));
        const void* s2 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 2, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 2, k, 0));
        const void* s3 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 3, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 3, k, 0));

        __m512 acc[MB][NB];
        for (int i = 0; i < MB; i++)
          for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          // 4 行权重解码一次, MB 个 token 共享
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          const __m512 sv0 = load_group_scales<GROUP_SIZE>(s0, g);
          const __m512 sv1 = load_group_scales<GROUP_SIZE>(s1, g);
          const __m512 sv2 = load_group_scales<GROUP_SIZE>(s2, g);
          const __m512 sv3 = load_group_scales<GROUP_SIZE>(s3, g);

#define V_FMA_ROW(M_I)                                                      \
  do {                                                                      \
    const ActivationBF16 a(a_rows[M_I][g]);                                 \
    acc[M_I][0] = _mm512_fmadd_ps(sv0, mxfp4_dot_bf16(d0, a), acc[M_I][0]); \
    acc[M_I][1] = _mm512_fmadd_ps(sv1, mxfp4_dot_bf16(d1, a), acc[M_I][1]); \
    acc[M_I][2] = _mm512_fmadd_ps(sv2, mxfp4_dot_bf16(d2, a), acc[M_I][2]); \
    acc[M_I][3] = _mm512_fmadd_ps(sv3, mxfp4_dot_bf16(d3, a), acc[M_I][3]); \
  } while (0)
          V_FMA_ROW(0);
          V_FMA_ROW(1);
          V_FMA_ROW(2);
          V_FMA_ROW(3);
#undef V_FMA_ROW
        }
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          reduce4<GROUP_SIZE>(acc[i][0], acc[i][1], acc[i][2], acc[i][3], c_row + (n_pos - n_start), bb);
        }
      }
      // N 尾巴: 单 N 列 × MB token (V4 不触发)
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const void* s = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos, k, 0))
                                         : static_cast<const void*>(bb->get_scale(n, n_pos, k, 0));
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          __m512 acc = _mm512_setzero_ps();
          for (int g = 0; g < kg_count; g++) {
            const ActivationBF16 a(a_rows[i][g]);
            const DequantizedWeight d(w[g]);
            acc = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s, g), mxfp4_dot_bf16(d, a), acc);
          }
          c_row[n_pos - n_start] = finalize<GROUP_SIZE>(_mm512_reduce_add_ps(acc), bb);
        }
      }
    }
    // M 尾巴: M 不是 MB 倍数时余下 token, 退回单 token mat-vec 内层 (V4 不触发)
    for (int mi = m_pos; mi < m; mi++) {
      float* c_row = bc->get_submat(m, n, mi, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, mi, 0);
      int n_pos = n_start;
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const void* s0 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 0, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 0, k, 0));
        const void* s1 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 1, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 1, k, 0));
        const void* s2 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 2, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 2, k, 0));
        const void* s3 = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos + 3, k, 0))
                                          : static_cast<const void*>(bb->get_scale(n, n_pos + 3, k, 0));
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          a0 = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s0, g), mxfp4_dot_bf16(d0, a), a0);
          a1 = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s1, g), mxfp4_dot_bf16(d1, a), a1);
          a2 = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s2, g), mxfp4_dot_bf16(d2, a), a2);
          a3 = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s3, g), mxfp4_dot_bf16(d3, a), a3);
        }
        reduce4<GROUP_SIZE>(a0, a1, a2, a3, c_row + (n_pos - n_start), bb);
      }
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const void* s = GROUP_SIZE == 16 ? static_cast<const void*>(bb->get_native_scale(n, n_pos, k, 0))
                                         : static_cast<const void*>(bb->get_scale(n, n_pos, k, 0));
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(load_group_scales<GROUP_SIZE>(s, g), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = finalize<GROUP_SIZE>(_mm512_reduce_add_ps(acc), bb);
      }
    }
  }

  static void fp4_mat_mat_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
#if defined(__AVX512BF16__)
    if (k_group_size == 16 && bb->blocked_nvfp4()) {
      if (nvfp4_decode_tile_batch() == 2) {
        fp4_mat_vec_nvfp4_blocked_two_tiles(m, n, k, ba, bb, bc, ith,
                                            nth);
      } else {
        fp4_mat_vec_nvfp4_blocked(m, n, k, ba, bb, bc, ith, nth);
      }
      return;
    }
#endif
    if (k_group_size == 16) {
      fp4_mat_mat_kgroup_impl<16>(m, n, k, ba, bb, bc, ith, nth);
    } else if (k_group_size == 32) {
      fp4_mat_mat_kgroup_impl<32>(m, n, k, ba, bb, bc, ith, nth);
    } else {
      throw std::runtime_error("FP4 AMX supports only group sizes 16 and 32");
    }
  }
};

// Dispatch functions
inline void vec_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
  GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
  GemmKernel224MXFP4SmallKGroup::fp4_mat_mat_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

// ============================================================================
// AMX_FP4_MOE_TP — CRTP class, identical structure to AMX_K2_MOE_TP
// ============================================================================
template <class T = amx::GemmKernel224MXFP4SmallKGroup>
class AMX_FP4_MOE_TP : public AMX_MOE_BASE<T, AMX_FP4_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_FP4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_FP4_MOE_TP() = default;
  AMX_FP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("MXFP4 MoE only supports KGroup FP4");
    }
    if ((quant_config.group_size == 16) != (quant_config.quant_method == "NVFP4")) {
      throw std::runtime_error("AMX FP4 group_size=16 is reserved for native ModelOpt NVFP4");
    }
    const bool blocked =
        quant_config.group_size == 16 && amx::use_nvfp4_blocked_layout();
    const char* layout = blocked ? "blocked-n16" : "row-major";
    printf(
        "Creating AMX_FP4_MOE_TP %d at numa %d (layout=%s, decode_tiles=%d, "
        "prefetch_groups=%d)\n",
        tp_part_idx, numa_node_of_cpu(sched_getcpu()), layout,
        blocked ? amx::nvfp4_decode_tile_batch() : 1,
        blocked && amx::nvfp4_decode_tile_batch() == 2
            ? amx::nvfp4_prefetch_groups()
            : 0);
  }

  ~AMX_FP4_MOE_TP() = default;

  // BufferA: raw BF16, no group_size needed
  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    }
  }

  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size == 0 || quant_config.zero_point)
      throw std::runtime_error("MXFP4 MoE only support KGroup FP4.");
    if (config_.gate_scale == nullptr) throw std::runtime_error("MXFP4 MoE only support load native weight.");

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          gate_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.gate_proj +
                  ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
          up_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.up_proj + ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          down_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.down_proj +
                  ((logical_expert_id * config_.hidden_size * config_.intermediate_size) >> 1),
              ith, nth);
        },
        nullptr);

    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_elem_count = (config_.hidden_size * config_.intermediate_size) / config_.quant_config.group_size;
          convert_or_copy(gate_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.gate_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          convert_or_copy(up_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.up_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          convert_or_copy(down_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.down_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
        },
        nullptr);
  }

  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; i++) {
      __m512i data = _mm512_loadu_si512((__m512i*)s);
      _mm512_storeu_si512((__m512i*)d, data);
      d += 64;
      s += 64;
    }
    if (bytes -= chunks * 64) std::memcpy(d, s, bytes);
  }

  static inline void fast_fp32_to_bf16(ggml_bf16_t* __restrict dst, const float* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 32 <= count; i += 32) {
      __m512 v0 = _mm512_loadu_ps(src + i);
      __m512 v1 = _mm512_loadu_ps(src + i + 16);
      __m512i i0 = _mm512_srli_epi32(_mm512_castps_si512(v0), 16);
      __m512i i1 = _mm512_srli_epi32(_mm512_castps_si512(v1), 16);
      __m512i packed = _mm512_packus_epi32(i0, i1);
      __m512i permuted = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), packed);
      _mm512_storeu_si512((__m512i*)(dst + i), permuted);
    }
    for (; i < count; i++) dst[i] = ggml_fp32_to_bf16(src[i]);
  }

  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    size_t cpu_tp_weight_elem_count = (size_t)config_.intermediate_size * config_.hidden_size;
    size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;
    size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;

    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;
    size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    if (cpu_tp_count >= gpu_tp_count) {
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);

      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];

      size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_bytes;
      size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int total_tasks = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;

      size_t weight_chunk_size = (cpu_tp_weight_bytes + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, num_down_tasks, expert_id, weight_chunk_size, offset_in_gpu_weight, offset_in_gpu_scale,
           gpu_tp_weight_bytes, gpu_tp_scale_elem_count, w13_weight_dst, w13_scale_dst, w2_weight_dst, w2_scale_dst,
           group_size](int task_id) {
            if (task_id < NUM_WEIGHT_TASKS) {
              int chunk_idx = task_id;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                fast_memcpy(w13_weight_dst + offset_in_gpu_weight + start, (uint8_t*)gate_bb_[expert_id]->b + start,
                            end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                fast_memcpy(w13_weight_dst + offset_in_gpu_weight + gpu_tp_weight_bytes + start,
                            (uint8_t*)up_bb_[expert_id]->b + start, end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_col = config_.intermediate_size >> 1;
              size_t scale_per_col = config_.intermediate_size / group_size;
              size_t gpu_weight_stride = (full_config.intermediate_size / gpu_tp_count) >> 1;
              size_t gpu_scale_stride = (full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gpu_weight_slice_offset = local_idx * weight_per_col;
              size_t gpu_scale_slice_offset = local_idx * scale_per_col;

              for (size_t col = col_start; col < col_end; col++) {
                fast_memcpy(w2_weight_dst + col * gpu_weight_stride + gpu_weight_slice_offset,
                            (uint8_t*)down_bb_[expert_id]->b + col * weight_per_col, weight_per_col);
                fast_fp32_to_bf16(w2_scale_dst + col * gpu_scale_stride + gpu_scale_slice_offset,
                                  down_bb_[expert_id]->d + col * scale_per_col, scale_per_col);
              }
            } else if (task_id == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fast_fp32_to_bf16(w13_scale_dst + offset_in_gpu_scale, gate_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            } else {
              fast_fp32_to_bf16(w13_scale_dst + offset_in_gpu_scale + gpu_tp_scale_elem_count, up_bb_[expert_id]->d,
                                cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;

      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int tasks_per_gpu_tp = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      int total_tasks = tasks_per_gpu_tp * gpu_tps_per_cpu_tp;

      size_t weight_chunk_size = (data_per_gpu_tp_weight + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, gpu_tps_per_cpu_tp, start_gpu_tp, data_per_gpu_tp_weight, data_per_gpu_tp_scale, num_down_tasks,
           tasks_per_gpu_tp, expert_id, weight_chunk_size, gpu_tp_weight_bytes, gpu_tp_scale_elem_count,
           group_size](int task_id) {
            int local_gpu_idx = task_id / tasks_per_gpu_tp;
            int task_type = task_id % tasks_per_gpu_tp;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;

            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];

            size_t cpu_offset_weight = local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = local_gpu_idx * data_per_gpu_tp_scale;

            if (task_type < NUM_WEIGHT_TASKS) {
              int chunk_idx = task_type;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                fast_memcpy(w13_weight_dst + start, (uint8_t*)gate_bb_[expert_id]->b + cpu_offset_weight + start,
                            end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                fast_memcpy(w13_weight_dst + gpu_tp_weight_bytes + start,
                            (uint8_t*)up_bb_[expert_id]->b + cpu_offset_weight + start, end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) >> 1;
              size_t scale_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size;

              for (size_t col = col_start; col < col_end; col++) {
                size_t col_offset_weight = (col * config_.intermediate_size / 2) +
                                           (local_gpu_idx * data_per_gpu_tp_weight / config_.hidden_size);
                size_t col_offset_scale = (col * (config_.intermediate_size / group_size)) +
                                          (local_gpu_idx * data_per_gpu_tp_scale / config_.hidden_size);

                fast_memcpy(w2_weight_dst + col * weight_per_gpu_col,
                            (uint8_t*)down_bb_[expert_id]->b + col_offset_weight, weight_per_gpu_col);
                fast_fp32_to_bf16(w2_scale_dst + col * scale_per_gpu_col, down_bb_[expert_id]->d + col_offset_scale,
                                  scale_per_gpu_col);
              }
            } else if (task_type == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fast_fp32_to_bf16(w13_scale_dst, gate_bb_[expert_id]->d + cpu_offset_scale, data_per_gpu_tp_scale);
            } else {
              fast_fp32_to_bf16(w13_scale_dst + gpu_tp_scale_elem_count, up_bb_[expert_id]->d + cpu_offset_scale,
                                data_per_gpu_tp_scale);
            }
          },
          nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_FP4_MOE_TP
// ============================================================================
template <typename K>
class TP_MOE<AMX_FP4_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>>;
  using Base::Base;

  void load_nvfp4_weights() {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    if (config.quant_config.group_size != 16) {
      throw std::runtime_error("ModelOpt NVFP4 requires group_size=16");
    }
    if (config.hidden_size % 32 != 0 || config.intermediate_size % this->tp_count != 0 ||
        (config.intermediate_size / this->tp_count) % 32 != 0) {
      throw std::runtime_error("ModelOpt NVFP4 requires every NUMA partition dimension to be divisible by 32");
    }
    if (config.gate_projs.empty() || config.up_projs.empty() || config.down_projs.empty() ||
        config.gate_scales.empty() || config.up_scales.empty() || config.down_scales.empty() ||
        config.gate_scale2s.empty() || config.up_scale2s.empty() || config.down_scale2s.empty()) {
      throw std::runtime_error("ModelOpt NVFP4 requires per-expert weight, block-scale, and tensor-scale pointers");
    }

    const int full_intermediate = config.intermediate_size;
    (void)e4m3fn_lut();
    pool->dispense_backend()->do_numa_job([&, this](int numa_id) {
      auto* tp = tps[numa_id].get();
      auto& tpc = tp->config_;
      const int local_intermediate = tpc.intermediate_size;
      const int hidden_groups = tpc.hidden_size / 16;
      const int local_intermediate_groups = local_intermediate / 16;
      const int full_intermediate_groups = full_intermediate / 16;
      auto subpool = pool->get_subpool(numa_id);

      subpool->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, numa_id, local_intermediate, hidden_groups, local_intermediate_groups,
           full_intermediate_groups](int expert_id) {
            if (tpc.should_skip_expert(expert_id)) return;
            const uint64_t logical_id = expert_map(physical_to_logical_map, expert_id);
            if (logical_id >= config.gate_projs[0].size() || logical_id >= config.up_projs[0].size() ||
                logical_id >= config.down_projs[0].size() || logical_id >= config.gate_scales[0].size() ||
                logical_id >= config.up_scales[0].size() || logical_id >= config.down_scales[0].size() ||
                logical_id >= config.gate_scale2s[0].size() || logical_id >= config.up_scale2s[0].size() ||
                logical_id >= config.down_scale2s[0].size()) {
              throw std::runtime_error("NVFP4 logical expert id is outside the source pointer tables");
            }
            if (config.gate_projs[0][logical_id] == nullptr || config.up_projs[0][logical_id] == nullptr ||
                config.down_projs[0][logical_id] == nullptr || config.gate_scales[0][logical_id] == nullptr ||
                config.up_scales[0][logical_id] == nullptr || config.down_scales[0][logical_id] == nullptr ||
                config.gate_scale2s[0][logical_id] == nullptr || config.up_scale2s[0][logical_id] == nullptr ||
                config.down_scale2s[0][logical_id] == nullptr) {
              throw std::runtime_error("NVFP4 source pointer table contains a null entry");
            }

            const size_t gate_weight_bytes = (size_t)local_intermediate * tpc.hidden_size / 2;
            const size_t gate_weight_offset = (size_t)numa_id * gate_weight_bytes;
            const size_t gate_scale_count = (size_t)local_intermediate * hidden_groups;
            const size_t gate_scale_offset = (size_t)numa_id * gate_scale_count;
            tp->gate_bb_[expert_id]->from_nvfp4_rows(
                (const uint8_t*)config.gate_projs[0][logical_id] + gate_weight_offset,
                tpc.hidden_size / 2,
                (const uint8_t*)config.gate_scales[0][logical_id] + gate_scale_offset,
                hidden_groups, 0, 1);
            tp->up_bb_[expert_id]->from_nvfp4_rows(
                (const uint8_t*)config.up_projs[0][logical_id] + gate_weight_offset,
                tpc.hidden_size / 2,
                (const uint8_t*)config.up_scales[0][logical_id] + gate_scale_offset,
                hidden_groups, 0, 1);
            tp->gate_bb_[expert_id]->tensor_scale = *(const float*)config.gate_scale2s[0][logical_id];
            tp->up_bb_[expert_id]->tensor_scale = *(const float*)config.up_scale2s[0][logical_id];

            const uint8_t* down_weight = (const uint8_t*)config.down_projs[0][logical_id];
            const uint8_t* down_scale = (const uint8_t*)config.down_scales[0][logical_id];
            const size_t local_down_row_bytes = (size_t)local_intermediate / 2;
            const size_t full_down_row_bytes = (size_t)full_intermediate / 2;
            tp->down_bb_[expert_id]->from_nvfp4_rows(
                down_weight + (size_t)numa_id * local_down_row_bytes,
                full_down_row_bytes,
                down_scale + (size_t)numa_id * local_intermediate_groups,
                full_intermediate_groups, 0, 1);
            tp->down_bb_[expert_id]->tensor_scale = *(const float*)config.down_scale2s[0][logical_id];
          },
          nullptr);
    });

    this->weights_loaded = true;
  }

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    if (config.quant_config.quant_method == "NVFP4") {
      load_nvfp4_weights();
      return;
    }

    bool use_per_expert_ptrs = !config.gate_projs.empty();

    if (config.gate_projs.empty() && config.gate_scale == nullptr)
      throw std::runtime_error("MXFP4 MoE only supports Packed FP4 with KGroup Scale");

    printf("From %s\n", use_per_expert_ptrs ? "per-expert pointers (gate_projs)" : "Packed FP4 with KGroup Scale");

    int& group_size = config.quant_config.group_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
      size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

      tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.gate_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      tpc.up_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      tpc.down_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];

      if (use_per_expert_ptrs) {
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int expert_id_) {
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              ggml_bf16_t* src_gate_scale = (ggml_bf16_t*)config.gate_scales[0][expert_id];
              ggml_bf16_t* src_up_scale = (ggml_bf16_t*)config.up_scales[0][expert_id];
              ggml_bf16_t* src_down_scale = (ggml_bf16_t*)config.down_scales[0][expert_id];

              memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                     src_gate + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                     src_up + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                     src_gate_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);
              memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                     src_up_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);

              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                       src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                       (tpc.intermediate_size >> 1));
                memcpy((ggml_bf16_t*)tpc.down_scale +
                           (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                       src_down_scale +
                           (col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                       sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
              }
            },
            nullptr);
      } else {
        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&, i](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.gate_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);
                memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.up_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                        col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                       1),
                         (tpc.intermediate_size >> 1));
                  memcpy((ggml_bf16_t*)tpc.down_scale +
                             (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                         (ggml_bf16_t*)config.down_scale +
                             ((expert_id * (config.intermediate_size / group_size) * config.hidden_size) +
                              col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                         sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
                }
              },
              nullptr);
        }
      }
      printf("TP %d load weight done.\n", i);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)(tpc.gate_proj);
      delete[] (uint8_t*)(tpc.up_proj);
      delete[] (uint8_t*)(tpc.down_proj);
      delete[] (ggml_bf16_t*)(tpc.gate_scale);
      delete[] (ggml_bf16_t*)(tpc.up_scale);
      delete[] (ggml_bf16_t*)(tpc.down_scale);
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->config.quant_config.quant_method == "NVFP4")
      throw std::runtime_error("Native NVFP4 supports static CPU/GPU expert placement only");
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (w13_weight_ptrs.size() != gpu_tp_count || w13_scale_ptrs.size() != gpu_tp_count ||
        w2_weight_ptrs.size() != gpu_tp_count || w2_scale_ptrs.size() != gpu_tp_count)
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FP4_MOE_H
