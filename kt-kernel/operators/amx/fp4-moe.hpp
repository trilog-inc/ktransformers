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
 *   Dot prod: native AMX-BF16 tiles for multi-token experts, AVX512-BF16
 *             for latency-sensitive tails (never an INT4 dot product)
 *   Scale:    FP32 per-group scale (weight only, no activation scale)
 **/
#ifndef CPUINFER_OPERATOR_AMX_FP4_MOE_H
#define CPUINFER_OPERATOR_AMX_FP4_MOE_H

#include <cstdlib>

#include "la/amx_raw_buffers.hpp"  // BufferABF16Impl
#include "moe_base.hpp"

namespace amx {

// ============================================================================
// MXFP4 kernel: FP4 E2M1 weights × BF16 activations → FP32 output.
//
// We deliberately keep the model's native nibble-packed E2M1 representation.
// The AMX path expands one 16x32 weight tile at a time into a small VNNI BF16
// scratch tile.  It does *not* materialize a BF16 copy of the model and does
// not reinterpret the values as integers.  Each UE8M0 group scale is applied
// in FP32 after the corresponding K=32 tile dot product, matching the AVX512
// path's quantization semantics.
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

  static constexpr int TILE_M = 16;
  static constexpr int TILE_N = 16;
  static constexpr int TILE_K = 32;
  static constexpr int VNNI_BLK = 2;

  static std::string name() { return "MXFP4_NATIVE_AMX_BF16"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static void config() {
#ifdef HAVE_AMX
    if (!enable_amx()) {
      throw std::runtime_error(
          "MXFP4 AMX-BF16 was compiled in, but Linux did not grant XTILEDATA permission");
    }
    TileConfig tile_config;
    tile_config.set_row_col(0, TILE_M, TILE_K * sizeof(ggml_bf16_t));
    for (int tile = 1; tile <= 2; ++tile) {
      tile_config.set_row_col(tile, TILE_K / VNNI_BLK,
                              TILE_N * VNNI_BLK * sizeof(ggml_bf16_t));
    }
    for (int tile = 3; tile <= 4; ++tile) {
      tile_config.set_row_col(tile, TILE_M, TILE_N * sizeof(float));
    }
    tile_config.set_config();
#endif
  }

  // Expert batches below this size remain on the AVX512-BF16 microkernel: the
  // fixed 16-row AMX tile and FP4 expansion cost dominate tiny decode batches.
  // Set to 0 to force AMX for validation/benchmarking.
  static int amx_min_tokens_per_expert() {
    static const int threshold = [] {
      constexpr int default_threshold = 4;
      const char* value = std::getenv("KT_MXFP4_AMX_MIN_TOKENS_PER_EXPERT");
      if (value == nullptr || *value == '\0') return default_threshold;
      char* end = nullptr;
      long parsed = std::strtol(value, &end, 10);
      if (end == value || *end != '\0' || parsed < 0 || parsed > 1024) {
        throw std::runtime_error(
            "KT_MXFP4_AMX_MIN_TOKENS_PER_EXPERT must be an integer in [0, 1024]");
      }
      return static_cast<int>(parsed);
    }();
    return threshold;
  }

  static bool should_use_amx(int m, int k_group_size) {
#ifdef HAVE_AMX
    // DeepSeek-V4-Flash uses one native UE8M0 scale per 32 E2M1 values.
    return k_group_size == TILE_K && m >= amx_min_tokens_per_expert();
#else
    (void)m;
    (void)k_group_size;
    return false;
#endif
  }

  // FP4 E2M1 → BF16 LUTs (16 entries each, for PSHUFB within 128-bit lanes)
  // E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
  alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,   //  0..7  positive
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};  //  8..15 negative
  alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {
      0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,   //  0..7  positive
      0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};  //  8..15 negative

  // Convert 16 packed FP4 bytes (32 values = 1 k_group) → 32 BF16 values (__m512i)
  // Output column order: [BF16(lo[0]),BF16(hi[0]), ..., BF16(lo[15]),BF16(hi[15])]
  __attribute__((always_inline)) static inline __m512i mxfp4_to_bf16_32(__m128i packed) {
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(packed, lo_mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

    __m128i lut_lo = _mm_load_si128((__m128i*)fp4_bf16_lo);
    __m128i lut_hi = _mm_load_si128((__m128i*)fp4_bf16_hi);

    // Look up low/high bytes for lo nibbles → 16 BF16 values
    __m128i l_lo = _mm_shuffle_epi8(lut_lo, lo);
    __m128i l_hi = _mm_shuffle_epi8(lut_hi, lo);
    __m128i lo_bf16_0 = _mm_unpacklo_epi8(l_lo, l_hi);  // BF16(lo[0..7])
    __m128i lo_bf16_1 = _mm_unpackhi_epi8(l_lo, l_hi);  // BF16(lo[8..15])

    // Look up low/high bytes for hi nibbles → 16 BF16 values
    __m128i h_lo = _mm_shuffle_epi8(lut_lo, hi);
    __m128i h_hi = _mm_shuffle_epi8(lut_hi, hi);
    __m128i hi_bf16_0 = _mm_unpacklo_epi8(h_lo, h_hi);  // BF16(hi[0..7])
    __m128i hi_bf16_1 = _mm_unpackhi_epi8(h_lo, h_hi);  // BF16(hi[8..15])

    // Interleave lo/hi at 16-bit: [lo[0],hi[0], lo[1],hi[1], ...] = column order
    __m128i p0 = _mm_unpacklo_epi16(lo_bf16_0, hi_bf16_0);  // cols  0..7
    __m128i p1 = _mm_unpackhi_epi16(lo_bf16_0, hi_bf16_0);  // cols  8..15
    __m128i p2 = _mm_unpacklo_epi16(lo_bf16_1, hi_bf16_1);  // cols 16..23
    __m128i p3 = _mm_unpackhi_epi16(lo_bf16_1, hi_bf16_1);  // cols 24..31

    __m256i q0 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
    __m256i q1 = _mm256_inserti128_si256(_mm256_castsi128_si256(p2), p3, 1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
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

  // Buffers retain native packed MXFP4 weights.  These aliases must precede
  // the AMX helper declarations that consume them.
  using BufferA = BufferABF16Impl<GemmKernel224MXFP4SmallKGroup>;        // raw BF16, no quant
  using BufferB = BufferBInt4KGroupImpl<GemmKernel224MXFP4SmallKGroup>;  // nibble-packed E2M1
  using BufferC = BufferCReduceImpl<GemmKernel224MXFP4SmallKGroup>;      // FP32 reduce

#ifdef HAVE_AMX
  struct alignas(64) AMXScratch {
    ggml_bf16_t a[TILE_M * TILE_K];
    ggml_bf16_t b[N_STEP * TILE_K];
    float partial[TILE_M * N_STEP];
    float accum[TILE_M * N_BLOCK];
  };

  // Decode N output-channel rows without applying their per-group scale.  The
  // 16x32 row-major BF16 tile is then transposed as 16x16 dwords into the VNNI
  // layout required by TDPBF16PS.
  static inline void prepare_amx_b_tile(BufferB* bb, int n, int k, int n_begin,
                                        int k_begin, int valid_n,
                                        ggml_bf16_t* dst) {
    std::memset(dst, 0, sizeof(ggml_bf16_t) * TILE_N * TILE_K);
    for (int ni = 0; ni < valid_n; ++ni) {
      const __m128i packed = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(bb->get_submat(n, k, n_begin + ni, k_begin)));
      _mm512_store_si512(reinterpret_cast<__m512i*>(dst + ni * TILE_K),
                         mxfp4_to_bf16_32(packed));
    }
    transpose_16x16_32bit(reinterpret_cast<__m512i*>(dst));
  }

  // Native MXFP4 AMX-BF16 GEMM.  Accumulation is deliberately split at every
  // K=32 MX group: AMX produces the unscaled FP32 partial, then an AVX512 FMA
  // applies the 16 independent output-channel scales.  This avoids rounding a
  // scale into BF16 and is numerically equivalent to the existing native
  // MXFP4 microkernel apart from the dot-product reduction order.
  static void fp4_mat_mat_amx(int m, int n, int k, BufferA* ba, BufferB* bb,
                              BufferC* bc, int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;

    AMXScratch scratch;
    alignas(64) float scales[N_STEP];

    for (int m_begin = 0; m_begin < m; m_begin += TILE_M) {
      const int valid_m = std::min(TILE_M, m - m_begin);
      const int panel_n = n_end - n_start;
      std::memset(scratch.accum, 0,
                  sizeof(float) * TILE_M * N_BLOCK);

      // The thread's N panel is at most N_BLOCK=256 columns. Accumulate its
      // scaled K=32 partials in a 16 KiB FP32 scratch panel, then write C once.
      // This retains A reuse across eight N_STEP tiles without turning every
      // MX group into a read-modify-write pass over the output buffer.
      for (int k_begin = 0; k_begin < k; k_begin += TILE_K) {
        if (valid_m == TILE_M) {
          const int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
          const int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          _tile_loadd(0, ba->get_submat(m, k, m_begin, k_begin),
                      k_block_size * sizeof(ggml_bf16_t));
        } else {
          std::memset(scratch.a, 0, sizeof(scratch.a));
          for (int mi = 0; mi < valid_m; ++mi) {
            const ggml_bf16_t* src =
                ba->get_submat(m, k, m_begin + mi, k_begin);
            _mm512_store_si512(
                reinterpret_cast<__m512i*>(scratch.a + mi * TILE_K),
                _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src)));
          }
          _tile_loadd(0, scratch.a, TILE_K * sizeof(ggml_bf16_t));
        }

        for (int n_begin = n_start; n_begin < n_end; n_begin += N_STEP) {
          const int valid_n = std::min(N_STEP, n_end - n_begin);
          const int valid_n0 = std::min(TILE_N, valid_n);
          const int valid_n1 = std::max(0, valid_n - TILE_N);
          prepare_amx_b_tile(bb, n, k, n_begin, k_begin, valid_n0, scratch.b);
          prepare_amx_b_tile(bb, n, k, n_begin + TILE_N, k_begin, valid_n1,
                             scratch.b + TILE_N * TILE_K);

          _tile_zero(3);
          _tile_zero(4);
          _tile_loadd(1, scratch.b,
                      TILE_N * VNNI_BLK * sizeof(ggml_bf16_t));
          _tile_loadd(2, scratch.b + TILE_N * TILE_K,
                      TILE_N * VNNI_BLK * sizeof(ggml_bf16_t));
          _tile_dpbf16ps(3, 0, 1);
          _tile_dpbf16ps(4, 0, 2);
          _tile_stored(3, scratch.partial, N_STEP * sizeof(float));
          _tile_stored(4, scratch.partial + TILE_N, N_STEP * sizeof(float));

          for (int ni = 0; ni < valid_n; ++ni) {
            scales[ni] = *bb->get_scale(n, n_begin + ni, k, k_begin);
          }
          for (int ni = valid_n; ni < N_STEP; ++ni) scales[ni] = 0.0f;
          const __m512 scale0 = _mm512_load_ps(scales);
          const __m512 scale1 = _mm512_load_ps(scales + TILE_N);

          for (int mi = 0; mi < valid_m; ++mi) {
            float* accum = scratch.accum + mi * N_BLOCK + (n_begin - n_start);
            if (valid_n == N_STEP) {
              const __m512 old0 = _mm512_load_ps(accum);
              const __m512 old1 = _mm512_load_ps(accum + TILE_N);
              const __m512 part0 = _mm512_load_ps(scratch.partial + mi * N_STEP);
              const __m512 part1 =
                  _mm512_load_ps(scratch.partial + mi * N_STEP + TILE_N);
              _mm512_store_ps(accum, _mm512_fmadd_ps(part0, scale0, old0));
              _mm512_store_ps(accum + TILE_N,
                              _mm512_fmadd_ps(part1, scale1, old1));
            } else {
              for (int ni = 0; ni < valid_n; ++ni) {
                accum[ni] += scratch.partial[mi * N_STEP + ni] * scales[ni];
              }
            }
          }
        }
      }

      for (int mi = 0; mi < valid_m; ++mi) {
        float* output = bc->get_submat(m, n, m_begin + mi, n_start);
        const float* accum = scratch.accum + mi * N_BLOCK;
        for (int ni = 0; ni < panel_n; ni += N_STEP) {
          const int valid_n = std::min(N_STEP, panel_n - ni);
          if (valid_n == N_STEP) {
            _mm512_storeu_ps(output + ni, _mm512_load_ps(accum + ni));
            _mm512_storeu_ps(output + ni + TILE_N,
                             _mm512_load_ps(accum + ni + TILE_N));
          } else {
            std::copy(accum + ni, accum + ni + valid_n, output + ni);
          }
        }
      }
    }
  }
#endif

  // 4 个 zmm 的 horizontal reduce → 4 个连续 fp32。
  // 4 次 reduce_add_ps 之间无依赖，编译器/CPU 可并行调度。
  __attribute__((always_inline)) static inline void reduce4(__m512 s0, __m512 s1, __m512 s2, __m512 s3, float* dst) {
    dst[0] = _mm512_reduce_add_ps(s0);
    dst[1] = _mm512_reduce_add_ps(s1);
    dst[2] = _mm512_reduce_add_ps(s2);
    dst[3] = _mm512_reduce_add_ps(s3);
  }

  // mat-vec: M 个独立 token，N 维 4 行一组累加，摊销 horizontal reduce。
  static void fp4_mat_vec_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
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
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);

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
          acc0 = _mm512_fmadd_ps(_mm512_set1_ps(s0[g]), mxfp4_dot_bf16(d0, a), acc0);
          acc1 = _mm512_fmadd_ps(_mm512_set1_ps(s1[g]), mxfp4_dot_bf16(d1, a), acc1);
          acc2 = _mm512_fmadd_ps(_mm512_set1_ps(s2[g]), mxfp4_dot_bf16(d2, a), acc2);
          acc3 = _mm512_fmadd_ps(_mm512_set1_ps(s3[g]), mxfp4_dot_bf16(d3, a), acc3);
        }
        reduce4(acc0, acc1, acc2, acc3, c_row + (n_pos - n_start));
      }
      // N 尾巴: N % 4 != 0 时单行 fallback
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }

  // mat-mat: 4×4 register tile (M_TILE=4, N_TILE=4 → 16 累加器)。
  // 每 K-group 解码 4 行 N 一次, 被 4 个 token 共享 → PSHUFB 解码开销 / 4。
  // M / N 尾巴回退到 mat-vec 单 token 内层 (V4 chunked-prefill 16/32/64 整数倍, 极少触发)。
  static void fp4_mat_mat_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
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
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc[MB][NB];
        for (int i = 0; i < MB; i++)
          for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          // 4 行权重解码一次, MB 个 token 共享
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          const __m512 sv0 = _mm512_set1_ps(s0[g]);
          const __m512 sv1 = _mm512_set1_ps(s1[g]);
          const __m512 sv2 = _mm512_set1_ps(s2[g]);
          const __m512 sv3 = _mm512_set1_ps(s3[g]);

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
          reduce4(acc[i][0], acc[i][1], acc[i][2], acc[i][3], c_row + (n_pos - n_start));
        }
      }
      // N 尾巴: 单 N 列 × MB token (V4 不触发)
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          __m512 acc = _mm512_setzero_ps();
          for (int g = 0; g < kg_count; g++) {
            const ActivationBF16 a(a_rows[i][g]);
            const DequantizedWeight d(w[g]);
            acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp4_dot_bf16(d, a), acc);
          }
          c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
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
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          a0 = _mm512_fmadd_ps(_mm512_set1_ps(s0[g]), mxfp4_dot_bf16(d0, a), a0);
          a1 = _mm512_fmadd_ps(_mm512_set1_ps(s1[g]), mxfp4_dot_bf16(d1, a), a1);
          a2 = _mm512_fmadd_ps(_mm512_set1_ps(s2[g]), mxfp4_dot_bf16(d2, a), a2);
          a3 = _mm512_fmadd_ps(_mm512_set1_ps(s3[g]), mxfp4_dot_bf16(d3, a), a3);
        }
        reduce4(a0, a1, a2, a3, c_row + (n_pos - n_start));
      }
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }
};

// Dispatch functions
inline void vec_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
  if (GemmKernel224MXFP4SmallKGroup::should_use_amx(m, k_group_size)) {
#ifdef HAVE_AMX
    GemmKernel224MXFP4SmallKGroup::fp4_mat_mat_amx(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
#endif
  } else {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
  }
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
  if (GemmKernel224MXFP4SmallKGroup::should_use_amx(m, k_group_size)) {
#ifdef HAVE_AMX
    GemmKernel224MXFP4SmallKGroup::fp4_mat_mat_amx(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
#endif
  } else {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_mat_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
  }
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
    constexpr const char* backend = amx::AMX_AVAILABLE ? "native-MXFP4/AMX-BF16+AVX512-tail"
                                                       : "native-MXFP4/AVX512-BF16";
    printf("Creating AMX_FP4_MOE_TP %d at numa %d (backend=%s, amx_min_tokens=%d)\n",
           tp_part_idx, numa_node_of_cpu(sched_getcpu()), backend,
           T::amx_min_tokens_per_expert());
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

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

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
              const size_t physical_expert_id = expert_id_;
              const size_t logical_expert_id =
                  expert_map(physical_to_logical_map, physical_expert_id);

              if (logical_expert_id >= config.gate_projs[0].size() ||
                  config.gate_projs[0][logical_expert_id] == nullptr ||
                  config.up_projs[0][logical_expert_id] == nullptr ||
                  config.down_projs[0][logical_expert_id] == nullptr ||
                  config.gate_scales[0][logical_expert_id] == nullptr ||
                  config.up_scales[0][logical_expert_id] == nullptr ||
                  config.down_scales[0][logical_expert_id] == nullptr) {
                throw std::runtime_error(
                    "MXFP4 compact load received a missing checkpoint pointer for logical expert " +
                    std::to_string(logical_expert_id));
              }

              uint8_t* src_gate =
                  (uint8_t*)config.gate_projs[0][logical_expert_id];
              uint8_t* src_up =
                  (uint8_t*)config.up_projs[0][logical_expert_id];
              uint8_t* src_down =
                  (uint8_t*)config.down_projs[0][logical_expert_id];
              ggml_bf16_t* src_gate_scale =
                  (ggml_bf16_t*)config.gate_scales[0][logical_expert_id];
              ggml_bf16_t* src_up_scale =
                  (ggml_bf16_t*)config.up_scales[0][logical_expert_id];
              ggml_bf16_t* src_down_scale =
                  (ggml_bf16_t*)config.down_scales[0][logical_expert_id];

              memcpy((uint8_t*)tpc.gate_proj + ((physical_expert_id * weight_elem_count) >> 1),
                     src_gate + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((uint8_t*)tpc.up_proj + ((physical_expert_id * weight_elem_count) >> 1),
                     src_up + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((ggml_bf16_t*)tpc.gate_scale + (physical_expert_id * scales_elem_count),
                     src_gate_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);
              memcpy((ggml_bf16_t*)tpc.up_scale + (physical_expert_id * scales_elem_count),
                     src_up_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);

              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj +
                           ((physical_expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                       src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                       (tpc.intermediate_size >> 1));
                memcpy((ggml_bf16_t*)tpc.down_scale +
                           (physical_expert_id * scales_elem_count +
                            col * (tpc.intermediate_size / group_size)),
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

    if (use_per_expert_ptrs) {
      // The staging arrays above are already packed in compact physical expert
      // order. Do not apply the checkpoint logical-id map a second time.
      pool->dispense_backend()->do_numa_job([&, this](int numa_id) {
        tps[numa_id]->config_.physical_to_logical_map = nullptr;
        tps[numa_id]->load_weights();
      });
    } else {
      DO_TPS_LOAD_WEIGHTS(pool);
    }

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
