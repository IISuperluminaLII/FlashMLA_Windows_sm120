/***************************************************************************************************
 * Copyright (c) 2025 FlashMLA Contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Safety Introspection: SFINAE-Based Mainloop Internal Checks
 *
 * This header provides compile-time safety checks for mainloop internals
 * (Points 1, 2, 8) using SFINAE detection idiom. Checks automatically activate
 * when mainloop classes expose required members as public API.
 *
 * Design Philosophy:
 * - Zero CUTLASS edits required for checks to compile
 * - Checks gracefully degrade to no-op when members unavailable
 * - When UniversalCopy exposes internals, checks automatically activate
 * - Type-safe: uses std::enable_if + detection idiom instead of macros
 *
 * Usage:
 *   #include "safety_introspection.hpp"
 *
 *   // After mainloop instantiation:
 *   SmemBudgetAssert<Mainloop, KernelTraits>::check();    // Point 1
 *   EpiloguePVAssert<Mainloop, KernelTraits>::check();    // Point 2
 *   AlignmentAssert<Mainloop>::check();                   // Point 8
 **************************************************************************************************/

#pragma once

#include <type_traits>
#include <cstdint>

// ============================================================================
// Detection Idiom (C++11-compatible void_t workaround)
// ============================================================================

namespace flash {
namespace safety {

// void_t for C++11 (native in C++17+)
template<typename...> struct make_void { using type = void; };
template<typename... Ts> using void_t = typename make_void<Ts...>::type;

// ============================================================================
// [SM120 SAFETY][P1] Point 1: SMEM Budget Validation Detection
// ============================================================================

// Detect if Mainloop exposes kSmemTotal
template<class T, class = void>
struct has_kSmemTotal : std::false_type {};

template<class T>
struct has_kSmemTotal<T, void_t<decltype(T::kSmemTotal)>> : std::true_type {};

// SFINAE-based budget checker
template <class Mainloop, class KernelTraits, bool Has = has_kSmemTotal<Mainloop>::value>
struct SmemBudgetAssert {
  // No kSmemTotal exposed: no-op (silent)
  static void check() {}
};

template <class Mainloop, class KernelTraits>
struct SmemBudgetAssert<Mainloop, KernelTraits, true> {
  // kSmemTotal exposed: perform validation
  static void check() {
    static_assert(Mainloop::kSmemTotal <= KernelTraits::kSharedMemLimit,
      "[SM120 SAFETY][P1] Total shared memory usage exceeds architecture limit. "
      "SM100a limit: 227KB (232448 bytes), SM120 limit: 99KB (101376 bytes). "
      "Reduce tile sizes, decrease pipeline stages, or use non-persistent scheduler. "
      "Current usage can be inspected via Mainloop::kSmemTotal. "
      "See SM120_FINAL_STATUS.md Point 1 for guidance.");
  }
};

// ============================================================================
// [SM120 SAFETY][P2] Point 2: Epilogue ↔ Mainloop Shape Compatibility Detection
// ============================================================================

// Detect if Mainloop exposes TileShapePV
template<class T, class = void>
struct has_TileShapePV : std::false_type {};

template<class T>
struct has_TileShapePV<T, void_t<typename T::TileShapePV>> : std::true_type {};

// Detect if Mainloop exposes HeadDimPV
template<class T, class = void>
struct has_HeadDimPV : std::false_type {};

template<class T>
struct has_HeadDimPV<T, void_t<decltype(T::HeadDimPV)>> : std::true_type {};

// SFINAE-based epilogue shape checker
template <class Mainloop, class KernelTraits,
          bool HasShape = has_TileShapePV<Mainloop>::value,
          bool HasHeadDim = has_HeadDimPV<Mainloop>::value>
struct EpiloguePVAssert {
  // Missing internals: no-op (silent)
  static void check() {}
};

template <class Mainloop, class KernelTraits>
struct EpiloguePVAssert<Mainloop, KernelTraits, true, true> {
  // TileShapePV + HeadDimPV exposed: perform validation
  static void check() {
    using PV  = typename Mainloop::TileShapePV;           // (M, N, Kpv)
    using Epi = typename KernelTraits::EpilogueTileShape; // (M, N, K)

    // Extract dimensions using cute::get<>
    using cute::get;

    // M dimension must match exactly
    static_assert(get<0>(PV{}) == get<0>(Epi{}),
      "[SM120 SAFETY][P2] Epilogue tile M dimension must match mainloop PV M dimension. "
      "Mismatch indicates desync between mainloop output and epilogue input. "
      "Mainloop produces (M,N,K)=PV shape, epilogue expects (M,N,K)=Epi shape. "
      "See SM120_FINAL_STATUS.md Point 2 for guidance.");

    // N dimension: epilogue must not exceed mainloop output
    static_assert(get<1>(PV{}) >= get<1>(Epi{}),
      "[SM120 SAFETY][P2] Epilogue tile N dimension must not exceed mainloop PV N dimension. "
      "Epilogue cannot process more columns than mainloop produces. "
      "This will cause out-of-bounds access or silent data corruption. "
      "See SM120_FINAL_STATUS.md Point 2 for guidance.");

    // K dimension: must match head dimension exactly
    static_assert(get<2>(Epi{}) == Mainloop::HeadDimPV,
      "[SM120 SAFETY][P2] Epilogue tile K dimension must equal mainloop PV head dimension. "
      "Projection dimension mismatch will cause silent data corruption. "
      "Mainloop outputs HeadDimPV-dimensional vectors, epilogue must consume same dimension. "
      "See SM120_FINAL_STATUS.md Point 2 for guidance.");
  }
};

// ============================================================================
// [SM120 SAFETY][P8] Point 8: Alignment & Vectorization Detection
// ============================================================================

// Detect if Mainloop exposes TmemAllocation
template<class T, class = void>
struct has_TmemAllocation : std::false_type {};

template<class T>
struct has_TmemAllocation<T, void_t<typename T::TmemAllocation>> : std::true_type {};

// SFINAE-based alignment checker
template <class Mainloop, bool Has = has_TmemAllocation<Mainloop>::value>
struct AlignmentAssert {
  // No TmemAllocation exposed: no-op (silent)
  static void check() {}
};

template <class Mainloop>
struct AlignmentAssert<Mainloop, true> {
  // TmemAllocation exposed: perform validation
  static void check() {
    using TA = typename Mainloop::TmemAllocation;

    // TMEM buffer offsets are in 32-bit (4-byte) units
    // 128-bit (16-byte) alignment requirement: offset % 4 == 0

    static_assert((uint32_t)TA::O0 % 4 == 0,
      "[SM120 SAFETY][P8] O0 TMEM buffer must be 16-byte aligned (128-bit vectorization requirement). "
      "TMA/TMEM operations require 128-bit alignment for optimal performance. "
      "Misaligned buffers cause 4x performance degradation or silent data corruption. "
      "Adjust TmemAllocation buffer sizes to ensure 16-byte alignment. "
      "See SM120_FINAL_STATUS.md Point 8 for guidance.");

    static_assert((uint32_t)TA::P0 % 4 == 0,
      "[SM120 SAFETY][P8] P0 TMEM buffer must be 16-byte aligned (128-bit vectorization requirement). "
      "TMA/TMEM operations require 128-bit alignment for optimal performance. "
      "Misaligned buffers cause 4x performance degradation or silent data corruption. "
      "Adjust TmemAllocation buffer sizes to ensure 16-byte alignment. "
      "See SM120_FINAL_STATUS.md Point 8 for guidance.");

    // Additional buffers (K0, V0, etc.) can be added when exposed
    // static_assert((uint32_t)TA::K0 % 4 == 0, "[SM120 SAFETY][P8] K0 must be 16-byte aligned");
    // static_assert((uint32_t)TA::V0 % 4 == 0, "[SM120 SAFETY][P8] V0 must be 16-byte aligned");
  }
};

// ============================================================================
// Convenience Wrapper: Check All Available Introspection Points
// ============================================================================

template <class Mainloop, class KernelTraits>
struct SafetyIntrospectionSuite {
  static void check_all() {
    // These checks gracefully degrade to no-op if internals unavailable
    SmemBudgetAssert<Mainloop, KernelTraits>::check();  // Point 1
    EpiloguePVAssert<Mainloop, KernelTraits>::check();  // Point 2
    AlignmentAssert<Mainloop>::check();                 // Point 8
  }
};

// ============================================================================
// Usage Example
// ============================================================================
//
// In fmha_cutlass_fwd_sm100.cuh, after mainloop instantiation:
//
// ```cpp
// #include "safety_introspection.hpp"
//
// template <class KernelTraits, ...>
// struct FMHAKernel {
//   using Mainloop = /* ... mainloop instantiation ... */;
//
//   // Constructor or static initialization
//   FMHAKernel() {
//     // Run all available introspection checks
//     flash::safety::SafetyIntrospectionSuite<Mainloop, KernelTraits>::check_all();
//
//     // Or run individual checks:
//     // flash::safety::SmemBudgetAssert<Mainloop, KernelTraits>::check();
//     // flash::safety::EpiloguePVAssert<Mainloop, KernelTraits>::check();
//     // flash::safety::AlignmentAssert<Mainloop>::check();
//   }
// };
// ```
//
// ============================================================================
// Future Work: When UniversalCopy Mainloop Exposes Internals
// ============================================================================
//
// To enable these checks, modify your UniversalCopy mainloop class to expose:
//
// ```cpp
// struct Sm120MlaFwdMainloopUniversal {
//   // [SM120 SAFETY][P1] Expose total SMEM for budget validation
//   static constexpr size_t kSmemTotal = kSmemQBytes + kSmemKBytes +
//                                        kSmemVBytes + kSmemEpilogueBytes;
//
//   // [SM120 SAFETY][P2] Expose PV tile shape for epilogue compatibility
//   using TileShapePV = Shape<_64, _32, _64>;  // (M, N, K)
//   static constexpr int HeadDimPV = 64;
//
//   // [SM120 SAFETY][P8] Expose TMEM allocation for alignment checks
//   struct TmemAllocation {
//     enum : uint32_t {
//       O0 = 0,
//       P0 = /* ... ensure P0 % 4 == 0 for 16-byte alignment ... */,
//       K0 = /* ... */,
//       V0 = /* ... */
//     };
//   };
// };
// ```
//
// ============================================================================

} // namespace safety
} // namespace flash
