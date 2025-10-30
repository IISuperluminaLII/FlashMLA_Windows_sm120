/***************************************************************************************************
 * Copyright (c) 2025 FlashMLA Contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SM120 Safety Gate: Centralized Fail-Fast Checks
 *
 * This header is designed to be used with NVCC's --pre-include flag to provide
 * uniform device-side safety checks across all translation units without modifying
 * individual source files.
 *
 * Usage:
 *   nvcc ... --pre-include sm120_safety_gate.hpp \
 *            -DFLASH_MLA_SM120_UNIVERSAL_PREFILL=0 \
 *            -DFLASH_MLA_SM120_DISABLE_BWD=1
 *
 * This enforces Points 4 & 5 from the safety assertion checklist:
 * - Point 4: UniversalCopy gating (forward path)
 * - Point 5: SM120 BWD dead-code elimination
 **************************************************************************************************/

#pragma once

// ============================================================================
// Feature Flag Defaults
// ============================================================================
// These are expected to be set from your build system via -D flags.
// Defaults here are conservative (block SM120 paths).

#ifndef FLASH_MLA_SM120_UNIVERSAL_PREFILL
#define FLASH_MLA_SM120_UNIVERSAL_PREFILL 0  // 0 = TMA/TMEM path (blocked on SM120)
                                              // 1 = UniversalCopy path (when implemented)
#endif

#ifndef FLASH_MLA_SM120_DISABLE_BWD
#define FLASH_MLA_SM120_DISABLE_BWD 1        // 1 = SM120 backward disabled
                                              // 0 = SM120 backward enabled (requires UniversalCopy BWD)
#endif

// ============================================================================
// [SM120 SAFETY][P4+P5] Device-Side Fail-Fast Gates
// ============================================================================
// These checks run during CUDA device code compilation (__CUDA_ARCH__ defined).
// They prevent SM120 from compiling with incompatible code paths.
//
// WHY: Template instantiation happens at compile-time. Runtime guards in .cu
// files don't prevent template expansion in headers. Device-side #error
// directives catch misconfigurations before mysterious CUTLASS layout errors.
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)

// ----------------------------------------------------------------------------
// [SM120 SAFETY][P4] Point 4: UniversalCopy Gating (Forward Path)
// ----------------------------------------------------------------------------
// SM120 requires UniversalCopy mainloop for forward pass.
// TMA/TMEM path requires:
// - 128-column TMEM layouts (hardware constraint)
// - Large TMEM buffers exceeding SM120's 99KB budget
// - Multi-stage persistent scheduling unavailable with 99KB limit
//
// When UniversalCopy forward is implemented, set FLASH_MLA_SM120_UNIVERSAL_PREFILL=1
#if !FLASH_MLA_SM120_UNIVERSAL_PREFILL
  #error "[SM120 SAFETY][P4] SM120 device code requires UniversalCopy mainloop implementation. \
TMA/TMEM path is incompatible with SM120's 99KB shared memory. \
Enable -DFLASH_MLA_SM120_UNIVERSAL_PREFILL=1 when UniversalCopy is implemented. \
See SM120_UNIVERSAL_COPY_PLAN.md for implementation guidance."
#endif

// ----------------------------------------------------------------------------
// [SM120 SAFETY][P5] Point 5: SM120 BWD Dead-Code Elimination
// ----------------------------------------------------------------------------
// SM120 backward pass is explicitly disabled during bring-up phase.
// Strategy: Implement forward UniversalCopy first, validate, then implement backward.
//
// When UniversalCopy backward is implemented, set FLASH_MLA_SM120_DISABLE_BWD=0
#if FLASH_MLA_SM120_DISABLE_BWD
  #error "[SM120 SAFETY][P5] SM120 backward pass is currently disabled pending UniversalCopy BWD implementation. \
Forward path must be fully validated before enabling backward. \
Set -DFLASH_MLA_SM120_DISABLE_BWD=0 only when UniversalCopy BWD is implemented. \
See SM120_UNIVERSAL_COPY_PLAN.md for backward implementation plan."
#endif

#endif // __CUDA_ARCH__ >= 1200

// ============================================================================
// Host-Side Configuration Validation
// ============================================================================
// These checks run during host compilation to validate build configuration

#if !defined(__CUDA_ARCH__)
// Validate that flags are set to known values
#if !defined(FLASH_MLA_SM120_UNIVERSAL_PREFILL)
  #warning "[SM120 SAFETY] FLASH_MLA_SM120_UNIVERSAL_PREFILL not defined, using default=0 (TMA/TMEM blocked)"
#endif

#if !defined(FLASH_MLA_SM120_DISABLE_BWD)
  #warning "[SM120 SAFETY] FLASH_MLA_SM120_DISABLE_BWD not defined, using default=1 (backward disabled)"
#endif

// Document active configuration
#if FLASH_MLA_SM120_UNIVERSAL_PREFILL
  #pragma message("[SM120 CONFIG] UniversalCopy forward: ENABLED")
#else
  #pragma message("[SM120 CONFIG] UniversalCopy forward: DISABLED (SM120 forward will fail to compile)")
#endif

#if FLASH_MLA_SM120_DISABLE_BWD
  #pragma message("[SM120 CONFIG] SM120 backward: DISABLED (SM120 backward will fail to compile)")
#else
  #pragma message("[SM120 CONFIG] SM120 backward: ENABLED")
#endif

#endif // !__CUDA_ARCH__

// ============================================================================
// Usage Notes
// ============================================================================
//
// ## Build System Integration
//
// ### PyTorch setup.py:
// ```python
// extra_compile_args = {
//     'nvcc': [
//         '--pre-include', 'csrc/sm100/prefill/dense/sm120_safety_gate.hpp',
//         '-DFLASH_MLA_SM120_UNIVERSAL_PREFILL=0',
//         '-DFLASH_MLA_SM120_DISABLE_BWD=1',
//     ]
// }
// ```
//
// ### CMake:
// ```cmake
// target_compile_options(flash_mla PRIVATE
//   $<$<COMPILE_LANGUAGE:CUDA>:
//     --pre-include ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sm100/prefill/dense/sm120_safety_gate.hpp
//     -DFLASH_MLA_SM120_UNIVERSAL_PREFILL=0
//     -DFLASH_MLA_SM120_DISABLE_BWD=1
//   >
// )
// ```
//
// ## Testing Strategy
//
// Verify fail-fast behavior with negative tests:
//
// 1. **SM120 forward blocked** (expected failure):
//    ```bash
//    nvcc ... --pre-include sm120_safety_gate.hpp \
//      -DFLASH_MLA_SM120_UNIVERSAL_PREFILL=0 \
//      -gencode=arch=compute_120,code=sm_120
//    # Expected: [SM120 SAFETY][P4] error during compilation
//    ```
//
// 2. **SM120 backward blocked** (expected failure):
//    ```bash
//    nvcc ... --pre-include sm120_safety_gate.hpp \
//      -DFLASH_MLA_SM120_UNIVERSAL_PREFILL=1 \
//      -DFLASH_MLA_SM120_DISABLE_BWD=1 \
//      -gencode=arch=compute_120,code=sm_120
//    # Expected: [SM120 SAFETY][P5] error during compilation
//    ```
//
// 3. **SM100a happy path** (expected success):
//    ```bash
//    nvcc ... --pre-include sm120_safety_gate.hpp \
//      -gencode=arch=compute_100,code=sm_100
//    # Expected: Clean compilation (no SM120 gates triggered)
//    ```
//
// ============================================================================
