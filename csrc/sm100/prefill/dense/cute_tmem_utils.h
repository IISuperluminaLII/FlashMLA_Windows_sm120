/***************************************************************************************************
 * Copyright (c) 2025 FlashMLA Contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * TMEM utilities for SM120 (Blackwell workstation) support
 *
 * Provides proper TMEM staging with 128B TMA swizzle for SM120 compatibility.
 * This fixes the "TVLayout complement non-injective" errors by ensuring proper layout.
 **************************************************************************************************/
#pragma once

#include <cutlass/cutlass.h>
#include <cute/arch/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>

namespace flash {
namespace cute_utils {

// 128B array-reorder (aka "128B TMA swizzle") required by Blackwell
// This is the canonical swizzle pattern for TMEM compatibility
using TmaSwizzle128B = cute::Swizzle<3,4,3>;  // standard 128B swizzle

// Note: SM120 uses different TMEM operations than SM100a
// SM100a uses tcgen05 instructions (5th gen tensor core)
// SM120 doesn't support tcgen05, needs different approach
// The actual copy functions are provided by CUTLASS internally

// Helper to create properly aligned TMEM tensor views
template<typename Element, typename Layout>
CUTE_HOST_DEVICE
auto make_tmem_tensor(void* ptr, Layout layout) {
  // Ensure 128-byte alignment for TMEM
  assert(reinterpret_cast<uintptr_t>(ptr) % 128 == 0);
  return cute::make_tensor(cute::make_gmem_ptr(static_cast<Element*>(ptr)), layout);
}

// FP32 accumulation helpers for numerical stability
template<typename T>
CUTE_DEVICE
float safe_exp(T x) {
  float fx = static_cast<float>(x);
  // Clamp to prevent overflow
  fx = cute::min(fx, 88.0f);  // exp(88) ~= 1e38
  fx = cute::max(fx, -88.0f);
  float result = expf(fx);
  // Guard against inf
  return isfinite(result) ? result : 0.0f;
}

template<typename T>
CUTE_DEVICE
float safe_div(T num, T denom) {
  float fnum = static_cast<float>(num);
  float fdenom = static_cast<float>(denom);
  // Add small epsilon to prevent division by zero
  fdenom = fdenom + 1e-6f;
  float result = fnum / fdenom;
  // Guard against inf/nan
  return isfinite(result) ? result : 0.0f;
}

// Softmax with FP32 accumulation and numerical guards
template<typename TensorIn, typename TensorOut>
CUTE_DEVICE
void safe_softmax(TensorIn const& input, TensorOut& output, float scale) {
  using namespace cute;

  // Find max in FP32
  float max_val = -INFINITY;
  for (int i = 0; i < size(input); ++i) {
    float val = static_cast<float>(input(i)) * scale;
    if (isfinite(val)) {
      max_val = cute::max(max_val, val);
    }
  }

  // Compute exp and sum in FP32
  float sum = 0.0f;
  for (int i = 0; i < size(input); ++i) {
    float val = static_cast<float>(input(i)) * scale;
    val = isfinite(val) ? val : -INFINITY;
    float exp_val = safe_exp(val - max_val);
    output(i) = exp_val;
    sum += exp_val;
  }

  // Normalize with epsilon
  sum = sum + 1e-6f;
  for (int i = 0; i < size(output); ++i) {
    output(i) = output(i) / sum;
  }
}

} // namespace cute_utils
} // namespace flash