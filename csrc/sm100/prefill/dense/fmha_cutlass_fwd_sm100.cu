#include "interface.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "common/mask.cuh"
#include "common/utils.hpp"

#include "sm100_kernel_traits.hpp"
#include "fmha_cutlass_fwd_sm100.cuh"

template <class Mask, class Varlen, class Element, class ElementOut, class Mla>
void call_run_fmha_fwd([[maybe_unused]] Mask mask, [[maybe_unused]] Varlen is_varlen,
                       [[maybe_unused]] Element in, [[maybe_unused]] ElementOut out,
                       [[maybe_unused]] Mla mla, at::Tensor workspace_buffer, at::Tensor q,
                       at::Tensor k, at::Tensor v, at::Tensor cumulative_seqlen_q,
                       at::Tensor cumulative_seqlen_kv, at::Tensor o, at::Tensor lse,
                       float softmax_scale, int max_seqlen_q, int max_seqlen_kv) {
  static constexpr bool IsVarlen = std::is_same_v<Varlen, true_type>;
  static constexpr bool IsMla = std::is_same_v<Mla, true_type>;
  static constexpr bool IsCausalMask = std::is_same_v<Mask, CausalMask<false>>;
  using Option =
      std::conditional_t<IsCausalMask || (IsVarlen), Option<Tag::kIsPersistent, false_type>,
                         Option<Tag::kIsPersistent, true_type>>;

  // Dual dispatch: Runtime architecture detection for SM100a (server) vs SM120 (workstation)
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  int sm_major = prop.major;
  int sm_minor = prop.minor;
  int sm_version = sm_major * 10 + sm_minor;

  // SM100a = compute_100a (10.0), SM120 = compute_120 (12.0)
  // Runtime guard: Only dispatch if architecture is supported
  if (sm_version >= 120) {
    // Workstation variant: SM120 (RTX 6000 Pro, RTX 50 series)
    run_fmha_fwd<flash::Sm120WorkstationConfig, Element, ElementOut, IsVarlen, IsMla, Mask, Option>(
        workspace_buffer, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse,
        softmax_scale, max_seqlen_q, max_seqlen_kv);
  }
#ifndef FLASH_MLA_DISABLE_SM100
  else if (sm_version >= 100) {
    // Server variant: SM100a (B100/B200)
    run_fmha_fwd<flash::Sm100ServerConfig, Element, ElementOut, IsVarlen, IsMla, Mask, Option>(
        workspace_buffer, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse,
        softmax_scale, max_seqlen_q, max_seqlen_kv);
  }
#endif
  else {
    FLASH_MLA_ASSERT(false && "Unsupported SM architecture: requires SM100a or SM120");
  }
}

void FMHACutlassSM100FwdRun(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k,
                            at::Tensor v, at::Tensor cumulative_seqlen_q,
                            at::Tensor cumulative_seqlen_kv, at::Tensor o, at::Tensor lse,
                            int mask_mode_code, float sm_scale, int max_seqlen_q,
                            int max_seqlen_kv, bool is_varlen) {
  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  CHECK(q.scalar_type() == k.scalar_type());
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();
  int head_dim_qk = q.size(-1);
  int head_dim_vo = v.size(-1);
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  if (scalar_type_in == at::ScalarType::BFloat16 &&
      scalar_type_out == at::ScalarType::BFloat16) {
    using Element = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;

    auto apply_config = [&](auto fn) {
      if (mask_mode == MaskMode::kCausal) {
        if (is_varlen) {
          fn(CausalMask<false>{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(CausalMask<false>{}, cute::false_type{}, Element{}, ElementOut{});
        }
      } else {
        if (is_varlen) {
          fn(ResidualMask{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(ResidualMask{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
    };

    apply_config([&](auto mask, auto varlen, auto in, auto out) {
      if (head_dim_qk == 192 && head_dim_vo == 128) {
        call_run_fmha_fwd(mask, varlen, in, out, true_type{}, workspace_buffer, q, k, v,
                          cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale,
                          max_seqlen_q, max_seqlen_kv);
      } else if (head_dim_qk == 128 && head_dim_vo == 128) {
        call_run_fmha_fwd(mask, varlen, in, out, false_type{}, workspace_buffer, q, k, v,
                          cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale,
                          max_seqlen_q, max_seqlen_kv);
      } else {
        std::cout << "No kernel instantiated for head_dim_qk=" << head_dim_qk
                  << " head_dim_vo=" << head_dim_vo << std::endl;
      }
    });

  } else {
    FLASH_MLA_ASSERT(false);
  }
}
