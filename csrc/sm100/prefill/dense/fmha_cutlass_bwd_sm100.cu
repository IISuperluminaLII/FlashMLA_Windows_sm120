#include "interface.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include "common/mask.cuh"
#include "common/utils.hpp"
#include "sm100_kernel_traits.hpp"

#include "fmha_cutlass_bwd_sm100.cuh"

template<class Mask, class Varlen, class Element, class ElementOut, class Mla>
void call_run_fmha_bwd([[maybe_unused]] Mask mask, [[maybe_unused]] Varlen is_varlen,
                      [[maybe_unused]] Element in, [[maybe_unused]] ElementOut out, [[maybe_unused]] Mla mla,
                  at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                  at::Tensor v, at::Tensor o, at::Tensor lse,
                  at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                  at::Tensor dq, at::Tensor dk, at::Tensor dv,
                  float softmax_scale, int max_seqlen_q, int total_seqlen_kv) {
  static constexpr bool IsVarlen = std::is_same_v<Varlen, true_type>;
  static constexpr bool IsMla = std::is_same_v<Mla, true_type>;

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
    using TileShape = std::conditional_t<IsMla,
                                         typename flash::Sm120WorkstationConfig::TileShapeMlaBwd,
                                         typename flash::Sm120WorkstationConfig::TileShapeFmhaBwd>;
    run_fmha_bwd<flash::Sm120WorkstationConfig, Element, IsVarlen, IsMla, TileShape, Mask>(
        workspace_buffer, d_o, q, k, v, o, lse,
        cumulative_seqlen_q, cumulative_seqlen_kv,
        dq, dk, dv,
        softmax_scale, max_seqlen_q, total_seqlen_kv);
  }
#ifndef FLASH_MLA_DISABLE_SM100
  else if (sm_version >= 100) {
    // Server variant: SM100a (B100/B200)
    using TileShape = std::conditional_t<IsMla,
                                         typename flash::Sm100ServerConfig::TileShapeMlaBwd,
                                         typename flash::Sm100ServerConfig::TileShapeFmhaBwd>;
    run_fmha_bwd<flash::Sm100ServerConfig, Element, IsVarlen, IsMla, TileShape, Mask>(
        workspace_buffer, d_o, q, k, v, o, lse,
        cumulative_seqlen_q, cumulative_seqlen_kv,
        dq, dk, dv,
        softmax_scale, max_seqlen_q, total_seqlen_kv);
  }
#endif
  else {
    FLASH_MLA_ASSERT(false && "Unsupported SM architecture: requires SM100a or SM120");
  }
}


void FMHACutlassSM100BwdRun(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                            at::Tensor v, at::Tensor o, at::Tensor lse,
                            at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                            at::Tensor dq, at::Tensor dk, at::Tensor dv,
                            int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) {

  const c10::cuda::OptionalCUDAGuard device_guard(q.device());

  int head_dim_qk = q.size(-1);
  int head_dim_vo = v.size(-1);
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();

  if(scalar_type_in == at::ScalarType::BFloat16 && scalar_type_out == at::ScalarType::BFloat16) {
    using Element = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;

    auto apply_config = [&](auto fn) {
      if (mask_mode == MaskMode::kCausal) {
        if(is_varlen) {
          fn(CausalForBackwardMask<false>{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(CausalForBackwardMask<false>{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
      else {
        if(is_varlen) {
          fn(ResidualMaskForBackward{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(ResidualMaskForBackward{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
    };

    apply_config([&](auto mask, auto varlen, auto in, auto out) {
      if (head_dim_qk == 192 && head_dim_vo == 128) {
        call_run_fmha_bwd(mask, varlen, in, out, true_type{}, workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, max_seqlen_kv);
      } else if (head_dim_qk == 128 && head_dim_vo == 128) {
        call_run_fmha_bwd(mask, varlen, in, out, false_type{}, workspace_buffer, d_o, q, k, v, o, lse,
                          cumulative_seqlen_q, cumulative_seqlen_kv,
                          dq, dk, dv,
                          softmax_scale, max_seqlen_q, max_seqlen_kv);      }
      else {
        std::cout << "No kernel instantiated for head_dim_qk=" << head_dim_qk << " head_dim_vo=" << head_dim_vo << std::endl;
      }
    });

  } else {
    FLASH_MLA_ASSERT(false);
  }
}
