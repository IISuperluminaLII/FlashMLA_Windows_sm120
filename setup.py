import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME
)


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]

def get_features_args(for_msvc=False):
    """Get feature flags. Use for_msvc=True to get /D flags for MSVC, otherwise -D for NVCC"""
    features_args = []
    prefix = "/D" if for_msvc else "-D"

    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        features_args.append(f"{prefix}FLASH_MLA_DISABLE_FP16")
    if is_flag_set("FLASH_MLA_DISABLE_SM90"):
        features_args.append(f"{prefix}FLASH_MLA_DISABLE_SM90")
    if is_flag_set("FLASH_MLA_DISABLE_SM100"):
        features_args.append(f"{prefix}FLASH_MLA_DISABLE_SM100")
    if is_flag_set("FLASH_MLA_SM120_DISABLE_BWD"):
        features_args.append(f"{prefix}FLASH_MLA_SM120_DISABLE_BWD")
    return features_args

def get_arch_flags():
    # Check NVCC Version
    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable. For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), '--version'], stderr=subprocess.STDOUT
    ).decode('utf-8')
    nvcc_version_number = nvcc_version.split('release ')[1].split(',')[0].strip()
    major, minor = map(int, nvcc_version_number.split('.'))
    print(f'Compiling using NVCC {major}.{minor}')

    DISABLE_SM100 = is_flag_set("FLASH_MLA_DISABLE_SM100")
    DISABLE_SM90 = True  # Hardcoded to True for Windows/MSVC compatibility
    if major < 12 or (major == 12 and minor <= 8):
        assert DISABLE_SM100, "sm100 compilation for Flash MLA requires NVCC 12.9 or higher. Please set FLASH_MLA_DISABLE_SM100=1 to disable sm100 compilation, or update your environment."

    arch_flags = []

    # Blackwell server (sm_100a) - keep if you want those kernels too
    if not DISABLE_SM100:
        arch_flags.extend([
            "-gencode", "arch=compute_100a,code=sm_100a",
            # Add PTX for future-proofing of the 100a path as well
            "-gencode", "arch=compute_100a,code=compute_100a",
        ])

    # Hopper (optional, keep if you also test on H100)
    if not DISABLE_SM90:
        arch_flags.extend([
            "-gencode", "arch=compute_90a,code=sm_90a",
            "-gencode", "arch=compute_90a,code=compute_90a",
        ])
        print("[ERROR] SM90 architecture flags unexpectedly included!")
    else:
        print("[OK] Excluding SM90 architecture flags (disabled for Windows)")

    # SM120 (Blackwell workstation - RTX 6000 Pro)
    # Strategy: Don't compile for SM120 directly due to TMEM layout incompatibilities
    # Build only for SM100a - SM120 will run SM100a binaries via architectural compatibility
    print("[OK] SM120 will use SM100a binaries (no sm_120a compilation)")

    return arch_flags

def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]

def get_nvcc_cxx_flags():
    """Get platform-specific flags to pass to NVCC's host compiler"""
    if IS_WINDOWS:
        return ["-Xcompiler", "/Zc:__cplusplus"]
    return []



this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    # Force-include msvc_compat.h to prevent std namespace ambiguity
    # Add Windows-friendly defines for MSVC compatibility
    cxx_args = ["/O2", "/std:c++17", "/Zc:__cplusplus", "/EHsc", "/permissive-",
                "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN", "/D_HAS_EXCEPTIONS=1",
                "/utf-8", "/DNDEBUG", "/W0", "/FImsvc_compat.h", "/DFLASH_MLA_FORCE_FALLBACK"]
else:
    cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

# Build source list based on enabled architectures
DISABLE_SM100 = is_flag_set("FLASH_MLA_DISABLE_SM100")
DISABLE_SM90 = True  # Always disable SM90 on Windows - has compilation errors

sources = [
    "csrc/pybind.cpp",
    "csrc/smxx/get_mla_metadata.cu",
    "csrc/smxx/mla_combine.cu",
]

# Only include sm90 sources if not disabled (they have Windows/MSVC incompatibilities)
# Use the hardcoded DISABLE_SM90 variable, not the environment check
if not DISABLE_SM90:
    sources.extend([
        "csrc/sm90/decode/dense/splitkv_mla.cu",
        "csrc/sm90/decode/sparse_fp8/splitkv_mla.cu",
        "csrc/sm90/prefill/sparse/fwd.cu",
    ])
    print("Including SM90 source files")
else:
    print("[OK] Excluding SM90 source files (disabled for Windows/MSVC compatibility)")

# SM100a training kernels (fwd/bwd)
# Strategy: Include these and build ONLY for SM100a architecture
# SM120 hardware will run SM100a binaries
if not DISABLE_SM100:
    sources.extend([
        "csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
        "csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",
    ])
    print("[OK] Including SM100 training kernels (will compile for sm_100a only)")
else:
    print("Excluding SM100 training kernels (FLASH_MLA_DISABLE_SM100 set)")

# SM100a-specific decode/sparse kernels use tcgen05 (SM100a-only instructions)
# Exclude these for now - not needed for basic training
print("[OK] Excluding SM100 decode/sparse files (not needed for training)")

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla.cuda",
        sources=sources,
        extra_compile_args={
            "cxx": cxx_args + get_features_args(for_msvc=IS_WINDOWS),
            "nvcc": [
                                # Additional NVCC flags for Windows/MSVC build

                "-include", "msvc_compat.h",  # Force-include MSVC compatibility shim for host passes
                "-O3",
                "-std=c++17",
                "-DNDEBUG",
                "-D_USE_MATH_DEFINES",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ptxas-options=-v,--register-usage-level=10",
                # Windows-specific NVCC flags for clang-cl compatibility
                "-Xcompiler", "/Zc:__cplusplus",
                "-Xcompiler", "/permissive-"
            ] + (["-DFLASH_MLA_FORCE_FALLBACK=1"] if IS_WINDOWS else [])
              + get_nvcc_cxx_flags()
              + get_features_args(for_msvc=False)
              + get_arch_flags()
              + get_nvcc_thread_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "sm90",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
        ],
    )
)

try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str


setup(
    name="flash_mla",
    version="1.0.0" + rev,
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
