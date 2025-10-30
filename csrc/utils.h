#pragma once

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)


#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                          \
            asm("trap;");                                                                                 \
        }                                                                                                 \
    } while(0)

#define println(fmt, ...) { print(fmt, ##__VA_ARGS__); print("\n"); }

template<typename T>
__inline__ __host__ __device__ T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

// For development, we define both IS_SM100 and IS_SM90 when using CLion or VSCode IDEs so code highlighting will be correct.
#if defined(__CLION_IDE__) || defined(__VSCODE_IDE__)
#define IS_SM100 1
#define IS_SM90 1
#else

// We define the following macros to detect the CUDA architecture, so that we can enable/disable certains kernels that depends on specific architectures.
// SM120 (Blackwell GeForce) uses same kernels as SM100 (Blackwell datacenter) with modifications
// Based on vLLM PR#17280 and community fixes for RTX 50 series support
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 1000) || (__CUDA_ARCH__ == 1200))
#define IS_SM100 1  // Enable SM100 kernels for both SM100a and SM120
#else
#define IS_SM100 0
#endif

// Separate SM120 detection for GeForce-specific handling
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200)
#define IS_SM120 1
#else
#define IS_SM120 0
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define IS_SM90 1
#else
#define IS_SM90 0
#endif

#endif  // defined(__CLION_IDE__) || defined(__VSCODE_IDE__)