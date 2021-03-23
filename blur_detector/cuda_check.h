#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHK(ans) { cudaCheck((ans), __FILE__, __LINE__); }

inline void cudaCheck(cudaError_t error, const char *file, int line, bool abort = true)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr,"CUDA error: %s %s:%d\n", cudaGetErrorString(error), file, line);
        cudaDeviceReset();
        if (abort)
        {
            getchar();
            exit(error);
        }
    }
}