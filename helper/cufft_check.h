#pragma once

#include <cufft.h>
#include <stdio.h>

static const char* _cufftGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

#define CUFFT_CHK(ans) { cufftCheck((ans), __FILE__, __LINE__); }

inline void cufftCheck(cufftResult error, const char *file, const int line, bool abort = true)
{
    if (error != CUFFT_SUCCESS)
    {
        fprintf(stderr,"CUFFT error: %s %s:%d\n", _cufftGetErrorEnum(error), file, line);
        cudaDeviceReset();
        if (abort)
        {
            getchar();
            exit(error);
        }
    }
}