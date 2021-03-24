#pragma once

#include <cufft.h>

class GpuImageBlur
{
public:
    GpuImageBlur(int image_rows, int image_cols, int max_images, int kernel_size);

    ~GpuImageBlur();

    void blur(float* blurred_image, float* images, int num_images);

private:
    void initGaussianFilter();

    int NY = 0;
    int NX = 0;
    int max_images = 0;
    int kernel_size = 0;
    cufftComplex* d_complex = nullptr;
    cufftComplex* d_gaussian_kernel = nullptr;
};