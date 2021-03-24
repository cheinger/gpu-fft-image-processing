#pragma once

#include <cufft.h>
#include <vector>

class GpuBlurImage
{
public:
    GpuBlurImage(int image_rows, int image_cols, int max_images, int kernel_size);

    ~GpuBlurImage();

    /**
     * Transform an image into a blurry image.
     *
     * Algorithm Overview
     *      1) Calculate the FFT of the image
     *      2) Point-wise Multiply the image FFT by the FFT of the padded Gaussian filter
     *      3) Calculate the inverse FFT
     *
     * @param blurred_images Resulting batch of blurred images
     * @param images Host allocated batch of images
     * @param num_images How many images to process
     */
    void blur(float* blurred_images, float* images, int num_images);

private:
    static std::vector<float> createGaussianFilter(int kernel_size);

    int NY = 0;
    int NX = 0;
    int max_images = 0;
    cufftComplex* d_complex = nullptr;
    cufftComplex* d_gaussian_kernel = nullptr;
    cufftHandle plan;
};