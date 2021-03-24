#pragma once

#include <cufft.h>

class GpuBlurDetector
{
public:
    GpuBlurDetector(int image_rows, int image_cols, int max_images, int hp_filter_size = -1);

    ~GpuBlurDetector();

    /**
     * Calculate how blurry an image is. The more blurry the image the lower the result will be.
     *
     * Algorithm Overview
     *      1) Calculate the FFT of the image
     *      2) Scale the FFT output (equivalent to opencv's DFT_SCALE flag)
     *      3) Move the FFT's low frequencies to the center of the image (fftShift)
     *      4) Run a high pass filter on the shifted FFT output
     *      5) Reconstruct the image by running an inverse FFT on the filtered FFT output
     *      6) Calculate mean of the magnitude of the reconstructed image
     *
     * @param blur_results How blurry the image is (sharper is larger, blurry is smaller)
     * @param images Host allocated batch of images
     * @param num_images How many images to process
     * @param vis Visualize the magnitude spectrum and reconstructed image
     */
    void detectBlur(float* blur_results, float* images, int num_images, bool vis=false);

private:
    void displayImage(int image);

    void displayMagnitude(int image);

    int NY = 0;
    int NX = 0;
    int max_images = 0;
    int hp_filter_size = 0; // High pass filter box dimensions (width == height)
    cufftComplex* d_complex = nullptr;
    cufftHandle plan;
};