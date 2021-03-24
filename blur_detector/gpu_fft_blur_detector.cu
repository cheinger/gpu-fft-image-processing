#include "gpu_fft_blur_detector.h"
#include "helper/cuda_check.h"
#include "helper/cufft_check.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cassert>

/**
 * Launches threads for the top-left and top-right quadrants, which will then respectively swap themselves
 * with the bottom-right and bottom-left quadrants.
 */
template<typename T>
__global__
void cufftShift_2D_kernel(T* data, int NY, int NX)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const int zIndex = blockIdx.z;

    const int NY_HALF = NY / 2;
    const int NX_HALF = NX / 2;

    if (yIndex >= NY_HALF || xIndex >= NX)
    {
        return;
    }

    int index = yIndex * NX + xIndex;

    data += (zIndex * NY * NX);

    // Swap top-left quadrant with bottom-right
    if (xIndex < NX_HALF)
    {
        T regTemp = data[index];

        // Set top-left quad
        const int bottomRightIndex = (yIndex + NY_HALF) * NX + xIndex + NX_HALF;
        data[index] = data[bottomRightIndex];

        // Set bottom-right
        data[bottomRightIndex] = regTemp;
    }
        // Swap top-right quadrant with bottom-left
    else
    {
        T regTemp = data[index];

        // Set top-right quad
        const int bottomLeftIndex = (yIndex + NY_HALF) * NX + (xIndex - NX_HALF);
        data[index] = data[bottomLeftIndex];

        data[bottomLeftIndex] = regTemp;
    }
}

/**
 * Normalizes the cuFFT output by dividing the results by the size of the output
 */
__global__ void cufftScale_kernel(cufftComplex* d_complex, int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    if (index < size)
    {
        d_complex[batch * size + index].x /= size;
        d_complex[batch * size + index].y /= size;
    }
}

/**
 * Zeros out the low frequencies in the center of the FFT
 */
__global__ void highPassFilter_kernel(cufftComplex* d_complex, int NY, int NX, int hp_filter_size)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const int zIndex = blockIdx.z;

    if (xIndex < hp_filter_size * 2 && yIndex < hp_filter_size * 2)
    {
        d_complex += (zIndex * NY * NX);

        const int NY_HALF = NY / 2;
        const int NX_HALF = NX / 2;
        const int y_offset = NY_HALF - hp_filter_size + yIndex;
        const int x_offset = NX_HALF - hp_filter_size + xIndex;
        cufftComplex zero { .x = 0.f, .y = 0.f };
        d_complex[y_offset * NX + x_offset] = zero;
    }
}

__global__ void cufftMagnitude_kernel(cufftComplex* d_complex, int size)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex < size)
    {
        float magnitude = logf(abs(d_complex[xIndex].x)) * 20;
        d_complex[xIndex].x = magnitude;
    }
}

GpuBlurDetector::GpuBlurDetector(int image_rows, int image_cols, int max_images, int hp_filter_size)
    : NY(image_rows)
    , NX(image_cols)
    , max_images(max_images)
    , hp_filter_size(hp_filter_size)
{
    if (hp_filter_size * 2 > image_rows || hp_filter_size * 2 > image_cols)
    {
        throw std::invalid_argument("High pass filter size cannot be larger than the image");
    }

    CUDA_CHK(cudaMalloc(&d_complex, sizeof(cufftComplex) * NY * NX * max_images));
}

GpuBlurDetector::~GpuBlurDetector()
{
    CUDA_CHK(cudaFree(d_complex));
}

void GpuBlurDetector::detectBlur(float* blur_results, float* images, int num_images, bool vis)
{
    assert(num_images <= max_images);
    const int total = NY * NX * num_images;

    // Set d_complex[*].y to 0.f
    CUDA_CHK(cudaMemset(d_complex, 0, sizeof(cufftComplex) * total));
    // Set d_complex[*].x
    CUDA_CHK(cudaMemcpy2D(d_complex, sizeof(cufftComplex), images, sizeof(float), sizeof(float), total, cudaMemcpyHostToDevice));

    int n[2] = {NY, NX};
    cufftHandle plan;
    CUFFT_CHK(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, num_images));
    CUFFT_CHK(cufftExecC2C(plan, d_complex, d_complex, CUFFT_FORWARD));

    {
        dim3 block(256);
        dim3 grid((NY * NX + block.x + 1) / block.x, num_images);
        cufftScale_kernel<<<grid, block>>>(d_complex, NY * NX);
    }

    {
        dim3 block(16, 16);
        dim3 grid((NX + block.x + 1) / block.x, (NY + block.y + 1) / block.y, num_images);
        cufftShift_2D_kernel<cufftComplex><<<grid, block>>>(d_complex, NY, NX);
    }

    {
        dim3 block(16, 16);
        dim3 grid((hp_filter_size * 2 + block.x + 1) / block.x, (hp_filter_size * 2 + block.y + 1) / block.y, num_images);
        highPassFilter_kernel<<<grid, block>>>(d_complex, NY, NX, hp_filter_size);
    }

    if( vis )
    {
        displayMagnitude(0);
    }

    CUFFT_CHK(cufftExecC2C(plan, d_complex, d_complex, CUFFT_INVERSE));

    {
        dim3 block(256);
        dim3 grid((total + block.x + 1) / block.x);
        cufftMagnitude_kernel<<<grid, block>>>(d_complex, total);
    }

    if( vis )
    {
        displayImage(0);
    }

    // TODO: Calculate mean on GPU
    for( int i = 0; i < num_images; ++i )
    {
        cv::Mat real(NY, NX, CV_32F);
        CUDA_CHK(cudaMemcpy2D(real.data, sizeof(float), &d_complex[i * NY * NX], sizeof(cufftComplex), sizeof(float), NY * NX, cudaMemcpyDeviceToHost));

        cv::Scalar result= cv::mean(real);

        blur_results[i] = result.val[0];
    }

}

void GpuBlurDetector::displayImage(int image)
{
    const auto d_image = d_complex + image * NY * NX;

    cv::Mat real(NY, NX, CV_32F);
    CUDA_CHK(cudaMemcpy2D(real.data, sizeof(float), d_image, sizeof(cufftComplex), sizeof(float), NY * NX, cudaMemcpyDeviceToHost));
    real.convertTo(real, CV_8U); // Back to 8-bits

    cv::imshow("Reconstructed Image", real);
    cv::waitKey();
}

void GpuBlurDetector::displayMagnitude(int image)
{
    const auto d_image = d_complex + image * NY * NX;

    cv::Mat complex_x(NY, NX, CV_32F);
    cv::Mat complex_y(NY, NX, CV_32F);
    CUDA_CHK(cudaMemcpy2D(complex_x.data, sizeof(float), d_image, sizeof(cufftComplex), sizeof(float), NY * NX, cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy2D(complex_y.data, sizeof(float), ((float*)d_image) + 1, sizeof(cufftComplex), sizeof(float), NY * NX, cudaMemcpyDeviceToHost));

    cv::Mat magI;
    cv::magnitude(complex_x, complex_y, magI); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    // switch to logarithmic scale: log(1 + magnitude)
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    // Transform the magnitude matrix into a viewable image (float values 0-1)
    cv::normalize(magI, magI, 1, 0, cv::NORM_INF);
    cv::imshow("Magnitude Spectrum", magI);
    cv::waitKey();
}