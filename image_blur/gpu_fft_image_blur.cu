#include "gpu_fft_image_blur.h"
#include "helper/cuda_check.h"
#include "helper/cufft_check.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cassert>

namespace kernel
{
/**
* Launches threads for the top-left and top-right quadrants, which will then respectively swap themselves
* with the bottom-right and bottom-left quadrants.
*/
template<typename T>
__global__
void cufftShift_2D(T* data, int NY, int NX)
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
* Zeros out the high frequencies in the center of the FFT
*/
__global__ void lowPassFilter_kernel(cufftComplex* d_complex, int NY, int NX, int lp_filter_size)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const int zIndex = blockIdx.z;

    if (xIndex < lp_filter_size * 2 && yIndex < lp_filter_size * 2)
    {
        d_complex += (zIndex * NY * NX);

        const int NY_HALF = NY / 2;
        const int NX_HALF = NX / 2;
        const int y_offset = NY_HALF - lp_filter_size + yIndex;
        const int x_offset = NX_HALF - lp_filter_size + xIndex;
        cufftComplex zero { .x = 0.f, .y = 0.f };
        d_complex[y_offset * NX + x_offset] = zero;
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

__global__ void complexPointwiseMulAndScale(cufftComplex *a, cufftComplex *b, int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    if (index < size)
    {
        float scale = 1.0f / (float)size;
        cufftComplex c = cuCmulf(a[batch * size + index], b[batch * size + index]);
        b[batch * size + index] = make_cuFloatComplex(scale * cuCrealf(c), scale * cuCimagf(c));
    }
}
}

GpuImageBlur::GpuImageBlur(int image_rows, int image_cols, int max_images, int kernel_size)
    : NY(image_rows)
    , NX(image_cols)
    , max_images(max_images)
    , kernel_size(kernel_size)
{
    CUDA_CHK(cudaMalloc(&d_complex, sizeof(cufftComplex) * NY * NX * max_images));
    CUDA_CHK(cudaMallocManaged(&d_gaussian_kernel, sizeof(cufftComplex) * NY * NX));

    initGaussianFilter();

    printf("before\n");
    for( int i = 0; i < NY; ++i) {
        for( int j = 0; j < NX; ++j) {
            printf("%f ", d_gaussian_kernel[i * NX + j].x);
        }
        printf("\n");
    }

    cufftHandle plan;
    CUFFT_CHK(cufftPlan2d(&plan, NY, NX, CUFFT_C2C));
    CUFFT_CHK(cufftExecC2C(plan, d_gaussian_kernel, d_gaussian_kernel, CUFFT_FORWARD));

    CUDA_CHK(cudaDeviceSynchronize());

    printf("after\n");
    for( int i = 0; i < NY; ++i) {
        for( int j = 0; j < NX; ++j) {
            printf("%f ", d_gaussian_kernel[i * NX + j].x);
        }
        printf("\n");
    }
}

GpuImageBlur::~GpuImageBlur()
{
    CUDA_CHK(cudaFree(d_complex));
    CUDA_CHK(cudaFree(d_gaussian_kernel));
}

void GpuImageBlur::blur(float *blurred_image, float *images, int num_images)
{
    assert(num_images <= max_images);
    const int total = NY * NX * num_images;

    const int lp_filter_size = 125;//std::min(NX, NY) * (blur_percent/2) * 2;
    printf("rows: %d, cols: %d, low_pass_filter_size: %d\n", NY, NX, lp_filter_size);

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
        kernel::complexPointwiseMulAndScale<<<grid, block>>>(d_gaussian_kernel, d_complex, NY * NX);
    }

//    {
//        dim3 block(256);
//        dim3 grid((NY * NX + block.x + 1) / block.x, num_images);
//        cufftScale_kernel<<<grid, block>>>(d_complex, NY * NX);
//    }
//
//    {
//        dim3 block(16, 16);
//        dim3 grid((lp_filter_size * 2 + block.x + 1) / block.x, (lp_filter_size * 2 + block.y + 1) / block.y, num_images);
//        lowPassFilter_kernel<<<grid, block>>>(d_complex, NY, NX, lp_filter_size);
//    }

    CUFFT_CHK(cufftExecC2C(plan, d_complex, d_complex, CUFFT_INVERSE));

//    {
//        dim3 block(16, 16);
//        dim3 grid((NX + block.x + 1) / block.x, (NY + block.y + 1) / block.y, num_images);
//        kernel::cufftShift_2D<cufftComplex><<<grid, block>>>(d_complex, NY, NX);
//    }

    // Copy down real data
    CUDA_CHK(cudaMemcpy2D(images, sizeof(float), d_complex, sizeof(cufftComplex), sizeof(float), total, cudaMemcpyDeviceToHost));

    CUFFT_CHK(cufftDestroy(plan));
}

void GpuImageBlur::initGaussianFilter()
{
    float sigma = 10.0;
    float q = 2.0 * sigma * sigma;
    float sum = 0.0; // sum is for normalization

    std::vector<float> kernel(kernel_size * kernel_size);

    int size = kernel_size / 2;

    for (int x = -size; x <= size; x++)
    {
        for (int y = -size; y <= size; y++)
        {
            float p = sqrt(x * x + y * y);
            kernel[(x + size) * kernel_size + (y + size)] = (exp(-(p * p) / q)) / (M_PI * q);
            sum += kernel[(x + size) * kernel_size + (y + size)];
        }
    }

    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            kernel[i * kernel_size + j] /= sum;
            printf("%f ", kernel[i * kernel_size + j]);
        }
        printf("\n");
    }

    // Set d_complex[*].y to 0.f
    CUDA_CHK(cudaMemset(d_gaussian_kernel, 0, sizeof(cufftComplex) * NY * NX));

    // Set d_complex[*].x
    for( int i = 0; i < kernel_size; ++i )
    {
        CUDA_CHK(cudaMemcpy2D(&d_gaussian_kernel[i * NX], sizeof(cufftComplex), &kernel[i * kernel_size], sizeof(float), sizeof(float), kernel_size, cudaMemcpyHostToDevice));
    }
}