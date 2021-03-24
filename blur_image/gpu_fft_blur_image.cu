#include "gpu_fft_blur_image.h"
#include "helper/cuda_check.h"
#include "helper/cufft_check.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cassert>

__global__ void complexPointwiseMulAndScale_kernel(cufftComplex *a, cufftComplex *b, int size)
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

GpuBlurImage::GpuBlurImage(int image_rows, int image_cols, int max_images, int kernel_size)
    : NY(image_rows)
    , NX(image_cols)
    , max_images(max_images)
{
    if (kernel_size > std::min(NY, NX))
    {
        throw std::invalid_argument("Kernel size cannot be larger than the image dimensions");
    }
    
    int n[2] = {NY, NX};
    CUFFT_CHK(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, max_images));

    CUDA_CHK(cudaMalloc(&d_complex, sizeof(cufftComplex) * NY * NX * max_images));
    CUDA_CHK(cudaMallocManaged(&d_gaussian_kernel, sizeof(cufftComplex) * NY * NX * max_images));

    std::vector<float> filter = createGaussianFilter(kernel_size);

    // Set d_gaussian_kernel[*].y to 0.f
    CUDA_CHK(cudaMemset(d_gaussian_kernel, 0, sizeof(cufftComplex) * NY * NX));

    // Set d_gaussian_kernel[*].x
    for( int i = 0; i < kernel_size; ++i )
    {
        CUDA_CHK(cudaMemcpy2D(&d_gaussian_kernel[i * NX], sizeof(cufftComplex), &filter[i * kernel_size], sizeof(float), sizeof(float), kernel_size, cudaMemcpyHostToDevice));
    }

    cufftHandle gaussian_plan;
    CUFFT_CHK(cufftPlan2d(&gaussian_plan, NY, NX, CUFFT_C2C));
    CUFFT_CHK(cufftExecC2C(gaussian_plan, d_gaussian_kernel, d_gaussian_kernel, CUFFT_FORWARD));
    CUDA_CHK(cudaDeviceSynchronize());
    CUFFT_CHK(cufftDestroy(gaussian_plan));

    // Copy the FFT gaussian kernel for every image in batch
    for (int i = 1; i < max_images; ++i)
    {
        CUDA_CHK(cudaMemcpy(&d_gaussian_kernel[i * NY * NX], d_gaussian_kernel, sizeof(cufftComplex) * NY * NX, cudaMemcpyDeviceToDevice));
    }
}

GpuBlurImage::~GpuBlurImage()
{
    CUFFT_CHK(cufftDestroy(plan));
    CUDA_CHK(cudaFree(d_complex));
    CUDA_CHK(cudaFree(d_gaussian_kernel));
}

void GpuBlurImage::blur(float *blurred_images, float *images, int num_images)
{
    assert(num_images <= max_images);
    const int total = NY * NX * num_images;

    // Set d_complex[*].y to 0.f
    CUDA_CHK(cudaMemset(d_complex, 0, sizeof(cufftComplex) * total));
    // Set d_complex[*].x
    CUDA_CHK(cudaMemcpy2D(d_complex, sizeof(cufftComplex), images, sizeof(float), sizeof(float), total, cudaMemcpyHostToDevice));

    CUFFT_CHK(cufftExecC2C(plan, d_complex, d_complex, CUFFT_FORWARD));

    {
        dim3 block(256);
        dim3 grid((NY * NX + block.x + 1) / block.x, num_images);
        complexPointwiseMulAndScale_kernel<<<grid, block>>>(d_gaussian_kernel, d_complex, NY * NX);
    }

    CUFFT_CHK(cufftExecC2C(plan, d_complex, d_complex, CUFFT_INVERSE));

    // Copy down real data
    CUDA_CHK(cudaMemcpy2D(blurred_images, sizeof(float), d_complex, sizeof(cufftComplex), sizeof(float), total, cudaMemcpyDeviceToHost));
}

std::vector<float> GpuBlurImage::createGaussianFilter(int kernel_size)
{
    const int radius = kernel_size / 2;
    const float sigma = kernel_size / 4; // By default, radius of kernel = 2 * sigma
    const float q = 2.0 * sigma * sigma;
    float sum = 0.0; // sum is for normalization

    std::vector<float> kernel(kernel_size * kernel_size);

    for (int y = 0; y < kernel_size; y++)
    {
        for (int x = 0; x < kernel_size; x++)
        {
            float p = sqrt((x - radius) * (x - radius) + (y - radius) * (y - radius));
            kernel[y * kernel_size + x] = (exp(-(p * p) / q)) / (M_PI * q);
            sum += kernel[y * kernel_size + x];
        }
    }

    for (auto& v : kernel)
    {
        v /= sum; // normalize
    }

    printf("Gaussian filter:\n");
    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            printf("%f ", kernel[i * kernel_size + j]);
        }
        printf("\n");
    }

    return kernel;
}