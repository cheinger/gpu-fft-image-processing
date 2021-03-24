#include "image_blur/gpu_fft_image_blur.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

int main(int argc, char* argv[])
{
    const int num_images = argc - 1;

    int rows = -1;
    int cols = -1;

    std::vector<std::string> files;
    for( int i = 0; i < num_images; ++i)
    {
        files.emplace_back(argv[1 + i]);

        cv::Mat image = cv::imread(files.back(), cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            fprintf(stderr, "Could not read the image %s\n", files.back().c_str());
            return -1;
        }

        if (i == 0)
        {
            rows = image.rows;
            cols = image.cols;
        }
        else if (rows != image.rows || cols != image.cols)
        {
            fprintf(stderr, "Images must be the same dimension\n");
            return -1;
        }
    }

    std::vector<float> images(num_images * rows * cols);
    for (int i = 0; i < files.size(); ++i)
    {
        cv::Mat image = cv::imread(files[i].c_str(), cv::IMREAD_GRAYSCALE);
        assert(!image.empty());
        cv::imshow("Original", image);

        image.convertTo(image, CV_32F);

        std::memcpy(images.data() + i * rows * cols, image.data, rows * cols * sizeof(float));
    }


    GpuImageBlur image_blur(rows, cols, num_images, 10);
    image_blur.blur(images.data(), images.data(), num_images);

    for (int i = 0; i < num_images; ++i)
    {
        cv::Mat image(rows, cols, CV_32F, images.data() + i * rows * cols);
        image.convertTo(image, CV_8U);
        cv::imshow("Blurred Result", image);
    }

    cv::waitKey();

    return 0;
}