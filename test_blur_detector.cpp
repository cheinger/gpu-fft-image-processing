#include "blur_detector/gpu_fft_blur_detector.h"

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

        cv::Mat fImage;
        image.convertTo(fImage, CV_32F);

        if (i == 0)
        {
            rows = fImage.rows;
            cols = fImage.cols;
        }
        else if (rows != fImage.rows || cols != fImage.cols)
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

        image.convertTo(image, CV_32F);

        std::memcpy(images.data() + i * rows * cols, image.data, rows * cols * sizeof(float));
    }


    std::vector<float> blur_results(num_images);

    GpuBlurDetector detector(rows, cols, num_images);
    detector.detectBlur(blur_results.data(), images.data(), num_images, false);

    for (int i = 0; i < num_images; ++i)
    {
        printf("image: %s blur_value: %f\n", files[i].c_str(), blur_results[i]);
    }

    return 0;
}