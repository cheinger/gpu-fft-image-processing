# GPU Image Processing via FFT

This is a suite of GPU image processing functions using Fast Fourier Transforms.
I was curious how FFTs are used in image processing so I decided to implement some transforms myself to deepen my understanding.

# Getting started

#### Dependencies
```
cuda
opencv
```

#### Build
```asm
cmake .
make
```

*Note: make sure the CMakeLists.txt contains the GPU architecture flags for your GPU if it doesn't already.*

#### Run the Blur Detector
```
$ ./test_blur_detector ./images/lena_1.jpg ./images/lena_2.jpg ./images/lena_3.jpg ./images/lena_4.jpg
```

#### Run the Blur Image
```
$ ./test_blur_image ./images/adrian_01.png ./images/adrian_02.png
```

*Note: when processing multiple images, each image must be the same dimension*

# Blur Detector

Detects how blurry an image is. 
The sharper the image the higher the result, conversely the blurrier the image the lower the result.
The API allows you to calculate blur values for multiple images on the GPU at once.

### API
```cpp
GpuBlurDetector detector(image_height, image_width, max_num_images);

std::vector<blur_values> blur_values(num_images);
float* images = ...

detector.detectBlur(blur_values, images, num_images);
```

# Blur Image

Makes an image blurry.
The larger the kernel/filter size the more blurry the image will be.
The API allows you to blur multiple images on the GPU at once.

### API
```cpp
GpuBlurImage blur_image(image_height, image_width, max_num_images, kernel_size);

float* images = ...
float* blurred_images = ...

blur_image.blur(blurred_images, images, num_images);
```