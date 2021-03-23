cmake_minimum_required(VERSION 3.12)
project(image-blur)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50")
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

cuda_add_library(gpu_fft_blur_detector blur_detector/gpu_fft_blur_detector.cu)
target_link_libraries(gpu_fft_blur_detector cufft opencv_core opencv_imgcodecs opencv_highgui)

add_executable(test_blur_detector test_blur_detector.cpp)
target_link_libraries(test_blur_detector gpu_fft_blur_detector)