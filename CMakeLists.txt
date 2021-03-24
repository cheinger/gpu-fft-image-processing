cmake_minimum_required(VERSION 3.12)
project(image-blur)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_35,code=sm_35\
                                            -gencode arch=compute_37,code=sm_37\
                                            -gencode arch=compute_50,code=sm_50\
                                            -gencode arch=compute_52,code=sm_52\
                                            -gencode arch=compute_60,code=sm_60\
                                            -gencode arch=compute_61,code=sm_61\
                                            -gencode arch=compute_70,code=sm_70\
                                            -gencode arch=compute_75,code=sm_75\
                                            -gencode arch=compute_80,code=sm_80\
                                            -gencode arch=compute_86,code=sm_86\
                                            -gencode arch=compute_86,code=compute_86")
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${CMAKE_SOURCE_DIR}/)

cuda_add_library(gpu_fft_blur_detector blur_detector/gpu_fft_blur_detector.cu)
target_link_libraries(gpu_fft_blur_detector cufft opencv_core opencv_imgcodecs opencv_highgui)

cuda_add_library(gpu_fft_image_blur blur_image/gpu_fft_blur_image.cu)
target_link_libraries(gpu_fft_image_blur cufft opencv_core opencv_imgcodecs opencv_highgui)

add_executable(test_blur_detector test_blur_detector.cpp)
target_link_libraries(test_blur_detector gpu_fft_blur_detector)

add_executable(test_blur_image test_blur_image.cpp)
target_link_libraries(test_blur_image gpu_fft_image_blur)