cmake_minimum_required(VERSION 2.6)

project(trt-infer)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)
# cudnn
#include_directories(/usr/local/cuda/cudnn/v8.0.4/include)
#link_directories(/usr/local/cuda/cudnn/v8.0.4/lib64)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# tensorrt
include_directories(/home/uisee/disk/TensorRT-8.2.1.8/include)
link_directories(/home/uisee/disk/TensorRT-8.2.1.8/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fsanitize=address -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O0")

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(
    trt-infer 
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/infer.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/yolo_detector.cpp
    ${PROJECT_SOURCE_DIR}/src/yolo_kernel.cu
)
target_link_libraries(trt-infer nvinfer)
target_link_libraries(trt-infer nvinfer_plugin)
target_link_libraries(trt-infer nvonnxparser)
target_link_libraries(trt-infer cudart)
target_link_libraries(trt-infer pthread)
target_link_libraries(trt-infer ${OpenCV_LIBS})

add_definitions(-O0 -pthread)
