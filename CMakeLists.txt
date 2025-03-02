cmake_minimum_required(VERSION 3.16)
project(yoloinfer VERSION 0.1)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(TENSORRT_DIR ~/download/TensorRT)

option(WITH_TEST "Build tests" OFF)
option(WITH_SAMPLES "Build samples" ON)


set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -w ${CUDA_GEN_CODE}")

# third libs
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64
                ${TENSORRT_DIR}/lib
)
include_directories(${TENSORRT_DIR}/include
                    ${CUDA_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS})
set(THIRD_LIBS nvinfer nvinfer_plugin cuda cublas cudart cudnn pthread ${OpenCV_LIBS})

add_subdirectory(src)
            
if(WITH_TEST)
    add_subdirectory(test)
endif()

if(WITH_SAMPLES)
    add_subdirectory(examples)
endif()