# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(unet_trt LANGUAGES CXX CUDA)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# -------- BUILDING --------

add_executable(unet_trt
    main.cpp
)

# Use C++11
target_compile_features(unet_trt PUBLIC cxx_std_14)

# Link TensorRT's nvinfer lib
target_link_libraries(unet_trt PUBLIC 
    ${NVINFER_LIB}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer nvparsers nvinfer_plugin nvonnxparser
)
target_include_directories(unet_trt PUBLIC
    ${CUDA_INCLUDE_DIRS}
)
