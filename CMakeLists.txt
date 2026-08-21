cmake_minimum_required(VERSION 3.24)

project(sgm_bridge LANGUAGES CXX CUDA)

set(ENABLE_SAMPLES OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_OPENCV_WRAPPER OFF CACHE BOOL "" FORCE)
set(LIBSGM_SHARED OFF CACHE BOOL "" FORCE)

add_subdirectory(libSGM)

add_library(sgm_py SHARED sgm_c.cpp)

target_link_libraries(sgm_py PRIVATE sgm)
target_include_directories(sgm_py PRIVATE libSGM/include)

target_compile_features(sgm_py PRIVATE cxx_std_17)