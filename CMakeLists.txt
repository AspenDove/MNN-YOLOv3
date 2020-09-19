cmake_minimum_required(VERSION 3.10)
FIND_PACKAGE(OpenCV REQUIRED)
project(Ultra-face-mnn)

set(CMAKE_CXX_STANDARD 17)

#find_package(OpenCV REQUIRED)
include_directories(
        mnn/include
        src
)

link_directories(mnn/lib)

add_executable(Ultra-face-mnn src/main.cpp)
target_link_libraries(Ultra-face-mnn MNN ${OpenCV_LIBS})
