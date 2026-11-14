cmake_minimum_required(VERSION 3.10)
project(hfnet_rknn_demo)

# 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启 O3 极速优化与 ARMv8 NEON 指令集
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=armv8-a")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
endif()

# 设置相对路径指向已存在的 RKNN NPU 运行时 SDK
set(RKNN_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rknn_model_zoo/3rdparty/rknpu2")
set(RKNN_HEADER_DIR "${RKNN_SDK_DIR}/include")

# 定位 Linux aarch64 平台对应的 librknnrt.so 动态链接库
set(RKNN_LIB "${RKNN_SDK_DIR}/Linux/aarch64/librknnrt.so")

# 定位已有的 stb_image 头文件搜索路径
set(STB_IMAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rknn_model_zoo/3rdparty/stb_image")
set(STB_RESIZE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rknn-toolkit2/rknpu2/examples/3rdparty/stb")

# 添加头文件搜索路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RKNN_HEADER_DIR}
    ${STB_IMAGE_DIR}
    ${STB_RESIZE_DIR}
)

# 收集源文件 (HF-Net NPU 流水线 + 完整 C++ 视觉惯性 SLAM 前后端与黑盒记录仪)
set(SOURCES
    src/main.cpp
    src/image_utils.cpp
    src/hfnet_postprocess.cpp
    src/rknn_engine.cpp
    src/hfnet_pipeline.cpp
    src/slam_math.cpp
    src/imu_preintegration.cpp
    src/slam_core.cpp
    src/slam_frontend.cpp
    src/slam_backend.cpp
    src/slam_tracker.cpp
    src/blackbox_logger.cpp
)

# 声明编译的可执行程序
add_executable(hfnet_demo ${SOURCES})

# 链接 RKNN 动态库与系统线程/数学库
target_link_libraries(hfnet_demo
    ${RKNN_LIB}
    dl
    pthread
    m
)
