#ifndef HFNET_TYPES_H
#define HFNET_TYPES_H

#include <vector>
#include <cstdint>
#include "imu_preintegration.h"

// 特征点结构体定义
struct Keypoint {
    float x;
    float y;
    float score;
    std::vector<float> descriptor;
};

// 预处理任务数据包 (Stage 1 -> Stage 2)
struct PreprocessPacket {
    int frame_idx = 0;
    double timestamp_sec = 0.0;
    std::vector<uint8_t> raw_nv12;
    std::vector<uint8_t> resized_y;
    std::vector<slam::ImuMeasurement> imu_batch;
    slam::ImuStatusSnapshot imu_status;
    bool is_eof = false;
};

// NPU 推理结果数据包 (Stage 2 -> Stage 3)
struct InferencePacket {
    int frame_idx = 0;
    double timestamp_sec = 0.0;
    std::vector<uint8_t> raw_nv12;
    std::vector<uint8_t> resized_y;
    std::vector<int8_t> logits_raw;
    std::vector<int8_t> desc_raw;
    std::vector<slam::ImuMeasurement> imu_batch;
    slam::ImuStatusSnapshot imu_status;
    double npu_time_ms = 0.0;
    int worker_id = 0;
    bool is_eof = false;
};

#endif // HFNET_TYPES_H
