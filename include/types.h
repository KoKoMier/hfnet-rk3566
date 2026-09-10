#ifndef HFNET_TYPES_H
#define HFNET_TYPES_H

#include <vector>
#include <cstdint>

// 特征点结构体定义
struct Keypoint {
    float x;
    float y;
    float score;
    std::vector<float> descriptor;
};

// 预处理任务数据包 (Stage 1 -> Stage 2)
struct PreprocessPacket {
    int frame_idx;
    std::vector<uint8_t> raw_nv12;
    std::vector<uint8_t> resized_y;
    bool is_eof = false;
};

// NPU 推理结果数据包 (Stage 2 -> Stage 3)
struct InferencePacket {
    int frame_idx;
    std::vector<uint8_t> raw_nv12;
    std::vector<int8_t> logits_raw;
    std::vector<int8_t> desc_raw;
    double npu_time_ms;
    int worker_id;
    bool is_eof = false;
};

#endif // HFNET_TYPES_H
