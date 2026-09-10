#ifndef HFNET_RKNN_ENGINE_H
#define HFNET_RKNN_ENGINE_H

#include <vector>
#include <cstdint>
#include <string>
#include "rknn_api.h"

// 辅助函数：将模型二进制文件读取进内存
unsigned char* load_model_file(const char* filename, int* model_size);

// 辅助函数：打印张量属性
void dump_tensor_attr(const rknn_tensor_attr* attr);

// 单个 NPU 推理工作实例封装 (管理独立的 rknn_context 与零拷贝物理内存)
class RknnWorker {
public:
    RknnWorker();
    ~RknnWorker();

    // 禁止拷贝
    RknnWorker(const RknnWorker&) = delete;
    RknnWorker& operator=(const RknnWorker&) = delete;

    // 初始化 Worker
    int init(int worker_id, unsigned char* model_data, int model_len, bool dump_attrs = false);

    // 释放资源
    void release();

    // 执行推理
    int run_inference(const uint8_t* resized_y,
                      std::vector<int8_t>& out_logits,
                      std::vector<int8_t>& out_desc,
                      double& npu_time_ms);

    // Getters
    int get_worker_id() const { return worker_id_; }
    int get_req_width() const { return req_width_; }
    int get_req_height() const { return req_height_; }
    int get_logits_idx() const { return logits_idx_; }
    int get_desc_idx() const { return desc_idx_; }
    bool is_logits_nchw() const { return logits_is_nchw_; }
    bool is_desc_nchw() const { return desc_is_nchw_; }
    float get_logits_scale() const { return logits_scale_; }
    int32_t get_logits_zp() const { return logits_zp_; }
    const std::vector<rknn_tensor_attr>& get_output_attrs() const { return output_attrs_; }

private:
    int worker_id_ = 0;
    rknn_context ctx_ = 0;
    bool initialized_ = false;

    rknn_tensor_mem* input_mem_ = nullptr;
    std::vector<rknn_tensor_mem*> output_mems_;

    rknn_input_output_num io_num_;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    int req_width_ = 0;
    int req_height_ = 0;
    int logits_idx_ = -1;
    int desc_idx_ = -1;
    bool logits_is_nchw_ = false;
    bool desc_is_nchw_ = false;
    float logits_scale_ = 1.0f;
    int32_t logits_zp_ = 0;
};

#endif // HFNET_RKNN_ENGINE_H
