#ifndef HFNET_PIPELINE_H
#define HFNET_PIPELINE_H

#include <string>
#include <vector>
#include <memory>
#include "types.h"
#include "safe_queue.hpp"
#include "rknn_engine.h"

// 流水线配置参数
struct PipelineConfig {
    std::string model_path;
    std::string input_path;
    std::string output_path = "./out4.nv12";
    int cam_width = 640;
    int cam_height = 480;
    int num_workers = 2;

    // 算法超参数
    float gamma = 0.65f;
    int max_kpts = 500;
    float min_score = 0.0003f;
    int nms_radius = 4;
    int grid_rows = 6;
    int grid_cols = 8;
};

// HF-Net 多线程三级并发流水线管理器
class HfnetPipeline {
public:
    HfnetPipeline();
    ~HfnetPipeline();

    // 运行流水线
    int run(const PipelineConfig& config);

private:
    void thread_preprocess(const PipelineConfig& config, int req_width, int req_height);
    void thread_npu_worker(RknnWorker* worker, int req_width, int req_height);
    void thread_postprocess(const PipelineConfig& config, const RknnWorker* worker_meta);

    SafeQueue<PreprocessPacket> queue_prep_to_npu_;
    SafeQueue<InferencePacket> queue_npu_to_post_;
};

#endif // HFNET_PIPELINE_H
