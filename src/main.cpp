#include <cstdio>
#include <cstdlib>
#include "hfnet_pipeline.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <rknn_model_path> <input_nv12_stream> <cam_width> <cam_height> [output_nv12_path]\n", argv[0]);
        printf("Example: %s ./hfnet.rknn ./out.nv12 640 480 ./out4.nv12\n", argv[0]);
        return -1;
    }

    PipelineConfig config;
    config.model_path  = argv[1];
    config.input_path  = argv[2];
    config.cam_width   = std::atoi(argv[3]);
    config.cam_height  = std::atoi(argv[4]);
    config.output_path = (argc > 5) ? argv[5] : "./out4.nv12";

    // 默认参数配置
    config.num_workers = 2;       // 双 Context 乒乓并发
    config.gamma       = 0.65f;   // 暗光自适应增强 Gamma
    config.max_kpts    = 500;     // 最大提取特征点数量
    config.min_score   = 0.0003f; // 最小得分阈值
    config.nms_radius  = 4;       // NMS 抑制半径
    config.grid_rows   = 6;       // 空间网格行数
    config.grid_cols   = 8;       // 空间网格列数

    HfnetPipeline pipeline;
    return pipeline.run(config);
}
