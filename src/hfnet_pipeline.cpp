#include "hfnet_pipeline.h"
#include "image_utils.h"
#include "hfnet_postprocess.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <map>
#include <algorithm>

HfnetPipeline::HfnetPipeline()
    : queue_prep_to_npu_(4), queue_npu_to_post_(4) {}

HfnetPipeline::~HfnetPipeline() = default;

void HfnetPipeline::thread_preprocess(const PipelineConfig& config, int req_width, int req_height) {
    FILE* fp_in = fopen(config.input_path.c_str(), "rb");
    if (fp_in == nullptr) {
        printf("[Stage 1] Failed to open input stream: %s\n", config.input_path.c_str());
        for (int i = 0; i < config.num_workers; ++i) {
            PreprocessPacket eof_pkt;
            eof_pkt.is_eof = true;
            queue_prep_to_npu_.push(std::move(eof_pkt));
        }
        return;
    }

    int frame_size = config.cam_width * config.cam_height * 3 / 2;
    int frame_idx = 0;

    while (true) {
        PreprocessPacket pkt;
        pkt.frame_idx = frame_idx;
        pkt.raw_nv12.resize(frame_size);
        pkt.resized_y.resize(req_width * req_height);
        pkt.is_eof = false;

        size_t read_bytes = fread(pkt.raw_nv12.data(), 1, frame_size, fp_in);
        if (read_bytes < (size_t)frame_size) {
            // 向所有 NPU Worker 投递结束信号
            for (int i = 0; i < config.num_workers; ++i) {
                PreprocessPacket eof_pkt;
                eof_pkt.is_eof = true;
                queue_prep_to_npu_.push(std::move(eof_pkt));
            }
            break;
        }

        // 快速缩放 Y 分量
        resize_y_channel(pkt.raw_nv12.data(), config.cam_width, config.cam_height,
                         pkt.resized_y.data(), req_width, req_height);

        // 暗部自适应增强 (激活暗光区域边缘与角点)
        apply_gamma_lut(pkt.resized_y.data(), req_width * req_height);

        queue_prep_to_npu_.push(std::move(pkt));
        frame_idx++;
    }

    fclose(fp_in);
    printf("[Stage 1] Preprocess thread finished. Total frames: %d\n", frame_idx);
}

void HfnetPipeline::thread_npu_worker(RknnWorker* worker, int req_width, int req_height) {
    int worker_id = worker->get_worker_id();

    while (true) {
        PreprocessPacket in_pkt;
        if (!queue_prep_to_npu_.pop(in_pkt) || in_pkt.is_eof) {
            InferencePacket out_pkt;
            out_pkt.worker_id = worker_id;
            out_pkt.is_eof = true;
            queue_npu_to_post_.push(std::move(out_pkt));
            break;
        }

        InferencePacket out_pkt;
        out_pkt.frame_idx = in_pkt.frame_idx;
        out_pkt.raw_nv12 = std::move(in_pkt.raw_nv12);
        out_pkt.worker_id = worker_id;
        out_pkt.is_eof = false;

        int ret = worker->run_inference(in_pkt.resized_y.data(),
                                        out_pkt.logits_raw,
                                        out_pkt.desc_raw,
                                        out_pkt.npu_time_ms);
        if (ret != RKNN_SUCC) {
            printf("[Stage 2: Worker %d] Inference failed at frame %d!\n", worker_id, in_pkt.frame_idx);
            break;
        }

        queue_npu_to_post_.push(std::move(out_pkt));
    }

    printf("[Stage 2: Worker %d] Finished.\n", worker_id);
}

void HfnetPipeline::thread_postprocess(const PipelineConfig& config, const RknnWorker* worker_meta) {
    FILE* fp_out = fopen(config.output_path.c_str(), "wb");
    if (fp_out == nullptr) {
        printf("[Stage 3] Failed to open output stream: %s\n", config.output_path.c_str());
        return;
    }

    int req_width = worker_meta->get_req_width();
    int req_height = worker_meta->get_req_height();
    bool logits_is_nchw = worker_meta->is_logits_nchw();
    bool desc_is_nchw = worker_meta->is_desc_nchw();
    float logits_scale = worker_meta->get_logits_scale();
    int32_t logits_zp = worker_meta->get_logits_zp();

    int expected_frame_idx = 0;
    int eof_workers_count = 0;
    std::map<int, InferencePacket> reorder_map;
    std::vector<float> scores_dense(req_height * req_width);
    auto t_start = std::chrono::high_resolution_clock::now();

    while (true) {
        InferencePacket pkt;
        if (!queue_npu_to_post_.pop(pkt)) {
            break;
        }

        if (pkt.is_eof) {
            eof_workers_count++;
            if (eof_workers_count >= config.num_workers) {
                break;
            }
            continue;
        }

        reorder_map[pkt.frame_idx] = std::move(pkt);

        // 严格按帧序列输出
        while (reorder_map.find(expected_frame_idx) != reorder_map.end()) {
            InferencePacket cur_pkt = std::move(reorder_map[expected_frame_idx]);
            reorder_map.erase(expected_frame_idx);

            auto t0 = std::chrono::high_resolution_clock::now();

            // 1. 计算 Softmax + DepthToSpace
            compute_scores_dense_from_logits(cur_pkt.logits_raw.data(), logits_scale, logits_zp,
                                             scores_dense.data(), req_height, req_width, logits_is_nchw);

            // 2. 提取特征点 (网格轮询均匀化分配)
            std::vector<Keypoint> kpts = extract_keypoints(scores_dense.data(), req_height, req_width,
                                                           config.max_kpts, config.min_score,
                                                           config.nms_radius, config.grid_rows, config.grid_cols);

            // 3. 描述子采样
            int H_desc = 60, W_desc = 80, C_desc = 256;
            float factor_x = (float)(req_width - 1) / (W_desc - 1);
            float factor_y = (float)(req_height - 1) / (H_desc - 1);
            for (size_t i = 0; i < kpts.size(); ++i) {
                float x_desc = kpts[i].x / factor_x;
                float y_desc = kpts[i].y / factor_y;
                kpts[i].descriptor = sample_descriptor_int8(cur_pkt.desc_raw.data(), x_desc, y_desc,
                                                           H_desc, W_desc, C_desc, desc_is_nchw);
            }

            // 4. NV12 直接绘制高亮荧光黄+白核点
            float scale_x = (float)config.cam_width / req_width;
            float scale_y = (float)config.cam_height / req_height;
            for (const auto& kp : kpts) {
                int draw_x = (int)(kp.x * scale_x);
                int draw_y = (int)(kp.y * scale_y);
                draw_circle_nv12(cur_pkt.raw_nv12.data(), config.cam_width, config.cam_height, draw_x, draw_y, 3);
            }

            // 5. 写入文件
            fwrite(cur_pkt.raw_nv12.data(), 1, cur_pkt.raw_nv12.size(), fp_out);

            auto t1 = std::chrono::high_resolution_clock::now();
            double post_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

            expected_frame_idx++;
            if (expected_frame_idx % 30 == 0 || expected_frame_idx <= 5) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double total_fps = expected_frame_idx / std::chrono::duration<double>(t_now - t_start).count();
                printf("[Dual-Context Frame %d | W%d] NPU = %.2f ms | CPU Post = %.2f ms | Keypoints = %zu | Total Pipeline FPS = \033[1;32m%.1f\033[0m\n",
                       cur_pkt.frame_idx, cur_pkt.worker_id, cur_pkt.npu_time_ms, post_time, kpts.size(), total_fps);
            }
        }
    }

    fclose(fp_out);
    printf("[Stage 3] Postprocess thread finished. Total frames: %d\n", expected_frame_idx);
}

int HfnetPipeline::run(const PipelineConfig& config) {
    // 1. 初始化 Gamma 暗光自适应增强映射表
    init_gamma_lut(config.gamma);

    // 2. 加载模型二进制
    int model_len = 0;
    unsigned char* model_data = load_model_file(config.model_path.c_str(), &model_len);
    if (model_data == nullptr) return -1;

    printf("========================================================================\n");
    printf("--> Initializing Multi-Context RKNN Environment (%d Parallel Workers)\n", config.num_workers);
    printf("========================================================================\n");

    std::vector<std::unique_ptr<RknnWorker>> workers;
    workers.reserve(config.num_workers);

    for (int w = 0; w < config.num_workers; ++w) {
        auto worker = std::unique_ptr<RknnWorker>(new RknnWorker());
        bool dump = (w == 0);
        int ret = worker->init(w, model_data, model_len, dump);
        if (ret != RKNN_SUCC) {
            printf("Failed to init RknnWorker %d\n", w);
            free(model_data);
            return -1;
        }
        workers.push_back(std::move(worker));
    }
    free(model_data);

    int req_width = workers[0]->get_req_width();
    int req_height = workers[0]->get_req_height();
    printf("Model input: %d x %d (Height x Width)\n", req_height, req_width);

    printf("========================================================================\n");
    printf("--> Starting Full Pipeline:\n");
    printf("    Stage 1: Async Preprocess Thread\n");
    printf("    Stage 2: %dx Parallel NPU Workers (Multi-Context Ping-Pong)\n", config.num_workers);
    printf("    Stage 3: Order-Preserving Postprocess & Fast NV12 Direct Render\n");
    printf("========================================================================\n");

    auto t_start = std::chrono::high_resolution_clock::now();

    // 启动线程
    std::thread t_prep(&HfnetPipeline::thread_preprocess, this, std::cref(config), req_width, req_height);

    std::vector<std::thread> npu_threads;
    for (int w = 0; w < config.num_workers; ++w) {
        npu_threads.emplace_back(&HfnetPipeline::thread_npu_worker, this, workers[w].get(), req_width, req_height);
    }

    std::thread t_post(&HfnetPipeline::thread_postprocess, this, std::cref(config), workers[0].get());

    t_prep.join();
    for (auto& t : npu_threads) {
        t.join();
    }
    t_post.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time_sec = std::chrono::duration<double>(t_end - t_start).count();

    printf("========================================================================\n");
    printf("Pipeline completed successfully in %.2f seconds!\n", total_time_sec);
    printf("========================================================================\n");

    return 0;
}
