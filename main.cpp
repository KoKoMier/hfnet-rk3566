#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <map>
#include <condition_variable>
#include <atomic>

#include "rknn_api.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

// NPU 并发实例数 (RK3566 推荐 2 实例乒乓并发，彻底消除单 Context 驱动阻塞)
static const int NUM_NPU_WORKERS = 2;

// ==========================================
// 1. 结构体与辅助工具定义
// ==========================================

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
    bool is_eof;
};

// NPU 推理结果数据包 (Stage 2 -> Stage 3)
struct InferencePacket {
    int frame_idx;
    std::vector<uint8_t> raw_nv12;
    std::vector<int8_t> logits_raw;
    std::vector<int8_t> desc_raw;
    double npu_time_ms;
    int worker_id;
    bool is_eof;
};

// 线程安全有界队列
template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;
    size_t max_size;
    bool finished = false;

public:
    SafeQueue(size_t capacity = 4) : max_size(capacity) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_push.wait(lock, [this]() { return q.size() < max_size || finished; });
        if (finished) return;
        q.push(std::move(item));
        cv_pop.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_pop.wait(lock, [this]() { return !q.empty() || finished; });
        if (q.empty() && finished) {
            return false;
        }
        item = std::move(q.front());
        q.pop();
        cv_push.notify_one();
        return true;
    }

    void shutdown() {
        std::unique_lock<std::mutex> lock(mtx);
        finished = true;
        cv_push.notify_all();
        cv_pop.notify_all();
    }
};

// 辅助函数：将模型二进制文件读取进内存
static unsigned char* load_model(const char* filename, int* model_size)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == nullptr) {
        printf("Failed to open model file: %s\n", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    unsigned char* data = (unsigned char*)malloc(size);
    if (data == nullptr) {
        printf("Failed to allocate memory of size %d bytes\n", size);
        fclose(fp);
        return nullptr;
    }
    
    int read_bytes = fread(data, 1, size, fp);
    if (read_bytes != size) {
        printf("Failed to read model file: read %d, expected %d\n", read_bytes, size);
        free(data);
        fclose(fp);
        return nullptr;
    }
    
    fclose(fp);
    *model_size = size;
    return data;
}

// 辅助函数：打印张量属性
static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  [Attr] index=%d, name=%s, n_dims=%d, dims=[", attr->index, attr->name, attr->n_dims);
    for (uint32_t i = 0; i < attr->n_dims; i++) {
        printf("%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
    printf("], size=%d, stride=[w=%d, h=%d], format=%s, type=%s, scale=%.6f, zp=%d\n",
           attr->size, attr->w_stride, attr->h_stride, get_format_string(attr->fmt), get_type_string(attr->type), attr->scale, attr->zp);
}

// =========================================================================
// =========================================================================
// 2. NV12 直接绘制高对比度彩色点（高亮荧光黄+白核，耗时 < 0.05ms，在明暗背景下均清晰醒目）
// =========================================================================
static void draw_circle_nv12(uint8_t* nv12, int W, int H, int cx, int cy, int radius = 3,
                             uint8_t y_val = 226, uint8_t u_val = 0, uint8_t v_val = 148) {
    uint8_t* y_plane = nv12;
    uint8_t* uv_plane = nv12 + W * H;
    
    int r2 = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy) {
        int py = cy + dy;
        if (py < 0 || py >= H) continue;
        int uv_row = (py / 2) * W;
        for (int dx = -radius; dx <= radius; ++dx) {
            int px = cx + dx;
            if (px < 0 || px >= W) continue;
            int dist2 = dx * dx + dy * dy;
            if (dist2 <= r2) {
                int uv_idx = uv_row + (px / 2) * 2;
                if (dist2 <= 1) {
                    // 中心核采用纯白 (Y=255)，精确定位
                    y_plane[py * W + px] = 255;
                    uv_plane[uv_idx]     = 128;
                    uv_plane[uv_idx + 1] = 128;
                } else {
                    // 外环采用高亮荧光黄 (Y=226, U=0, V=148)，高亮度高饱和度，彻底告别暗红隐形
                    y_plane[py * W + px] = y_val;
                    uv_plane[uv_idx]     = u_val;
                    uv_plane[uv_idx + 1] = v_val;
                }
            }
        }
    }
}

// 描述子特征采样 (在 INT8 上计算 L2-Norm)
static std::vector<float> sample_descriptor_int8(const int8_t* desc_map, float x_desc, float y_desc,
                                                 int H, int W, int C, bool is_nchw) {
    std::vector<float> desc(C, 0.0f);
    
    int x0 = (int)floor(x_desc);
    int y0 = (int)floor(y_desc);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    x0 = std::max(0, std::min(x0, W - 1));
    x1 = std::max(0, std::min(x1, W - 1));
    y0 = std::max(0, std::min(y0, H - 1));
    y1 = std::max(0, std::min(y1, H - 1));
    
    float tx = x_desc - x0;
    float ty = y_desc - y0;
    
    float wa = (1.0f - tx) * (1.0f - ty);
    float wb = tx * (1.0f - ty);
    float wc = (1.0f - tx) * ty;
    float wd = tx * ty;
    
    float sum_sq = 0.0f;
    for (int c = 0; c < C; ++c) {
        float val00 = is_nchw ? (float)desc_map[c * H * W + y0 * W + x0] : (float)desc_map[y0 * W * C + x0 * C + c];
        float val01 = is_nchw ? (float)desc_map[c * H * W + y0 * W + x1] : (float)desc_map[y0 * W * C + x1 * C + c];
        float val10 = is_nchw ? (float)desc_map[c * H * W + y1 * W + x0] : (float)desc_map[y1 * W * C + x0 * C + c];
        float val11 = is_nchw ? (float)desc_map[c * H * W + y1 * W + x1] : (float)desc_map[y1 * W * C + x1 * C + c];
        
        float val = wa * val00 + wb * val01 + wc * val10 + wd * val11;
        desc[c] = val;
        sum_sq += val * val;
    }
    
    float norm = sqrt(sum_sq);
    if (norm > 1e-6f) {
        for (int c = 0; c < C; ++c) {
            desc[c] /= norm;
        }
    }
    
    return desc;
}

// 预生成 Gamma 0.65 暗光自适应增强映射表 (提升暗部对比度与梯度，耗时 < 0.1ms，显著激发暗区角点)
static uint8_t g_gamma_lut[256];
static void init_gamma_lut(float gamma = 0.65f) {
    for (int i = 0; i < 256; ++i) {
        float norm = (float)i / 255.0f;
        float enhanced = powf(norm, gamma);
        int val = (int)(enhanced * 255.0f + 0.5f);
        g_gamma_lut[i] = (uint8_t)std::min(255, std::max(0, val));
    }
}

// 提取局部极大值作为特征点 (采用网格轮询均匀化分配 Grid Round-Robin Bucketing，彻底解决特征点局部扎堆聚集问题)
static std::vector<Keypoint> extract_keypoints(const float* scores_dense, int H, int W, 
                                              int max_kpts = 500, float min_score = 0.0003f, 
                                              int radius = 4, int grid_rows = 6, int grid_cols = 8) {
    int num_cells = grid_rows * grid_cols;
    std::vector<std::vector<Keypoint>> grid_cells(num_cells);
    
    // 1. 局部极大值检测并按空间坐标落入对应网格
    for (int y = radius; y < H - radius; ++y) {
        int gy = std::min(grid_rows - 1, y * grid_rows / H);
        for (int x = radius; x < W - radius; ++x) {
            float val = scores_dense[y * W + x];
            if (val < min_score) continue;
            
            bool is_max = true;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (scores_dense[(y + dy) * W + (x + dx)] > val) {
                        is_max = false;
                        break;
                    }
                }
                if (!is_max) break;
            }
            
            if (is_max) {
                int gx = std::min(grid_cols - 1, x * grid_cols / W);
                Keypoint kp;
                kp.x = (float)x;
                kp.y = (float)y;
                kp.score = val;
                grid_cells[gy * grid_cols + gx].push_back(kp);
            }
        }
    }
    
    // 2. 对每个网格单元内的候选点按得分降序排序 (每个网格仅几十个元素，耗时微秒级)
    for (int i = 0; i < num_cells; ++i) {
        std::sort(grid_cells[i].begin(), grid_cells[i].end(), [](const Keypoint& a, const Keypoint& b) {
            return a.score > b.score;
        });
    }
    
    // 3. 跨网格轮询均匀提取 (Round-Robin Bucketing)
    // 每一轮让全图所有具备特征的网格各出 1 个当前最高分点，天然实现全图空间均布，杜绝局部扎堆
    std::vector<Keypoint> result;
    result.reserve(max_kpts);
    
    size_t round = 0;
    bool has_more = true;
    while (result.size() < (size_t)max_kpts && has_more) {
        has_more = false;
        for (int i = 0; i < num_cells; ++i) {
            if (round < grid_cells[i].size()) {
                result.push_back(grid_cells[i][round]);
                if (result.size() >= (size_t)max_kpts) break;
                if (round + 1 < grid_cells[i].size()) {
                    has_more = true;
                }
            }
        }
        round++;
    }
    
    return result;
}

// 从 INT8 logits (65 通道) 计算 scores_dense (Softmax + DepthToSpace 在 CPU 上快速计算)
static void compute_scores_dense_from_logits(const int8_t* logits_i8, float scale, int32_t zp,
                                             float* scores_dense, int H, int W, bool is_nchw) {
    int H_grid = H / 8; // 60
    int W_grid = W / 8; // 80
    
    for (int gh = 0; gh < H_grid; ++gh) {
        for (int gw = 0; gw < W_grid; ++gw) {
            float exp_vals[65];
            float sum_exp = 0.0f;
            float max_val = -1e9f;
            
            for (int c = 0; c < 65; ++c) {
                int8_t raw = is_nchw ? logits_i8[c * (H_grid * W_grid) + gh * W_grid + gw]
                                     : logits_i8[gh * (W_grid * 65) + gw * 65 + c];
                float val = (float)(raw - zp) * scale;
                exp_vals[c] = val;
                if (val > max_val) max_val = val;
            }
            
            for (int c = 0; c < 65; ++c) {
                exp_vals[c] = expf(exp_vals[c] - max_val);
                sum_exp += exp_vals[c];
            }
            
            float inv_sum = 1.0f / sum_exp;
            
            for (int dh = 0; dh < 8; ++dh) {
                for (int dw = 0; dw < 8; ++dw) {
                    int c = dh * 8 + dw;
                    int y = gh * 8 + dh;
                    int x = gw * 8 + dw;
                    scores_dense[y * W + x] = exp_vals[c] * inv_sum;
                }
            }
        }
    }
}

// =========================================================================
// 3. 多线程双 Context 乒乓并发流水线
// =========================================================================

// 全局流水线队列
static SafeQueue<PreprocessPacket> g_queue_prep_to_npu(4);
static SafeQueue<InferencePacket>  g_queue_npu_to_post(4);

// [Stage 1] 预处理线程：逐帧读取 NV12 并进行 Y 通道缩放
void thread_preprocess(const char* input_path, int cam_width, int cam_height, int req_width, int req_height) {
    FILE* fp_in = fopen(input_path, "rb");
    if (fp_in == nullptr) {
        printf("[Stage 1] Failed to open input stream: %s\n", input_path);
        for (int i = 0; i < NUM_NPU_WORKERS; ++i) {
            PreprocessPacket eof_pkt;
            eof_pkt.is_eof = true;
            g_queue_prep_to_npu.push(std::move(eof_pkt));
        }
        return;
    }

    int frame_size = cam_width * cam_height * 3 / 2;
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
            for (int i = 0; i < NUM_NPU_WORKERS; ++i) {
                PreprocessPacket eof_pkt;
                eof_pkt.is_eof = true;
                g_queue_prep_to_npu.push(std::move(eof_pkt));
            }
            break;
        }

        // 快速缩放 Y 分量
        stbir_resize_uint8(pkt.raw_nv12.data(), cam_width, cam_height, 0,
                           pkt.resized_y.data(), req_width, req_height, 0, 1);

        // 对送入 NPU 的特征图执行硬件友好的暗部自适应增强 (激活暗光区域边缘与角点，耗时 < 0.15ms)
        for (int i = 0; i < req_width * req_height; ++i) {
            pkt.resized_y[i] = g_gamma_lut[pkt.resized_y[i]];
        }

        g_queue_prep_to_npu.push(std::move(pkt));
        frame_idx++;
    }

    fclose(fp_in);
    printf("[Stage 1] Preprocess thread finished. Total frames: %d\n", frame_idx);
}

// [Stage 2] NPU 乒乓工作线程 (Worker 0 与 Worker 1 并发竞争推断)
void thread_npu_worker(int worker_id, rknn_context ctx, rknn_tensor_mem* input_mem,
                       const std::vector<rknn_tensor_mem*>& output_mems,
                       const std::vector<rknn_tensor_attr>& input_attrs,
                       const std::vector<rknn_tensor_attr>& output_attrs,
                       int req_width, int req_height) {
    int stride = input_attrs[0].w_stride;
    int logits_idx = -1;
    int desc_idx = -1;

    for (size_t i = 0; i < output_attrs.size(); ++i) {
        if (strcmp(output_attrs[i].name, "logits") == 0) logits_idx = i;
        else if (strcmp(output_attrs[i].name, "local_descriptor_map") == 0) desc_idx = i;
    }

    while (true) {
        PreprocessPacket in_pkt;
        if (!g_queue_prep_to_npu.pop(in_pkt) || in_pkt.is_eof) {
            InferencePacket out_pkt;
            out_pkt.worker_id = worker_id;
            out_pkt.is_eof = true;
            g_queue_npu_to_post.push(std::move(out_pkt));
            break;
        }

        // 填充专用 Zero-Copy 输入内存
        if (stride == req_width || stride == 0) {
            memcpy(input_mem->virt_addr, in_pkt.resized_y.data(), req_width * req_height);
        } else {
            const uint8_t* src = in_pkt.resized_y.data();
            uint8_t* dst = (uint8_t*)input_mem->virt_addr;
            for (int h = 0; h < req_height; ++h) {
                memcpy(dst, src, req_width);
                src += req_width;
                dst += stride;
            }
        }

        // 硬件 NPU 推理
        auto t0 = std::chrono::high_resolution_clock::now();
        int ret = rknn_run(ctx, nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double npu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (ret != RKNN_SUCC) {
            printf("[Stage 2: Worker %d] rknn_run failed at frame %d! ret=%d\n", worker_id, in_pkt.frame_idx, ret);
            break;
        }

        InferencePacket out_pkt;
        out_pkt.frame_idx = in_pkt.frame_idx;
        out_pkt.raw_nv12 = std::move(in_pkt.raw_nv12);
        out_pkt.npu_time_ms = npu_ms;
        out_pkt.worker_id = worker_id;
        out_pkt.is_eof = false;

        // 快速提取输出数据
        if (logits_idx >= 0) {
            int size = output_attrs[logits_idx].n_elems;
            out_pkt.logits_raw.resize(size);
            memcpy(out_pkt.logits_raw.data(), output_mems[logits_idx]->virt_addr, size);
        }
        if (desc_idx >= 0) {
            int size = output_attrs[desc_idx].n_elems;
            out_pkt.desc_raw.resize(size);
            memcpy(out_pkt.desc_raw.data(), output_mems[desc_idx]->virt_addr, size);
        }

        g_queue_npu_to_post.push(std::move(out_pkt));
    }

    printf("[Stage 2: Worker %d] Finished.\n", worker_id);
}

// [Stage 3] 严格保序的后处理与写盘线程 (通过 Frame Index 保障输出视频顺序绝对不错乱)
void thread_postprocess(const char* output_path, int cam_width, int cam_height,
                        int req_width, int req_height,
                        const std::vector<rknn_tensor_attr>& output_attrs) {
    FILE* fp_out = fopen(output_path, "wb");
    if (fp_out == nullptr) {
        printf("[Stage 3] Failed to open output stream: %s\n", output_path);
        return;
    }

    int logits_idx = -1;
    int desc_idx = -1;
    for (size_t i = 0; i < output_attrs.size(); ++i) {
        if (strcmp(output_attrs[i].name, "logits") == 0) logits_idx = i;
        else if (strcmp(output_attrs[i].name, "local_descriptor_map") == 0) desc_idx = i;
    }

    bool logits_is_nchw = (logits_idx >= 0) ? (output_attrs[logits_idx].dims[1] == 65) : false;
    bool desc_is_nchw = (desc_idx >= 0) ? (output_attrs[desc_idx].dims[1] == 256) : false;
    float logits_scale = (logits_idx >= 0) ? output_attrs[logits_idx].scale : 1.0f;
    int32_t logits_zp = (logits_idx >= 0) ? output_attrs[logits_idx].zp : 0;

    int expected_frame_idx = 0;
    int eof_workers_count = 0;
    std::map<int, InferencePacket> reorder_map;
    std::vector<float> scores_dense(req_height * req_width);
    auto t_start = std::chrono::high_resolution_clock::now();

    while (true) {
        InferencePacket pkt;
        if (!g_queue_npu_to_post.pop(pkt)) {
            break;
        }

        if (pkt.is_eof) {
            eof_workers_count++;
            if (eof_workers_count >= NUM_NPU_WORKERS) {
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

            // 2. 提取特征点 (启用网格轮询均匀分配 Grid Round-Robin Bucketing，覆盖暗区弱纹理)
            std::vector<Keypoint> kpts = extract_keypoints(scores_dense.data(), req_height, req_width, 500, 0.0003f, 4, 6, 8);

            // 3. 描述子采样
            int H_desc = 60, W_desc = 80, C_desc = 256;
            float factor_x = (float)(req_width - 1) / (W_desc - 1);
            float factor_y = (float)(req_height - 1) / (H_desc - 1);
            for (size_t i = 0; i < kpts.size(); ++i) {
                float x_desc = kpts[i].x / factor_x;
                float y_desc = kpts[i].y / factor_y;
                kpts[i].descriptor = sample_descriptor_int8(cur_pkt.desc_raw.data(), x_desc, y_desc, H_desc, W_desc, C_desc, desc_is_nchw);
            }

            // 4. NV12 直接绘制红点
            float scale_x = (float)cam_width / req_width;
            float scale_y = (float)cam_height / req_height;
            for (const auto& kp : kpts) {
                int draw_x = (int)(kp.x * scale_x);
                int draw_y = (int)(kp.y * scale_y);
                draw_circle_nv12(cur_pkt.raw_nv12.data(), cam_width, cam_height, draw_x, draw_y, 3);
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

// =========================================================================
// 4. Main 入口
// =========================================================================

int main(int argc, char** argv)
{
    if (argc < 5) {
        printf("Usage: %s <rknn_model_path> <input_nv12_stream> <cam_width> <cam_height> [output_nv12_path]\n", argv[0]);
        printf("Example: %s ./hfnet.rknn ./out.nv12 640 480 ./out4.nv12\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* input_path = argv[2];
    int cam_width = atoi(argv[3]);
    int cam_height = atoi(argv[4]);
    const char* output_path = (argc > 5) ? argv[5] : "./out4.nv12";

    // 初始化暗光增强映射表 (Gamma 0.65 映射，激活暗部特征响应)
    init_gamma_lut(0.65f);

    int ret;
    int model_len = 0;
    unsigned char* model_data = load_model(model_path, &model_len);
    if (model_data == nullptr) return -1;

    printf("========================================================================\n");
    printf("--> Initializing Dual-Context RKNN Environment (%d Parallel Workers)\n", NUM_NPU_WORKERS);
    printf("========================================================================\n");

    std::vector<rknn_context> contexts(NUM_NPU_WORKERS, 0);
    std::vector<rknn_tensor_mem*> input_mems(NUM_NPU_WORKERS, nullptr);
    std::vector<std::vector<rknn_tensor_mem*>> output_mems(NUM_NPU_WORKERS);

    rknn_input_output_num io_num;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;

    for (int w = 0; w < NUM_NPU_WORKERS; ++w) {
        ret = rknn_init(&contexts[w], model_data, model_len, 0, nullptr);
        if (ret != RKNN_SUCC) {
            printf("rknn_init for worker %d failed! ret=%d\n", w, ret);
            return -1;
        }

        if (w == 0) {
            rknn_query(contexts[w], RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
            input_attrs.resize(io_num.n_input);
            for (uint32_t i = 0; i < io_num.n_input; i++) {
                input_attrs[i].index = i;
                rknn_query(contexts[w], RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
                dump_tensor_attr(&(input_attrs[i]));
            }
            output_attrs.resize(io_num.n_output);
            for (uint32_t i = 0; i < io_num.n_output; i++) {
                output_attrs[i].index = i;
                rknn_query(contexts[w], RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
                dump_tensor_attr(&(output_attrs[i]));
            }
        }

        // 独立为每个 Worker 创建零拷贝物理内存
        input_attrs[0].type = RKNN_TENSOR_UINT8;
        input_mems[w] = rknn_create_mem(contexts[w], input_attrs[0].size_with_stride);
        rknn_set_io_mem(contexts[w], input_mems[w], &input_attrs[0]);

        output_mems[w].resize(io_num.n_output, nullptr);
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
            output_attrs[i].type = RKNN_TENSOR_INT8;
            output_mems[w][i] = rknn_create_mem(contexts[w], output_attrs[i].n_elems * sizeof(int8_t));
            rknn_set_io_mem(contexts[w], output_mems[w][i], &output_attrs[i]);
        }
    }
    free(model_data);

    int req_height = (input_attrs[0].fmt == RKNN_TENSOR_NCHW) ? input_attrs[0].dims[2] : input_attrs[0].dims[1];
    int req_width  = (input_attrs[0].fmt == RKNN_TENSOR_NCHW) ? input_attrs[0].dims[3] : input_attrs[0].dims[2];
    printf("Model input: %d x %d (Height x Width)\n", req_height, req_width);

    printf("========================================================================\n");
    printf("--> Starting Full Pipeline:\n");
    printf("    Stage 1: Async Preprocess Thread\n");
    printf("    Stage 2: 2x Parallel NPU Workers (Dual-Context Ping-Pong)\n");
    printf("    Stage 3: Order-Preserving Postprocess & Fast NV12 Direct Render\n");
    printf("========================================================================\n");

    auto t_start = std::chrono::high_resolution_clock::now();

    // 启动线程
    std::thread t_prep(thread_preprocess, input_path, cam_width, cam_height, req_width, req_height);
    
    std::vector<std::thread> npu_workers;
    for (int w = 0; w < NUM_NPU_WORKERS; ++w) {
        npu_workers.emplace_back(thread_npu_worker, w, contexts[w], input_mems[w],
                                 std::ref(output_mems[w]), std::ref(input_attrs),
                                 std::ref(output_attrs), req_width, req_height);
    }

    std::thread t_post(thread_postprocess, output_path, cam_width, cam_height, req_width, req_height, std::ref(output_attrs));

    t_prep.join();
    for (auto& t : npu_workers) {
        t.join();
    }
    t_post.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time_sec = std::chrono::duration<double>(t_end - t_start).count();

    printf("========================================================================\n");
    printf("Pipeline completed successfully in %.2f seconds!\n", total_time_sec);
    printf("========================================================================\n");

    // 释放资源
    for (int w = 0; w < NUM_NPU_WORKERS; ++w) {
        if (input_mems[w]) rknn_destroy_mem(contexts[w], input_mems[w]);
        for (auto mem : output_mems[w]) {
            if (mem) rknn_destroy_mem(contexts[w], mem);
        }
        rknn_destroy(contexts[w]);
    }

    return 0;
}