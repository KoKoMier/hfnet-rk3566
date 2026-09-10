#include "hfnet_postprocess.h"

#include <cmath>
#include <algorithm>

// 描述子特征采样 (在 INT8 上计算双线性插值与 L2-Norm)
std::vector<float> sample_descriptor_int8(const int8_t* desc_map, float x_desc, float y_desc,
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
        float inv_norm = 1.0f / norm;
        for (int c = 0; c < C; ++c) {
            desc[c] *= inv_norm;
        }
    }

    return desc;
}

// 提取局部极大值作为特征点 (采用网格轮询均匀化分配 Grid Round-Robin Bucketing)
std::vector<Keypoint> extract_keypoints(const float* scores_dense, int H, int W,
                                        int max_kpts, float min_score,
                                        int radius, int grid_rows, int grid_cols) {
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

    // 2. 对每个网格单元内的候选点按得分降序排序
    for (int i = 0; i < num_cells; ++i) {
        std::sort(grid_cells[i].begin(), grid_cells[i].end(), [](const Keypoint& a, const Keypoint& b) {
            return a.score > b.score;
        });
    }

    // 3. 跨网格轮询均匀提取 (Round-Robin Bucketing)
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

// 从 INT8 logits (65 通道) 计算 scores_dense (Softmax + DepthToSpace)
void compute_scores_dense_from_logits(const int8_t* logits_i8, float scale, int32_t zp,
                                     float* scores_dense, int H, int W, bool is_nchw) {
    int H_grid = H / 8; // e.g. 480 / 8 = 60
    int W_grid = W / 8; // e.g. 640 / 8 = 80

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
