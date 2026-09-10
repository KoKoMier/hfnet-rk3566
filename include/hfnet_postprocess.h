#ifndef HFNET_POSTPROCESS_H
#define HFNET_POSTPROCESS_H

#include <vector>
#include <cstdint>
#include "types.h"

// 从 INT8 logits (65 通道) 计算 scores_dense (Softmax + DepthToSpace)
void compute_scores_dense_from_logits(const int8_t* logits_i8, float scale, int32_t zp,
                                     float* scores_dense, int H, int W, bool is_nchw);

// 提取局部极大值作为特征点 (采用网格轮询均匀化分配 Grid Round-Robin Bucketing)
std::vector<Keypoint> extract_keypoints(const float* scores_dense, int H, int W,
                                        int max_kpts = 500, float min_score = 0.0003f,
                                        int radius = 4, int grid_rows = 6, int grid_cols = 8);

// 描述子特征采样 (双线性插值采样并在 INT8 空间计算后做 L2-Norm)
std::vector<float> sample_descriptor_int8(const int8_t* desc_map, float x_desc, float y_desc,
                                         int H, int W, int C, bool is_nchw);

#endif // HFNET_POSTPROCESS_H
