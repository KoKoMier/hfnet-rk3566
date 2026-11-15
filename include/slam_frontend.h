#ifndef HFNET_SLAM_FRONTEND_H
#define HFNET_SLAM_FRONTEND_H

#include "slam_core.h"

namespace slam {

class FrontendMatcher {
public:
    // 构建 4 层 2x 下采样高斯金字塔 (用于纯 C++ 金字塔 LK 光流)
    static void build_pyramid(const uint8_t* gray, int w, int h, int levels,
                              std::vector<std::vector<uint8_t>>& pyr,
                              std::vector<int>& widths, std::vector<int>& heights);

    // 4 层双向金字塔 LK 光流跟踪 (带 IMU 预测位姿先验 T_cw_pred 与反向光流一致性校验)
    static void track_optical_flow(
        const Frame& last_frame,
        const std::vector<uint8_t>& curr_gray,
        std::vector<std::vector<uint8_t>>& curr_pyr,
        std::vector<int>& curr_pyr_w,
        std::vector<int>& curr_pyr_h,
        const StereoCamera& camera,
        const SE3* T_cw_pred,
        std::vector<Vec2>& out_kp_pts,
        std::vector<std::shared_ptr<MapPoint>>& out_tracked_mps,
        std::vector<Vec2>& out_prev_pts
    );

    // 单目初始化阶段: 双向 4 层金字塔 LK 光流跟踪参考帧 F0 -> 当前帧 Fk (fb_err <= 1.5 px)
    static void track_init_optical_flow(
        const Frame& init_frame,
        const std::vector<std::vector<uint8_t>>& curr_pyr,
        const std::vector<int>& curr_pyr_w,
        const std::vector<int>& curr_pyr_h,
        const StereoCamera& camera,
        std::vector<int>& out_matched_idx0,
        std::vector<Vec2>& out_matched_pts0,
        std::vector<Vec2>& out_matched_pts1
    );

    // 256 维归一化 HF-Net 描述子余弦距离 (0.0 表示完全相同, 2.0 表示相反)
    static float descriptor_distance(const std::vector<float>& d1, const std::vector<float>& d2);

    // 局部地图点投影匹配 (64x48 空间网格 O(1) 邻域搜索 + 次优比检验)
    static int search_by_projection(Frame& curr_frame,
                                    const std::vector<std::shared_ptr<MapPoint>>& candidates,
                                    double th_radius = 8.0);

    // 关键帧间对极几何约束匹配 (用于新建三角化 3D 地图点)
    static std::vector<std::pair<int, int>> search_for_triangulation(
        const KeyFrame& kf1, const KeyFrame& kf2
    );

    // 二维梯度结构张量 (Shi-Tomasi 最小特征值与各向同性比) 亚像素角点精化与快速网格补点检测
    static void detect_grid_corners_with_scores(
        const uint8_t* gray, const float* scores_dense, int w, int h,
        const bool* occ_grid, int occ_w, int occ_h,
        int cell_counts[6][8], int max_per_cell, int need_total,
        std::vector<Vec2>& out_added_pts
    );
};

} // namespace slam

#endif // HFNET_SLAM_FRONTEND_H
