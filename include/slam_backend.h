#ifndef HFNET_SLAM_BACKEND_H
#define HFNET_SLAM_BACKEND_H

#include "slam_core.h"
#include "slam_frontend.h"

namespace slam {

// 显式解析雅可比与稀疏舒尔补后端优化器 (1:1 对齐 Python src/backend/optimizer.py)
class Optimizer {
public:
    static constexpr double CHI2_MONO = 5.991;
    static constexpr double CHI2_STEREO = 7.815;
    static constexpr double HUBER_MONO = 2.4476519358;   // sqrt(5.991)
    static constexpr double HUBER_STEREO = 2.7955321497; // sqrt(7.815)

    // 4 轮 Huber 鲁棒核 + 卡方检验的当前帧 6DoF 位姿优化 (Motion-only BA + IMU 旋转先验紧耦合)
    static int pose_optimization(Frame& frame, int n_its_per_round = 6,
                                 const Mat3* R_cw_prior = nullptr,
                                 double rot_prior_weight = 0.0);

    // 局部滑动窗口捆集调整 (Local BA: 基于显式稀疏舒尔补 Sparse Schur Complement)
    static void local_bundle_adjustment(std::shared_ptr<KeyFrame> curr_kf,
                                        SlamMap& slam_map, int n_iterations = 5);
};

// 局部建图器 (1:1 对齐 Python src/backend/local_mapper.py)
class LocalMapper {
public:
    SlamMap* slam_map = nullptr;
    std::vector<std::shared_ptr<MapPoint>> recent_added_mappoints;

    explicit LocalMapper(SlamMap* map_ptr) : slam_map(map_ptr) {}

    void process_new_keyframe(std::shared_ptr<KeyFrame> curr_kf);

private:
    void mappoint_culling(std::shared_ptr<KeyFrame> curr_kf);
    void create_new_mappoints(std::shared_ptr<KeyFrame> curr_kf);
    void keyframe_culling(std::shared_ptr<KeyFrame> curr_kf);
};

} // namespace slam

#endif // HFNET_SLAM_BACKEND_H
