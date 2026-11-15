#ifndef HFNET_SLAM_TRACKER_H
#define HFNET_SLAM_TRACKER_H

#include "slam_core.h"
#include "slam_frontend.h"
#include "slam_backend.h"
#include "imu_preintegration.h"

namespace slam {

enum class TrackingState {
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    LOST = 3
};

struct TrackerDiagnostics {
    double trans_gate = 1.0;
    double eff_gyro_deg_s = 0.0;
    double raw_flow_px = 0.0;
    double derot_res_px = 0.0;
    double radial_med_px = 0.0;
    double frontend_ms = 0.0;
    double backend_ms = 0.0;
    double total_ms = 0.0;
};

// 前端实时位姿跟踪状态机 (1:1 对齐 Python src/frontend/tracker.py)
class Tracker {
public:
    StereoCamera camera;
    SlamMap* slam_map = nullptr;
    LocalMapper* local_mapper = nullptr;

    TrackingState state = TrackingState::NO_IMAGES_YET;

    Frame curr_frame;
    Frame last_frame;
    bool has_last_frame = false;
    Frame init_frame;
    bool has_init_frame = false;

    std::shared_ptr<KeyFrame> ref_kf;
    std::vector<KeyFrame*> local_keyframes;
    std::vector<std::shared_ptr<MapPoint>> local_mappoints;

    SE3 velocity = SE3::identity();
    bool has_velocity = false;
    int last_kf_frame_id = 0;

    ImuPreintegrator imu_preintegrator;
    ImuPreintegrator init_imu_preintegrator;
    std::vector<Vec3> gyro_vec_hist_;

    bool last_is_keyframe = false;
    bool last_used_optical_flow = false;
    int last_inliers = 0;
    TrackerDiagnostics last_diag;

    // 用于在 NV12 画面上直接绘制光流轨迹线与内点状态
    std::vector<Vec2> last_tracked_curr_pts;
    std::vector<Vec2> last_tracked_prev_pts;

    Tracker(const StereoCamera& cam, SlamMap* map_ptr, LocalMapper* mapper_ptr);

    SE3 grab_image_monocular_inertial(
        const std::vector<uint8_t>& gray_l,
        const float* scores_dense,
        const int8_t* desc_raw,
        bool desc_is_nchw,
        const std::vector<Keypoint>& hfnet_kps,
        double timestamp,
        const std::vector<ImuMeasurement>& imu_measurements
    );

    static const char* state_to_string(TrackingState s);

private:
    void monocular_initialization(
        const std::vector<uint8_t>& gray_l,
        const std::vector<Keypoint>& hfnet_kps,
        double timestamp,
        const std::vector<ImuMeasurement>& imu_measurements
    );

    SE3 compute_predicted_pose(const std::vector<ImuMeasurement>& imu_measurements);

    void estimate_rotation_and_trans_gate(
        const std::vector<Vec2>& flow_pts,
        const std::vector<std::shared_ptr<MapPoint>>& flow_mps,
        const std::vector<ImuMeasurement>& imu_measurements,
        bool& has_rot_prior_inout,
        Mat3& R_cw_prior_inout,
        double& trans_gate_out
    );

    void merge_flow_and_hfnet_features(
        const std::vector<Vec2>& flow_pts,
        const std::vector<std::shared_ptr<MapPoint>>& flow_mps,
        Frame& frame
    );

    void match_from_last_frame();
    int track_local_map(const Mat3* R_cw_prior, double rot_prior_weight);
    void update_local_map();
    void search_local_points();

    int replenish_live_flow_features(
        const std::vector<uint8_t>& gray_l,
        const float* scores_dense,
        const int8_t* desc_raw,
        bool desc_is_nchw,
        int n_inliers
    );

    bool need_new_keyframe(int n_tracked_inliers, bool is_flow_only = false) const;
    void create_new_keyframe();
};

} // namespace slam

#endif // HFNET_SLAM_TRACKER_H
