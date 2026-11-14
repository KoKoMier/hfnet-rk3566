#ifndef HFNET_SLAM_CORE_H
#define HFNET_SLAM_CORE_H

#include "slam_math.h"
#include "types.h"
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>

namespace slam {

class KeyFrame;
class MapPoint;
class SlamMap;

// 相机内参模型 (1:1 对齐 Python src/core/camera.py)
class StereoCamera {
public:
    double fx = 622.0;
    double fy = 622.0;
    double cx = 320.0;
    double cy = 240.0;
    double inv_fx = 1.0 / 622.0;
    double inv_fy = 1.0 / 622.0;
    double baseline = 0.060;
    double bf = 622.0 * 0.060;
    int width = 640;
    int height = 480;
    double th_depth = 2.40;
    double fps = 30.0;

    StereoCamera() = default;
    StereoCamera(double fx_, double fy_, double cx_, double cy_, double baseline_,
                 int width_ = 640, int height_ = 480, double th_depth_factor = 40.0, double fps_ = 30.0);

    static StereoCamera from_calib_json(const std::string& json_path, int default_w = 640, int default_h = 480);

    Vec2 project(const Vec3& p_c) const {
        double inv_z = 1.0 / p_c.z;
        return Vec2(fx * p_c.x * inv_z + cx, fy * p_c.y * inv_z + cy);
    }
    Vec3 project_stereo(const Vec3& p_c) const {
        double inv_z = 1.0 / p_c.z;
        double u = fx * p_c.x * inv_z + cx;
        double v = fy * p_c.y * inv_z + cy;
        double ur = u - bf * inv_z;
        return Vec3(u, v, ur);
    }
    Vec3 unproject(double u, double v, double z = 1.0) const {
        return Vec3((u - cx) * inv_fx * z, (v - cy) * inv_fy * z, z);
    }
    bool is_in_image(double u, double v, double margin = 0.0) const {
        return (u >= margin) && (u < width - margin) && (v >= margin) && (v < height - margin);
    }
};

// 3D 地图点类 (1:1 对齐 Python src/core/mappoint.py)
class MapPoint : public std::enable_shared_from_this<MapPoint> {
public:
    static int next_id_;

    int id = 0;
    Vec3 pos_w;
    Vec3 normal = Vec3(0.0, 0.0, 1.0);
    std::vector<float> descriptor; // 256-D L2-normalized HF-Net descriptor

    std::shared_ptr<KeyFrame> ref_kf;
    int first_kf_id = -1;
    int first_frame_id = -1;

    std::map<KeyFrame*, int> observations;
    int visible_count = 1;
    int found_count = 1;

    double min_distance = 0.1;
    double max_distance = 100.0;

    bool is_bad = false;
    bool is_temp_flow_point = false;

    // 前端投影跟踪缓存变量
    double track_proj_x = 0.0;
    double track_proj_y = 0.0;
    double track_proj_xr = 0.0;
    bool track_in_view = false;
    int track_scale_level = 0;
    double track_view_cos = 1.0;
    int last_frame_seen = -1;

    MapPoint(const Vec3& pos, std::shared_ptr<KeyFrame> ref = nullptr,
             int frame_id = -1, const std::vector<float>& desc = {});

    int num_observations() const { return (int)observations.size(); }
    void add_observation(KeyFrame* kf, int idx);
    void erase_observation(KeyFrame* kf, SlamMap* slam_map = nullptr);
    void set_bad_flag(SlamMap* slam_map = nullptr);
    void increase_visible(int n = 1) { visible_count += n; }
    void increase_found(int n = 1) { found_count += n; }
    double get_found_ratio() const {
        return (visible_count > 0) ? ((double)found_count / (double)visible_count) : 0.0;
    }
    void compute_distinctive_descriptors();
    void update_normal_and_depth();
};

// 普通图像帧类 (64x48 空间网格索引, 1:1 对齐 Python src/core/frame.py)
class Frame {
public:
    static int next_id_;
    static constexpr int GRID_COLS = 64;
    static constexpr int GRID_ROWS = 48;

    int id = 0;
    double timestamp = 0.0;
    StereoCamera camera;

    std::vector<uint8_t> gray_left;
    std::vector<std::vector<uint8_t>> pyramid_left;
    std::vector<int> pyr_widths;
    std::vector<int> pyr_heights;

    std::vector<Vec2> kp_pts;
    std::vector<int> kp_octaves;
    std::vector<float> kp_scores;
    std::vector<std::vector<float>> descriptors; // (N, 256)
    std::vector<float> u_right;
    std::vector<float> depths;
    int num_kps = 0;

    std::vector<std::shared_ptr<MapPoint>> map_points;
    std::vector<bool> outliers;

    SE3 T_cw;
    bool has_pose = false;
    Vec3 v_w = Vec3(0, 0, 0);
    Vec3 bg = Vec3(0, 0, 0);
    Vec3 ba = Vec3(0, 0, 0);
    bool has_imu_rot_prior = false;
    Mat3 R_cw_imu_prior = Mat3::identity();

    std::shared_ptr<KeyFrame> ref_kf;

    double grid_element_width_inv = 1.0;
    double grid_element_height_inv = 1.0;
    std::vector<int> grid[GRID_COLS][GRID_ROWS];

    Frame() = default;
    Frame(double ts, const StereoCamera& cam, const std::vector<uint8_t>& gray,
          const std::vector<Keypoint>& kps);

    void set_pose(const SE3& T) { T_cw = T; has_pose = true; }
    SE3 get_pose_inverse() const { return T_cw.inverse(); }
    Vec3 get_camera_center() const { return T_cw.camera_center(); }

    void assign_features_to_grid();
    std::vector<int> get_features_in_area(double x, double y, double r) const;
    Vec3 unproject_stereo(int i) const;
};

// 关键帧与共视图节点类 (1:1 对齐 Python src/core/keyframe.py)
class KeyFrame : public std::enable_shared_from_this<KeyFrame> {
public:
    static int next_id_;

    int id = 0;
    int frame_id = 0;
    double timestamp = 0.0;
    StereoCamera camera;

    std::vector<Vec2> kp_pts;
    std::vector<int> kp_octaves;
    std::vector<float> kp_scores;
    std::vector<std::vector<float>> descriptors;
    std::vector<float> u_right;
    std::vector<float> depths;
    int num_kps = 0;

    std::vector<std::shared_ptr<MapPoint>> map_points;

    SE3 T_cw;
    Vec3 v_w = Vec3(0, 0, 0);
    Vec3 bg = Vec3(0, 0, 0);
    Vec3 ba = Vec3(0, 0, 0);
    bool has_imu_rot_prior = false;
    Mat3 R_cw_imu_prior = Mat3::identity();

    double grid_element_width_inv = 1.0;
    double grid_element_height_inv = 1.0;
    std::vector<int> grid[Frame::GRID_COLS][Frame::GRID_ROWS];

    std::map<KeyFrame*, int> connected_kf_weights;
    std::vector<KeyFrame*> ordered_connected_kfs;
    std::vector<int> ordered_weights;

    bool first_connection = true;
    KeyFrame* parent = nullptr;
    std::set<KeyFrame*> children;
    std::set<KeyFrame*> loop_edges;
    bool is_bad = false;

    explicit KeyFrame(const Frame& frame);

    void set_pose(const SE3& T) { T_cw = T; }
    SE3 get_pose() const { return T_cw; }
    SE3 get_pose_inverse() const { return T_cw.inverse(); }
    Vec3 get_camera_center() const { return T_cw.camera_center(); }

    void add_mappoint(std::shared_ptr<MapPoint> mp, int idx);
    void erase_mappoint_match(int idx);
    int tracked_mappoints(int min_obs = 1) const;
    std::vector<int> get_features_in_area(double x, double y, double r) const;

    void add_connection(KeyFrame* kf, int weight);
    void erase_connection(KeyFrame* kf);
    void update_best_covisibles();
    void update_connections();
    std::vector<KeyFrame*> get_best_covisibility_keyframes(int n) const;
};

// 全局共视地图容器 (1:1 对齐 Python src/core/map.py)
class SlamMap {
public:
    std::map<int, std::shared_ptr<KeyFrame>> keyframes;
    std::map<int, std::shared_ptr<MapPoint>> mappoints;
    std::vector<std::shared_ptr<MapPoint>> reference_mappoints;

    void add_keyframe(std::shared_ptr<KeyFrame> kf) { keyframes[kf->id] = kf; }
    void add_mappoint(std::shared_ptr<MapPoint> mp) { mappoints[mp->id] = mp; }
    void erase_mappoint(MapPoint* mp) { if (mp) mappoints.erase(mp->id); }
    void erase_keyframe(KeyFrame* kf) { if (kf) keyframes.erase(kf->id); }

    int num_keyframes() const { return (int)keyframes.size(); }
    int num_mappoints() const { return (int)mappoints.size(); }

    std::vector<std::shared_ptr<KeyFrame>> get_all_keyframes() const;
    std::vector<std::shared_ptr<MapPoint>> get_all_mappoints() const;
};

} // namespace slam

#endif // HFNET_SLAM_CORE_H
