#include "slam_core.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace slam {

int MapPoint::next_id_ = 0;
int Frame::next_id_ = 0;
int KeyFrame::next_id_ = 0;

StereoCamera::StereoCamera(double fx_, double fy_, double cx_, double cy_, double baseline_,
                           int width_, int height_, double th_depth_factor, double fps_)
    : fx(fx_), fy(fy_), cx(cx_), cy(cy_),
      inv_fx(1.0 / fx_), inv_fy(1.0 / fy_),
      baseline(baseline_), bf(fx_ * baseline_),
      width(width_), height(height_),
      th_depth(baseline_ * th_depth_factor), fps(fps_) {}

static bool parse_json_double(const std::string& content, const std::string& key, double& out_val) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = content.find(pattern);
    if (pos == std::string::npos) return false;
    pos = content.find(':', pos + pattern.size());
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n' || content[pos] == '\r')) pos++;
    char* endptr = nullptr;
    double val = std::strtod(content.c_str() + pos, &endptr);
    if (endptr == content.c_str() + pos) return false;
    out_val = val;
    return true;
}

StereoCamera StereoCamera::from_calib_json(const std::string& json_path, int default_w, int default_h) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        ifs.open("../config/custom_stereo_imu.json");
    }
    if (!ifs.is_open()) {
        ifs.open("../../config/custom_stereo_imu.json");
    }
    if (!ifs.is_open()) {
        return StereoCamera(622.0, 622.0, default_w * 0.5, default_h * 0.5, 0.060, default_w, default_h, 40.0, 30.0);
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string txt = ss.str();

    double fx_v = 622.0, fy_v = 622.0, cx_v = default_w * 0.5, cy_v = default_h * 0.5, bl_v = 0.060;
    double w_v = default_w, h_v = default_h, fps_v = 30.0;
    parse_json_double(txt, "fx", fx_v);
    parse_json_double(txt, "fy", fy_v);
    parse_json_double(txt, "cx", cx_v);
    parse_json_double(txt, "cy", cy_v);
    parse_json_double(txt, "baseline", bl_v);
    parse_json_double(txt, "width", w_v);
    parse_json_double(txt, "height", h_v);
    parse_json_double(txt, "fps", fps_v);
    return StereoCamera(fx_v, fy_v, cx_v, cy_v, bl_v, (int)w_v, (int)h_v, 40.0, fps_v);
}

MapPoint::MapPoint(const Vec3& pos, std::shared_ptr<KeyFrame> ref,
                   int frame_id, const std::vector<float>& desc)
    : id(next_id_++), pos_w(pos), descriptor(desc), ref_kf(ref),
      first_kf_id(ref ? ref->id : -1), first_frame_id(frame_id) {}

void MapPoint::add_observation(KeyFrame* kf, int idx) {
    if (!kf || is_bad) return;
    observations[kf] = idx;
}

void MapPoint::erase_observation(KeyFrame* kf, SlamMap* slam_map) {
    auto it = observations.find(kf);
    if (it != observations.end()) {
        observations.erase(it);
        if (ref_kf.get() == kf && !observations.empty()) {
            // 维持有效参考关键帧
        }
        if ((int)observations.size() <= 2 && !is_temp_flow_point && first_kf_id >= 0 && observations.empty()) {
            set_bad_flag(slam_map);
        }
    }
}

void MapPoint::set_bad_flag(SlamMap* slam_map) {
    is_bad = true;
    auto obs_copy = observations;
    observations.clear();
    for (const auto& kv : obs_copy) {
        if (kv.first && !kv.first->is_bad) {
            kv.first->erase_mappoint_match(kv.second);
        }
    }
    if (slam_map) {
        slam_map->erase_mappoint(this);
    }
}

void MapPoint::compute_distinctive_descriptors() {
    if (is_bad || observations.empty()) return;
    std::vector<const std::vector<float>*> descs;
    descs.reserve(observations.size());
    for (const auto& kv : observations) {
        KeyFrame* kf = kv.first;
        int idx = kv.second;
        if (kf && !kf->is_bad && idx >= 0 && idx < (int)kf->descriptors.size() && !kf->descriptors[idx].empty()) {
            descs.push_back(&kf->descriptors[idx]);
        }
    }
    if (descs.empty()) return;
    if (descs.size() == 1) {
        descriptor = *descs[0];
        return;
    }

    int N = (int)descs.size();
    int best_idx = 0;
    float best_median = 1e9f;
    std::vector<float> dists(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) { dists[j] = 0.0f; continue; }
            const auto& d1 = *descs[i];
            const auto& d2 = *descs[j];
            float dot = 0.0f;
            int C = (int)std::min(d1.size(), d2.size());
            for (int c = 0; c < C; ++c) dot += d1[c] * d2[c];
            dists[j] = std::max(0.0f, 1.0f - dot);
        }
        std::sort(dists.begin(), dists.end());
        float med = dists[N / 2];
        if (med < best_median) {
            best_median = med;
            best_idx = i;
        }
    }
    descriptor = *descs[best_idx];
}

void MapPoint::update_normal_and_depth() {
    if (is_bad || observations.empty()) return;
    Vec3 normal_sum(0.0, 0.0, 0.0);
    int n = 0;
    for (const auto& kv : observations) {
        KeyFrame* kf = kv.first;
        if (!kf || kf->is_bad) continue;
        Vec3 o_w = kf->get_camera_center();
        Vec3 v = pos_w - o_w;
        normal_sum += v.normalized();
        n++;
    }
    if (n > 0) {
        normal = (normal_sum / (double)n).normalized();
    }
    if (ref_kf && !ref_kf->is_bad) {
        double dist = (pos_w - ref_kf->get_camera_center()).norm();
        max_distance = dist * 1.4;
        min_distance = dist / 1.4;
    }
}

Frame::Frame(double ts, const StereoCamera& cam, const std::vector<uint8_t>& gray,
             const std::vector<Keypoint>& kps)
    : id(next_id_++), timestamp(ts), camera(cam), gray_left(gray) {
    num_kps = (int)kps.size();
    kp_pts.resize(num_kps);
    kp_octaves.assign(num_kps, 0);
    kp_scores.resize(num_kps);
    descriptors.resize(num_kps);
    u_right.assign(num_kps, -1.0f);
    depths.assign(num_kps, -1.0f);
    map_points.assign(num_kps, nullptr);
    outliers.assign(num_kps, false);

    for (int i = 0; i < num_kps; ++i) {
        kp_pts[i] = Vec2(kps[i].x, kps[i].y);
        kp_scores[i] = kps[i].score;
        descriptors[i] = kps[i].descriptor;
    }

    grid_element_width_inv = (double)GRID_COLS / (double)std::max(1, camera.width);
    grid_element_height_inv = (double)GRID_ROWS / (double)std::max(1, camera.height);
    assign_features_to_grid();
}

void Frame::assign_features_to_grid() {
    for (int c = 0; c < GRID_COLS; ++c)
        for (int r = 0; r < GRID_ROWS; ++r)
            grid[c][r].clear();

    for (int i = 0; i < num_kps; ++i) {
        int gx = (int)std::floor(kp_pts[i].x * grid_element_width_inv);
        int gy = (int)std::floor(kp_pts[i].y * grid_element_height_inv);
        if (gx >= 0 && gx < GRID_COLS && gy >= 0 && gy < GRID_ROWS) {
            grid[gx][gy].push_back(i);
        }
    }
}

std::vector<int> Frame::get_features_in_area(double x, double y, double r) const {
    std::vector<int> indices;
    if (r <= 0.0) return indices;

    int min_cell_x = std::max(0, (int)std::floor((x - r) * grid_element_width_inv));
    int max_cell_x = std::min(GRID_COLS - 1, (int)std::ceil((x + r) * grid_element_width_inv));
    int min_cell_y = std::max(0, (int)std::floor((y - r) * grid_element_height_inv));
    int max_cell_y = std::min(GRID_ROWS - 1, (int)std::ceil((y + r) * grid_element_height_inv));

    double r2 = r * r;
    for (int ix = min_cell_x; ix <= max_cell_x; ++ix) {
        for (int iy = min_cell_y; iy <= max_cell_y; ++iy) {
            for (int idx : grid[ix][iy]) {
                double dx = kp_pts[idx].x - x;
                double dy = kp_pts[idx].y - y;
                if (dx * dx + dy * dy <= r2) {
                    indices.push_back(idx);
                }
            }
        }
    }
    return indices;
}

Vec3 Frame::unproject_stereo(int i) const {
    double z = depths[i];
    if (z <= 0.0) return Vec3(0, 0, 0);
    Vec3 p_c = camera.unproject(kp_pts[i].x, kp_pts[i].y, z);
    return T_cw.inverse().transform(p_c);
}

KeyFrame::KeyFrame(const Frame& frame)
    : id(next_id_++), frame_id(frame.id), timestamp(frame.timestamp), camera(frame.camera),
      kp_pts(frame.kp_pts), kp_octaves(frame.kp_octaves), kp_scores(frame.kp_scores),
      descriptors(frame.descriptors), u_right(frame.u_right), depths(frame.depths),
      num_kps(frame.num_kps), map_points(frame.map_points),
      T_cw(frame.T_cw), v_w(frame.v_w), bg(frame.bg), ba(frame.ba),
      has_imu_rot_prior(frame.has_imu_rot_prior), R_cw_imu_prior(frame.R_cw_imu_prior),
      grid_element_width_inv(frame.grid_element_width_inv),
      grid_element_height_inv(frame.grid_element_height_inv) {
    for (int c = 0; c < Frame::GRID_COLS; ++c)
        for (int r = 0; r < Frame::GRID_ROWS; ++r)
            grid[c][r] = frame.grid[c][r];
}

void KeyFrame::add_mappoint(std::shared_ptr<MapPoint> mp, int idx) {
    if (idx >= 0 && idx < num_kps) {
        map_points[idx] = mp;
    }
}

void KeyFrame::erase_mappoint_match(int idx) {
    if (idx >= 0 && idx < num_kps) {
        map_points[idx] = nullptr;
    }
}

int KeyFrame::tracked_mappoints(int min_obs) const {
    int count = 0;
    for (const auto& mp : map_points) {
        if (mp && !mp->is_bad) {
            if (min_obs <= 1 || mp->num_observations() >= min_obs) count++;
        }
    }
    return count;
}

std::vector<int> KeyFrame::get_features_in_area(double x, double y, double r) const {
    std::vector<int> indices;
    if (r <= 0.0) return indices;

    int min_cell_x = std::max(0, (int)std::floor((x - r) * grid_element_width_inv));
    int max_cell_x = std::min(Frame::GRID_COLS - 1, (int)std::ceil((x + r) * grid_element_width_inv));
    int min_cell_y = std::max(0, (int)std::floor((y - r) * grid_element_height_inv));
    int max_cell_y = std::min(Frame::GRID_ROWS - 1, (int)std::ceil((y + r) * grid_element_height_inv));

    double r2 = r * r;
    for (int ix = min_cell_x; ix <= max_cell_x; ++ix) {
        for (int iy = min_cell_y; iy <= max_cell_y; ++iy) {
            for (int idx : grid[ix][iy]) {
                double dx = kp_pts[idx].x - x;
                double dy = kp_pts[idx].y - y;
                if (dx * dx + dy * dy <= r2) {
                    indices.push_back(idx);
                }
            }
        }
    }
    return indices;
}

void KeyFrame::add_connection(KeyFrame* kf, int weight) {
    connected_kf_weights[kf] = weight;
    update_best_covisibles();
}

void KeyFrame::erase_connection(KeyFrame* kf) {
    if (connected_kf_weights.erase(kf) > 0) {
        update_best_covisibles();
    }
}

void KeyFrame::update_best_covisibles() {
    std::vector<std::pair<int, KeyFrame*>> pairs;
    for (const auto& kv : connected_kf_weights) {
        if (kv.first && !kv.first->is_bad) {
            pairs.emplace_back(kv.second, kv.first);
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const std::pair<int, KeyFrame*>& a, const std::pair<int, KeyFrame*>& b) {
        return a.first > b.first;
    });
    ordered_connected_kfs.clear();
    ordered_weights.clear();
    for (const auto& p : pairs) {
        ordered_connected_kfs.push_back(p.second);
        ordered_weights.push_back(p.first);
    }
}

void KeyFrame::update_connections() {
    std::map<KeyFrame*, int> kf_counter;
    for (const auto& mp : map_points) {
        if (!mp || mp->is_bad) continue;
        for (const auto& kv : mp->observations) {
            KeyFrame* other = kv.first;
            if (!other || other == this || other->is_bad) continue;
            kf_counter[other]++;
        }
    }
    if (kf_counter.empty()) return;

    int max_w = 0;
    KeyFrame* best_kf = nullptr;
    const int th = 15;
    connected_kf_weights.clear();
    for (const auto& kv : kf_counter) {
        if (kv.second > max_w) {
            max_w = kv.second;
            best_kf = kv.first;
        }
        if (kv.second >= th) {
            connected_kf_weights[kv.first] = kv.second;
            kv.first->add_connection(this, kv.second);
        }
    }
    if (connected_kf_weights.empty() && best_kf != nullptr) {
        connected_kf_weights[best_kf] = max_w;
        best_kf->add_connection(this, max_w);
    }
    update_best_covisibles();

    if (first_connection && id != 0 && best_kf != nullptr) {
        parent = best_kf;
        parent->children.insert(this);
        first_connection = false;
    }
}

std::vector<KeyFrame*> KeyFrame::get_best_covisibility_keyframes(int n) const {
    if ((int)ordered_connected_kfs.size() <= n) return ordered_connected_kfs;
    return std::vector<KeyFrame*>(ordered_connected_kfs.begin(), ordered_connected_kfs.begin() + n);
}

std::vector<std::shared_ptr<KeyFrame>> SlamMap::get_all_keyframes() const {
    std::vector<std::shared_ptr<KeyFrame>> res;
    res.reserve(keyframes.size());
    for (const auto& kv : keyframes) {
        if (kv.second && !kv.second->is_bad) res.push_back(kv.second);
    }
    return res;
}

std::vector<std::shared_ptr<MapPoint>> SlamMap::get_all_mappoints() const {
    std::vector<std::shared_ptr<MapPoint>> res;
    res.reserve(mappoints.size());
    for (const auto& kv : mappoints) {
        if (kv.second && !kv.second->is_bad) res.push_back(kv.second);
    }
    return res;
}

} // namespace slam
