#include "slam_tracker.h"
#include "hfnet_postprocess.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <set>

namespace slam {

Tracker::Tracker(const StereoCamera& cam, SlamMap* map_ptr, LocalMapper* mapper_ptr)
    : camera(cam), slam_map(map_ptr), local_mapper(mapper_ptr) {}

const char* Tracker::state_to_string(TrackingState s) {
    switch (s) {
        case TrackingState::NO_IMAGES_YET: return "NO_IMAGES_YET";
        case TrackingState::NOT_INITIALIZED: return "NOT_INITIALIZED";
        case TrackingState::OK: return "OK";
        case TrackingState::LOST: return "LOST";
    }
    return "UNKNOWN";
}

SE3 Tracker::compute_predicted_pose(const std::vector<ImuMeasurement>& imu_measurements) {
    if (!has_last_frame || !last_frame.has_pose) return SE3::identity();

    if (!imu_measurements.empty()) {
        imu_preintegrator.reset_state();
        imu_preintegrator.integrate_batch(imu_measurements, last_frame.timestamp, curr_frame.timestamp);
        if (imu_preintegrator.dt_total > 0.0) {
            SE3 T_wc_i = last_frame.get_pose_inverse();
            // 将机体系陀螺仪旋转变化量映射至相机光学系:
            // 机身系 (X右, Y前, Z上) -> 相机光学系 (X右, Y下, Z前): R_cb = [[1,0,0],[0,0,-1],[0,1,0]]
            Mat3 R_cb;
            R_cb(0, 0) = 1.0; R_cb(0, 1) =  0.0; R_cb(0, 2) =  0.0;
            R_cb(1, 0) = 0.0; R_cb(1, 1) =  0.0; R_cb(1, 2) = -1.0;
            R_cb(2, 0) = 0.0; R_cb(2, 1) =  1.0; R_cb(2, 2) =  0.0;
            Mat3 dR_cam = R_cb * imu_preintegrator.delta_R * R_cb.transpose();
            Mat3 R_wc_j = SO3::normalize_rotation(T_wc_i.R * dR_cam);

            Vec3 p_w_j = T_wc_i.t;
            if (has_velocity) {
                double dt = std::max(1e-3, curr_frame.timestamp - last_frame.timestamp);
                p_w_j += last_frame.v_w * dt;
            }
            return SE3(R_wc_j, p_w_j).inverse();
        }
    }

    if (has_velocity) {
        return velocity * last_frame.T_cw;
    }
    return last_frame.T_cw;
}

void Tracker::monocular_initialization(
    const std::vector<uint8_t>& gray_l,
    const std::vector<Keypoint>& hfnet_kps,
    double timestamp,
    const std::vector<ImuMeasurement>& imu_measurements
) {
    bool has_imu = !imu_measurements.empty();

    curr_frame = Frame(timestamp, camera, gray_l, hfnet_kps);
    FrontendMatcher::build_pyramid(gray_l.data(), camera.width, camera.height, 4,
                                   curr_frame.pyramid_left, curr_frame.pyr_widths, curr_frame.pyr_heights);

    if (has_imu && has_last_frame) {
        curr_frame.set_pose(compute_predicted_pose(imu_measurements));
    } else {
        curr_frame.set_pose(SE3::identity());
    }

    if (!has_init_frame) {
        if (curr_frame.num_kps >= 60) {
            init_frame = curr_frame;
            has_init_frame = true;
            last_frame = curr_frame;
            has_last_frame = true;
            init_imu_preintegrator.reset_state();
        }
        return;
    }

    if (has_imu && has_last_frame) {
        init_imu_preintegrator.integrate_batch(imu_measurements, last_frame.timestamp, timestamp);
    }

    // 使用 HF-Net 256 维描述子在 init_frame 与 curr_frame 之间匹配
    std::vector<int> matches12(init_frame.num_kps, -1);
    std::vector<double> disp_list;
    for (int i1 = 0; i1 < init_frame.num_kps; ++i1) {
        if (init_frame.descriptors[i1].empty()) continue;
        const Vec2& p1 = init_frame.kp_pts[i1];
        std::vector<int> cands = curr_frame.get_features_in_area(p1.x, p1.y, 80.0);
        float best_d = 10.0f, second_d = 10.0f;
        int best_i2 = -1;
        for (int i2 : cands) {
            if (curr_frame.descriptors[i2].empty()) continue;
            float d = FrontendMatcher::descriptor_distance(init_frame.descriptors[i1], curr_frame.descriptors[i2]);
            if (d < best_d) {
                second_d = best_d;
                best_d = d;
                best_i2 = i2;
            } else if (d < second_d) {
                second_d = d;
            }
        }
        if (best_i2 >= 0 && best_d < 0.28f && best_d < 0.85f * second_d) {
            matches12[i1] = best_i2;
            disp_list.push_back((curr_frame.kp_pts[best_i2] - p1).norm());
        }
    }

    int n_matched = (int)disp_list.size();
    if (n_matched < 45) {
        init_frame = curr_frame;
        last_frame = curr_frame;
        init_imu_preintegrator.reset_state();
        return;
    }

    std::sort(disp_list.begin(), disp_list.end());
    double med_disp = disp_list[disp_list.size() / 2];
    int frames_wait = curr_frame.id - init_frame.id;

    // 实机单目+IMU模式快速冷启动 (等待 2 帧或视差 >= 6px 即可建立初始关键帧与深度先验地图点)
    if (med_disp < 6.0 && frames_wait < 2) {
        last_frame = curr_frame;
        return;
    }

    init_frame.set_pose(SE3::identity());
    Mat3 R_10 = Mat3::identity();
    if (init_imu_preintegrator.dt_total > 0.0) {
        Mat3 R_cb;
        R_cb(0, 0) = 1.0; R_cb(0, 1) =  0.0; R_cb(0, 2) =  0.0;
        R_cb(1, 0) = 0.0; R_cb(1, 1) =  0.0; R_cb(1, 2) = -1.0;
        R_cb(2, 0) = 0.0; R_cb(2, 1) =  1.0; R_cb(2, 2) =  0.0;
        Mat3 dR_cam = R_cb * init_imu_preintegrator.delta_R * R_cb.transpose();
        R_10 = dR_cam.transpose();
    }

    // 估计初始平移基线 (静止微视差时采用零位移 + 1.2m 室内物理深度先验)
    Vec3 t_10(0.0, 0.0, 0.0);
    if (med_disp >= 8.0) {
        double sum_du = 0.0, sum_dv = 0.0;
        int cnt = 0;
        for (int i1 = 0; i1 < init_frame.num_kps; ++i1) {
            int i2 = matches12[i1];
            if (i2 < 0) continue;
            sum_du += (curr_frame.kp_pts[i2].x - init_frame.kp_pts[i1].x);
            sum_dv += (curr_frame.kp_pts[i2].y - init_frame.kp_pts[i1].y);
            cnt++;
        }
        if (cnt > 0) {
            double du_m = (sum_du / cnt) * camera.inv_fx * 1.2;
            double dv_m = (sum_dv / cnt) * camera.inv_fy * 1.2;
            double step_norm = std::hypot(du_m, dv_m);
            if (step_norm > 0.04) {
                du_m *= (0.04 / step_norm);
                dv_m *= (0.04 / step_norm);
            }
            t_10 = Vec3(du_m, dv_m, 0.0);
        }
    }
    curr_frame.set_pose(SE3(R_10, t_10));

    auto kf0 = std::make_shared<KeyFrame>(init_frame);
    auto kf1 = std::make_shared<KeyFrame>(curr_frame);
    slam_map->add_keyframe(kf0);
    slam_map->add_keyframe(kf1);

    int created_mps = 0;
    const double default_z = 1.20;
    for (int i1 = 0; i1 < init_frame.num_kps; ++i1) {
        int i2 = matches12[i1];
        if (i2 < 0) continue;

        Vec3 xn1 = camera.unproject(init_frame.kp_pts[i1].x, init_frame.kp_pts[i1].y, 1.0);
        Vec3 xn2 = camera.unproject(curr_frame.kp_pts[i2].x, curr_frame.kp_pts[i2].y, 1.0);
        Vec3 pos_w;
        if (!triangulate_two_views(kf0->T_cw, kf1->T_cw, xn1, xn2, pos_w)) {
            pos_w = camera.unproject(curr_frame.kp_pts[i2].x, curr_frame.kp_pts[i2].y, default_z);
            pos_w = kf1->get_pose_inverse().transform(pos_w);
        }

        auto mp = std::make_shared<MapPoint>(pos_w, kf1, curr_frame.id, curr_frame.descriptors[i2]);
        kf0->add_mappoint(mp, i1);
        kf1->add_mappoint(mp, i2);
        mp->add_observation(kf0.get(), i1);
        mp->add_observation(kf1.get(), i2);
        mp->compute_distinctive_descriptors();
        mp->update_normal_and_depth();
        slam_map->add_mappoint(mp);
        curr_frame.map_points[i2] = mp;
        created_mps++;
    }

    kf0->update_connections();
    kf1->update_connections();

    ref_kf = kf1;
    curr_frame.ref_kf = kf1;
    last_frame = curr_frame;
    has_last_frame = true;
    last_kf_frame_id = curr_frame.id;
    state = TrackingState::OK;
}

void Tracker::estimate_rotation_and_trans_gate(
    const std::vector<Vec2>& flow_pts,
    const std::vector<std::shared_ptr<MapPoint>>& flow_mps,
    const std::vector<ImuMeasurement>& imu_measurements,
    bool& has_rot_prior_inout,
    Mat3& R_cw_prior_inout,
    double& trans_gate_out
) {
    int n_pts = (int)flow_pts.size();
    if (!has_last_frame || !last_frame.has_pose || n_pts < 8) {
        trans_gate_out = 1.0;
        last_diag.trans_gate = 1.0;
        last_diag.eff_gyro_deg_s = 0.0;
        last_diag.raw_flow_px = 0.0;
        last_diag.derot_res_px = 0.0;
        last_diag.radial_med_px = 0.0;
        return;
    }

    double fx = camera.fx, fy = camera.fy, cx = camera.cx, cy = camera.cy;
    Mat3 R_last = last_frame.T_cw.R;
    Vec3 p_last = last_frame.get_camera_center();

    std::vector<Vec2> prev_uvs;
    std::vector<Vec2> curr_uvs;
    std::vector<Vec3> dirs_last;
    std::vector<Vec3> dirs_curr;
    std::vector<double> raw_flows;

    prev_uvs.reserve(n_pts);
    curr_uvs.reserve(n_pts);
    dirs_last.reserve(n_pts);
    dirs_curr.reserve(n_pts);
    raw_flows.reserve(n_pts);

    for (int i = 0; i < n_pts; ++i) {
        const auto& mp = flow_mps[i];
        if (!mp || mp->is_bad) continue;
        Vec3 v_c_last = R_last * (mp->pos_w - p_last);
        if (v_c_last.z <= 0.15) continue;
        Vec2 uv_prev = camera.project(v_c_last);
        Vec2 uv_curr = flow_pts[i];
        double flow_len = (uv_curr - uv_prev).norm();
        if (flow_len > 80.0) continue;

        prev_uvs.push_back(uv_prev);
        curr_uvs.push_back(uv_curr);
        raw_flows.push_back(flow_len);
        dirs_last.push_back(v_c_last.normalized());
        dirs_curr.push_back(Vec3((uv_curr.x - cx) / fx, (uv_curr.y - cy) / fy, 1.0).normalized());
    }

    int m = (int)dirs_last.size();
    if (m < 8) {
        trans_gate_out = 1.0;
        return;
    }

    std::sort(raw_flows.begin(), raw_flows.end());
    double raw_flow_med = raw_flows[m / 2];

    // 1. 视觉消旋迭代求解帧间纯旋转矩阵 R_rel_vis (使得 dirs_curr ≈ R_rel_vis * dirs_last)
    Mat3 R_rel_vis = has_rot_prior_inout ? (R_cw_prior_inout * R_last.transpose()) : Mat3::identity();
    for (int it = 0; it < 3; ++it) {
        Mat3 H_r = Mat3::diag(1e-4, 1e-4, 1e-4);
        Vec3 b_r(0, 0, 0);
        for (int i = 0; i < m; ++i) {
            Vec3 d_rot = R_rel_vis * dirs_last[i];
            Vec3 err = d_rot.cross(dirs_curr[i]); // 小角度旋转残差
            double e_n = err.norm();
            double w = (e_n > 0.015) ? (0.015 / e_n) : 1.0;
            for (int r = 0; r < 3; ++r) {
                b_r[r] += w * err[r];
                H_r(r, r) += w;
            }
        }
        Vec3 dphi;
        if (solve_linear_3x3(H_r, b_r, dphi)) {
            R_rel_vis = SO3::normalize_rotation(SO3::exp(dphi) * R_rel_vis);
        }
    }

    // 2. 计算消旋后的残差光流中位数 derot_res_px 与径向发散/收缩流 radial_med_px
    std::vector<double> derot_res_list;
    std::vector<double> radial_list;
    derot_res_list.reserve(m);
    radial_list.reserve(m);

    for (int i = 0; i < m; ++i) {
        Vec3 d_rot = R_rel_vis * dirs_last[i];
        if (d_rot.z <= 0.1) continue;
        double u_rot = fx * (d_rot.x / d_rot.z) + cx;
        double v_rot = fy * (d_rot.y / d_rot.z) + cy;
        double res_u = curr_uvs[i].x - u_rot;
        double res_v = curr_uvs[i].y - v_rot;
        derot_res_list.push_back(std::hypot(res_u, res_v));

        double rx = prev_uvs[i].x - cx;
        double ry = prev_uvs[i].y - cy;
        double r_len = std::hypot(rx, ry);
        if (r_len > 30.0) {
            radial_list.push_back((res_u * rx + res_v * ry) / r_len);
        }
    }

    double derot_res_px = 0.0;
    if (!derot_res_list.empty()) {
        std::sort(derot_res_list.begin(), derot_res_list.end());
        derot_res_px = derot_res_list[derot_res_list.size() / 2];
    }
    double radial_med_px = 0.0;
    if (!radial_list.empty()) {
        std::sort(radial_list.begin(), radial_list.end());
        radial_med_px = std::abs(radial_list[radial_list.size() / 2]);
    }

    // 3. 计算有效旋转角速度 eff_gyro_deg_s (融合 IMU 陀螺仪与视觉旋转角)
    double dt = std::max(1e-3, curr_frame.timestamp - last_frame.timestamp);
    double vis_rot_deg = std::acos(std::min(1.0, std::max(-1.0, 0.5 * (R_rel_vis.trace() - 1.0)))) * (180.0 / M_PI);
    double vis_gyro_deg_s = vis_rot_deg / dt;
    double imu_gyro_deg_s = 0.0;
    if (!imu_measurements.empty()) {
        for (const auto& im : imu_measurements) {
            imu_gyro_deg_s = std::max(imu_gyro_deg_s, im.gyro.norm() * (180.0 / M_PI));
        }
    }
    double eff_gyro_deg_s = std::max(imu_gyro_deg_s, vis_gyro_deg_s);

    // 4. 纯旋转平移门控 (Pure-Rotation Translation Gating):
    // 当检测到显著角速度 (eff_gyro_deg_s > 6.5 deg/s) 且消旋残差占总光流比例很低时，闭锁伪平移
    double trans_gate = 1.0;
    if (eff_gyro_deg_s > 6.5) {
        double rot_explain_ratio = derot_res_px / std::max(1.0, raw_flow_med);
        if (eff_gyro_deg_s > 12.0 || rot_explain_ratio < 0.42) {
            trans_gate = 0.0;
        } else {
            trans_gate = std::max(0.0, std::min(1.0, (12.0 - eff_gyro_deg_s) / 5.5));
        }
    } else if (raw_flow_med < 0.55 && derot_res_px < 0.45 && radial_med_px < 0.25) {
        // 亚像素静止死区门控
        trans_gate = 0.15;
    }

    if (!has_rot_prior_inout) {
        R_cw_prior_inout = SO3::normalize_rotation(R_rel_vis * R_last);
        has_rot_prior_inout = true;
    } else if (trans_gate < 0.35) {
        // 纯旋转期间融合视觉精细消旋角与 IMU 重力水平约束
        R_cw_prior_inout = SO3::normalize_rotation(R_rel_vis * R_last);
    }

    trans_gate_out = trans_gate;
    last_diag.trans_gate = trans_gate;
    last_diag.eff_gyro_deg_s = eff_gyro_deg_s;
    last_diag.raw_flow_px = raw_flow_med;
    last_diag.derot_res_px = derot_res_px;
    last_diag.radial_med_px = radial_med_px;
}

void Tracker::merge_flow_and_hfnet_features(
    const std::vector<Vec2>& flow_pts,
    const std::vector<std::shared_ptr<MapPoint>>& flow_mps,
    Frame& frame
) {
    if (flow_pts.empty()) return;
    int n_hf = frame.num_kps;

    for (size_t i = 0; i < flow_pts.size(); ++i) {
        const auto& mp = flow_mps[i];
        if (!mp || mp->is_bad || mp->descriptor.empty()) continue;
        const Vec2& pt = flow_pts[i];

        int best_j = -1;
        double best_d2 = 9.0; // 3.0 px 半径内吸附并修正为亚像素光流坐标
        for (int j = 0; j < n_hf; ++j) {
            if (frame.map_points[j] != nullptr) continue;
            double d2 = (frame.kp_pts[j] - pt).squared_norm();
            if (d2 < best_d2) {
                best_d2 = d2;
                best_j = j;
            }
        }
        if (best_j >= 0) {
            frame.kp_pts[best_j] = pt;
            frame.map_points[best_j] = mp;
        } else {
            frame.kp_pts.push_back(pt);
            frame.kp_octaves.push_back(0);
            frame.kp_scores.push_back(0.01f);
            frame.descriptors.push_back(mp->descriptor);
            frame.u_right.push_back(-1.0f);
            frame.depths.push_back(-1.0f);
            frame.map_points.push_back(mp);
            frame.outliers.push_back(false);
        }
    }
    frame.num_kps = (int)frame.kp_pts.size();
    frame.assign_features_to_grid();
}

void Tracker::match_from_last_frame() {
    if (!has_last_frame) return;
    std::set<MapPoint*> already;
    for (const auto& mp : curr_frame.map_points) {
        if (mp) already.insert(mp.get());
    }

    std::vector<std::shared_ptr<MapPoint>> cands;
    for (const auto& mp : last_frame.map_points) {
        if (!mp || mp->is_bad || already.count(mp.get())) continue;
        Vec3 p_c = curr_frame.T_cw.transform(mp->pos_w);
        if (p_c.z <= 0.15) continue;
        Vec2 uv = camera.project(p_c);
        if (camera.is_in_image(uv.x, uv.y, 4.0)) {
            mp->track_proj_x = uv.x;
            mp->track_proj_y = uv.y;
            mp->track_in_view = true;
            cands.push_back(mp);
        }
    }
    if (!cands.empty()) {
        FrontendMatcher::search_by_projection(curr_frame, cands, 10.0);
    }
}

int Tracker::track_local_map(const Mat3* R_cw_prior, double rot_prior_weight) {
    update_local_map();
    search_local_points();

    int n_inliers = Optimizer::pose_optimization(curr_frame, 6, R_cw_prior, rot_prior_weight);
    int final_good = 0;
    for (int i = 0; i < curr_frame.num_kps; ++i) {
        auto mp = curr_frame.map_points[i];
        if (!mp) continue;
        if (curr_frame.outliers[i]) {
            if (mp->num_observations() == 0) {
                mp->set_bad_flag(slam_map);
            }
            curr_frame.map_points[i] = nullptr;
        } else {
            mp->increase_found();
            final_good++;
        }
    }
    return std::max(n_inliers, final_good);
}

void Tracker::update_local_map() {
    std::map<KeyFrame*, int> kf_counter;
    for (const auto& mp : curr_frame.map_points) {
        if (mp && !mp->is_bad) {
            for (const auto& kv : mp->observations) {
                if (kv.first && !kv.first->is_bad) {
                    kf_counter[kv.first]++;
                }
            }
        }
    }
    if (kf_counter.empty() && ref_kf && !ref_kf->is_bad) {
        kf_counter[ref_kf.get()] = 1;
    }

    std::set<KeyFrame*> local_kf_set;
    int max_votes = 0;
    KeyFrame* best_ref = ref_kf.get();
    for (const auto& kv : kf_counter) {
        local_kf_set.insert(kv.first);
        if (kv.second > max_votes) {
            max_votes = kv.second;
            best_ref = kv.first;
        }
    }
    for (KeyFrame* kf : std::vector<KeyFrame*>(local_kf_set.begin(), local_kf_set.end())) {
        if (local_kf_set.size() >= 20) break;
        for (KeyFrame* neigh : kf->get_best_covisibility_keyframes(6)) {
            if (neigh && !neigh->is_bad) local_kf_set.insert(neigh);
        }
        if (kf->parent && !kf->parent->is_bad) local_kf_set.insert(kf->parent);
    }

    if (best_ref) {
        auto it = slam_map->keyframes.find(best_ref->id);
        if (it != slam_map->keyframes.end()) {
            ref_kf = it->second;
            curr_frame.ref_kf = ref_kf;
        }
    }

    local_keyframes.assign(local_kf_set.begin(), local_kf_set.end());
    std::set<MapPoint*> mp_set;
    local_mappoints.clear();
    for (KeyFrame* kf : local_keyframes) {
        for (const auto& mp : kf->map_points) {
            if (mp && !mp->is_bad && mp_set.insert(mp.get()).second) {
                local_mappoints.push_back(mp);
            }
        }
    }
    slam_map->reference_mappoints = local_mappoints;
}

void Tracker::search_local_points() {
    int fid = curr_frame.id;
    for (const auto& mp : curr_frame.map_points) {
        if (mp && !mp->is_bad) {
            mp->increase_visible();
            mp->last_frame_seen = fid;
        }
    }

    Vec3 o_w = curr_frame.get_camera_center();
    std::vector<std::shared_ptr<MapPoint>> to_project;
    for (const auto& mp : local_mappoints) {
        if (!mp || mp->is_bad || mp->last_frame_seen == fid) continue;
        mp->track_in_view = false;
        Vec3 p_c = curr_frame.T_cw.transform(mp->pos_w);
        if (p_c.z <= 0.15) continue;
        Vec2 uv = camera.project(p_c);
        if (!camera.is_in_image(uv.x, uv.y, 4.0)) continue;
        Vec3 vec_po = mp->pos_w - o_w;
        double dist = vec_po.norm();
        if (dist < 0.7 * mp->min_distance || dist > 1.3 * mp->max_distance) continue;

        mp->track_in_view = true;
        mp->track_proj_x = uv.x;
        mp->track_proj_y = uv.y;
        mp->increase_visible();
        to_project.push_back(mp);
    }
    if (!to_project.empty()) {
        FrontendMatcher::search_by_projection(curr_frame, to_project, 7.0);
    }
}

int Tracker::replenish_live_flow_features(
    const std::vector<uint8_t>& gray_l,
    const float* scores_dense,
    const int8_t* desc_raw,
    bool desc_is_nchw,
    int n_inliers
) {
    if (!ref_kf) return n_inliers;
    int w = camera.width;
    int h = camera.height;
    int cell_counts[6][8];
    std::memset(cell_counts, 0, sizeof(cell_counts));
    int occ_w = w / 12 + 2;
    int occ_h = h / 12 + 2;
    std::vector<bool> occ_grid(occ_w * occ_h, false);

    std::vector<Vec2> surviving_pts;
    std::vector<std::shared_ptr<MapPoint>> surviving_mps;
    std::vector<std::vector<float>> surviving_descs;
    std::vector<double> valid_zs;

    float cell_w = (float)w / 8.0f;
    float cell_h = (float)h / 6.0f;

    for (int i = 0; i < curr_frame.num_kps; ++i) {
        auto mp = curr_frame.map_points[i];
        if (mp && !mp->is_bad && !curr_frame.outliers[i]) {
            Vec2 pt = curr_frame.kp_pts[i];
            surviving_pts.push_back(pt);
            surviving_mps.push_back(mp);
            surviving_descs.push_back(mp->descriptor);
            int cr = std::min(5, std::max(0, (int)(pt.y / cell_h)));
            int cc = std::min(7, std::max(0, (int)(pt.x / cell_w)));
            cell_counts[cr][cc]++;
            int gy = (int)pt.y / 12, gx = (int)pt.x / 12;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int ny = gy + dy, nx = gx + dx;
                    if (ny >= 0 && ny < occ_h && nx >= 0 && nx < occ_w) {
                        occ_grid[ny * occ_w + nx] = true;
                    }
                }
            }
            double zc = curr_frame.T_cw.transform(mp->pos_w).z;
            if (zc > 0.25 && zc < 6.0) valid_zs.push_back(zc);
        }
    }

    int need_total = 230 - (int)surviving_pts.size();
    if (need_total <= 8) return (int)surviving_pts.size();

    std::vector<Vec2> added_pts;
    bool* occ_raw = new bool[occ_w * occ_h];
    for (int k = 0; k < occ_w * occ_h; ++k) occ_raw[k] = occ_grid[k];
    FrontendMatcher::detect_grid_corners_with_scores(
        gray_l.data(), scores_dense, w, h, occ_raw, occ_w, occ_h,
        cell_counts, 5, need_total, added_pts
    );
    delete[] occ_raw;

    if (added_pts.empty()) return (int)surviving_pts.size();

    double z_med = 1.20;
    if (valid_zs.size() >= 6) {
        std::sort(valid_zs.begin(), valid_zs.end());
        z_med = std::min(2.20, std::max(0.70, valid_zs[valid_zs.size() / 2]));
    }

    SE3 T_wc = curr_frame.get_pose_inverse();
    const int H_desc = 60, W_desc = 80, C_desc = 256;
    float factor_x = (float)(w - 1) / (float)(W_desc - 1);
    float factor_y = (float)(h - 1) / (float)(H_desc - 1);

    for (const auto& pt : added_pts) {
        Vec3 p_c = camera.unproject(pt.x, pt.y, z_med);
        Vec3 pos_w = T_wc.transform(p_c);
        std::vector<float> desc;
        if (desc_raw != nullptr) {
            desc = sample_descriptor_int8(desc_raw, (float)pt.x / factor_x, (float)pt.y / factor_y,
                                          H_desc, W_desc, C_desc, desc_is_nchw);
        }
        auto mp = std::make_shared<MapPoint>(pos_w, ref_kf, curr_frame.id, desc);
        mp->is_temp_flow_point = true;
        mp->increase_visible();
        mp->increase_found();
        slam_map->add_mappoint(mp);
        surviving_pts.push_back(pt);
        surviving_mps.push_back(mp);
        surviving_descs.push_back(desc);
    }

    int n_tot = (int)surviving_pts.size();
    curr_frame.kp_pts = std::move(surviving_pts);
    curr_frame.num_kps = n_tot;
    curr_frame.kp_octaves.assign(n_tot, 0);
    curr_frame.kp_scores.assign(n_tot, 0.01f);
    curr_frame.descriptors = std::move(surviving_descs);
    curr_frame.u_right.assign(n_tot, -1.0f);
    curr_frame.depths.assign(n_tot, -1.0f);
    curr_frame.map_points = std::move(surviving_mps);
    curr_frame.outliers.assign(n_tot, false);
    curr_frame.assign_features_to_grid();
    return n_tot;
}

bool Tracker::need_new_keyframe(int n_tracked_inliers, bool is_flow_only) const {
    int frames_since_last_kf = curr_frame.id - last_kf_frame_id;
    if (frames_since_last_kf < 5) return false;

    int min_inliers_live = is_flow_only ? 22 : 35;
    if (n_tracked_inliers < min_inliers_live) return true;

    if (ref_kf && frames_since_last_kf >= 18) {
        double trans_gate = last_diag.trans_gate;
        double dp_ref = (curr_frame.get_camera_center() - ref_kf->get_camera_center()).norm();
        Mat3 dR_ref = curr_frame.T_cw.R * ref_kf->T_cw.R.transpose();
        double rot_deg = std::acos(std::min(1.0, std::max(-1.0, 0.5 * (dR_ref.trace() - 1.0)))) * (180.0 / M_PI);

        if (dp_ref >= 0.045) return true;
        if (trans_gate >= 0.35 && dp_ref >= 0.020 && rot_deg >= 10.0) return true;
        if (rot_deg >= 28.0 && frames_since_last_kf >= 35 && n_tracked_inliers < 85) return true;
    }
    return false;
}

void Tracker::create_new_keyframe() {
    auto new_kf = std::make_shared<KeyFrame>(curr_frame);
    ref_kf = new_kf;
    curr_frame.ref_kf = new_kf;
    last_kf_frame_id = curr_frame.id;
    if (local_mapper) {
        local_mapper->process_new_keyframe(new_kf);
    }
    curr_frame.set_pose(new_kf->T_cw);
    for (int i = 0; i < curr_frame.num_kps; ++i) {
        if (new_kf->map_points[i] && !new_kf->map_points[i]->is_bad) {
            curr_frame.map_points[i] = new_kf->map_points[i];
        }
    }
}

SE3 Tracker::grab_image_monocular_inertial(
    const std::vector<uint8_t>& gray_l,
    const float* scores_dense,
    const int8_t* desc_raw,
    bool desc_is_nchw,
    const std::vector<Keypoint>& hfnet_kps,
    double timestamp,
    const std::vector<ImuMeasurement>& imu_measurements
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    if (state == TrackingState::NO_IMAGES_YET || state == TrackingState::NOT_INITIALIZED) {
        state = TrackingState::NOT_INITIALIZED;
        monocular_initialization(gray_l, hfnet_kps, timestamp, imu_measurements);
        auto t_end = std::chrono::high_resolution_clock::now();
        last_is_keyframe = (state == TrackingState::OK);
        last_used_optical_flow = false;
        last_inliers = 0;
        last_tracked_curr_pts.clear();
        last_tracked_prev_pts.clear();
        for (int i = 0; i < curr_frame.num_kps; ++i) {
            if (curr_frame.map_points[i] && !curr_frame.map_points[i]->is_bad) {
                last_inliers++;
                last_tracked_curr_pts.push_back(curr_frame.kp_pts[i]);
                last_tracked_prev_pts.push_back(curr_frame.kp_pts[i]);
            }
        }
        double tot_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        last_diag.frontend_ms = tot_ms;
        last_diag.backend_ms = 0.0;
        last_diag.total_ms = tot_ms;
        return curr_frame.T_cw;
    }

    auto t_fe_start = std::chrono::high_resolution_clock::now();
    bool need_kf_or_retrack = false;
    int n_inliers = 0;

    // 预构建当前帧结构体 (预先用于计算预测位姿时间戳)
    curr_frame.timestamp = timestamp;
    SE3 T_cw_pred_init = compute_predicted_pose(imu_measurements);
    bool has_imu_prior = (!imu_measurements.empty() && imu_preintegrator.dt_total > 0.0);
    Mat3 R_cw_imu_prior = has_imu_prior ? T_cw_pred_init.R : Mat3::identity();
    double w_rot_imu = 2.0e6;
    double trans_gate = 1.0;

    bool has_flow_prior_pose = false;
    SE3 T_cw_flow_prior = T_cw_pred_init;

    std::vector<Vec2> flow_pts;
    std::vector<std::shared_ptr<MapPoint>> flow_mps;
    std::vector<Vec2> flow_prev_pts;
    std::vector<std::vector<uint8_t>> curr_pyr;
    std::vector<int> curr_pyr_w, curr_pyr_h;

    if (has_last_frame) {
        FrontendMatcher::track_optical_flow(
            last_frame, gray_l, curr_pyr, curr_pyr_w, curr_pyr_h,
            camera, &T_cw_pred_init, flow_pts, flow_mps, flow_prev_pts
        );

        estimate_rotation_and_trans_gate(
            flow_pts, flow_mps, imu_measurements, has_imu_prior, R_cw_imu_prior, trans_gate
        );

        if (has_imu_prior) {
            Vec3 p_pred = T_cw_pred_init.camera_center();
            T_cw_pred_init = SE3(R_cw_imu_prior, -(R_cw_imu_prior * p_pred));
        }

        if ((int)flow_pts.size() >= 18) {
            std::vector<Keypoint> flow_kps(flow_pts.size());
            for (size_t i = 0; i < flow_pts.size(); ++i) {
                flow_kps[i].x = (float)flow_pts[i].x;
                flow_kps[i].y = (float)flow_pts[i].y;
                flow_kps[i].score = 0.02f;
                if (flow_mps[i]) flow_kps[i].descriptor = flow_mps[i]->descriptor;
            }
            curr_frame = Frame(timestamp, camera, gray_l, flow_kps);
            curr_frame.pyramid_left = curr_pyr;
            curr_frame.pyr_widths = curr_pyr_w;
            curr_frame.pyr_heights = curr_pyr_h;
            curr_frame.map_points = flow_mps;
            curr_frame.ref_kf = ref_kf;
            curr_frame.has_imu_rot_prior = has_imu_prior;
            curr_frame.R_cw_imu_prior = R_cw_imu_prior;
            curr_frame.set_pose(T_cw_pred_init);

            n_inliers = Optimizer::pose_optimization(
                curr_frame, 6, has_imu_prior ? &R_cw_imu_prior : nullptr, w_rot_imu
            );

            for (int i = 0; i < curr_frame.num_kps; ++i) {
                auto mp = curr_frame.map_points[i];
                if (!mp) continue;
                if (curr_frame.outliers[i]) {
                    if (mp->num_observations() == 0) mp->set_bad_flag(slam_map);
                    curr_frame.map_points[i] = nullptr;
                } else {
                    mp->increase_visible();
                    mp->increase_found();
                }
            }

            if (n_inliers >= 18) {
                Vec3 p_flow_opt = curr_frame.get_camera_center();
                Mat3 R_flow_opt = has_imu_prior ? R_cw_imu_prior : curr_frame.T_cw.R;
                T_cw_flow_prior = SE3(R_flow_opt, -(R_flow_opt * p_flow_opt));
                has_flow_prior_pose = true;
            }

            int frames_since_kf = curr_frame.id - last_kf_frame_id;
            if (frames_since_kf <= 3 && n_inliers >= 18) {
                need_kf_or_retrack = false;
            } else if (n_inliers >= 45 && !need_new_keyframe(n_inliers, true)) {
                need_kf_or_retrack = false;
            } else if (need_new_keyframe(n_inliers, true) || n_inliers < 22) {
                need_kf_or_retrack = true;
            }
        } else {
            need_kf_or_retrack = true;
        }
    } else {
        need_kf_or_retrack = true;
    }

    if (need_kf_or_retrack) {
        curr_frame = Frame(timestamp, camera, gray_l, hfnet_kps);
        if (curr_pyr.empty()) {
            FrontendMatcher::build_pyramid(gray_l.data(), camera.width, camera.height, 4,
                                           curr_pyr, curr_pyr_w, curr_pyr_h);
        }
        curr_frame.pyramid_left = curr_pyr;
        curr_frame.pyr_widths = curr_pyr_w;
        curr_frame.pyr_heights = curr_pyr_h;

        merge_flow_and_hfnet_features(flow_pts, flow_mps, curr_frame);
        curr_frame.ref_kf = ref_kf;
        curr_frame.has_imu_rot_prior = has_imu_prior;
        curr_frame.R_cw_imu_prior = R_cw_imu_prior;
        curr_frame.set_pose(has_flow_prior_pose ? T_cw_flow_prior : T_cw_pred_init);

        match_from_last_frame();
        int pre_matches = 0;
        for (const auto& mp : curr_frame.map_points) if (mp && !mp->is_bad) pre_matches++;
        if (pre_matches >= 6) {
            Optimizer::pose_optimization(
                curr_frame, 6, has_imu_prior ? &R_cw_imu_prior : nullptr, w_rot_imu
            );
        }
        n_inliers = track_local_map(has_imu_prior ? &R_cw_imu_prior : nullptr, w_rot_imu);
    }

    // 连续位姿锚定、纯旋转平移门控与光流比例步长保护 (1:1 对齐 Python tracker.py)
    if (has_last_frame && last_frame.has_pose) {
        Vec3 p_last_ref = last_frame.get_camera_center();
        Mat3 R_safe = has_imu_prior ? R_cw_imu_prior : curr_frame.T_cw.R;
        Vec3 p_curr_raw = curr_frame.get_camera_center();
        if (has_flow_prior_pose) {
            Vec3 p_flow = T_cw_flow_prior.camera_center();
            Vec3 diff_ba = p_curr_raw - p_flow;
            double diff_norm = diff_ba.norm();
            if (diff_norm > 0.0050) {
                p_curr_raw = p_flow + diff_ba * (0.0050 / diff_norm);
            }
        }

        double eff_trans_px = std::max({last_diag.raw_flow_px, last_diag.radial_med_px * 3.2, last_diag.derot_res_px * 1.8});
        double max_step_m = std::min(0.0150, std::max(0.0004, eff_trans_px * 0.0025)) * trans_gate;
        if (n_inliers < 18) max_step_m = std::min(max_step_m, 0.0030);

        Vec3 dp_vec = (p_curr_raw - p_last_ref) * trans_gate;
        double dp_norm = dp_vec.norm();
        Vec3 p_safe;
        if (n_inliers < 12 || trans_gate <= 1e-4) {
            p_safe = p_last_ref;
        } else if (dp_norm > max_step_m) {
            p_safe = p_last_ref + dp_vec * (max_step_m / dp_norm);
        } else {
            p_safe = p_last_ref + dp_vec;
        }
        curr_frame.set_pose(SE3(R_safe, -(R_safe * p_safe)));

        if (!need_kf_or_retrack && n_inliers < 210) {
            n_inliers = replenish_live_flow_features(gray_l, scores_dense, desc_raw, desc_is_nchw, n_inliers);
        }
    }

    auto t_fe_end = std::chrono::high_resolution_clock::now();
    auto t_be_start = std::chrono::high_resolution_clock::now();

    bool created_kf = false;
    if (need_kf_or_retrack && need_new_keyframe(n_inliers, false)) {
        Vec3 p_before_kf = curr_frame.get_camera_center();
        Mat3 R_before_kf = curr_frame.T_cw.R;
        create_new_keyframe();
        created_kf = true;

        Vec3 p_after_kf = curr_frame.get_camera_center();
        Vec3 dp_kf = p_after_kf - p_before_kf;
        double dp_kf_norm = dp_kf.norm();
        double max_kf_dp = 0.0030 * trans_gate;
        Vec3 p_clamped = (dp_kf_norm > max_kf_dp)
            ? ((max_kf_dp <= 1e-6) ? p_before_kf : (p_before_kf + dp_kf * (max_kf_dp / dp_kf_norm)))
            : p_after_kf;
        curr_frame.set_pose(SE3(R_before_kf, -(R_before_kf * p_clamped)));
        if (ref_kf) ref_kf->set_pose(curr_frame.T_cw);
    } else if (need_kf_or_retrack && n_inliers < 210) {
        n_inliers = replenish_live_flow_features(gray_l, scores_dense, desc_raw, desc_is_nchw, n_inliers);
    }

    auto t_be_end = std::chrono::high_resolution_clock::now();

    // 更新阻尼恒速模型线速度与角速度
    if (curr_frame.has_pose && has_last_frame && last_frame.has_pose) {
        double dt_frame = std::max(1e-3, curr_frame.timestamp - last_frame.timestamp);
        Vec3 p_curr = curr_frame.get_camera_center();
        Vec3 p_last = last_frame.get_camera_center();
        Vec3 dp_frame = p_curr - p_last;
        double dp_len = dp_frame.norm();
        if (dp_len > 0.015) dp_frame *= (0.015 / dp_len);
        double vel_scale = 0.50 * trans_gate;
        Vec3 dp_damped = dp_frame * vel_scale;
        Mat3 R_rel = curr_frame.T_cw.R * last_frame.T_cw.R.transpose();
        Vec3 t_vel = -(curr_frame.T_cw.R * (p_curr - R_rel.transpose() * p_last)) * vel_scale;
        velocity = SE3(R_rel, t_vel);
        curr_frame.v_w = dp_damped / dt_frame;
        has_velocity = true;
    }

    // 动态回收滑出视野的临时光流点 (防止地图点超过 1100 个膨胀)
    if (has_last_frame) {
        std::set<int> curr_mp_ids;
        for (const auto& mp : curr_frame.map_points) {
            if (mp) curr_mp_ids.insert(mp->id);
        }
        for (const auto& mp_old : last_frame.map_points) {
            if (mp_old && !mp_old->is_bad && curr_mp_ids.find(mp_old->id) == curr_mp_ids.end()) {
                int n_obs = mp_old->num_observations();
                if (n_obs == 0 || (mp_old->is_temp_flow_point && n_obs <= 1)) {
                    mp_old->set_bad_flag(slam_map);
                }
            }
        }
        if (slam_map->num_mappoints() > 1100) {
            auto all_mps = slam_map->get_all_mappoints();
            for (const auto& mp_cand : all_mps) {
                if (slam_map->num_mappoints() <= 950) break;
                if (curr_mp_ids.find(mp_cand->id) == curr_mp_ids.end() &&
                    mp_cand->is_temp_flow_point && mp_cand->num_observations() <= 1) {
                    mp_cand->set_bad_flag(slam_map);
                }
            }
        }
    }

    // 整理当前帧有效内点用于 NV12 HUD 可视化
    last_tracked_curr_pts.clear();
    last_tracked_prev_pts.clear();
    for (size_t i = 0; i < flow_pts.size(); ++i) {
        if (flow_mps[i] && !flow_mps[i]->is_bad) {
            last_tracked_curr_pts.push_back(flow_pts[i]);
            last_tracked_prev_pts.push_back(flow_prev_pts[i]);
        }
    }

    last_is_keyframe = created_kf;
    last_used_optical_flow = !need_kf_or_retrack;
    last_inliers = n_inliers;
    last_frame = curr_frame;
    has_last_frame = true;

    last_diag.frontend_ms = std::chrono::duration<double, std::milli>(t_fe_end - t_fe_start).count();
    last_diag.backend_ms  = std::chrono::duration<double, std::milli>(t_be_end - t_be_start).count();
    last_diag.total_ms    = std::chrono::duration<double, std::milli>(t_be_end - t_start).count();

    return curr_frame.T_cw;
}

} // namespace slam
