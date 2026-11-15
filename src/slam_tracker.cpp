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
            // 左相机 (X右, Y下, Z前) 到水平机身系 Body (X右, Y前, Z上) 外参旋转矩阵 R_bc = [[1,0,0],[0,0,1],[0,-1,0]]
            Mat3 R_bc;
            R_bc(0, 0) = 1.0; R_bc(0, 1) =  0.0; R_bc(0, 2) = 0.0;
            R_bc(1, 0) = 0.0; R_bc(1, 1) =  0.0; R_bc(1, 2) = 1.0;
            R_bc(2, 0) = 0.0; R_bc(2, 1) = -1.0; R_bc(2, 2) = 0.0;
            Mat3 dR_cam = R_bc.transpose() * imu_preintegrator.delta_R * R_bc;
            Mat3 R_cw_pred = SO3::normalize_rotation(dR_cam.transpose() * last_frame.T_cw.R);

            // 实机单目+IMU (640x480) 闭环重力方向约束 (VIO Roll/Pitch Observability Constraint, 1:1 对齐 tracker.py):
            // 在重力场中 Yaw 绕竖轴不可观 (完全由陀螺仪积分驱动)，而 Roll 与 Pitch 100% 受重力向量约束；
            // 利用加速度计实测重力方向实时锚定 Roll/Pitch，彻底消除陀螺仪开环积分倾角漂移
            Vec3 acc_mean(0.0, 0.0, 0.0);
            Vec3 gyro_mean(0.0, 0.0, 0.0);
            for (const auto& m : imu_measurements) {
                acc_mean += m.acc;
                gyro_mean += m.gyro;
            }
            double inv_n = 1.0 / (double)imu_measurements.size();
            acc_mean *= inv_n;
            gyro_mean *= inv_n;
            double acc_norm = acc_mean.norm();

            if (acc_norm > 8.2 && acc_norm < 11.5 && acc_mean.z > 5.0) {
                Mat3 R_wc_gyro = R_cw_pred.transpose();
                Mat3 R_wb_gyro = R_bc * R_wc_gyro * R_bc.transpose();
                double sy = std::min(1.0, std::max(-1.0, -R_wb_gyro(2, 0)));
                double pitch_gyro = std::asin(sy);
                double roll_gyro  = std::atan2(R_wb_gyro(2, 1), R_wb_gyro(2, 2));
                double yaw_gyro   = std::atan2(R_wb_gyro(1, 0), R_wb_gyro(0, 0));

                double roll_acc  = std::atan2(acc_mean.y, acc_mean.z);
                double pitch_acc = std::atan2(-acc_mean.x, std::hypot(acc_mean.y, acc_mean.z));
                double tilt_gyro = std::hypot(gyro_mean.x, gyro_mean.y);
                const double rad2deg = 180.0 / M_PI;

                double roll_target, pitch_target, alpha_g;
                if (tilt_gyro < 0.30 && (
                    (std::abs(roll_acc * rad2deg) < 6.5 && std::abs(pitch_acc * rad2deg) < 6.5) ||
                    (std::abs(gyro_mean.z) > 0.30 && std::abs(roll_gyro * rad2deg) < 5.0 && std::abs(pitch_gyro * rad2deg) < 5.0)
                )) {
                    roll_target  = 0.08 * roll_acc;
                    pitch_target = 0.08 * pitch_acc;
                    alpha_g      = 0.50;
                } else {
                    roll_target  = roll_acc;
                    pitch_target = pitch_acc;
                    alpha_g      = 0.25;
                }

                double r_f = (1.0 - alpha_g) * roll_gyro + alpha_g * roll_target;
                double p_f = (1.0 - alpha_g) * pitch_gyro + alpha_g * pitch_target;
                double cr = std::cos(r_f), sr = std::sin(r_f);
                double cp = std::cos(p_f), sp = std::sin(p_f);
                double cy = std::cos(yaw_gyro), sy_y = std::sin(yaw_gyro);

                Mat3 Rx = Mat3::identity();
                Rx(1, 1) = cr; Rx(1, 2) = -sr;
                Rx(2, 1) = sr; Rx(2, 2) =  cr;

                Mat3 Ry = Mat3::identity();
                Ry(0, 0) =  cp; Ry(0, 2) = sp;
                Ry(2, 0) = -sp; Ry(2, 2) = cp;

                Mat3 Rz = Mat3::identity();
                Rz(0, 0) = cy;   Rz(0, 1) = -sy_y;
                Rz(1, 0) = sy_y; Rz(1, 1) =  cy;

                Mat3 R_wb_grav = Rz * Ry * Rx;
                Mat3 R_wc_grav = R_bc.transpose() * R_wb_grav * R_bc;
                R_cw_pred = SO3::normalize_rotation(R_wc_grav.transpose());
            }

            Vec3 p_last = last_frame.get_camera_center();
            Vec3 p_w_pred = has_velocity ? (p_last + last_frame.v_w * imu_preintegrator.dt_total) : p_last;
            return SE3(R_cw_pred, -(R_cw_pred * p_w_pred));
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

    SE3 T_cw_init_pred = (has_imu && has_last_frame)
        ? compute_predicted_pose(imu_measurements)
        : SE3::identity();
    curr_frame.set_pose(T_cw_init_pred);
    if (has_imu) {
        curr_frame.has_imu_rot_prior = true;
        curr_frame.R_cw_imu_prior = T_cw_init_pred.R;
    }

    if (!has_init_frame || init_frame.num_kps < 60) {
        init_frame = curr_frame;
        has_init_frame = (curr_frame.num_kps >= 60);
        last_frame = curr_frame;
        has_last_frame = true;
        init_imu_preintegrator.reset_state();
        return;
    }

    if (has_imu && has_last_frame) {
        init_imu_preintegrator.integrate_batch(imu_measurements, last_frame.timestamp, timestamp);
    }

    // 1:1 对齐 Python tracker.py: 使用双向 4 层金字塔 LK 光流跟踪参考帧 F0 -> 当前帧 Fk (fb_err <= 1.5px)
    std::vector<int> matched_idx0;
    std::vector<Vec2> matched_pts0;
    std::vector<Vec2> matched_pts1;
    FrontendMatcher::track_init_optical_flow(
        init_frame, curr_frame.pyramid_left, curr_frame.pyr_widths, curr_frame.pyr_heights,
        camera, matched_idx0, matched_pts0, matched_pts1
    );

    if ((int)matched_pts0.size() < 45) {
        init_frame = curr_frame;
        last_frame = curr_frame;
        init_imu_preintegrator.reset_state();
        return;
    }

    std::vector<double> parallax_list(matched_pts0.size());
    for (size_t i = 0; i < matched_pts0.size(); ++i) {
        parallax_list[i] = (matched_pts1[i] - matched_pts0[i]).norm();
    }
    std::sort(parallax_list.begin(), parallax_list.end());
    double med_parallax = parallax_list[parallax_list.size() / 2];

    if ((curr_frame.id - init_frame.id) < 2) {
        last_frame = curr_frame;
        return;
    }
    if (!has_imu && med_parallax < 6.0 && (curr_frame.id - init_frame.id) < 3) {
        last_frame = curr_frame;
        return;
    }

    // 实机桌面单目+IMU模式 (640x480): 利用 IMU 陀螺仪去旋转光流 + 室内物理渐变深度先验完成平滑冷启动
    Mat3 R_10 = T_cw_init_pred.R * init_frame.T_cw.R.transpose();
    double fx = camera.fx, fy = camera.fy, cx = camera.cx, cy = camera.cy;
    size_t n_m = matched_idx0.size();
    std::vector<Vec3> x0_n(n_m);
    std::vector<double> derot_dx(n_m), derot_dy(n_m);
    for (size_t k = 0; k < n_m; ++k) {
        x0_n[k] = Vec3((matched_pts0[k].x - cx) * camera.inv_fx,
                       (matched_pts0[k].y - cy) * camera.inv_fy, 1.0);
        Vec3 x1_k((matched_pts1[k].x - cx) * camera.inv_fx,
                  (matched_pts1[k].y - cy) * camera.inv_fy, 1.0);
        Vec3 x0_rot = R_10 * x0_n[k];
        derot_dx[k] = x1_k.x - x0_rot.x;
        derot_dy[k] = x1_k.y - x0_rot.y;
    }
    std::sort(derot_dx.begin(), derot_dx.end());
    std::sort(derot_dy.begin(), derot_dy.end());
    const double z_prior = 1.20;
    double tx_est = derot_dx[n_m / 2] * z_prior;
    double ty_est = derot_dy[n_m / 2] * z_prior;
    Vec3 t_raw(tx_est, ty_est, 0.0);
    double t_norm = t_raw.norm();
    if (t_norm > 0.03) {
        t_raw *= (0.03 / t_norm);
    }
    Vec3 t_10_scaled = t_raw;

    // 将光流跟踪点合并至当前帧 F1 的特征列表 (1:1 对齐 Python lines 896-945)
    int base_n = curr_frame.num_kps;
    for (size_t k = 0; k < n_m; ++k) {
        int idx0 = matched_idx0[k];
        curr_frame.kp_pts.push_back(matched_pts1[k]);
        curr_frame.kp_octaves.push_back(0);
        curr_frame.kp_scores.push_back(init_frame.kp_scores[idx0]);
        curr_frame.descriptors.push_back(init_frame.descriptors[idx0]);
        curr_frame.u_right.push_back(-1.0f);
        curr_frame.depths.push_back(-1.0f);
        curr_frame.map_points.push_back(nullptr);
        curr_frame.outliers.push_back(false);
    }
    curr_frame.num_kps = (int)curr_frame.kp_pts.size();
    curr_frame.assign_features_to_grid();

    init_frame.set_pose(SE3::identity());
    curr_frame.set_pose(SE3(R_10, t_10_scaled));
    if (has_imu) {
        curr_frame.has_imu_rot_prior = true;
        curr_frame.R_cw_imu_prior = R_10;
        init_frame.has_imu_rot_prior = true;
        init_frame.R_cw_imu_prior = Mat3::identity();
    }

    auto kf0 = std::make_shared<KeyFrame>(init_frame);
    auto kf1 = std::make_shared<KeyFrame>(curr_frame);
    slam_map->add_keyframe(kf0);
    slam_map->add_keyframe(kf1);

    for (size_t k = 0; k < n_m; ++k) {
        int idx0 = matched_idx0[k];
        int idx1 = base_n + (int)k;
        double z0 = z_prior * (0.85 + 0.30 * (1.0 - matched_pts0[k].y / (double)camera.height));
        Vec3 pos_w = x0_n[k] * z0;
        if (!has_imu && med_parallax >= 8.0) {
            Vec3 xn1 = camera.unproject(matched_pts0[k].x, matched_pts0[k].y, 1.0);
            Vec3 xn2 = camera.unproject(matched_pts1[k].x, matched_pts1[k].y, 1.0);
            Vec3 tri_w;
            if (triangulate_two_views(kf0->T_cw, kf1->T_cw, xn1, xn2, tri_w)) {
                pos_w = tri_w;
            }
        }

        auto mp = std::make_shared<MapPoint>(pos_w, kf0, init_frame.id, init_frame.descriptors[idx0]);
        mp->add_observation(kf0.get(), idx0);
        mp->add_observation(kf1.get(), idx1);
        kf0->add_mappoint(mp, idx0);
        kf1->add_mappoint(mp, idx1);
        curr_frame.map_points[idx1] = mp;
        mp->compute_distinctive_descriptors();
        mp->update_normal_and_depth();
        slam_map->add_mappoint(mp);
    }

    kf0->update_connections();
    kf1->update_connections();

    // 初始化完成后立即执行 3 轮 Local BA 并同步位姿与阻尼初速度 (1:1 对齐 Python lines 948-963)
    Optimizer::local_bundle_adjustment(kf1, *slam_map, 3);
    curr_frame.set_pose(kf1->T_cw);

    ref_kf = kf1;
    curr_frame.ref_kf = kf1;
    last_frame = curr_frame;
    has_last_frame = true;
    last_kf_frame_id = curr_frame.id;
    local_keyframes = {kf0.get(), kf1.get()};
    local_mappoints = slam_map->get_all_mappoints();

    double n_init_span = (double)std::max(2, curr_frame.id - init_frame.id);
    SE3 T_delta_init = kf1->T_cw * kf0->T_cw.inverse();
    double damp_factor = 0.25 / n_init_span;
    velocity = SE3(T_delta_init.R, T_delta_init.t * damp_factor);
    has_velocity = true;
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
    if (!has_last_frame || !last_frame.has_pose) {
        trans_gate_out = 1.0;
        return;
    }

    // 1. 维护 3 帧陀螺仪三维向量滑动均值 (1:1 对齐 Python lines 1065-1075: 仅信任真实陀螺仪角速度，绝不混入伪视觉角速度)
    Vec3 curr_gyro_vec(0.0, 0.0, 0.0);
    if (!imu_measurements.empty()) {
        for (const auto& m : imu_measurements) curr_gyro_vec += m.gyro;
        curr_gyro_vec *= (1.0 / (double)imu_measurements.size());
    }
    gyro_vec_hist_.push_back(curr_gyro_vec);
    if (gyro_vec_hist_.size() > 3) {
        gyro_vec_hist_.erase(gyro_vec_hist_.begin());
    }
    Vec3 avg_gyro_vec(0.0, 0.0, 0.0);
    for (const auto& g : gyro_vec_hist_) avg_gyro_vec += g;
    avg_gyro_vec *= (1.0 / (double)gyro_vec_hist_.size());

    const double rad2deg = 180.0 / M_PI;
    double avg_gyro_deg_s  = avg_gyro_vec.norm() * rad2deg;
    double curr_gyro_deg_s = curr_gyro_vec.norm() * rad2deg;
    double eff_gyro_deg_s  = std::max(avg_gyro_deg_s, 0.70 * curr_gyro_deg_s);

    // 2. 收集上一帧与当前帧光流匹配点对 (1:1 对齐 Python lines 1077-1088)
    std::map<int, Vec2> last_mp_to_pt;
    for (int i_l = 0; i_l < last_frame.num_kps; ++i_l) {
        const auto& mp_l = last_frame.map_points[i_l];
        if (mp_l && !mp_l->is_bad && !last_frame.outliers[i_l]) {
            last_mp_to_pt[mp_l->id] = last_frame.kp_pts[i_l];
        }
    }

    std::vector<Vec2> pts0;
    std::vector<Vec2> pts1;
    pts0.reserve(flow_pts.size());
    pts1.reserve(flow_pts.size());
    for (size_t i = 0; i < flow_pts.size(); ++i) {
        const auto& mp1 = flow_mps[i];
        if (mp1) {
            auto it = last_mp_to_pt.find(mp1->id);
            if (it != last_mp_to_pt.end()) {
                pts0.push_back(it->second);
                pts1.push_back(flow_pts[i]);
            }
        }
    }

    double raw_flow_px = 0.0;
    double derot_res_px = 0.0;
    double radial_med_px = 0.0;

    int m = (int)pts0.size();
    if (m >= 12) {
        double fx = camera.fx, fy = camera.fy, cx = camera.cx, cy = camera.cy;
        std::vector<double> raw_flows(m);
        std::vector<Vec3> x0(m), x1(m), b0(m), b1(m);
        for (int i = 0; i < m; ++i) {
            raw_flows[i] = (pts1[i] - pts0[i]).norm();
            x0[i] = Vec3((pts0[i].x - cx) * camera.inv_fx, (pts0[i].y - cy) * camera.inv_fy, 1.0);
            x1[i] = Vec3((pts1[i].x - cx) * camera.inv_fx, (pts1[i].y - cy) * camera.inv_fy, 1.0);
            b0[i] = x0[i].normalized();
            b1[i] = x1[i].normalized();
        }
        std::vector<double> rf_sorted = raw_flows;
        std::sort(rf_sorted.begin(), rf_sorted.end());
        raw_flow_px = rf_sorted[m / 2];

        // 两轮 IRLS 加权求解视觉纯旋转 dR_vis (仅用于计算去旋转光流残差，绝不覆盖 R_cw_imu_prior)
        Mat3 dR_vis = Mat3::identity();
        std::vector<double> w_irls(m, 1.0);
        for (int pass = 0; pass < 2; ++pass) {
            for (int it = 0; it < 3; ++it) {
                Mat3 H_r = Mat3::diag(1e-5, 1e-5, 1e-5);
                Vec3 b_r(0, 0, 0);
                for (int i = 0; i < m; ++i) {
                    Vec3 d_rot = dR_vis * b0[i];
                    Vec3 err = d_rot.cross(b1[i]);
                    double w = w_irls[i];
                    for (int r = 0; r < 3; ++r) {
                        b_r[r] += w * err[r];
                        H_r(r, r) += w;
                    }
                }
                Vec3 dphi;
                if (solve_linear_3x3(H_r, b_r, dphi)) {
                    dR_vis = SO3::normalize_rotation(SO3::exp(dphi) * dR_vis);
                }
            }
            for (int i = 0; i < m; ++i) {
                Vec3 x0_rot = dR_vis * x0[i];
                double z_rot = std::max(1e-4, x0_rot.z);
                double u1_rot = fx * (x0_rot.x / z_rot) + cx;
                double v1_rot = fy * (x0_rot.y / z_rot) + cy;
                double res_i = std::hypot(pts1[i].x - u1_rot, pts1[i].y - v1_rot);
                w_irls[i] = 1.0 / std::max(1.0, res_i / 1.5);
            }
        }

        std::vector<double> res_px(m);
        std::vector<double> derot_du(m), derot_dv(m);
        for (int i = 0; i < m; ++i) {
            Vec3 x0_rot = dR_vis * x0[i];
            double z_rot = std::max(1e-4, x0_rot.z);
            double u1_rot = fx * (x0_rot.x / z_rot) + cx;
            double v1_rot = fy * (x0_rot.y / z_rot) + cy;
            derot_du[i] = pts1[i].x - u1_rot;
            derot_dv[i] = pts1[i].y - v1_rot;
            res_px[i] = std::hypot(derot_du[i], derot_dv[i]);
        }
        std::vector<double> res_sorted = res_px;
        std::sort(res_sorted.begin(), res_sorted.end());
        derot_res_px = res_sorted[m / 2];

        // 平稳平移阶段 (eff_gyro_deg_s < 6.0 deg/s 且 prev_gate >= 0.85) 结合 IMU 真实物理去旋转光流残差 (对齐 Python lines 1132-1140)
        double prev_gate_est = last_diag.trans_gate;
        if (has_rot_prior_inout && eff_gyro_deg_s < 6.0 && prev_gate_est >= 0.85) {
            Mat3 dR_imu = R_cw_prior_inout * last_frame.T_cw.R.transpose();
            std::vector<double> imu_res_list(m);
            for (int i = 0; i < m; ++i) {
                Vec3 x0_imu = dR_imu * x0[i];
                double z_imu = std::max(1e-4, x0_imu.z);
                double u1_imu = fx * (x0_imu.x / z_imu) + cx;
                double v1_imu = fy * (x0_imu.y / z_imu) + cy;
                imu_res_list[i] = std::hypot(pts1[i].x - u1_imu, pts1[i].y - v1_imu);
            }
            std::sort(imu_res_list.begin(), imu_res_list.end());
            double imu_res_px = imu_res_list[m / 2];
            derot_res_px = std::max(derot_res_px, std::min(imu_res_px, raw_flow_px));
        }

        // 计算去旋转后的径向膨胀/收缩量 radial_med_px (对齐 Python lines 1142-1153: r0_norm > 50.0 且点数 >= 8)
        std::vector<double> rad_proj;
        rad_proj.reserve(m);
        for (int i = 0; i < m; ++i) {
            double rx0 = pts0[i].x - cx;
            double ry0 = pts0[i].y - cy;
            double r0_norm = std::hypot(rx0, ry0);
            if (r0_norm > 50.0) {
                rad_proj.push_back((derot_du[i] * rx0 + derot_dv[i] * ry0) / r0_norm);
            }
        }
        if ((int)rad_proj.size() >= 8) {
            std::sort(rad_proj.begin(), rad_proj.end());
            radial_med_px = std::abs(rad_proj[rad_proj.size() / 2]);
        }
    }

    // 3. 门控判据 (1:1 对齐 Python lines 1159-1179):
    // - 静止死区 (raw_flow_px < 0.48px 且 eff_gyro_deg_s < 1.0): 锁定零平移与零姿态漂移
    // - 手持平移运动 (eff_gyro_deg_s < 9.5 deg/s): 完全放开平移 (trans_gate = 1.0)，绝不将横向平移误判为旋转
    // - 确认旋转 (eff_gyro_deg_s >= 9.5 deg/s): 仅当陀螺仪确有显著角速度时才启动旋转平移门控
    double max_rot_res = std::max({2.20, 0.14 * raw_flow_px, 0.11 * eff_gyro_deg_s});
    bool is_stationary = (
        raw_flow_px < 0.48 && derot_res_px < 0.25 && radial_med_px < 0.12 && eff_gyro_deg_s < 1.0
    );
    double prev_gate = last_diag.trans_gate;
    double trans_gate = 1.0;
    if (is_stationary) {
        trans_gate = 0.0;
    } else if (eff_gyro_deg_s < 9.5) {
        trans_gate = (prev_gate < 0.50 && eff_gyro_deg_s >= 5.5) ? 0.0 : 1.0;
    } else if (eff_gyro_deg_s < 16.0 && (
        derot_res_px > max_rot_res || (radial_med_px > 0.85 && radial_med_px > 0.22 * raw_flow_px)
    )) {
        trans_gate = 1.0;
    } else {
        trans_gate = 0.0;
    }

    // 静止死区保持上一帧姿态稳定；其余全部阶段严格信任 IMU 旋转先验 R_cw_prior_inout (严禁用 dR_vis 覆盖)
    if (is_stationary) {
        R_cw_prior_inout = last_frame.T_cw.R;
        has_rot_prior_inout = true;
    }

    trans_gate_out = trans_gate;
    last_diag.trans_gate = trans_gate;
    last_diag.eff_gyro_deg_s = eff_gyro_deg_s;
    last_diag.raw_flow_px = raw_flow_px;
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
        double best_d2 = 2.25; // 1.5 px 半径内吸附并修正为亚像素光流坐标 (1:1 对齐 Python d2 <= 2.25)
        for (int j = 0; j < n_hf; ++j) {
            if (frame.map_points[j] != nullptr) continue;
            double d2 = (frame.kp_pts[j] - pt).squared_norm();
            if (d2 <= best_d2) {
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
    Mat3 R_cw_imu_prior = has_imu_prior
        ? T_cw_pred_init.R
        : ((has_last_frame && last_frame.has_pose) ? last_frame.T_cw.R : Mat3::identity());
    bool has_rot_prior = has_imu_prior || (has_last_frame && last_frame.has_pose);
    double w_rot_imu = has_rot_prior ? 2.0e6 : 0.0;
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
            flow_pts, flow_mps, imu_measurements, has_rot_prior, R_cw_imu_prior, trans_gate
        );

        if (has_rot_prior) {
            Vec3 p_pred = T_cw_pred_init.camera_center();
            T_cw_pred_init = SE3(R_cw_imu_prior, -(R_cw_imu_prior * p_pred));
        }

        if ((int)flow_pts.size() >= 28) {
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
            curr_frame.has_imu_rot_prior = has_rot_prior;
            curr_frame.R_cw_imu_prior = R_cw_imu_prior;
            curr_frame.set_pose(T_cw_pred_init);

            n_inliers = Optimizer::pose_optimization(
                curr_frame, 6, has_rot_prior ? &R_cw_imu_prior : nullptr, w_rot_imu
            );

            // 实机 640x480 旋转阶段 (trans_gate < 0.85): 串口 IMU 与 USB 摄像头存在 10~30ms 包相位差，
            // 对已通过双向 LK 光流一致性校验 (fb_err <= 1.5px) 的跟踪点豁免因 IMU 瞬时相位差导致的卡方误剔除 (1:1 对齐 tracker.py lines 502-506)
            if (trans_gate < 0.85 && n_inliers < (int)(0.75 * flow_pts.size())) {
                std::fill(curr_frame.outliers.begin(), curr_frame.outliers.end(), false);
                n_inliers = 0;
                for (const auto& mp : curr_frame.map_points) {
                    if (mp && !mp->is_bad) n_inliers++;
                }
            }

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
                Mat3 R_flow_opt = has_rot_prior ? R_cw_imu_prior : curr_frame.T_cw.R;
                T_cw_flow_prior = SE3(R_flow_opt, -(R_flow_opt * p_flow_opt));
                has_flow_prior_pose = true;
                flow_mps = curr_frame.map_points;
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
        curr_frame.has_imu_rot_prior = has_rot_prior;
        curr_frame.R_cw_imu_prior = R_cw_imu_prior;
        curr_frame.set_pose(has_flow_prior_pose ? T_cw_flow_prior : T_cw_pred_init);

        match_from_last_frame();
        int pre_matches = 0;
        for (const auto& mp : curr_frame.map_points) if (mp && !mp->is_bad) pre_matches++;
        if (pre_matches >= 6) {
            Optimizer::pose_optimization(
                curr_frame, 6, has_rot_prior ? &R_cw_imu_prior : nullptr, w_rot_imu
            );
        }
        n_inliers = track_local_map(has_rot_prior ? &R_cw_imu_prior : nullptr, w_rot_imu);
    }

    // 连续位姿锚定、纯旋转平移门控与光流比例步长保护 (1:1 对齐 Python tracker.py)
    if (has_last_frame && last_frame.has_pose) {
        Vec3 p_last_ref = last_frame.get_camera_center();
        Mat3 R_safe = has_rot_prior ? R_cw_imu_prior : curr_frame.T_cw.R;
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
        int valid_after_kf = 0;
        for (const auto& mp : curr_frame.map_points) {
            if (mp && !mp->is_bad) valid_after_kf++;
        }
        n_inliers = std::max(n_inliers, valid_after_kf);
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
