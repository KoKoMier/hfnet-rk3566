#include "blackbox_logger.h"

#include <ctime>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _WIN32
#include <direct.h>
#define MKDIR_P(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define MKDIR_P(p) mkdir(p, 0755)
#endif

namespace slam {

static void make_dirs_recursive(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        char c = path[i];
        if (c == '/' || c == '\\') {
            if (!cur.empty()) MKDIR_P(cur.c_str());
        }
        cur.push_back(c);
    }
    if (!cur.empty()) MKDIR_P(cur.c_str());
}

static std::string current_iso_time(bool compact = false) {
    std::time_t t = std::time(nullptr);
    std::tm tm_val;
#ifdef _WIN32
    localtime_s(&tm_val, &t);
#else
    localtime_r(&t, &tm_val);
#endif
    char buf[64];
    if (compact) {
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_val);
    } else {
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_val);
    }
    return std::string(buf);
}

BlackboxLogger::BlackboxLogger(const std::string& base_dir, const std::string& mode)
    : base_dir_(base_dir), mode_(mode) {
    start_time_iso_ = current_iso_time(false);
    run_id_ = "run_" + current_iso_time(true) + "_" + mode_;
    run_dir_ = base_dir_ + "/" + run_id_;
    make_dirs_recursive(run_dir_);

    csv_path_ = run_dir_ + "/frames.csv";
    events_path_ = run_dir_ + "/events.jsonl";
    summary_path_ = run_dir_ + "/summary.json";
    report_path_ = run_dir_ + "/report.md";

    fp_csv_ = std::fopen(csv_path_.c_str(), "w");
    if (fp_csv_) {
        std::fprintf(fp_csv_,
            "frame_idx,timestamp_s,dt_s,state,is_kf,used_flow,inliers,num_kps,num_kfs,num_mps,"
            "pos_x_m,pos_y_m,pos_z_m,step_dist_m,total_travel_m,cam_roll_deg,cam_pitch_deg,cam_yaw_deg,"
            "vel_x_mps,vel_y_mps,vel_z_mps,speed_mps,trans_gate,eff_gyro_deg_s,raw_flow_px,derot_res_px,radial_med_px,"
            "imu_batch_n,imu_total_pkts,imu_ax_mps2,imu_ay_mps2,imu_az_mps2,imu_acc_norm_mps2,"
            "imu_gx_rads,imu_gy_rads,imu_gz_rads,imu_gyro_norm_degs,"
            "imu_roll_deg,imu_pitch_deg,imu_yaw_deg,imu_raw_roll_deg,imu_raw_pitch_deg,imu_raw_yaw_deg,"
            "frontend_ms,backend_ms,total_ms,fps_inst\n"
        );
        std::fflush(fp_csv_);
    }

    fp_events_ = std::fopen(events_path_.c_str(), "w");
    log_event("SESSION_START", "\"mode\":\"" + mode_ + "\"");
}

BlackboxLogger::~BlackboxLogger() {
    close();
}

void BlackboxLogger::log_event(const std::string& event_type, const std::string& json_payload) {
    if (!fp_events_) return;
    std::fprintf(fp_events_, "{\"wall_time\":\"%s\",\"frame_idx\":%d,\"event\":\"%s\",%s}\n",
                 current_iso_time(false).c_str(), frame_count_, event_type.c_str(), json_payload.c_str());
    std::fflush(fp_events_);
}

void BlackboxLogger::log_frame(int frame_idx, double timestamp, const Tracker& tracker,
                               const SlamMap& slam_map, int imu_batch_size,
                               const ImuStatusSnapshot& imu_status) {
    if (!has_first_ts_) {
        first_ts_ = timestamp;
        has_first_ts_ = true;
    }
    double dt_s = (frame_count_ > 0 && timestamp > prev_ts_) ? (timestamp - prev_ts_) : 0.033;
    prev_ts_ = timestamp;

    std::string state_name = Tracker::state_to_string(tracker.state);
    Vec3 pos(0, 0, 0);
    Vec3 cam_rpy(0, 0, 0);
    Vec3 vel(0, 0, 0);
    int num_kps = tracker.curr_frame.num_kps;

    if (tracker.curr_frame.has_pose) {
        SE3 T_wc = tracker.curr_frame.get_pose_inverse();
        pos = T_wc.t;
        cam_rpy = SO3::R_to_body_euler_deg(T_wc.R);
        vel = tracker.curr_frame.v_w;
    }

    double step_dist = 0.0;
    if (has_prev_pos_ && tracker.state == TrackingState::OK) {
        step_dist = (pos - prev_pos_).norm();
    }
    if (tracker.state == TrackingState::OK) {
        prev_pos_ = pos;
        has_prev_pos_ = true;
        total_travel_m_ += step_dist;
    }

    double speed_mps = vel.norm();
    int is_kf = tracker.last_is_keyframe ? 1 : 0;
    int used_flow = tracker.last_used_optical_flow ? 1 : 0;
    int inliers = tracker.last_inliers;
    int num_kfs = slam_map.num_keyframes();
    int num_mps = slam_map.num_mappoints();

    const auto& diag = tracker.last_diag;
    double acc_norm = imu_status.acc_mps2.norm();
    double gyro_norm_degs = imu_status.gyro_rads.norm() * (180.0 / M_PI);
    double tot_ms = std::max(0.1, diag.total_ms);
    double inst_fps = 1000.0 / tot_ms;

    if (fp_csv_) {
        std::fprintf(fp_csv_,
            "%d,%.4f,%.4f,%s,%d,%d,%d,%d,%d,%d,"
            "%.5f,%.5f,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,"
            "%.4f,%.4f,%.4f,%.4f,%.3f,%.2f,%.2f,%.2f,%.2f,"
            "%d,%d,%.3f,%.3f,%.3f,%.3f,"
            "%.4f,%.4f,%.4f,%.2f,"
            "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
            "%.2f,%.2f,%.2f,%.1f\n",
            frame_idx, timestamp, dt_s, state_name.c_str(), is_kf, used_flow, inliers, num_kps, num_kfs, num_mps,
            pos.x, pos.y, pos.z, step_dist, total_travel_m_, cam_rpy.x, cam_rpy.y, cam_rpy.z,
            vel.x, vel.y, vel.z, speed_mps, diag.trans_gate, diag.eff_gyro_deg_s,
            diag.raw_flow_px, diag.derot_res_px, diag.radial_med_px,
            imu_batch_size, imu_status.total_packets,
            imu_status.acc_mps2.x, imu_status.acc_mps2.y, imu_status.acc_mps2.z, acc_norm,
            imu_status.gyro_rads.x, imu_status.gyro_rads.y, imu_status.gyro_rads.z, gyro_norm_degs,
            imu_status.euler_deg.x, imu_status.euler_deg.y, imu_status.euler_deg.z,
            imu_status.raw_euler_deg.x, imu_status.raw_euler_deg.y, imu_status.raw_euler_deg.z,
            diag.frontend_ms, diag.backend_ms, diag.total_ms, inst_fps
        );
        std::fflush(fp_csv_);
    }

    frame_count_++;
    pos_history_.push_back(pos);
    fe_ms_history_.push_back(diag.frontend_ms);
    tot_ms_history_.push_back(tot_ms);
    inliers_history_.push_back(inliers);
    gyro_degs_history_.push_back(std::max(diag.eff_gyro_deg_s, gyro_norm_degs));
    if (acc_norm > 0.1) acc_norm_history_.push_back(acc_norm);
    last_num_kfs_ = num_kfs;
    last_num_mps_ = num_mps;
    last_imu_packets_ = imu_status.total_packets;

    if (diag.trans_gate < 0.50) {
        rot_gated_frames_++;
        rot_gated_disp_m_ += step_dist;
    } else if (step_dist > 0.0015 || diag.raw_flow_px > 1.5) {
        translation_frames_++;
        translation_disp_m_ += step_dist;
    } else {
        stationary_frames_++;
    }

    if (state_name != prev_state_) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "\"from_state\":\"%s\",\"to_state\":\"%s\",\"num_kfs\":%d,\"num_mps\":%d",
                      prev_state_.c_str(), state_name.c_str(), num_kfs, num_mps);
        log_event("STATE_CHANGE", buf);
        prev_state_ = state_name;
    }
    if (is_kf) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "\"num_kfs\":%d,\"num_mps\":%d,\"inliers\":%d,\"backend_ms\":%.2f",
                      num_kfs, num_mps, inliers, diag.backend_ms);
        log_event("KEYFRAME_CREATED", buf);
    }

    if (frame_count_ % 30 == 0 || frame_count_ == 1) {
        write_summary_and_report(false);
    }
}

void BlackboxLogger::write_summary_and_report(bool is_final) {
    double duration_s = has_first_ts_ ? std::max(0.0, prev_ts_ - first_ts_) : 0.0;
    double mean_fe = 0.0, mean_tot = 1.0, mean_inl = 0.0, max_gyro = 0.0, mean_g = 9.81;
    if (!fe_ms_history_.empty()) {
        mean_fe = std::accumulate(fe_ms_history_.begin(), fe_ms_history_.end(), 0.0) / fe_ms_history_.size();
    }
    if (!tot_ms_history_.empty()) {
        mean_tot = std::accumulate(tot_ms_history_.begin(), tot_ms_history_.end(), 0.0) / tot_ms_history_.size();
    }
    if (!inliers_history_.empty()) {
        mean_inl = std::accumulate(inliers_history_.begin(), inliers_history_.end(), 0.0) / inliers_history_.size();
    }
    if (!gyro_degs_history_.empty()) {
        max_gyro = *std::max_element(gyro_degs_history_.begin(), gyro_degs_history_.end());
    }
    if (!acc_norm_history_.empty()) {
        mean_g = std::accumulate(acc_norm_history_.begin(), acc_norm_history_.end(), 0.0) / acc_norm_history_.size();
    }
    double avg_fps = 1000.0 / std::max(1e-3, mean_tot);
    Vec3 final_pos = pos_history_.empty() ? Vec3(0, 0, 0) : pos_history_.back();

    FILE* fp_sum = std::fopen(summary_path_.c_str(), "w");
    if (fp_sum) {
        std::fprintf(fp_sum,
            "{\n"
            "  \"run_id\": \"%s\",\n"
            "  \"mode\": \"%s\",\n"
            "  \"is_final\": %s,\n"
            "  \"stats\": {\n"
            "    \"total_frames\": %d,\n"
            "    \"duration_s\": %.3f,\n"
            "    \"final_state\": \"%s\",\n"
            "    \"active_keyframes\": %d,\n"
            "    \"active_mappoints\": %d,\n"
            "    \"imu_total_packets\": %d,\n"
            "    \"mean_gravity_mps2\": %.4f,\n"
            "    \"mean_inliers\": %.1f,\n"
            "    \"frontend_mean_ms\": %.2f,\n"
            "    \"average_fps\": %.1f\n"
            "  },\n"
            "  \"trajectory\": {\n"
            "    \"total_travel_m\": %.4f,\n"
            "    \"final_pos_m\": [%.4f, %.4f, %.4f]\n"
            "  },\n"
            "  \"motion_breakdown\": {\n"
            "    \"stationary_frames\": %d,\n"
            "    \"translation_frames\": %d,\n"
            "    \"translation_distance_m\": %.4f,\n"
            "    \"rotation_gated_frames\": %d,\n"
            "    \"rotation_gated_total_drift_m\": %.4f,\n"
            "    \"max_gyro_deg_s\": %.2f\n"
            "  }\n"
            "}\n",
            run_id_.c_str(), mode_.c_str(), is_final ? "true" : "false",
            frame_count_, duration_s, prev_state_.c_str(),
            last_num_kfs_, last_num_mps_, last_imu_packets_,
            mean_g, mean_inl, mean_fe, avg_fps,
            total_travel_m_, final_pos.x, final_pos.y, final_pos.z,
            stationary_frames_, translation_frames_, translation_disp_m_,
            rot_gated_frames_, rot_gated_disp_m_, max_gyro
        );
        std::fclose(fp_sum);
    }

    FILE* fp_rep = std::fopen(report_path_.c_str(), "w");
    if (fp_rep) {
        std::fprintf(fp_rep,
            "# RK3566 C++ Visual-Inertial SLAM Blackbox Report (`%s`)\n\n"
            "- **Total Frames**: `%d` (`%.2f s`)\n"
            "- **Final State**: `%s` (`%d` KeyFrames, `%d` MapPoints)\n"
            "- **Frontend Mean Time**: `%.2f ms` (`%.1f FPS`)\n"
            "- **Total Travel**: `%.4f m` | **Final Position**: `(%.4f, %.4f, %.4f) m`\n"
            "- **Pure-Rotation Gated Frames**: `%d` (Drift: `%.4f m`, Max Gyro: `%.2f deg/s`)\n",
            run_id_.c_str(), frame_count_, duration_s,
            prev_state_.c_str(), last_num_kfs_, last_num_mps_,
            mean_fe, avg_fps, total_travel_m_, final_pos.x, final_pos.y, final_pos.z,
            rot_gated_frames_, rot_gated_disp_m_, max_gyro
        );
        std::fclose(fp_rep);
    }
}

void BlackboxLogger::close() {
    if (finalized_) return;
    finalized_ = true;
    log_event("SESSION_END", "\"total_frames\":" + std::to_string(frame_count_));
    write_summary_and_report(true);
    if (fp_csv_) { std::fclose(fp_csv_); fp_csv_ = nullptr; }
    if (fp_events_) { std::fclose(fp_events_); fp_events_ = nullptr; }
}

} // namespace slam
