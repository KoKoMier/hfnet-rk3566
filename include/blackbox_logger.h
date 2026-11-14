#ifndef HFNET_BLACKBOX_LOGGER_H
#define HFNET_BLACKBOX_LOGGER_H

#include "slam_tracker.h"
#include <cstdio>
#include <string>
#include <vector>

namespace slam {

// 实时运行黑盒记录仪 (1:1 对齐 Python src/blackbox_logger.py，输出 frames.csv / events.jsonl / summary.json / report.md)
class BlackboxLogger {
public:
    explicit BlackboxLogger(const std::string& base_dir = "outputs/blackbox_logs",
                            const std::string& mode = "rk3566_live_mono_imu");
    ~BlackboxLogger();

    void log_frame(int frame_idx, double timestamp, const Tracker& tracker,
                   const SlamMap& slam_map, int imu_batch_size,
                   const ImuStatusSnapshot& imu_status);

    void close();

    std::string get_run_dir() const { return run_dir_; }

private:
    void log_event(const std::string& event_type, const std::string& json_payload);
    void write_summary_and_report(bool is_final);

    std::string base_dir_;
    std::string mode_;
    std::string run_id_;
    std::string run_dir_;
    std::string start_time_iso_;

    std::string csv_path_;
    std::string events_path_;
    std::string summary_path_;
    std::string report_path_;

    FILE* fp_csv_ = nullptr;
    FILE* fp_events_ = nullptr;
    bool finalized_ = false;

    int frame_count_ = 0;
    bool has_first_ts_ = false;
    double first_ts_ = 0.0;
    double prev_ts_ = 0.0;
    bool has_prev_pos_ = false;
    Vec3 prev_pos_ = Vec3(0, 0, 0);
    std::string prev_state_ = "NO_IMAGES_YET";
    double total_travel_m_ = 0.0;

    std::vector<Vec3> pos_history_;
    std::vector<double> fe_ms_history_;
    std::vector<double> tot_ms_history_;
    std::vector<int> inliers_history_;
    std::vector<double> gyro_degs_history_;
    std::vector<double> acc_norm_history_;

    int rot_gated_frames_ = 0;
    double rot_gated_disp_m_ = 0.0;
    int translation_frames_ = 0;
    double translation_disp_m_ = 0.0;
    int stationary_frames_ = 0;
    int last_num_kfs_ = 0;
    int last_num_mps_ = 0;
    int last_imu_packets_ = 0;
};

} // namespace slam

#endif // HFNET_BLACKBOX_LOGGER_H
