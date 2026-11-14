#ifndef HFNET_IMU_PREINTEGRATION_H
#define HFNET_IMU_PREINTEGRATION_H

#include "slam_math.h"
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>

namespace slam {

struct ImuMeasurement {
    double timestamp = 0.0;
    Vec3 acc;   // m/s^2
    Vec3 gyro;  // rad/s
    ImuMeasurement() = default;
    ImuMeasurement(double ts, const Vec3& a, const Vec3& g)
        : timestamp(ts), acc(a), gyro(g) {}
};

// Forster 流形预积分器 (1:1 对齐 Python src/frontend/imu_preintegration.py)
class ImuPreintegrator {
public:
    Vec3 bg_bar;
    Vec3 ba_bar;
    double sigma_g2;
    double sigma_a2;
    double sigma_bg2;
    double sigma_ba2;

    Mat3 delta_R;
    Vec3 delta_v;
    Vec3 delta_p;
    double dt_total;

    Mat3 J_dR_bg;
    Mat3 J_dv_bg;
    Mat3 J_dv_ba;
    Mat3 J_dp_bg;
    Mat3 J_dp_ba;

    double cov[15][15];

    explicit ImuPreintegrator(
        const Vec3& bg = Vec3(0, 0, 0),
        const Vec3& ba = Vec3(0, 0, 0),
        double gyro_noise = 1.7e-4,
        double acc_noise = 2.0e-3,
        double gyro_walk = 1.9e-5,
        double acc_walk = 3.0e-3
    );

    void reset(const Vec3& new_bg = nullptr ? Vec3() : Vec3(), bool update_bias = false, const Vec3& new_ba = Vec3());
    void reset_state();
    void integrate_measurement(const Vec3& acc_m, const Vec3& gyro_m, double dt);
    void integrate_batch(const std::vector<ImuMeasurement>& measurements, double t_start, double t_end);
    void get_corrected_measurements(const Vec3& bg_new, const Vec3& ba_new,
                                    Mat3& dR_corr, Vec3& dv_corr, Vec3& dp_corr) const;
    void predict_state(const Mat3& R_wb_i, const Vec3& v_w_i, const Vec3& p_w_i,
                       Mat3& R_wb_j, Vec3& v_w_j, Vec3& p_w_j,
                       const Vec3& g_w = Vec3(0.0, 0.0, -9.81)) const;
};

struct ImuStatusSnapshot {
    bool connected = false;
    std::string port;
    int baud = 9600;
    int total_packets = 0;
    Vec3 acc_mps2 = Vec3(0.0, 0.0, 9.81);
    Vec3 gyro_rads = Vec3(0.0, 0.0, 0.0);
    Vec3 euler_deg = Vec3(0.0, 0.0, 0.0);
    Vec3 raw_acc_mps2 = Vec3(0.0, 0.0, 9.81);
    Vec3 raw_gyro_rads = Vec3(0.0, 0.0, 0.0);
    Vec3 raw_euler_deg = Vec3(0.0, 0.0, 0.0);
    double temp_c = 25.0;
};

// WitMotion 11字节 0x55 协议串口 IMU 读取器 (支持 RK3566 Linux termios /dev/ttyUSB* 与 Windows COM*)
class WitMotionSerialImu {
public:
    WitMotionSerialImu(const std::string& port = "/dev/ttyUSB0", int baud = 9600);
    ~WitMotionSerialImu();

    bool start();
    void close();
    std::vector<ImuMeasurement> get_measurements_between(double t_prev, double t_curr);
    ImuStatusSnapshot get_status_snapshot();

private:
    void read_loop();
    bool parse_buffer(std::vector<uint8_t>& buf, bool has_acc);

    std::string port_;
    int baud_;
    std::atomic<bool> is_connected_{false};
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex lock_;

    int fd_ = -1;
    void* win_handle_ = nullptr;

    Mat3 R_mount_;
    Vec2 ref_tilt_deg_;
    int mount_calib_count_ = 0;

    Vec3 latest_acc_ = Vec3(0.0, 0.0, 9.81);
    Vec3 latest_gyro_ = Vec3(0.0, 0.0, 0.0);
    Vec3 latest_euler_deg_ = Vec3(0.0, 0.0, 0.0);
    Vec3 latest_raw_acc_ = Vec3(0.0, 0.0, 9.81);
    Vec3 latest_raw_gyro_ = Vec3(0.0, 0.0, 0.0);
    Vec3 latest_raw_euler_deg_ = Vec3(0.0, 0.0, 0.0);
    double latest_temp_c_ = 25.0;
    int total_packets_ = 0;
    std::vector<ImuMeasurement> measurements_;
};

} // namespace slam

#endif // HFNET_IMU_PREINTEGRATION_H
