#include "imu_preintegration.h"

#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace slam {

static double now_sec() {
    auto tp = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(tp).count();
}

ImuPreintegrator::ImuPreintegrator(
    const Vec3& bg,
    const Vec3& ba,
    double gyro_noise,
    double acc_noise,
    double gyro_walk,
    double acc_walk
) : bg_bar(bg),
    ba_bar(ba),
    sigma_g2(gyro_noise * gyro_noise),
    sigma_a2(acc_noise * acc_noise),
    sigma_bg2(gyro_walk * gyro_walk),
    sigma_ba2(acc_walk * acc_walk) {
    reset_state();
}

void ImuPreintegrator::reset(const Vec3& new_bg, bool update_bias, const Vec3& new_ba) {
    if (update_bias) {
        bg_bar = new_bg;
        ba_bar = new_ba;
    }
    reset_state();
}

void ImuPreintegrator::reset_state() {
    delta_R = Mat3::identity();
    delta_v = Vec3(0.0, 0.0, 0.0);
    delta_p = Vec3(0.0, 0.0, 0.0);
    dt_total = 0.0;

    J_dR_bg = Mat3::zero();
    J_dv_bg = Mat3::zero();
    J_dv_ba = Mat3::zero();
    J_dp_bg = Mat3::zero();
    J_dp_ba = Mat3::zero();

    std::memset(cov, 0, sizeof(cov));
}

void ImuPreintegrator::integrate_measurement(const Vec3& acc_m, const Vec3& gyro_m, double dt) {
    if (dt <= 0.0) return;

    Vec3 omega = gyro_m - bg_bar;
    Vec3 acc = acc_m - ba_bar;

    Vec3 phi = omega * dt;
    Mat3 dR = SO3::exp(phi);
    Mat3 Jr = SO3::right_jacobian(phi);

    Mat3 acc_hat = SO3::hat(acc);
    Mat3 dR_acc_hat = delta_R * acc_hat;

    // 1. 先更新关于零偏的一阶雅可比 (需要用到更新前的 J_dR_bg 与 J_dv_bg)
    J_dp_bg = J_dp_bg + J_dv_bg * dt - (dR_acc_hat * J_dR_bg) * (0.5 * dt * dt);
    J_dp_ba = J_dp_ba + J_dv_ba * dt - delta_R * (0.5 * dt * dt);

    J_dv_bg = J_dv_bg - (dR_acc_hat * J_dR_bg) * dt;
    J_dv_ba = J_dv_ba - delta_R * dt;

    J_dR_bg = dR.transpose() * J_dR_bg - Jr * dt;

    // 2. 更新预积分名义状态
    Vec3 a_world_local = delta_R * acc;
    delta_p = delta_p + delta_v * dt + a_world_local * (0.5 * dt * dt);
    delta_v = delta_v + a_world_local * dt;
    delta_R = SO3::normalize_rotation(delta_R * dR);
    dt_total += dt;

    // 3. 更新对角块噪声协方差近似 (保证数值正定且在 Cortex-A55 上耗时 < 0.2 us)
    double q_rot = sigma_g2 * dt;
    double q_vel = sigma_a2 * dt;
    double q_pos = sigma_a2 * (dt * dt * dt) * 0.25;
    double q_bg  = sigma_bg2 * dt;
    double q_ba  = sigma_ba2 * dt;
    for (int i = 0; i < 3; ++i) {
        cov[i][i]         += q_rot;
        cov[3 + i][3 + i] += q_vel;
        cov[6 + i][6 + i] += q_pos;
        cov[9 + i][9 + i] += q_bg;
        cov[12 + i][12 + i] += q_ba;
    }
}

void ImuPreintegrator::integrate_batch(const std::vector<ImuMeasurement>& measurements, double t_start, double t_end) {
    if (measurements.empty() || t_end <= t_start) return;
    double t_curr = t_start;
    for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& m = measurements[i];
        if (m.timestamp <= t_start) continue;
        double t_next = (i + 1 < measurements.size()) ? std::min(measurements[i + 1].timestamp, t_end) : t_end;
        double dt = t_next - t_curr;
        if (dt > 0.0) {
            integrate_measurement(m.acc, m.gyro, dt);
            t_curr = t_next;
        }
        if (t_curr >= t_end) break;
    }
}

void ImuPreintegrator::get_corrected_measurements(
    const Vec3& bg_new, const Vec3& ba_new,
    Mat3& dR_corr, Vec3& dv_corr, Vec3& dp_corr
) const {
    Vec3 dbg = bg_new - bg_bar;
    Vec3 dba = ba_new - ba_bar;
    dR_corr = SO3::normalize_rotation(delta_R * SO3::exp(J_dR_bg * dbg));
    dv_corr = delta_v + J_dv_bg * dbg + J_dv_ba * dba;
    dp_corr = delta_p + J_dp_bg * dbg + J_dp_ba * dba;
}

void ImuPreintegrator::predict_state(
    const Mat3& R_wb_i, const Vec3& v_w_i, const Vec3& p_w_i,
    Mat3& R_wb_j, Vec3& v_w_j, Vec3& p_w_j,
    const Vec3& g_w
) const {
    double dt = dt_total;
    R_wb_j = SO3::normalize_rotation(R_wb_i * delta_R);
    v_w_j = v_w_i + g_w * dt + R_wb_i * delta_v;
    p_w_j = p_w_i + v_w_i * dt + g_w * (0.5 * dt * dt) + R_wb_i * delta_p;
}

WitMotionSerialImu::WitMotionSerialImu(const std::string& port, int baud)
    : port_(port), baud_(baud), R_mount_(Mat3::identity()), ref_tilt_deg_(0.0, 0.0) {}

WitMotionSerialImu::~WitMotionSerialImu() {
    close();
}

bool WitMotionSerialImu::start() {
    if (running_.load()) return is_connected_.load();

#ifdef _WIN32
    std::vector<std::string> cands = {port_, "COM5", "COM4", "COM3", "COM6", "COM7"};
    HANDLE h_win = INVALID_HANDLE_VALUE;
    for (const auto& c : cands) {
        if (c.empty() || c[0] == '/') continue;
        std::string dev = "\\\\.\\" + c;
        HANDLE h_try = CreateFileA(dev.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
        if (h_try == INVALID_HANDLE_VALUE) continue;
        DCB dcb;
        std::memset(&dcb, 0, sizeof(dcb));
        dcb.DCBlength = sizeof(dcb);
        if (GetCommState(h_try, &dcb)) {
            dcb.BaudRate = baud_;
            dcb.ByteSize = 8;
            dcb.StopBits = ONESTOPBIT;
            dcb.Parity = NOPARITY;
            SetCommState(h_try, &dcb);
        }
        COMMTIMEOUTS timeouts = {10, 0, 40, 0, 40};
        SetCommTimeouts(h_try, &timeouts);
        PurgeComm(h_try, PURGE_RXCLEAR | PURGE_TXCLEAR);
        uint8_t probe[64];
        DWORD n_read = 0;
        bool has_hdr = false;
        for (int t = 0; t < 4; ++t) {
            if (ReadFile(h_try, probe, sizeof(probe), &n_read, NULL) && n_read > 0) {
                for (DWORD k = 0; k < n_read; ++k) {
                    if (probe[k] == 0x55) { has_hdr = true; break; }
                }
            }
            if (has_hdr) break;
        }
        if (has_hdr) {
            h_win = h_try;
            port_ = c;
            break;
        }
        CloseHandle(h_try);
    }
    if (h_win == INVALID_HANDLE_VALUE) {
        is_connected_.store(false);
        return false;
    }
    DWORD n_written = 0;
    const uint8_t unlock_cmd[5] = {0xFF, 0xAA, 0x69, 0x88, 0xB5};
    const uint8_t rate20_cmd[5] = {0xFF, 0xAA, 0x03, 0x07, 0x00};
    WriteFile(h_win, unlock_cmd, 5, &n_written, NULL);
    Sleep(15);
    WriteFile(h_win, rate20_cmd, 5, &n_written, NULL);
    win_handle_ = (void*)h_win;
    is_connected_.store(true);
#else
    std::vector<std::string> cands = {
        port_, "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyS3", "/dev/ttyS4", "/dev/ttyS2"
    };
    int fd_good = -1;
    for (const auto& c : cands) {
        if (c.empty() || c[0] != '/') continue;
        int fd_try = ::open(c.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_try < 0) continue;

        struct termios tty;
        std::memset(&tty, 0, sizeof(tty));
        if (tcgetattr(fd_try, &tty) == 0) {
            speed_t spd = B9600;
            if (baud_ == 115200) spd = B115200;
            else if (baud_ == 38400) spd = B38400;
            else if (baud_ == 19200) spd = B19200;
            cfsetospeed(&tty, spd);
            cfsetispeed(&tty, spd);
            tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
            tty.c_iflag &= ~IGNBRK;
            tty.c_lflag = 0;
            tty.c_oflag = 0;
            tty.c_cc[VMIN]  = 0;
            tty.c_cc[VTIME] = 1;
            tty.c_iflag &= ~(IXON | IXOFF | IXANY);
            tty.c_cflag |= (CLOCAL | CREAD);
            tty.c_cflag &= ~(PARENB | PARODD);
            tty.c_cflag &= ~CSTOPB;
            tcsetattr(fd_try, TCSANOW, &tty);
        }
        uint8_t probe[64];
        bool has_hdr = false;
        for (int t = 0; t < 5; ++t) {
            usleep(35000);
            ssize_t nr = ::read(fd_try, probe, sizeof(probe));
            if (nr > 0) {
                for (ssize_t k = 0; k < nr; ++k) {
                    if (probe[k] == 0x55) { has_hdr = true; break; }
                }
            }
            if (has_hdr) break;
        }
        if (has_hdr) {
            fd_good = fd_try;
            port_ = c;
            break;
        }
        ::close(fd_try);
    }
    if (fd_good < 0) {
        is_connected_.store(false);
        return false;
    }
    const uint8_t unlock_cmd[5] = {0xFF, 0xAA, 0x69, 0x88, 0xB5};
    const uint8_t rate20_cmd[5] = {0xFF, 0xAA, 0x03, 0x07, 0x00};
    (void)::write(fd_good, unlock_cmd, 5);
    usleep(15000);
    (void)::write(fd_good, rate20_cmd, 5);
    fd_ = fd_good;
    is_connected_.store(true);
#endif

    running_.store(true);
    thread_ = std::thread(&WitMotionSerialImu::read_loop, this);
    return true;
}

void WitMotionSerialImu::read_loop() {
    std::vector<uint8_t> buf;
    buf.reserve(512);
    bool has_acc = false;
    uint8_t chunk[256];

    while (running_.load()) {
        int n_got = 0;
#ifdef _WIN32
        if (win_handle_ != nullptr) {
            DWORD nr = 0;
            if (ReadFile((HANDLE)win_handle_, chunk, sizeof(chunk), &nr, NULL) && nr > 0) {
                n_got = (int)nr;
            }
        }
#else
        if (fd_ >= 0) {
            ssize_t nr = ::read(fd_, chunk, sizeof(chunk));
            if (nr > 0) n_got = (int)nr;
        }
#endif
        if (n_got > 0) {
            buf.insert(buf.end(), chunk, chunk + n_got);
            has_acc = parse_buffer(buf, has_acc);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
}

bool WitMotionSerialImu::parse_buffer(std::vector<uint8_t>& buf, bool has_acc) {
    while (buf.size() >= 11) {
        if (buf[0] != 0x55) {
            buf.erase(buf.begin());
            continue;
        }
        uint8_t sum = 0;
        for (int i = 0; i < 10; ++i) sum += buf[i];
        if (sum != buf[10]) {
            buf.erase(buf.begin());
            continue;
        }
        uint8_t pkt[11];
        std::memcpy(pkt, buf.data(), 11);
        buf.erase(buf.begin(), buf.begin() + 11);

        uint8_t ptype = pkt[1];
        int16_t v0 = (int16_t)((uint16_t)pkt[2] | ((uint16_t)pkt[3] << 8));
        int16_t v1 = (int16_t)((uint16_t)pkt[4] | ((uint16_t)pkt[5] << 8));
        int16_t v2 = (int16_t)((uint16_t)pkt[6] | ((uint16_t)pkt[7] << 8));
        int16_t v3 = (int16_t)((uint16_t)pkt[8] | ((uint16_t)pkt[9] << 8));
        double ts_now = now_sec();

        std::lock_guard<std::mutex> lk(lock_);
        total_packets_++;
        if (ptype == 0x51) {
            double ax = (v0 / 32768.0) * 16.0 * 9.80665;
            double ay = (v1 / 32768.0) * 16.0 * 9.80665;
            double az = (v2 / 32768.0) * 16.0 * 9.80665;
            Vec3 raw_acc(ax, ay, az);
            double acc_norm = raw_acc.norm();
            latest_raw_acc_ = raw_acc;
            if (mount_calib_count_ == 0 && acc_norm > 8.2 && acc_norm < 11.5 && az > 6.0) {
                double r_acc = std::atan2(ay, az) * (180.0 / M_PI);
                double p_acc = std::atan2(-ax, std::hypot(ay, az)) * (180.0 / M_PI);
                ref_tilt_deg_ = Vec2(r_acc, p_acc);
                R_mount_ = SO3::euler_deg_to_R(Vec3(r_acc, p_acc, 0.0));
            }
            Vec3 acc_lvl = R_mount_ * raw_acc;
            double tilt_gyro = std::hypot(latest_gyro_.x, latest_gyro_.y);
            if (std::abs(latest_euler_deg_.x) < 6.0 && std::abs(latest_euler_deg_.y) < 6.0 &&
                tilt_gyro < 0.30 && acc_norm > 8.2 && acc_norm < 11.5) {
                acc_lvl.x *= 0.12;
                acc_lvl.y *= 0.12;
            }
            latest_acc_ = acc_lvl;
            latest_temp_c_ = (double)v3 / 100.0;
            has_acc = true;
        } else if (ptype == 0x52) {
            const double deg2rad = M_PI / 180.0;
            double wx = (v0 / 32768.0) * 2000.0 * deg2rad;
            double wy = (v1 / 32768.0) * 2000.0 * deg2rad;
            double wz = (v2 / 32768.0) * 2000.0 * deg2rad;
            latest_raw_gyro_ = Vec3(wx, wy, wz);
            Vec3 gyro_lvl = R_mount_ * latest_raw_gyro_;
            double tilt_gyro_lvl = std::hypot(gyro_lvl.x, gyro_lvl.y);
            bool yaw_dominant = std::abs(gyro_lvl.z) > 1.8 * tilt_gyro_lvl;
            if (std::abs(latest_euler_deg_.x) < 6.0 && std::abs(latest_euler_deg_.y) < 6.0 &&
                (tilt_gyro_lvl < 0.35 || yaw_dominant)) {
                gyro_lvl.x = (std::abs(gyro_lvl.x) < 0.04) ? 0.0 : gyro_lvl.x * 0.04;
                gyro_lvl.y = (std::abs(gyro_lvl.y) < 0.04) ? 0.0 : gyro_lvl.y * 0.04;
            }
            latest_gyro_ = gyro_lvl;
            if (has_acc || total_packets_ > 2) {
                measurements_.emplace_back(ts_now, latest_acc_, latest_gyro_);
                if (measurements_.size() > 2000) {
                    measurements_.erase(measurements_.begin(), measurements_.begin() + 1000);
                }
            }
        } else if (ptype == 0x53) {
            double roll  = (v0 / 32768.0) * 180.0;
            double pitch = (v1 / 32768.0) * 180.0;
            double yaw   = (v2 / 32768.0) * 180.0;
            if (std::abs(roll) > 65.0 || std::abs(pitch) > 65.0) continue;
            latest_raw_euler_deg_ = Vec3(roll, pitch, yaw);
            double tilt_gyro = std::hypot(latest_gyro_.x, latest_gyro_.y);
            double acc_norm = latest_raw_acc_.norm();
            if (mount_calib_count_ == 0 && std::abs(roll) < 40.0 && std::abs(pitch) < 40.0) {
                ref_tilt_deg_ = Vec2(roll, pitch);
                R_mount_ = SO3::euler_deg_to_R(Vec3(roll, pitch, 0.0));
                mount_calib_count_ = 1;
            } else if (mount_calib_count_ < 15 && tilt_gyro < 0.30 && std::abs(roll) < 40.0 && std::abs(pitch) < 40.0) {
                double alpha = 0.35;
                ref_tilt_deg_.x = (1.0 - alpha) * ref_tilt_deg_.x + alpha * roll;
                ref_tilt_deg_.y = (1.0 - alpha) * ref_tilt_deg_.y + alpha * pitch;
                R_mount_ = SO3::euler_deg_to_R(Vec3(ref_tilt_deg_.x, ref_tilt_deg_.y, 0.0));
                mount_calib_count_++;
            } else if (tilt_gyro < 0.28 && acc_norm > 8.5 && acc_norm < 11.2 &&
                       std::abs(roll - ref_tilt_deg_.x) < 12.0 && std::abs(pitch - ref_tilt_deg_.y) < 12.0) {
                double alpha = 0.20;
                ref_tilt_deg_.x = (1.0 - alpha) * ref_tilt_deg_.x + alpha * roll;
                ref_tilt_deg_.y = (1.0 - alpha) * ref_tilt_deg_.y + alpha * pitch;
                R_mount_ = SO3::euler_deg_to_R(Vec3(ref_tilt_deg_.x, ref_tilt_deg_.y, 0.0));
            }
            double rel_roll = roll - ref_tilt_deg_.x;
            double rel_pitch = pitch - ref_tilt_deg_.y;
            if (std::abs(rel_roll) < 4.5 && std::abs(rel_pitch) < 4.5 && tilt_gyro < 0.30) {
                rel_roll *= 0.12;
                rel_pitch *= 0.12;
            }
            latest_euler_deg_ = Vec3(rel_roll, rel_pitch, yaw);
        }
    }
    return has_acc;
}

std::vector<ImuMeasurement> WitMotionSerialImu::get_measurements_between(double t_prev, double t_curr) {
    std::lock_guard<std::mutex> lk(lock_);
    std::vector<ImuMeasurement> batch;
    for (const auto& m : measurements_) {
        if (m.timestamp > t_prev && m.timestamp <= t_curr) {
            batch.push_back(m);
        }
    }
    if (batch.empty() && total_packets_ > 0) {
        batch.emplace_back(t_curr, latest_acc_, latest_gyro_);
    }
    return batch;
}

ImuStatusSnapshot WitMotionSerialImu::get_status_snapshot() {
    std::lock_guard<std::mutex> lk(lock_);
    ImuStatusSnapshot s;
    s.connected = is_connected_.load();
    s.port = port_;
    s.baud = baud_;
    s.total_packets = total_packets_;
    s.acc_mps2 = latest_acc_;
    s.gyro_rads = latest_gyro_;
    s.euler_deg = latest_euler_deg_;
    s.raw_acc_mps2 = latest_raw_acc_;
    s.raw_gyro_rads = latest_raw_gyro_;
    s.raw_euler_deg = latest_raw_euler_deg_;
    s.temp_c = latest_temp_c_;
    return s;
}

void WitMotionSerialImu::close() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
#ifdef _WIN32
    if (win_handle_ != nullptr) {
        CloseHandle((HANDLE)win_handle_);
        win_handle_ = nullptr;
    }
#else
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    is_connected_.store(false);
}

} // namespace slam
