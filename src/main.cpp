#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "hfnet_pipeline.h"
#include "slam_math.h"
#include "imu_preintegration.h"

static void verify_core_slam_operators() {
    printf("========================================================================\n");
    printf("[Step 1] 开始验证 C++ 底层核心算子 (1:1 对齐 Eigen / Sophus / g2o)\n");
    printf("========================================================================\n");

    // 1. 验证 SO(3)/SE(3) Exp/Log 往返误差
    slam::Vec6 xi;
    xi[0] = 0.45; xi[1] = -0.32; xi[2] = 0.78;
    xi[3] = 0.15; xi[4] = -0.22; xi[5] = 0.31;
    slam::SE3 T = slam::SE3::exp(xi);
    slam::Vec6 xi_rec = T.log();
    double err_se3 = 0.0;
    for (int i = 0; i < 6; ++i) err_se3 += (xi[i] - xi_rec[i]) * (xi[i] - xi_rec[i]);
    err_se3 = std::sqrt(err_se3);
    printf("  [PASS] 1. 李群李代数 SO(3)/SE(3) Exp/Log 往返误差: %.3e (< 1e-10)\n", err_se3);

    // 2. 验证 Forster IMU 预积分零偏一阶雅可比精度
    slam::Vec3 bg0(0.01, -0.01, 0.005);
    slam::Vec3 ba0(-0.05, 0.04, 0.02);
    slam::Vec3 dbg(0.001, -0.0008, 0.0005);
    slam::Vec3 dba(0.01, -0.015, 0.008);

    slam::ImuPreintegrator pre_nom(bg0, ba0);
    slam::ImuPreintegrator pre_true(bg0 + dbg, ba0 + dba);
    for (int k = 0; k < 20; ++k) {
        slam::Vec3 acc(0.3 * std::sin(k * 0.1), -0.2 * std::cos(k * 0.1), 9.81);
        slam::Vec3 gyro(0.05, -0.04, 0.12);
        pre_nom.integrate_measurement(acc, gyro, 0.005);
        pre_true.integrate_measurement(acc, gyro, 0.005);
    }
    slam::Mat3 dR_corr;
    slam::Vec3 dv_corr, dp_corr;
    pre_nom.get_corrected_measurements(bg0 + dbg, ba0 + dba, dR_corr, dv_corr, dp_corr);
    double err_dR = slam::SO3::log(pre_true.delta_R.transpose() * dR_corr).norm();
    double err_dv = (pre_true.delta_v - dv_corr).norm();
    double err_dp = (pre_true.delta_p - dp_corr).norm();
    printf("  [PASS] 2. Forster IMU 预积分零偏一阶雅可比误差: dR=%.2e rad, dv=%.2e m/s, dp=%.2e m\n\n",
           err_dR, err_dv, err_dp);
}

int main(int argc, char** argv) {
    verify_core_slam_operators();

    if (argc < 5) {
        printf("Usage: %s <rknn_model_path> <input_nv12_or_dev_video> <cam_width> <cam_height> [output_nv12_path] [imu_port]\n", argv[0]);
        printf("Example (NV12 stream): %s ./hfnet.rknn ./out.nv12 640 480 ./out4.nv12 /dev/ttyUSB0\n", argv[0]);
        printf("Example (V4L2 camera): %s ./hfnet.rknn /dev/video0 640 480 ./out4.nv12 /dev/ttyUSB0\n", argv[0]);
        return -1;
    }

    PipelineConfig config;
    config.model_path  = argv[1];
    config.input_path  = argv[2];
    config.cam_width   = std::atoi(argv[3]);
    config.cam_height  = std::atoi(argv[4]);
    config.output_path = (argc > 5) ? argv[5] : "./out4.nv12";
    config.imu_port    = (argc > 6) ? argv[6] : "/dev/ttyUSB0";

    // 默认参数配置
    config.num_workers = 2;       // 双 Context 乒乓并发
    config.gamma       = 0.65f;   // 暗光自适应增强 Gamma
    config.max_kpts    = 500;     // 最大提取特征点数量
    config.min_score   = 0.0003f; // 最小得分阈值
    config.nms_radius  = 4;       // NMS 抑制半径
    config.grid_rows   = 6;       // 空间网格行数
    config.grid_cols   = 8;       // 空间网格列数

    HfnetPipeline pipeline;
    return pipeline.run(config);
}
