# HF-Net + Visual-Inertial SLAM for Rockchip RK3566 (Pure C++11 + RKNN NPU)

面向 **Rockchip RK3566 (4× ARM Cortex-A55 + 0.8 TOPS RKNN NPU)** 的全栈 C++11 视觉/视觉惯性 SLAM 系统。
在原有三级并发 RKNN NPU 特征提取流水线 (`HfnetPipeline`) 基础上，1:1 完整移植了 Python 视觉惯性 SLAM 的全部核心模块（**零 OpenCV / Eigen / Sophus / g2o 第三方库依赖**，直接链接 `librknnrt.so` + `pthread` + `m` 即可在板端编译运行）。

---

## 模块架构

| 模块文件 | 对应原 Python 模块 | 核心功能 |
| :--- | :--- | :--- |
| `include/slam_math.h`<br>`src/slam_math.cpp` | `src/core/lie_algebra.py` | $\text{SO}(3)/\text{SE}(3)$ 李群李代数指数/对数映射、右雅可比 $\mathbf{J}_r$、$6\times 6$ / $N\times N$ 列主元线性求解器、双视图射线中点三角化 |
| `include/imu_preintegration.h`<br>`src/imu_preintegration.cpp` | `src/frontend/imu_preintegration.py`<br>`src/dataset.py` | Forster 流形预积分 ($\Delta\mathbf{R}, \Delta\mathbf{v}, \Delta\mathbf{p}$)、$15\times 15$ 协方差递推、零偏一阶雅可比修正；Linux `termios` `/dev/ttyUSB*` WitMotion 串口 IMU 异步读取与 20Hz 倍频指令 |
| `include/slam_core.h`<br>`src/slam_core.cpp` | `src/core/camera.py`, `frame.py`, `keyframe.py`, `mappoint.py`, `map.py` | 相机模型 (`custom_stereo_imu.json` 解析)、$64\times 48$ 空间网格索引 `Frame`、共视图与生成树 `KeyFrame`、256 维描述子 `MapPoint` 与全局地图 `SlamMap` |
| `include/slam_frontend.h`<br>`src/slam_frontend.cpp` | `src/frontend/stereo_matcher.py`<br>`src/frontend/hfnet_extractor.py` | 4 层高斯金字塔构建、4 层双向 Inverse-Compositional 金字塔 LK 光流跟踪 (`fb_err < 1.5 px`)、二维梯度结构张量 (Shi-Tomasi $\lambda_{\min}$ + 各向同性比) 网格补点与 256 维描述子投影/极线匹配 |
| `include/slam_backend.h`<br>`src/slam_backend.cpp` | `src/backend/optimizer.py`<br>`src/backend/local_mapper.py` | 4 轮 Huber 鲁棒核 + 卡方检验的 6DoF 位姿优化 (`pose_optimization`) + IMU 旋转紧耦合先验；基于显式**稀疏舒尔补 (`Sparse Schur Complement`)** 的局部 BA (`local_bundle_adjustment`) 与 `LocalMapper` |
| `include/slam_tracker.h`<br>`src/slam_tracker.cpp` | `src/frontend/tracker.py` | 单目+IMU冷启动初始化、混合光流+HF-Net特征无损融合、视觉消旋与纯旋转平移门控 (`trans_gate`)、光流比例步长限幅、临时流点动态回收与关键帧状态机 |
| `include/blackbox_logger.h`<br>`src/blackbox_logger.cpp` | `src/blackbox_logger.py` | 实时生成 47 列 `frames.csv`、`events.jsonl`、`summary.json` 与 `report.md` 黑盒诊断报告 |
| `include/image_utils.h`<br>`src/image_utils.cpp` | `src/visualizer.py` | NV12 零拷贝直接绘制光流跟踪轨迹线、内点圆环，以及右上角半透明 **2D 俯视轨迹 + 3D 地图点雷达小地图 HUD** |

---

## RK3566 板端编译与运行指南

```bash
mkdir -p build && cd build
cmake ..
make -j4

# 1. 使用录制好的 NV12 视频流 + 串口 IMU 运行 SLAM 并输出带 2D 轨迹雷达 HUD 的 NV12 流
./hfnet_demo ./hfnet_backbone_rk3566_i8.rknn ./out.nv12 640 480 ./out_slam.nv12 /dev/ttyUSB0

# 2. 直接读取板端 V4L2 实时摄像头 (/dev/video0) + 串口 IMU (/dev/ttyUSB0)
./hfnet_demo ./hfnet_backbone_rk3566_i8.rknn /dev/video0 640 480 ./out_slam.nv12 /dev/ttyUSB0
```
