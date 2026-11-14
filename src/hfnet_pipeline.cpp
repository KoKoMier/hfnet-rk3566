#include "hfnet_pipeline.h"
#include "image_utils.h"
#include "hfnet_postprocess.h"
#include "slam_tracker.h"
#include "blackbox_logger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <map>
#include <algorithm>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#endif

HfnetPipeline::HfnetPipeline()
    : queue_prep_to_npu_(4), queue_npu_to_post_(4) {}

HfnetPipeline::~HfnetPipeline() = default;

void HfnetPipeline::thread_preprocess(const PipelineConfig& config, int req_width, int req_height,
                                      slam::WitMotionSerialImu* imu_reader) {
    int frame_size = config.cam_width * config.cam_height * 3 / 2;
    int frame_idx = 0;
    auto t0_wall = std::chrono::steady_clock::now();
    double prev_ts_abs = std::chrono::duration<double>(t0_wall.time_since_epoch()).count();

#if defined(__linux__)
    // 若 input_path 为 /dev/video* 设备节点，则启用 V4L2 实时摄像头采集 (支持 NV12 / YUYV)
    if (config.input_path.rfind("/dev/video", 0) == 0) {
        int fd = ::open(config.input_path.c_str(), O_RDWR);
        if (fd >= 0) {
            struct v4l2_format fmt;
            std::memset(&fmt, 0, sizeof(fmt));
            fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            fmt.fmt.pix.width = config.cam_width;
            fmt.fmt.pix.height = config.cam_height;
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
            fmt.fmt.pix.field = V4L2_FIELD_NONE;
            if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0 || fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_NV12) {
                fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
                ioctl(fd, VIDIOC_S_FMT, &fmt);
            }
            bool is_yuyv = (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV);

            struct v4l2_requestbuffers req;
            std::memset(&req, 0, sizeof(req));
            req.count = 4;
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            req.memory = V4L2_MEMORY_MMAP;
            if (ioctl(fd, VIDIOC_REQBUFS, &req) == 0 && req.count >= 2) {
                struct MmapBuf { void* start = nullptr; size_t length = 0; };
                std::vector<MmapBuf> bufs(req.count);
                for (uint32_t i = 0; i < req.count; ++i) {
                    struct v4l2_buffer b;
                    std::memset(&b, 0, sizeof(b));
                    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                    b.memory = V4L2_MEMORY_MMAP;
                    b.index = i;
                    ioctl(fd, VIDIOC_QUERYBUF, &b);
                    bufs[i].length = b.length;
                    bufs[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, b.m.offset);
                    ioctl(fd, VIDIOC_QBUF, &b);
                }
                enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                ioctl(fd, VIDIOC_STREAMON, &type);
                printf("[Stage 1] V4L2 Live Camera started on %s (%dx%d, %s)\n",
                       config.input_path.c_str(), config.cam_width, config.cam_height, is_yuyv ? "YUYV->NV12" : "NV12");

                while (frame_idx < 3000) {
                    struct v4l2_buffer b;
                    std::memset(&b, 0, sizeof(b));
                    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                    b.memory = V4L2_MEMORY_MMAP;
                    if (ioctl(fd, VIDIOC_DQBUF, &b) < 0) break;

                    auto t_now = std::chrono::steady_clock::now();
                    double cur_ts_abs = std::chrono::duration<double>(t_now.time_since_epoch()).count();
                    double rel_ts = std::chrono::duration<double>(t_now - t0_wall).count();

                    PreprocessPacket pkt;
                    pkt.frame_idx = frame_idx;
                    pkt.timestamp_sec = rel_ts;
                    pkt.raw_nv12.resize(frame_size);
                    pkt.resized_y.resize(req_width * req_height);
                    pkt.is_eof = false;

                    const uint8_t* raw_ptr = (const uint8_t*)bufs[b.index].start;
                    if (!is_yuyv) {
                        std::memcpy(pkt.raw_nv12.data(), raw_ptr, std::min((size_t)frame_size, (size_t)b.bytesused));
                    } else {
                        int W = config.cam_width, H = config.cam_height;
                        uint8_t* y_dst = pkt.raw_nv12.data();
                        uint8_t* uv_dst = pkt.raw_nv12.data() + W * H;
                        for (int y = 0; y < H; ++y) {
                            const uint8_t* row = raw_ptr + y * W * 2;
                            for (int x = 0; x < W; x += 2) {
                                y_dst[y * W + x]     = row[x * 2 + 0];
                                y_dst[y * W + x + 1] = row[x * 2 + 2];
                                if ((y & 1) == 0) {
                                    uv_dst[(y / 2) * W + x]     = row[x * 2 + 1];
                                    uv_dst[(y / 2) * W + x + 1] = row[x * 2 + 3];
                                }
                            }
                        }
                    }
                    ioctl(fd, VIDIOC_QBUF, &b);

                    if (imu_reader != nullptr) {
                        pkt.imu_batch = imu_reader->get_measurements_between(prev_ts_abs, cur_ts_abs);
                        for (auto& m : pkt.imu_batch) {
                            m.timestamp = std::max(0.0, m.timestamp - (cur_ts_abs - rel_ts));
                        }
                        pkt.imu_status = imu_reader->get_status_snapshot();
                    }
                    prev_ts_abs = cur_ts_abs;

                    resize_y_channel(pkt.raw_nv12.data(), config.cam_width, config.cam_height,
                                     pkt.resized_y.data(), req_width, req_height);
                    apply_gamma_lut(pkt.resized_y.data(), req_width * req_height);

                    queue_prep_to_npu_.push(std::move(pkt));
                    frame_idx++;
                }

                ioctl(fd, VIDIOC_STREAMOFF, &type);
                for (auto& mb : bufs) if (mb.start) munmap(mb.start, mb.length);
                ::close(fd);
                for (int i = 0; i < config.num_workers; ++i) {
                    PreprocessPacket eof_pkt;
                    eof_pkt.is_eof = true;
                    queue_prep_to_npu_.push(std::move(eof_pkt));
                }
                return;
            }
            ::close(fd);
        }
    }
#endif

    FILE* fp_in = fopen(config.input_path.c_str(), "rb");
    if (fp_in == nullptr) {
        printf("[Stage 1] Failed to open input stream: %s\n", config.input_path.c_str());
        for (int i = 0; i < config.num_workers; ++i) {
            PreprocessPacket eof_pkt;
            eof_pkt.is_eof = true;
            queue_prep_to_npu_.push(std::move(eof_pkt));
        }
        return;
    }

    while (true) {
        PreprocessPacket pkt;
        pkt.frame_idx = frame_idx;
        pkt.timestamp_sec = frame_idx * (1.0 / 30.0);
        pkt.raw_nv12.resize(frame_size);
        pkt.resized_y.resize(req_width * req_height);
        pkt.is_eof = false;

        size_t read_bytes = fread(pkt.raw_nv12.data(), 1, frame_size, fp_in);
        if (read_bytes < (size_t)frame_size) {
            for (int i = 0; i < config.num_workers; ++i) {
                PreprocessPacket eof_pkt;
                eof_pkt.is_eof = true;
                queue_prep_to_npu_.push(std::move(eof_pkt));
            }
            break;
        }

        auto t_now = std::chrono::steady_clock::now();
        double cur_ts_abs = std::chrono::duration<double>(t_now.time_since_epoch()).count();
        if (imu_reader != nullptr && imu_reader->get_status_snapshot().connected) {
            pkt.imu_batch = imu_reader->get_measurements_between(prev_ts_abs, cur_ts_abs);
            for (auto& m : pkt.imu_batch) {
                m.timestamp = pkt.timestamp_sec;
            }
            pkt.imu_status = imu_reader->get_status_snapshot();
        }
        prev_ts_abs = cur_ts_abs;

        // 快速缩放 Y 分量
        resize_y_channel(pkt.raw_nv12.data(), config.cam_width, config.cam_height,
                         pkt.resized_y.data(), req_width, req_height);

        // 暗部自适应增强 (激活暗光区域边缘与角点)
        apply_gamma_lut(pkt.resized_y.data(), req_width * req_height);

        queue_prep_to_npu_.push(std::move(pkt));
        frame_idx++;
    }

    fclose(fp_in);
    printf("[Stage 1] Preprocess thread finished. Total frames: %d\n", frame_idx);
}

void HfnetPipeline::thread_npu_worker(RknnWorker* worker, int req_width, int req_height) {
    int worker_id = worker->get_worker_id();

    while (true) {
        PreprocessPacket in_pkt;
        if (!queue_prep_to_npu_.pop(in_pkt) || in_pkt.is_eof) {
            InferencePacket out_pkt;
            out_pkt.worker_id = worker_id;
            out_pkt.is_eof = true;
            queue_npu_to_post_.push(std::move(out_pkt));
            break;
        }

        InferencePacket out_pkt;
        out_pkt.frame_idx = in_pkt.frame_idx;
        out_pkt.timestamp_sec = in_pkt.timestamp_sec;
        out_pkt.raw_nv12 = std::move(in_pkt.raw_nv12);
        out_pkt.resized_y = std::move(in_pkt.resized_y);
        out_pkt.imu_batch = std::move(in_pkt.imu_batch);
        out_pkt.imu_status = in_pkt.imu_status;
        out_pkt.worker_id = worker_id;
        out_pkt.is_eof = false;

        int ret = worker->run_inference(out_pkt.resized_y.data(),
                                        out_pkt.logits_raw,
                                        out_pkt.desc_raw,
                                        out_pkt.npu_time_ms);
        if (ret != RKNN_SUCC) {
            printf("[Stage 2: Worker %d] Inference failed at frame %d!\n", worker_id, in_pkt.frame_idx);
            break;
        }

        queue_npu_to_post_.push(std::move(out_pkt));
    }

    printf("[Stage 2: Worker %d] Finished.\n", worker_id);
}

void HfnetPipeline::thread_postprocess(const PipelineConfig& config, const RknnWorker* worker_meta) {
    FILE* fp_out = fopen(config.output_path.c_str(), "wb");
    if (fp_out == nullptr) {
        printf("[Stage 3] Failed to open output stream: %s\n", config.output_path.c_str());
        return;
    }

    int req_width = worker_meta->get_req_width();
    int req_height = worker_meta->get_req_height();
    bool logits_is_nchw = worker_meta->is_logits_nchw();
    bool desc_is_nchw = worker_meta->is_desc_nchw();
    float logits_scale = worker_meta->get_logits_scale();
    int32_t logits_zp = worker_meta->get_logits_zp();

    // 初始化 C++ 视觉惯性 SLAM 系统组件 (Camera + SlamMap + LocalMapper + Tracker + BlackboxLogger)
    slam::StereoCamera camera = slam::StereoCamera::from_calib_json(config.calib_json, req_width, req_height);
    slam::SlamMap slam_map;
    slam::LocalMapper local_mapper(&slam_map);
    slam::Tracker tracker(camera, &slam_map, &local_mapper);
    slam::BlackboxLogger blackbox(config.blackbox_dir, "rk3566_cpp_mono_imu");
    printf("  [Blackbox] C++ 实时黑盒记录已启动 -> %s\n", blackbox.get_run_dir().c_str());

    std::vector<slam::Vec3> traj_history;
    int expected_frame_idx = 0;
    int eof_workers_count = 0;
    std::map<int, InferencePacket> reorder_map;
    std::vector<float> scores_dense(req_height * req_width);
    auto t_start = std::chrono::high_resolution_clock::now();

    while (true) {
        InferencePacket pkt;
        if (!queue_npu_to_post_.pop(pkt)) {
            break;
        }

        if (pkt.is_eof) {
            eof_workers_count++;
            if (eof_workers_count >= config.num_workers) {
                break;
            }
            continue;
        }

        reorder_map[pkt.frame_idx] = std::move(pkt);

        // 严格按帧序列输出并推进 SLAM 状态机
        while (reorder_map.find(expected_frame_idx) != reorder_map.end()) {
            InferencePacket cur_pkt = std::move(reorder_map[expected_frame_idx]);
            reorder_map.erase(expected_frame_idx);

            auto t0 = std::chrono::high_resolution_clock::now();

            // 1. 计算 Softmax + DepthToSpace
            compute_scores_dense_from_logits(cur_pkt.logits_raw.data(), logits_scale, logits_zp,
                                             scores_dense.data(), req_height, req_width, logits_is_nchw);

            // 2. 提取特征点 (网格轮询均匀化分配)
            std::vector<Keypoint> kpts = extract_keypoints(scores_dense.data(), req_height, req_width,
                                                           config.max_kpts, config.min_score,
                                                           config.nms_radius, config.grid_rows, config.grid_cols);

            // 3. 描述子采样 (256维 L2归一化)
            int H_desc = 60, W_desc = 80, C_desc = 256;
            float factor_x = (float)(req_width - 1) / (W_desc - 1);
            float factor_y = (float)(req_height - 1) / (H_desc - 1);
            for (size_t i = 0; i < kpts.size(); ++i) {
                float x_desc = kpts[i].x / factor_x;
                float y_desc = kpts[i].y / factor_y;
                kpts[i].descriptor = sample_descriptor_int8(cur_pkt.desc_raw.data(), x_desc, y_desc,
                                                           H_desc, W_desc, C_desc, desc_is_nchw);
            }

            // 4. 执行 C++ 视觉惯性 SLAM 跟踪与建图 (金字塔光流 + HF-Net 特征融合 + 姿态优化 + 局部 BA)
            slam::SE3 T_cw = tracker.grab_image_monocular_inertial(
                cur_pkt.resized_y,
                scores_dense.data(),
                cur_pkt.desc_raw.data(),
                desc_is_nchw,
                kpts,
                cur_pkt.timestamp_sec,
                cur_pkt.imu_batch
            );
            slam::SE3 T_wc = T_cw.inverse();
            slam::Vec3 pos_w = T_wc.t;
            slam::Vec3 cam_rpy = slam::SO3::R_to_body_euler_deg(T_wc.R);
            if (tracker.state == slam::TrackingState::OK) {
                traj_history.push_back(pos_w);
            }

            // 5. 记录黑盒日志 (frames.csv, events.jsonl, summary.json, report.md)
            blackbox.log_frame(cur_pkt.frame_idx, cur_pkt.timestamp_sec, tracker, slam_map,
                               (int)cur_pkt.imu_batch.size(), cur_pkt.imu_status);

            // 6. NV12 直接绘制:
            //    - 黄色圆点: HF-Net 候选特征点
            //    - 翠绿圆点 + 青色光流运动线: 当前帧成功跟踪的 3D 地图点内点
            //    - 右上角 2D 俯视轨迹与 3D 地图点雷达小地图 HUD
            float scale_x = (float)config.cam_width / req_width;
            float scale_y = (float)config.cam_height / req_height;
            for (const auto& kp : kpts) {
                int draw_x = (int)(kp.x * scale_x);
                int draw_y = (int)(kp.y * scale_y);
                draw_circle_nv12(cur_pkt.raw_nv12.data(), config.cam_width, config.cam_height, draw_x, draw_y, 2, 210, 16, 146);
            }
            for (size_t i = 0; i < tracker.last_tracked_curr_pts.size(); ++i) {
                int x1 = (int)(tracker.last_tracked_curr_pts[i].x * scale_x);
                int y1 = (int)(tracker.last_tracked_curr_pts[i].y * scale_y);
                int x0 = (int)(tracker.last_tracked_prev_pts[i].x * scale_x);
                int y0 = (int)(tracker.last_tracked_prev_pts[i].y * scale_y);
                draw_line_nv12(cur_pkt.raw_nv12.data(), config.cam_width, config.cam_height, x0, y0, x1, y1, 180, 170, 30);
                draw_circle_nv12(cur_pkt.raw_nv12.data(), config.cam_width, config.cam_height, x1, y1, 3, 195, 54, 34);
            }

            std::vector<slam::Vec3> active_mp_coords;
            auto all_mps = slam_map.get_all_mappoints();
            active_mp_coords.reserve(all_mps.size());
            for (const auto& mp : all_mps) {
                active_mp_coords.push_back(mp->pos_w);
            }
            draw_slam_hud_nv12(cur_pkt.raw_nv12.data(), config.cam_width, config.cam_height,
                               traj_history, active_mp_coords, tracker.last_diag.trans_gate < 0.35);

            // 7. 写入渲染后的 NV12 视频流
            fwrite(cur_pkt.raw_nv12.data(), 1, cur_pkt.raw_nv12.size(), fp_out);

            auto t1 = std::chrono::high_resolution_clock::now();
            double post_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

            expected_frame_idx++;
            if (expected_frame_idx % 15 == 0 || expected_frame_idx <= 5 || tracker.last_is_keyframe) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double total_fps = expected_frame_idx / std::chrono::duration<double>(t_now - t_start).count();
                printf("[SLAM Frame %4d | W%d | %-15s] KFs=%2d | MPs=%4d | Inliers=%3d | Pos=(%+.3f,%+.3f,%+.3f)m | RPY=(%+.1f,%+.1f,%+.1f)deg | Gate=%.2f | NPU=%.1fms | Post+SLAM=%.1fms | FPS=\033[1;32m%.1f\033[0m\n",
                       cur_pkt.frame_idx, cur_pkt.worker_id,
                       slam::Tracker::state_to_string(tracker.state),
                       slam_map.num_keyframes(), slam_map.num_mappoints(), tracker.last_inliers,
                       pos_w.x, pos_w.y, pos_w.z,
                       cam_rpy.x, cam_rpy.y, cam_rpy.z,
                       tracker.last_diag.trans_gate,
                       cur_pkt.npu_time_ms, post_time, total_fps);
            }
        }
    }

    blackbox.close();
    fclose(fp_out);
    printf("[Stage 3] SLAM & Postprocess thread finished. Total frames: %d | KFs: %d | MPs: %d\n",
           expected_frame_idx, slam_map.num_keyframes(), slam_map.num_mappoints());
    printf("          Blackbox saved to: %s\n", blackbox.get_run_dir().c_str());
}

int HfnetPipeline::run(const PipelineConfig& config) {
    // 1. 初始化 Gamma 暗光自适应增强映射表
    init_gamma_lut(config.gamma);

    // 2. 启动串口 IMU (若已接入则自动锁定端口并开启 20Hz 倍频采集)
    slam::WitMotionSerialImu imu_reader(config.imu_port, config.imu_baud);
    if (imu_reader.start()) {
        auto st = imu_reader.get_status_snapshot();
        printf("  [IMU   ] 已连接串口 %s @ %d bps | 初始加速度 |g|=%.3f m/s^2\n",
               st.port.c_str(), st.baud, st.acc_mps2.norm());
    } else {
        printf("  [IMU   ] 未检测到串口 IMU (%s)，将以纯视觉 + 消旋运动门控模式运行\n",
               config.imu_port.c_str());
    }

    // 3. 加载模型二进制
    int model_len = 0;
    unsigned char* model_data = load_model_file(config.model_path.c_str(), &model_len);
    if (model_data == nullptr) return -1;

    printf("========================================================================\n");
    printf("--> Initializing Multi-Context RKNN Environment (%d Parallel Workers)\n", config.num_workers);
    printf("========================================================================\n");

    std::vector<std::unique_ptr<RknnWorker>> workers;
    workers.reserve(config.num_workers);

    for (int w = 0; w < config.num_workers; ++w) {
        auto worker = std::unique_ptr<RknnWorker>(new RknnWorker());
        bool dump = (w == 0);
        int ret = worker->init(w, model_data, model_len, dump);
        if (ret != RKNN_SUCC) {
            printf("Failed to init RknnWorker %d\n", w);
            free(model_data);
            return -1;
        }
        workers.push_back(std::move(worker));
    }
    free(model_data);

    int req_width = workers[0]->get_req_width();
    int req_height = workers[0]->get_req_height();
    printf("Model input: %d x %d (Height x Width)\n", req_height, req_width);

    printf("========================================================================\n");
    printf("--> Starting Full RK3566 HF-Net Visual-Inertial SLAM Pipeline:\n");
    printf("    Stage 1: Async Preprocess & V4L2/NV12 + WitMotion IMU Capture\n");
    printf("    Stage 2: %dx Parallel NPU Workers (Multi-Context Ping-Pong)\n", config.num_workers);
    printf("    Stage 3: Hybrid Optical Flow + HF-Net SLAM Tracker, Schur BA & NV12 HUD\n");
    printf("========================================================================\n");

    auto t_start = std::chrono::high_resolution_clock::now();

    // 启动线程
    std::thread t_prep(&HfnetPipeline::thread_preprocess, this, std::cref(config), req_width, req_height, &imu_reader);

    std::vector<std::thread> npu_threads;
    for (int w = 0; w < config.num_workers; ++w) {
        npu_threads.emplace_back(&HfnetPipeline::thread_npu_worker, this, workers[w].get(), req_width, req_height);
    }

    std::thread t_post(&HfnetPipeline::thread_postprocess, this, std::cref(config), workers[0].get());

    t_prep.join();
    for (auto& t : npu_threads) {
        t.join();
    }
    t_post.join();
    imu_reader.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time_sec = std::chrono::duration<double>(t_end - t_start).count();

    printf("========================================================================\n");
    printf("SLAM Pipeline completed successfully in %.2f seconds!\n", total_time_sec);
    printf("========================================================================\n");

    return 0;
}
