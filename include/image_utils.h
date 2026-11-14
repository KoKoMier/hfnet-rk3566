#ifndef HFNET_IMAGE_UTILS_H
#define HFNET_IMAGE_UTILS_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include "slam_math.h"

// 初始化 Gamma 暗光自适应增强映射表
void init_gamma_lut(float gamma = 0.65f);

// 对内存区域应用 Gamma 映射
void apply_gamma_lut(uint8_t* data, size_t size);

// Y 分量缩放（基于 stb_image_resize）
void resize_y_channel(const uint8_t* src, int src_w, int src_h,
                      uint8_t* dst, int dst_w, int dst_h);

// NV12 直接绘制高对比度彩色点（高亮荧光黄+白核，耗时 < 0.05ms，在明暗背景下均清晰醒目）
void draw_circle_nv12(uint8_t* nv12, int W, int H, int cx, int cy, int radius = 3,
                      uint8_t y_val = 226, uint8_t u_val = 0, uint8_t v_val = 148);

// NV12 直接绘制 Bresenham 彩色线段（用于绘制光流运动尾迹与轨迹连线）
void draw_line_nv12(uint8_t* nv12, int W, int H, int x0, int y0, int x1, int y1,
                    uint8_t y_val = 180, uint8_t u_val = 44, uint8_t v_val = 21);

// NV12 右上角直接绘制 2D 俯视轨迹与局部 3D 地图点雷达小地图 (HUD Overlay)
void draw_slam_hud_nv12(uint8_t* nv12, int W, int H,
                        const std::vector<slam::Vec3>& traj_history,
                        const std::vector<slam::Vec3>& map_points_w,
                        bool is_Pure_Rotation_Gated);

#endif // HFNET_IMAGE_UTILS_H
