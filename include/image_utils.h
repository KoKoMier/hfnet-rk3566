#ifndef HFNET_IMAGE_UTILS_H
#define HFNET_IMAGE_UTILS_H

#include <cstdint>
#include <cstddef>

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

#endif // HFNET_IMAGE_UTILS_H
