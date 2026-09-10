#include "image_utils.h"

#include <cmath>
#include <algorithm>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

// 预生成 Gamma 暗光自适应增强映射表
static uint8_t g_gamma_lut[256];

void init_gamma_lut(float gamma) {
    for (int i = 0; i < 256; ++i) {
        float norm = (float)i / 255.0f;
        float enhanced = powf(norm, gamma);
        int val = (int)(enhanced * 255.0f + 0.5f);
        g_gamma_lut[i] = (uint8_t)std::min(255, std::max(0, val));
    }
}

void apply_gamma_lut(uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = g_gamma_lut[data[i]];
    }
}

void resize_y_channel(const uint8_t* src, int src_w, int src_h,
                      uint8_t* dst, int dst_w, int dst_h) {
    stbir_resize_uint8(src, src_w, src_h, 0,
                       dst, dst_w, dst_h, 0, 1);
}

void draw_circle_nv12(uint8_t* nv12, int W, int H, int cx, int cy, int radius,
                      uint8_t y_val, uint8_t u_val, uint8_t v_val) {
    uint8_t* y_plane = nv12;
    uint8_t* uv_plane = nv12 + W * H;

    int r2 = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy) {
        int py = cy + dy;
        if (py < 0 || py >= H) continue;
        int uv_row = (py / 2) * W;
        for (int dx = -radius; dx <= radius; ++dx) {
            int px = cx + dx;
            if (px < 0 || px >= W) continue;
            int dist2 = dx * dx + dy * dy;
            if (dist2 <= r2) {
                int uv_idx = uv_row + (px / 2) * 2;
                if (dist2 <= 1) {
                    // 中心核采用纯白 (Y=255)，精确定位
                    y_plane[py * W + px] = 255;
                    uv_plane[uv_idx]     = 128;
                    uv_plane[uv_idx + 1] = 128;
                } else {
                    // 外环采用高亮荧光黄 (Y=226, U=0, V=148)
                    y_plane[py * W + px] = y_val;
                    uv_plane[uv_idx]     = u_val;
                    uv_plane[uv_idx + 1] = v_val;
                }
            }
        }
    }
}
