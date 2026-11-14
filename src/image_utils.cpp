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
                    y_plane[py * W + px] = 255;
                    uv_plane[uv_idx]     = 128;
                    uv_plane[uv_idx + 1] = 128;
                } else {
                    y_plane[py * W + px] = y_val;
                    uv_plane[uv_idx]     = u_val;
                    uv_plane[uv_idx + 1] = v_val;
                }
            }
        }
    }
}

void draw_line_nv12(uint8_t* nv12, int W, int H, int x0, int y0, int x1, int y1,
                    uint8_t y_val, uint8_t u_val, uint8_t v_val) {
    uint8_t* y_plane = nv12;
    uint8_t* uv_plane = nv12 + W * H;

    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
            y_plane[y0 * W + x0] = y_val;
            int uv_idx = (y0 / 2) * W + (x0 / 2) * 2;
            uv_plane[uv_idx]     = u_val;
            uv_plane[uv_idx + 1] = v_val;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void draw_slam_hud_nv12(uint8_t* nv12, int W, int H,
                        const std::vector<slam::Vec3>& traj_history,
                        const std::vector<slam::Vec3>& map_points_w,
                        bool is_pure_rotation_gated) {
    if (W < 240 || H < 180) return;

    const int box_w = 156;
    const int box_h = 156;
    const int x_left = W - box_w - 10;
    const int y_top = 10;

    uint8_t* y_plane = nv12;
    uint8_t* uv_plane = nv12 + W * H;

    // 1. 半透明暗化右上角雷达小地图区域并绘制边框 (纯旋转门控激活时边框显示橙色，正常跟踪时显示青色)
    uint8_t border_y = is_pure_rotation_gated ? 170 : 185;
    uint8_t border_u = is_pure_rotation_gated ? 30  : 170;
    uint8_t border_v = is_pure_rotation_gated ? 220 : 30;

    for (int py = y_top; py < y_top + box_h; ++py) {
        for (int px = x_left; px < x_left + box_w; ++px) {
            bool is_border = (py == y_top || py == y_top + box_h - 1 ||
                              px == x_left || px == x_left + box_w - 1);
            if (is_border) {
                y_plane[py * W + px] = border_y;
                int uv_idx = (py / 2) * W + (px / 2) * 2;
                uv_plane[uv_idx] = border_u;
                uv_plane[uv_idx + 1] = border_v;
            } else {
                y_plane[py * W + px] = (uint8_t)(y_plane[py * W + px] / 3 + 12);
            }
        }
    }

    // 2. 自适应计算 X-Z 俯视平面坐标缩放比例
    double min_x = -0.35, max_x = 0.35;
    double min_z = -0.35, max_z = 0.35;
    for (const auto& p : traj_history) {
        min_x = std::min(min_x, p.x - 0.15);
        max_x = std::max(max_x, p.x + 0.15);
        min_z = std::min(min_z, p.z - 0.15);
        max_z = std::max(max_z, p.z + 0.15);
    }
    double span = std::max(0.50, std::max(max_x - min_x, max_z - min_z));
    double cx_w = 0.5 * (min_x + max_x);
    double cz_w = 0.5 * (min_z + max_z);

    auto world_to_hud = [&](double wx, double wz, int& hx, int& hy) {
        double nx = (wx - cx_w) / span + 0.5;
        double nz = 0.5 - (wz - cz_w) / span;
        hx = x_left + 12 + (int)(nx * (box_w - 24));
        hy = y_top + 12 + (int)(nz * (box_h - 24));
        hx = std::max(x_left + 2, std::min(x_left + box_w - 3, hx));
        hy = std::max(y_top + 2, std::min(y_top + box_h - 3, hy));
    };

    // 3. 绘制局部地图点云 (浅蓝色点)
    size_t mp_step = std::max<size_t>(1, map_points_w.size() / 200);
    for (size_t i = 0; i < map_points_w.size(); i += mp_step) {
        int hx, hy;
        world_to_hud(map_points_w[i].x, map_points_w[i].z, hx, hy);
        y_plane[hy * W + hx] = 150;
        int uv_idx = (hy / 2) * W + (hx / 2) * 2;
        uv_plane[uv_idx] = 190;
        uv_plane[uv_idx + 1] = 90;
    }

    // 4. 绘制历史轨迹连线 (翠绿色) 与当前相机位置 (亮红/白核圆点)
    for (size_t i = 1; i < traj_history.size(); ++i) {
        int x0, y0, x1, y1;
        world_to_hud(traj_history[i - 1].x, traj_history[i - 1].z, x0, y0);
        world_to_hud(traj_history[i].x, traj_history[i].z, x1, y1);
        draw_line_nv12(nv12, W, H, x0, y0, x1, y1, 195, 54, 34);
    }
    if (!traj_history.empty()) {
        int cur_x, cur_y;
        world_to_hud(traj_history.back().x, traj_history.back().z, cur_x, cur_y);
        draw_circle_nv12(nv12, W, H, cur_x, cur_y, 3, 180, 90, 240);
    }
}
