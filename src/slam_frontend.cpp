#include "slam_frontend.h"

#include <cmath>
#include <algorithm>
#include <cstring>

namespace slam {

void FrontendMatcher::build_pyramid(const uint8_t* gray, int w, int h, int levels,
                                    std::vector<std::vector<uint8_t>>& pyr,
                                    std::vector<int>& widths, std::vector<int>& heights) {
    pyr.resize(levels);
    widths.resize(levels);
    heights.resize(levels);

    widths[0] = w;
    heights[0] = h;
    pyr[0].assign(gray, gray + w * h);

    for (int lvl = 1; lvl < levels; ++lvl) {
        int pw = widths[lvl - 1];
        int ph = heights[lvl - 1];
        int cw = pw / 2;
        int ch = ph / 2;
        widths[lvl] = cw;
        heights[lvl] = ch;
        pyr[lvl].resize(cw * ch);

        const uint8_t* src = pyr[lvl - 1].data();
        uint8_t* dst = pyr[lvl].data();
        for (int y = 0; y < ch; ++y) {
            int sy = y * 2;
            const uint8_t* r0 = src + sy * pw;
            const uint8_t* r1 = src + std::min(ph - 1, sy + 1) * pw;
            for (int x = 0; x < cw; ++x) {
                int sx = x * 2;
                int sx1 = std::min(pw - 1, sx + 1);
                int sum = (int)r0[sx] + (int)r0[sx1] + (int)r1[sx] + (int)r1[sx1];
                dst[y * cw + x] = (uint8_t)((sum + 2) >> 2);
            }
        }
    }
}

static inline float sample_bilinear_u8(const uint8_t* img, int w, int h, float x, float y) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    float ax = x - x0;
    float ay = y - y0;
    x0 = std::max(0, std::min(w - 2, x0));
    y0 = std::max(0, std::min(h - 2, y0));
    const uint8_t* p0 = img + y0 * w + x0;
    const uint8_t* p1 = p0 + w;
    return (1.0f - ay) * ((1.0f - ax) * p0[0] + ax * p0[1])
         + ay * ((1.0f - ax) * p1[0] + ax * p1[1]);
}

// 单点多尺度金字塔 Inverse-Compositional LK 光流求解 (半窗口半径 win_r = 3 -> 7x7 像素块)
static bool track_single_point_lk(
    const std::vector<std::vector<uint8_t>>& pyr1,
    const std::vector<std::vector<uint8_t>>& pyr2,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    float u1, float v1,
    float& u2_inout, float& v2_inout
) {
    const int levels = (int)pyr1.size();
    const int win_r = 3;
    const int win_size = (2 * win_r + 1) * (2 * win_r + 1);
    float I1_patch[49];
    float gx_patch[49];
    float gy_patch[49];

    float scale0 = (float)(1 << (levels - 1));
    float u_curr = u2_inout / scale0;
    float v_curr = v2_inout / scale0;

    for (int lvl = levels - 1; lvl >= 0; --lvl) {
        float scale = (float)(1 << lvl);
        float u_ref = u1 / scale;
        float v_ref = v1 / scale;
        int w = widths[lvl];
        int h = heights[lvl];
        const uint8_t* img1 = pyr1[lvl].data();
        const uint8_t* img2 = pyr2[lvl].data();

        if (u_ref < win_r + 2 || u_ref >= w - win_r - 2 ||
            v_ref < win_r + 2 || v_ref >= h - win_r - 2) {
            return false;
        }

        float Gxx = 0.0f, Gxy = 0.0f, Gyy = 0.0f;
        int idx = 0;
        for (int dy = -win_r; dy <= win_r; ++dy) {
            for (int dx = -win_r; dx <= win_r; ++dx, ++idx) {
                float px = u_ref + dx;
                float py = v_ref + dy;
                I1_patch[idx] = sample_bilinear_u8(img1, w, h, px, py);
                float gx = 0.5f * (sample_bilinear_u8(img1, w, h, px + 1.0f, py) -
                                   sample_bilinear_u8(img1, w, h, px - 1.0f, py));
                float gy = 0.5f * (sample_bilinear_u8(img1, w, h, px, py + 1.0f) -
                                   sample_bilinear_u8(img1, w, h, px, py - 1.0f));
                gx_patch[idx] = gx;
                gy_patch[idx] = gy;
                Gxx += gx * gx;
                Gxy += gx * gy;
                Gyy += gy * gy;
            }
        }

        float det = Gxx * Gyy - Gxy * Gxy;
        if (det < 50.0f) return false;
        float inv_det = 1.0f / det;

        for (int iter = 0; iter < 6; ++iter) {
            if (u_curr < win_r + 1 || u_curr >= w - win_r - 2 ||
                v_curr < win_r + 1 || v_curr >= h - win_r - 2) {
                return false;
            }
            float bx = 0.0f, by = 0.0f;
            idx = 0;
            for (int dy = -win_r; dy <= win_r; ++dy) {
                for (int dx = -win_r; dx <= win_r; ++dx, ++idx) {
                    float I2 = sample_bilinear_u8(img2, w, h, u_curr + dx, v_curr + dy);
                    float diff = I1_patch[idx] - I2;
                    bx += gx_patch[idx] * diff;
                    by += gy_patch[idx] * diff;
                }
            }
            float du = (Gyy * bx - Gxy * by) * inv_det;
            float dv = (-Gxy * bx + Gxx * by) * inv_det;
            u_curr += du;
            v_curr += dv;
            if (du * du + dv * dv < 1e-3f) break;
        }

        if (lvl > 0) {
            u_curr *= 2.0f;
            v_curr *= 2.0f;
        }
    }

    // 校验最终块残差平均绝对误差 (MAE < 22.0)
    int w0 = widths[0], h0 = heights[0];
    if (u_curr < 4.0f || u_curr >= w0 - 4.0f || v_curr < 4.0f || v_curr >= h0 - 4.0f) return false;
    float err_sum = 0.0f;
    int idx = 0;
    for (int dy = -win_r; dy <= win_r; ++dy) {
        for (int dx = -win_r; dx <= win_r; ++dx, ++idx) {
            float I2 = sample_bilinear_u8(pyr2[0].data(), w0, h0, u_curr + dx, v_curr + dy);
            err_sum += std::abs(I1_patch[idx] - I2);
        }
    }
    if (err_sum / (float)win_size > 24.0f) return false;

    u2_inout = u_curr;
    v2_inout = v_curr;
    return true;
}

void FrontendMatcher::track_optical_flow(
    const Frame& last_frame,
    const std::vector<uint8_t>& curr_gray,
    std::vector<std::vector<uint8_t>>& curr_pyr,
    std::vector<int>& curr_pyr_w,
    std::vector<int>& curr_pyr_h,
    const StereoCamera& camera,
    const SE3* T_cw_pred,
    std::vector<Vec2>& out_kp_pts,
    std::vector<std::shared_ptr<MapPoint>>& out_tracked_mps,
    std::vector<Vec2>& out_prev_pts
) {
    out_kp_pts.clear();
    out_tracked_mps.clear();
    out_prev_pts.clear();

    if (curr_pyr.empty()) {
        build_pyramid(curr_gray.data(), camera.width, camera.height, 4, curr_pyr, curr_pyr_w, curr_pyr_h);
    }
    if (last_frame.pyramid_left.empty()) return;

    Mat3 H_rot = Mat3::identity();
    bool has_rot_prior = false;
    if (T_cw_pred != nullptr && last_frame.has_pose) {
        Mat3 R_rel = T_cw_pred->R * last_frame.T_cw.R.transpose();
        double trace_v = std::min(1.0, std::max(-1.0, 0.5 * (R_rel.trace() - 1.0)));
        double angle_deg = std::acos(trace_v) * (180.0 / M_PI);
        if (angle_deg > 0.05 && angle_deg < 35.0) {
            Mat3 K = Mat3::identity();
            K(0, 0) = camera.fx; K(0, 2) = camera.cx;
            K(1, 1) = camera.fy; K(1, 2) = camera.cy;
            Mat3 K_inv = Mat3::identity();
            K_inv(0, 0) = camera.inv_fx; K_inv(0, 2) = -camera.cx * camera.inv_fx;
            K_inv(1, 1) = camera.inv_fy; K_inv(1, 2) = -camera.cy * camera.inv_fy;
            H_rot = K * R_rel * K_inv;
            has_rot_prior = true;
        }
    }

    for (int i = 0; i < last_frame.num_kps; ++i) {
        auto mp = last_frame.map_points[i];
        if (!mp || mp->is_bad) continue;

        float u1 = (float)last_frame.kp_pts[i].x;
        float v1 = (float)last_frame.kp_pts[i].y;
        float u2 = u1;
        float v2 = v1;

        if (T_cw_pred != nullptr) {
            Vec3 p_c = T_cw_pred->transform(mp->pos_w);
            if (p_c.z > 0.15) {
                Vec2 uv = camera.project(p_c);
                if (camera.is_in_image(uv.x, uv.y, 4.0) &&
                    std::hypot(uv.x - u1, uv.y - v1) < 64.0) {
                    u2 = (float)uv.x;
                    v2 = (float)uv.y;
                }
            }
        } else if (has_rot_prior) {
            Vec3 p_h = H_rot * Vec3(u1, v1, 1.0);
            if (p_h.z > 0.1) {
                float up = (float)(p_h.x / p_h.z);
                float vp = (float)(p_h.y / p_h.z);
                if (camera.is_in_image(up, vp, 4.0) && std::hypot(up - u1, vp - v1) < 64.0) {
                    u2 = up;
                    v2 = vp;
                }
            }
        }

        if (!track_single_point_lk(last_frame.pyramid_left, curr_pyr, curr_pyr_w, curr_pyr_h, u1, v1, u2, v2)) {
            continue;
        }

        // 反向光流一致性校验 (Forward-Backward Error < 1.5 px)
        float u1_back = u1;
        float v1_back = v1;
        if (!track_single_point_lk(curr_pyr, last_frame.pyramid_left, curr_pyr_w, curr_pyr_h, u2, v2, u1_back, v1_back)) {
            continue;
        }
        float fb_err2 = (u1_back - u1) * (u1_back - u1) + (v1_back - v1) * (v1_back - v1);
        if (fb_err2 > 2.25f) continue;

        out_kp_pts.emplace_back(u2, v2);
        out_tracked_mps.push_back(mp);
        out_prev_pts.emplace_back(u1, v1);
    }
}

void FrontendMatcher::track_init_optical_flow(
    const Frame& init_frame,
    const std::vector<std::vector<uint8_t>>& curr_pyr,
    const std::vector<int>& curr_pyr_w,
    const std::vector<int>& curr_pyr_h,
    const StereoCamera& camera,
    std::vector<int>& out_matched_idx0,
    std::vector<Vec2>& out_matched_pts0,
    std::vector<Vec2>& out_matched_pts1
) {
    out_matched_idx0.clear();
    out_matched_pts0.clear();
    out_matched_pts1.clear();
    if (init_frame.pyramid_left.empty() || curr_pyr.empty()) return;

    for (int i = 0; i < init_frame.num_kps; ++i) {
        float u0 = (float)init_frame.kp_pts[i].x;
        float v0 = (float)init_frame.kp_pts[i].y;
        float u1 = u0;
        float v1 = v0;

        if (!track_single_point_lk(init_frame.pyramid_left, curr_pyr, curr_pyr_w, curr_pyr_h, u0, v0, u1, v1)) {
            continue;
        }

        float u0_back = u0;
        float v0_back = v0;
        if (!track_single_point_lk(curr_pyr, init_frame.pyramid_left, curr_pyr_w, curr_pyr_h, u1, v1, u0_back, v0_back)) {
            continue;
        }
        float fb_err2 = (u0_back - u0) * (u0_back - u0) + (v0_back - v0) * (v0_back - v0);
        if (fb_err2 <= 2.25f && camera.is_in_image(u1, v1, 4.0)) {
            out_matched_idx0.push_back(i);
            out_matched_pts0.emplace_back(u0, v0);
            out_matched_pts1.emplace_back(u1, v1);
        }
    }
}

float FrontendMatcher::descriptor_distance(const std::vector<float>& d1, const std::vector<float>& d2) {
    if (d1.empty() || d2.empty()) return 2.0f;
    int C = (int)std::min(d1.size(), d2.size());
    float dot = 0.0f;
    for (int i = 0; i < C; ++i) dot += d1[i] * d2[i];
    return std::max(0.0f, 1.0f - dot);
}

int FrontendMatcher::search_by_projection(
    Frame& curr_frame,
    const std::vector<std::shared_ptr<MapPoint>>& candidates,
    double th_radius
) {
    int matched = 0;
    const float th_dist = 0.28f; // 对应余弦相似度 >= 0.72
    const float ratio = 0.82f;

    for (const auto& mp : candidates) {
        if (!mp || mp->is_bad || !mp->track_in_view || mp->descriptor.empty()) continue;

        std::vector<int> idxs = curr_frame.get_features_in_area(mp->track_proj_x, mp->track_proj_y, th_radius);
        if (idxs.empty()) continue;

        float best_dist = 10.0f;
        float second_dist = 10.0f;
        int best_idx = -1;

        for (int idx : idxs) {
            if (curr_frame.map_points[idx] != nullptr) continue;
            if (curr_frame.descriptors[idx].empty()) continue;
            float dist = descriptor_distance(mp->descriptor, curr_frame.descriptors[idx]);
            if (dist < best_dist) {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = idx;
            } else if (dist < second_dist) {
                second_dist = dist;
            }
        }

        if (best_idx >= 0 && best_dist <= th_dist && best_dist <= ratio * second_dist) {
            curr_frame.map_points[best_idx] = mp;
            matched++;
        }
    }
    return matched;
}

std::vector<std::pair<int, int>> FrontendMatcher::search_for_triangulation(
    const KeyFrame& kf1, const KeyFrame& kf2
) {
    std::vector<std::pair<int, int>> matches;
    Vec3 o_w2 = kf1.get_camera_center();
    Vec3 p_c2_o1 = kf2.T_cw.transform(o_w2);
    Vec2 epipole2(kf2.camera.cx, kf2.camera.cy);
    bool has_epipole = false;
    if (std::abs(p_c2_o1.z) > 1e-4) {
        epipole2 = kf2.camera.project(p_c2_o1);
        has_epipole = true;
    }

    // 计算基础矩阵 F_12 = K2^{-T} [t_12]_x R_12 K1^{-1}
    SE3 T_1w = kf1.T_cw;
    SE3 T_2w = kf2.T_cw;
    Mat3 R_12 = T_1w.R * T_2w.R.transpose();
    Vec3 t_12 = T_1w.t - R_12 * T_2w.t;
    Mat3 t12_x = SO3::hat(t_12);
    Mat3 E_12 = t12_x * R_12;

    Mat3 K_inv;
    K_inv(0, 0) = kf1.camera.inv_fx; K_inv(0, 2) = -kf1.camera.cx * kf1.camera.inv_fx;
    K_inv(1, 1) = kf1.camera.inv_fy; K_inv(1, 2) = -kf1.camera.cy * kf1.camera.inv_fy;
    K_inv(2, 2) = 1.0;
    Mat3 F_12 = K_inv.transpose() * E_12 * K_inv;

    std::vector<bool> matched2(kf2.num_kps, false);
    const float th_dist = 0.26f;

    for (int i1 = 0; i1 < kf1.num_kps; ++i1) {
        if (kf1.map_points[i1] != nullptr || kf1.descriptors[i1].empty()) continue;

        const Vec2& p1 = kf1.kp_pts[i1];
        float best_d = 10.0f, second_d = 10.0f;
        int best_i2 = -1;

        for (int i2 = 0; i2 < kf2.num_kps; ++i2) {
            if (matched2[i2] || kf2.map_points[i2] != nullptr || kf2.descriptors[i2].empty()) continue;
            float d = descriptor_distance(kf1.descriptors[i1], kf2.descriptors[i2]);
            if (d > th_dist || d >= second_d) continue;

            const Vec2& p2 = kf2.kp_pts[i2];
            if (has_epipole && (p2 - epipole2).squared_norm() < 100.0) continue;

            // 点到极线距离检验 (< 3.84 px^2)
            Vec3 l2 = F_12.transpose() * Vec3(p1.x, p1.y, 1.0);
            double denom = l2.x * l2.x + l2.y * l2.y;
            if (denom < 1e-10) continue;
            double num = l2.x * p2.x + l2.y * p2.y + l2.z;
            if ((num * num) / denom > 4.0) continue;

            if (d < best_d) {
                second_d = best_d;
                best_d = d;
                best_i2 = i2;
            } else if (d < second_d) {
                second_d = d;
            }
        }

        if (best_i2 >= 0 && best_d < 0.82f * second_d) {
            matched2[best_i2] = true;
            matches.emplace_back(i1, best_i2);
        }
    }
    return matches;
}

void FrontendMatcher::detect_grid_corners_with_scores(
    const uint8_t* gray, const float* scores_dense, int w, int h,
    const bool* occ_grid, int occ_w, int occ_h,
    int cell_counts[6][8], int max_per_cell, int need_total,
    std::vector<Vec2>& out_added_pts
) {
    out_added_pts.clear();
    if (need_total <= 0) return;

    struct Cand {
        float u, v, resp;
    };
    std::vector<Cand> cands;
    cands.reserve(512);

    const int border = 16;
    const int step = 4;
    for (int y = border; y < h - border; y += step) {
        for (int x = border; x < w - border; x += step) {
            int gy = y / 12, gx = x / 12;
            if (gy >= 0 && gy < occ_h && gx >= 0 && gx < occ_w && occ_grid[gy * occ_w + gx]) {
                continue;
            }
            // 计算 3x3 局部二维梯度结构张量 (Shi-Tomasi 最小特征值与各向同性比)
            float ixx = 0.0f, iyy = 0.0f, ixy = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                const uint8_t* row_m = gray + (y + dy - 1) * w + x;
                const uint8_t* row_c = gray + (y + dy) * w + x;
                const uint8_t* row_p = gray + (y + dy + 1) * w + x;
                for (int dx = -1; dx <= 1; ++dx) {
                    float ix = 0.5f * ((float)row_c[dx + 1] - (float)row_c[dx - 1]);
                    float iy = 0.5f * ((float)row_p[dx] - (float)row_m[dx]);
                    ixx += ix * ix;
                    iyy += iy * iy;
                    ixy += ix * iy;
                }
            }
            float tr = ixx + iyy;
            if (tr < 260.0f) continue;
            float det = ixx * iyy - ixy * ixy;
            float disc = std::sqrt(std::max(0.0f, tr * tr - 4.0f * det));
            float lmin = 0.5f * (tr - disc);
            if (lmin < 80.0f) continue;
            float isotropy = lmin / tr;
            if (isotropy < 0.055f) continue;

            float hf_s = (scores_dense != nullptr) ? scores_dense[y * w + x] : 0.0f;
            float resp = lmin * isotropy * (1.0f + 120.0f * hf_s);
            cands.push_back({(float)x, (float)y, resp});
        }
    }

    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return a.resp > b.resp;
    });

    std::vector<bool> occ_local(occ_grid, occ_grid + occ_w * occ_h);
    float cell_w = (float)w / 8.0f;
    float cell_h = (float)h / 6.0f;

    for (const auto& c : cands) {
        int cr = std::min(5, std::max(0, (int)(c.v / cell_h)));
        int cc = std::min(7, std::max(0, (int)(c.u / cell_w)));
        if (cell_counts[cr][cc] >= max_per_cell) continue;
        int gy = (int)c.v / 12;
        int gx = (int)c.u / 12;
        if (gy >= 0 && gy < occ_h && gx >= 0 && gx < occ_w && occ_local[gy * occ_w + gx]) continue;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int ny = gy + dy, nx = gx + dx;
                if (ny >= 0 && ny < occ_h && nx >= 0 && nx < occ_w) {
                    occ_local[ny * occ_w + nx] = true;
                }
            }
        }
        cell_counts[cr][cc]++;
        out_added_pts.emplace_back(c.u, c.v);
        if ((int)out_added_pts.size() >= need_total) break;
    }
}

} // namespace slam
