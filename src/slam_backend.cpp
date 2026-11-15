#include "slam_backend.h"

#include <cmath>
#include <algorithm>
#include <set>
#include <map>

namespace slam {

int Optimizer::pose_optimization(Frame& frame, int n_its_per_round,
                                 const Mat3* R_cw_prior, double rot_prior_weight) {
    double fx = frame.camera.fx;
    double fy = frame.camera.fy;
    double cx = frame.camera.cx;
    double cy = frame.camera.cy;

    std::vector<int> valid_indices;
    valid_indices.reserve(frame.num_kps);
    for (int i = 0; i < frame.num_kps; ++i) {
        if (frame.map_points[i] && !frame.map_points[i]->is_bad) {
            frame.outliers[i] = false;
            valid_indices.push_back(i);
        } else {
            frame.outliers[i] = true;
        }
    }
    if ((int)valid_indices.size() < 4) return 0;

    SE3 T_cw = frame.T_cw;

    for (int round_idx = 0; round_idx < 4; ++round_idx) {
        for (int it = 0; it < n_its_per_round; ++it) {
            Mat6 H;
            Vec6 b;
            int n_used = 0;

            for (int idx : valid_indices) {
                if (frame.outliers[idx]) continue;
                const auto& mp = frame.map_points[idx];
                Vec3 p_c = T_cw.transform(mp->pos_w);
                double X = p_c.x, Y = p_c.y, Z = p_c.z;
                if (Z <= 1e-4) continue;

                double inv_z = 1.0 / Z;
                double inv_z2 = inv_z * inv_z;

                double u_pred = fx * X * inv_z + cx;
                double v_pred = fy * Y * inv_z + cy;
                double eu = frame.kp_pts[idx].x - u_pred;
                double ev = frame.kp_pts[idx].y - v_pred;

                double e2 = eu * eu + ev * ev;
                double w_huber = 1.0;
                double e_norm = std::sqrt(e2);
                if (e_norm > HUBER_MONO) {
                    w_huber = HUBER_MONO / e_norm;
                }

                // 2x6 误差雅可比 J = -d(pi)/d(xi):
                // d(u)/d(rho) = [fx/Z, 0, -fx*X/Z^2], d(u)/d(phi) = [-fx*X*Y/Z^2, fx*(1+X^2/Z^2), -fx*Y/Z]
                // d(v)/d(rho) = [0, fy/Z, -fy*Y/Z^2], d(v)/d(phi) = [-fy*(1+Y^2/Z^2), fy*X*Y/Z^2, fy*X/Z]
                double Ju[6] = {
                    -fx * inv_z,
                    0.0,
                    fx * X * inv_z2,
                    fx * X * Y * inv_z2,
                    -fx * (1.0 + X * X * inv_z2),
                    fx * Y * inv_z
                };
                double Jv[6] = {
                    0.0,
                    -fy * inv_z,
                    fy * Y * inv_z2,
                    fy * (1.0 + Y * Y * inv_z2),
                    -fy * X * Y * inv_z2,
                    -fy * X * inv_z
                };

                for (int r = 0; r < 6; ++r) {
                    b[r] -= w_huber * (Ju[r] * eu + Jv[r] * ev);
                    for (int c = r; c < 6; ++c) {
                        H(r, c) += w_huber * (Ju[r] * Ju[c] + Jv[r] * Jv[c]);
                    }
                }
                n_used++;
            }

            if (n_used < 4) break;
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < r; ++c) H(r, c) = H(c, r);
                H(r, r) += 1e-3;
            }

            // 紧耦合 IMU 旋转先验约束: e_rot = Log(R_cw * R_prior^T)
            if (R_cw_prior != nullptr && rot_prior_weight > 0.0) {
                Vec3 e_rot = SO3::log(T_cw.R * R_cw_prior->transpose());
                for (int k = 0; k < 3; ++k) {
                    H(3 + k, 3 + k) += rot_prior_weight;
                    b[3 + k] -= rot_prior_weight * e_rot[k];
                }
            }

            Vec6 dx;
            if (!solve_linear_6x6(H, b, dx)) break;
            T_cw = SE3::exp(dx) * T_cw;
            if (dx.squared_norm() < 1e-10) break;
        }

        // 每轮迭代结束后执行卡方检验标记/恢复外点
        for (int idx : valid_indices) {
            const auto& mp = frame.map_points[idx];
            Vec3 p_c = T_cw.transform(mp->pos_w);
            if (p_c.z <= 1e-4) {
                frame.outliers[idx] = true;
                continue;
            }
            Vec2 uv = frame.camera.project(p_c);
            double e2 = (frame.kp_pts[idx] - uv).squared_norm();
            frame.outliers[idx] = (e2 > CHI2_MONO);
        }
    }

    int inliers = 0;
    for (int idx : valid_indices) {
        if (!frame.outliers[idx]) inliers++;
    }
    if (inliers >= 4) {
        frame.set_pose(T_cw);
    }
    return inliers;
}

void Optimizer::local_bundle_adjustment(std::shared_ptr<KeyFrame> curr_kf,
                                        SlamMap& slam_map, int n_iterations) {
    if (!curr_kf || curr_kf->is_bad) return;

    // 1. 收集局部活跃关键帧 (当前关键帧 + 共视邻居)
    std::vector<KeyFrame*> opt_kfs;
    std::set<KeyFrame*> opt_kf_set;
    opt_kfs.push_back(curr_kf.get());
    opt_kf_set.insert(curr_kf.get());

    for (KeyFrame* neigh : curr_kf->get_best_covisibility_keyframes(5)) {
        if (neigh && !neigh->is_bad) {
            opt_kfs.push_back(neigh);
            opt_kf_set.insert(neigh);
        }
    }

    bool has_imu_in_window = false;
    for (KeyFrame* kf : opt_kfs) {
        if (kf->has_imu_rot_prior) {
            has_imu_in_window = true;
            break;
        }
    }

    // 2. 收集局部地图点
    std::vector<MapPoint*> local_mps;
    std::set<MapPoint*> mp_set;
    for (KeyFrame* kf : opt_kfs) {
        for (const auto& mp : kf->map_points) {
            if (mp && !mp->is_bad && !mp->is_temp_flow_point && mp->num_observations() >= 2) {
                if (mp_set.insert(mp.get()).second) {
                    local_mps.push_back(mp.get());
                }
            }
        }
    }
    if (local_mps.empty()) return;

    // 3. 收集固定关键帧:
    // 第 0 号世界原点关键帧永远固定；单目+IMU模式下同时固定第 1 号初始化关键帧以锁定初始物理基线尺度 (1:1 对齐 optimizer.py line 278)
    std::set<KeyFrame*> fixed_kfs;
    for (MapPoint* mp : local_mps) {
        for (const auto& kv : mp->observations) {
            KeyFrame* kf_obs = kv.first;
            if (kf_obs && !kf_obs->is_bad && opt_kf_set.find(kf_obs) == opt_kf_set.end()) {
                fixed_kfs.insert(kf_obs);
            }
        }
    }
    for (auto it = opt_kfs.begin(); it != opt_kfs.end();) {
        if ((*it)->id == 0 || (has_imu_in_window && (*it)->id == 1)) {
            fixed_kfs.insert(*it);
            opt_kf_set.erase(*it);
            it = opt_kfs.erase(it);
        } else {
            ++it;
        }
    }
    if (opt_kfs.empty()) return;

    int N_p = (int)opt_kfs.size();
    int dim_p = 6 * N_p;
    std::map<KeyFrame*, int> kf_to_pidx;
    for (int i = 0; i < N_p; ++i) kf_to_pidx[opt_kfs[i]] = i;

    const double w_kf_rot = 5.0e5;
    const double w_kf_trans_reg = 35.0;

    // 辅助 Lambda: 计算当前局部窗口总代价 (视觉重投影 + IMU 旋转先验)
    auto compute_total_cost = [&](bool use_huber, double chi2_gate) -> double {
        double cost = 0.0;
        for (MapPoint* mp : local_mps) {
            if (mp->is_bad) continue;
            for (const auto& kv : mp->observations) {
                KeyFrame* kf = kv.first;
                int kp_idx = kv.second;
                if (!kf || kf->is_bad || kp_idx < 0 || kp_idx >= kf->num_kps) continue;
                Vec3 p_c = kf->T_cw.transform(mp->pos_w);
                if (p_c.z <= 1e-3) continue;
                double inv_z = 1.0 / p_c.z;
                double u_pred = kf->camera.fx * p_c.x * inv_z + kf->camera.cx;
                double v_pred = kf->camera.fy * p_c.y * inv_z + kf->camera.cy;
                double eu = kf->kp_pts[kp_idx].x - u_pred;
                double ev = kf->kp_pts[kp_idx].y - v_pred;
                double e2 = eu * eu + ev * ev;
                if (e2 > chi2_gate) continue;
                double w = (use_huber && e2 > CHI2_MONO) ? (HUBER_MONO / std::sqrt(e2)) : 1.0;
                cost += w * e2;
            }
        }
        if (has_imu_in_window) {
            for (KeyFrame* kf : opt_kfs) {
                if (kf->has_imu_rot_prior) {
                    Vec3 e_r = SO3::log(kf->T_cw.R * kf->R_cw_imu_prior.transpose());
                    cost += w_kf_rot * e_r.squared_norm();
                }
            }
        }
        return cost;
    };

    // 4. 两阶段稀疏舒尔补 LM 迭代 (第一阶段带 Huber 核，第二阶段卡方筛查后精细收敛，含步长信赖域与代价回滚保护)
    double chi2_gate = 35.0;
    for (int stage = 0; stage < 2; ++stage) {
        bool use_huber = (stage == 0);
        int stage_iters = (stage == 0) ? n_iterations : std::max(2, n_iterations / 2);
        double damping = 1e-3;
        if (stage == 1) chi2_gate = CHI2_MONO;

        for (int iter = 0; iter < stage_iters; ++iter) {
            std::vector<double> H_pp(dim_p * dim_p, 0.0);
            std::vector<double> b_p(dim_p, 0.0);

            // 为每个可优化关键帧加入阻尼、微基线平移正则与 IMU 旋转流形先验 (对齐 optimizer.py lines 467-477)
            for (int i = 0; i < N_p; ++i) {
                KeyFrame* kf = opt_kfs[i];
                int base = 6 * i;
                for (int d = 0; d < 3; ++d) {
                    H_pp[(base + d) * dim_p + (base + d)] += damping + (has_imu_in_window ? w_kf_trans_reg : 0.1);
                }
                for (int d = 3; d < 6; ++d) {
                    H_pp[(base + d) * dim_p + (base + d)] += damping;
                }
                if (kf->has_imu_rot_prior) {
                    Vec3 e_rot = SO3::log(kf->T_cw.R * kf->R_cw_imu_prior.transpose());
                    for (int k = 0; k < 3; ++k) {
                        H_pp[(base + 3 + k) * dim_p + (base + 3 + k)] += w_kf_rot;
                        b_p[base + 3 + k] -= w_kf_rot * e_rot[k];
                    }
                }
            }

            struct MpUpdateCache {
                MapPoint* mp;
                Mat3 H_mm_inv;
                Vec3 b_m;
                std::vector<std::pair<int, double[3][6]>> H_mp_blocks;
            };
            std::vector<MpUpdateCache> mp_caches;
            mp_caches.reserve(local_mps.size());

            for (MapPoint* mp : local_mps) {
                if (mp->is_bad) continue;
                Mat3 H_mm = Mat3::diag(damping, damping, damping);
                Vec3 b_m(0, 0, 0);
                MpUpdateCache cache;
                cache.mp = mp;

                int valid_obs = 0;
                for (const auto& kv : mp->observations) {
                    KeyFrame* kf = kv.first;
                    int kp_idx = kv.second;
                    if (!kf || kf->is_bad || kp_idx < 0 || kp_idx >= kf->num_kps) continue;

                    Vec3 p_c = kf->T_cw.transform(mp->pos_w);
                    double X = p_c.x, Y = p_c.y, Z = p_c.z;
                    if (Z <= 1e-3) continue;
                    double fx = kf->camera.fx, fy = kf->camera.fy;
                    double inv_z = 1.0 / Z, inv_z2 = inv_z * inv_z;

                    double u_pred = fx * X * inv_z + kf->camera.cx;
                    double v_pred = fy * Y * inv_z + kf->camera.cy;
                    double eu = kf->kp_pts[kp_idx].x - u_pred;
                    double ev = kf->kp_pts[kp_idx].y - v_pred;
                    double e2 = eu * eu + ev * ev;
                    if (e2 > chi2_gate) continue;
                    double w = (use_huber && e2 > CHI2_MONO) ? (HUBER_MONO / std::sqrt(e2)) : 1.0;

                    double J_pc_u[3] = {-fx * inv_z, 0.0, fx * X * inv_z2};
                    double J_pc_v[3] = {0.0, -fy * inv_z, fy * Y * inv_z2};
                    double Jm_u[3] = {0, 0, 0}, Jm_v[3] = {0, 0, 0};
                    for (int c = 0; c < 3; ++c) {
                        for (int k = 0; k < 3; ++k) {
                            Jm_u[c] += J_pc_u[k] * kf->T_cw.R(k, c);
                            Jm_v[c] += J_pc_v[k] * kf->T_cw.R(k, c);
                        }
                    }

                    for (int r = 0; r < 3; ++r) {
                        b_m[r] -= w * (Jm_u[r] * eu + Jm_v[r] * ev);
                        for (int c = 0; c < 3; ++c) {
                            H_mm(r, c) += w * (Jm_u[r] * Jm_u[c] + Jm_v[r] * Jm_v[c]);
                        }
                    }
                    valid_obs++;

                    auto it_p = kf_to_pidx.find(kf);
                    if (it_p != kf_to_pidx.end()) {
                        int pidx = it_p->second;
                        int base = 6 * pidx;
                        double Jp_u[6] = {
                            -fx * inv_z, 0.0, fx * X * inv_z2,
                            fx * X * Y * inv_z2, -fx * (1.0 + X * X * inv_z2), fx * Y * inv_z
                        };
                        double Jp_v[6] = {
                            0.0, -fy * inv_z, fy * Y * inv_z2,
                            fy * (1.0 + Y * Y * inv_z2), -fy * X * Y * inv_z2, -fy * X * inv_z
                        };
                        for (int r = 0; r < 6; ++r) {
                            b_p[base + r] -= w * (Jp_u[r] * eu + Jp_v[r] * ev);
                            for (int c = 0; c < 6; ++c) {
                                H_pp[(base + r) * dim_p + (base + c)] += w * (Jp_u[r] * Jp_u[c] + Jp_v[r] * Jp_v[c]);
                            }
                        }
                        cache.H_mp_blocks.emplace_back();
                        cache.H_mp_blocks.back().first = pidx;
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 6; ++c) {
                                cache.H_mp_blocks.back().second[r][c] = w * (Jm_u[r] * Jp_u[c] + Jm_v[r] * Jp_v[c]);
                            }
                        }
                    }
                }

                if (valid_obs < 2) continue;
                cache.H_mm_inv = H_mm.inverse();
                cache.b_m = b_m;

                // 舒尔补消元: H_pp -= H_pm * H_mm^{-1} * H_mp, b_p -= H_pm * H_mm^{-1} * b_m
                Vec3 Hmm_inv_bm = cache.H_mm_inv * b_m;
                int nb = (int)cache.H_mp_blocks.size();
                for (int bi = 0; bi < nb; ++bi) {
                    int pi = cache.H_mp_blocks[bi].first;
                    const auto& H_mp_i = cache.H_mp_blocks[bi].second;
                    for (int r = 0; r < 6; ++r) {
                        double s_b = 0.0;
                        for (int k = 0; k < 3; ++k) s_b += H_mp_i[k][r] * Hmm_inv_bm[k];
                        b_p[6 * pi + r] -= s_b;
                    }
                    double W_inv[6][3];
                    for (int r = 0; r < 6; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            W_inv[r][c] = H_mp_i[0][r] * cache.H_mm_inv(0, c)
                                        + H_mp_i[1][r] * cache.H_mm_inv(1, c)
                                        + H_mp_i[2][r] * cache.H_mm_inv(2, c);
                        }
                    }
                    for (int bj = 0; bj < nb; ++bj) {
                        int pj = cache.H_mp_blocks[bj].first;
                        const auto& H_mp_j = cache.H_mp_blocks[bj].second;
                        for (int r = 0; r < 6; ++r) {
                            for (int c = 0; c < 6; ++c) {
                                double sub = W_inv[r][0] * H_mp_j[0][c]
                                           + W_inv[r][1] * H_mp_j[1][c]
                                           + W_inv[r][2] * H_mp_j[2][c];
                                H_pp[(6 * pi + r) * dim_p + (6 * pj + c)] -= sub;
                            }
                        }
                    }
                }
                mp_caches.push_back(std::move(cache));
            }

            for (int d = 0; d < dim_p; ++d) H_pp[d * dim_p + d] += damping;

            std::vector<double> dx_p;
            if (!solve_linear_nxn(dim_p, H_pp, b_p, dx_p)) break;

            // 回代求解 3D 地图点增量 dx_m = H_mm^{-1} * (b_m - H_mp * dx_p)
            std::vector<Vec3> dx_m_list(mp_caches.size());
            double max_dx_p = 0.0;
            for (double v : dx_p) max_dx_p = std::max(max_dx_p, std::abs(v));
            double max_dx_m = 0.0;
            for (size_t ci = 0; ci < mp_caches.size(); ++ci) {
                const auto& cache = mp_caches[ci];
                Vec3 rhs = cache.b_m;
                for (const auto& blk : cache.H_mp_blocks) {
                    int pidx = blk.first;
                    for (int r = 0; r < 3; ++r) {
                        double s = 0.0;
                        for (int c = 0; c < 6; ++c) s += blk.second[r][c] * dx_p[6 * pidx + c];
                        rhs[r] -= s;
                    }
                }
                dx_m_list[ci] = cache.H_mm_inv * rhs;
                max_dx_m = std::max({max_dx_m, std::abs(dx_m_list[ci].x), std::abs(dx_m_list[ci].y), std::abs(dx_m_list[ci].z)});
            }

            // Levenberg-Marquardt 步长信赖域保护 (对齐 optimizer.py lines 500-502)
            if (max_dx_p > 0.15 || max_dx_m > 0.8) {
                damping *= 10.0;
                continue;
            }

            double prev_cost = compute_total_cost(use_huber, chi2_gate);
            std::vector<SE3> saved_poses(N_p);
            for (int i = 0; i < N_p; ++i) saved_poses[i] = opt_kfs[i]->T_cw;
            std::vector<Vec3> saved_pts(mp_caches.size());
            for (size_t ci = 0; ci < mp_caches.size(); ++ci) saved_pts[ci] = mp_caches[ci].mp->pos_w;

            // 试更新位姿与地图点
            for (int i = 0; i < N_p; ++i) {
                Vec6 dxi;
                for (int d = 0; d < 6; ++d) dxi[d] = dx_p[6 * i + d];
                opt_kfs[i]->T_cw = SE3::exp(dxi) * opt_kfs[i]->T_cw;
            }
            for (size_t ci = 0; ci < mp_caches.size(); ++ci) {
                mp_caches[ci].mp->pos_w += dx_m_list[ci];
            }

            double new_cost = compute_total_cost(use_huber, chi2_gate);
            if (new_cost > prev_cost) {
                // 拒绝劣质步，恢复状态并增大阻尼因子 (对齐 optimizer.py lines 542-551)
                for (int i = 0; i < N_p; ++i) opt_kfs[i]->T_cw = saved_poses[i];
                for (size_t ci = 0; ci < mp_caches.size(); ++ci) mp_caches[ci].mp->pos_w = saved_pts[ci];
                damping *= 10.0;
                continue;
            } else {
                damping = std::max(1e-5, damping * 0.5);
            }

            double dx_norm2 = 0.0;
            for (double v : dx_p) dx_norm2 += v * v;
            if (dx_norm2 < 1e-9) break;
        }
    }

    // 5. 从共视图和地图点中解除卡方超标的外点观测，并更新地图点法向与深度 (对齐 optimizer.py lines 578-589)
    for (MapPoint* mp : local_mps) {
        if (!mp || mp->is_bad) continue;
        std::vector<std::pair<KeyFrame*, int>> bad_obs;
        for (const auto& kv : mp->observations) {
            KeyFrame* kf = kv.first;
            int kp_idx = kv.second;
            if (!kf || kf->is_bad || kp_idx < 0 || kp_idx >= kf->num_kps) continue;
            Vec3 p_c = kf->T_cw.transform(mp->pos_w);
            if (p_c.z <= 1e-3) {
                bad_obs.emplace_back(kf, kp_idx);
                continue;
            }
            Vec2 uv = kf->camera.project(p_c);
            double e2 = (kf->kp_pts[kp_idx] - uv).squared_norm();
            if (e2 > CHI2_MONO) {
                bad_obs.emplace_back(kf, kp_idx);
            }
        }
        for (const auto& bo : bad_obs) {
            bo.first->erase_mappoint_match(bo.second);
            mp->erase_observation(bo.first, &slam_map);
        }
        if (!mp->is_bad) {
            mp->update_normal_and_depth();
        }
    }
}

void LocalMapper::process_new_keyframe(std::shared_ptr<KeyFrame> curr_kf) {
    if (!curr_kf || !slam_map) return;

    // 1. 绑定当前关键帧中已跟踪到的地图点观测
    for (int i = 0; i < curr_kf->num_kps; ++i) {
        auto mp = curr_kf->map_points[i];
        if (mp && !mp->is_bad) {
            if (mp->observations.find(curr_kf.get()) == mp->observations.end()) {
                mp->add_observation(curr_kf.get(), i);
                mp->update_normal_and_depth();
                mp->compute_distinctive_descriptors();
            } else {
                recent_added_mappoints.push_back(mp);
            }
        }
    }

    // 2. 更新共视图连接并加入地图
    curr_kf->update_connections();
    slam_map->add_keyframe(curr_kf);

    // 3. 剔除近期劣质地图点
    mappoint_culling(curr_kf);

    // 4. 与共视邻居关键帧对极三角化新建 3D 地图点 + 惯性深度锚点补充
    create_new_mappoints(curr_kf);

    // 5. 当地图关键帧数 > 2 且具备足够平移基线 (>= 2.0cm) 时执行稀疏舒尔补局部 BA (1:1 对齐 local_mapper.py lines 64-87)
    if (slam_map->num_keyframes() > 2) {
        SE3 T_cw_before = curr_kf->T_cw;
        Vec3 p_curr = curr_kf->get_camera_center();
        double max_base = 0.0;
        for (KeyFrame* k : curr_kf->get_best_covisibility_keyframes(3)) {
            if (k && !k->is_bad) {
                max_base = std::max(max_base, (p_curr - k->get_camera_center()).norm());
            }
        }
        if (max_base >= 0.020) {
            Optimizer::local_bundle_adjustment(curr_kf, *slam_map, 2);
            SE3 T_delta_w = curr_kf->T_cw.inverse() * T_cw_before;
            for (const auto& mp : curr_kf->map_points) {
                if (mp && !mp->is_bad && mp->num_observations() == 1) {
                    mp->pos_w = T_delta_w.transform(mp->pos_w);
                }
            }
        }
    }

    // 6. 剔除冗余关键帧
    keyframe_culling(curr_kf);
}

void LocalMapper::mappoint_culling(std::shared_ptr<KeyFrame> curr_kf) {
    int curr_kf_id = curr_kf->id;
    for (auto it = recent_added_mappoints.begin(); it != recent_added_mappoints.end();) {
        auto mp = *it;
        if (!mp || mp->is_bad) {
            it = recent_added_mappoints.erase(it);
        } else if (mp->get_found_ratio() < 0.25) {
            mp->set_bad_flag(slam_map);
            it = recent_added_mappoints.erase(it);
        } else if ((curr_kf_id - mp->first_kf_id) >= 2 && mp->num_observations() < 2 && !mp->is_temp_flow_point) {
            mp->set_bad_flag(slam_map);
            it = recent_added_mappoints.erase(it);
        } else if ((curr_kf_id - mp->first_kf_id) >= 3) {
            it = recent_added_mappoints.erase(it);
        } else {
            ++it;
        }
    }
}

void LocalMapper::create_new_mappoints(std::shared_ptr<KeyFrame> curr_kf) {
    // Part A: 与共视关键帧极线搜索 + 对极三角化
    std::vector<KeyFrame*> neigh_kfs = curr_kf->get_best_covisibility_keyframes(3);
    if (neigh_kfs.empty()) {
        for (const auto& kv : slam_map->keyframes) {
            if (kv.second && !kv.second->is_bad && kv.second.get() != curr_kf.get()) {
                neigh_kfs.push_back(kv.second.get());
                if (neigh_kfs.size() >= 2) break;
            }
        }
    }

    Vec3 o_w1 = curr_kf->get_camera_center();
    for (KeyFrame* kf2 : neigh_kfs) {
        if (!kf2 || kf2->is_bad) continue;
        Vec3 o_w2 = kf2->get_camera_center();
        double baseline_dist = (o_w1 - o_w2).norm();
        if (baseline_dist < 0.015) continue;

        auto matches = FrontendMatcher::search_for_triangulation(*curr_kf, *kf2);
        for (const auto& m : matches) {
            int idx1 = m.first;
            int idx2 = m.second;
            if (curr_kf->map_points[idx1] != nullptr || kf2->map_points[idx2] != nullptr) continue;

            Vec3 xn1 = curr_kf->camera.unproject(curr_kf->kp_pts[idx1].x, curr_kf->kp_pts[idx1].y, 1.0);
            Vec3 xn2 = kf2->camera.unproject(kf2->kp_pts[idx2].x, kf2->kp_pts[idx2].y, 1.0);

            Vec3 pos_w;
            if (!triangulate_two_views(curr_kf->T_cw, kf2->T_cw, xn1, xn2, pos_w)) continue;

            Vec3 p_c1 = curr_kf->T_cw.transform(pos_w);
            Vec3 p_c2 = kf2->T_cw.transform(pos_w);
            if (p_c1.z <= 0.08 || p_c2.z <= 0.08 || p_c1.z > 30.0 || p_c2.z > 30.0) continue;

            Vec2 uv1 = curr_kf->camera.project(p_c1);
            Vec2 uv2 = kf2->camera.project(p_c2);
            if ((uv1 - curr_kf->kp_pts[idx1]).squared_norm() > Optimizer::CHI2_MONO) continue;
            if ((uv2 - kf2->kp_pts[idx2]).squared_norm() > Optimizer::CHI2_MONO) continue;

            auto mp = std::make_shared<MapPoint>(pos_w, curr_kf, curr_kf->frame_id, curr_kf->descriptors[idx1]);
            curr_kf->add_mappoint(mp, idx1);
            kf2->add_mappoint(mp, idx2);
            mp->add_observation(curr_kf.get(), idx1);
            mp->add_observation(kf2, idx2);
            mp->compute_distinctive_descriptors();
            mp->update_normal_and_depth();
            slam_map->add_mappoint(mp);
            recent_added_mappoints.push_back(mp);
        }
    }

    // Part B: 实机单目+IMU模式 (640x480) 下，若相机发生纯旋转/微基线平移导致 DLT 无法三角化新视场特征点，
    // 利用当前局部深度中位数对新视场内未匹配特征点补充生成惯性深度锚点，防止大幅转头时 Inliers 跌零 (1:1 对齐 local_mapper.py lines 259-296)
    if (curr_kf->has_imu_rot_prior) {
        std::vector<double> valid_depths;
        for (const auto& mp : curr_kf->map_points) {
            if (mp && !mp->is_bad) {
                double zc = curr_kf->T_cw.transform(mp->pos_w).z;
                if (zc > 0.25 && zc < 8.0) valid_depths.push_back(zc);
            }
        }
        int n_curr_valid = (int)valid_depths.size();
        if (n_curr_valid < 230) {
            double z_med = 1.20;
            if (n_curr_valid >= 8) {
                std::sort(valid_depths.begin(), valid_depths.end());
                z_med = std::min(2.50, std::max(0.65, valid_depths[n_curr_valid / 2]));
            }
            int need_n = std::min(150, 230 - n_curr_valid);
            std::vector<int> unmatched_indices;
            for (int i = 0; i < curr_kf->num_kps; ++i) {
                if (curr_kf->map_points[i] == nullptr && curr_kf->kp_octaves[i] <= 3) {
                    unmatched_indices.push_back(i);
                }
            }
            int step = std::max(1, (int)unmatched_indices.size() / std::max(1, need_n));
            SE3 T_wc = curr_kf->get_pose_inverse();
            double h_img = (double)curr_kf->camera.height;
            int added_cnt = 0;
            for (size_t k = 0; k < unmatched_indices.size() && added_cnt < need_n; k += step) {
                int idx_u = unmatched_indices[k];
                double u_p = curr_kf->kp_pts[idx_u].x;
                double v_p = curr_kf->kp_pts[idx_u].y;
                double z_pt = z_med * (0.88 + 0.24 * (1.0 - v_p / h_img));
                Vec3 p_c = curr_kf->camera.unproject(u_p, v_p, z_pt);
                Vec3 pos_w = T_wc.transform(p_c);
                auto mp = std::make_shared<MapPoint>(pos_w, curr_kf, curr_kf->frame_id, curr_kf->descriptors[idx_u]);
                mp->add_observation(curr_kf.get(), idx_u);
                curr_kf->add_mappoint(mp, idx_u);
                mp->compute_distinctive_descriptors();
                mp->update_normal_and_depth();
                slam_map->add_mappoint(mp);
                recent_added_mappoints.push_back(mp);
                added_cnt++;
            }
        }
    }
}

void LocalMapper::keyframe_culling(std::shared_ptr<KeyFrame> curr_kf) {
    for (KeyFrame* kf : curr_kf->ordered_connected_kfs) {
        if (!kf || kf->id <= 1 || kf->is_bad || (curr_kf->id - kf->id) <= 2) continue;
        int n_mps = 0;
        int n_redundant = 0;
        for (const auto& mp : kf->map_points) {
            if (mp && !mp->is_bad && !mp->is_temp_flow_point) {
                n_mps++;
                if (mp->num_observations() > 3) n_redundant++;
            }
        }
        if (n_mps > 15 && n_redundant > (int)(0.90 * n_mps)) {
            kf->is_bad = true;
            slam_map->erase_keyframe(kf);
        }
    }
}

} // namespace slam
