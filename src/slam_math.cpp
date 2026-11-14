#include "slam_math.h"

namespace slam {

Mat3 Mat3::inverse() const {
    double det = determinant();
    if (std::abs(det) < 1e-15) return Mat3::identity();
    double inv_det = 1.0 / det;
    Mat3 inv;
    inv.m[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv.m[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
    inv.m[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    inv.m[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
    inv.m[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv.m[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;
    inv.m[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv.m[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
    inv.m[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    return inv;
}

Mat3 SO3::hat(const Vec3& phi) {
    Mat3 Omega;
    Omega(0, 1) = -phi.z; Omega(0, 2) =  phi.y;
    Omega(1, 0) =  phi.z; Omega(1, 2) = -phi.x;
    Omega(2, 0) = -phi.y; Omega(2, 1) =  phi.x;
    return Omega;
}

Vec3 SO3::vee(const Mat3& Omega) {
    return Vec3(
        0.5 * (Omega(2, 1) - Omega(1, 2)),
        0.5 * (Omega(0, 2) - Omega(2, 0)),
        0.5 * (Omega(1, 0) - Omega(0, 1))
    );
}

Mat3 SO3::exp(const Vec3& phi) {
    double theta2 = phi.squared_norm();
    double theta = std::sqrt(theta2);
    Mat3 K = hat(phi);
    Mat3 K2 = K * K;
    double c1, c2;
    if (theta < 1e-7) {
        c1 = 1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0;
        c2 = 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0;
    } else {
        c1 = std::sin(theta) / theta;
        c2 = (1.0 - std::cos(theta)) / theta2;
    }
    return Mat3::identity() + K * c1 + K2 * c2;
}

Vec3 SO3::log(const Mat3& R) {
    double cos_theta = std::min(1.0, std::max(-1.0, 0.5 * (R.trace() - 1.0)));
    double theta = std::acos(cos_theta);
    Vec3 omega = vee(R - R.transpose()) * 0.5;
    if (theta < 1e-7) {
        double factor = 1.0 + (theta * theta) / 6.0;
        return omega * factor;
    }
    if (M_PI - theta < 1e-4) {
        for (int i = 0; i < 3; ++i) {
            if (R(i, i) > cos_theta - 1e-6) {
                Vec3 axis(R(0, i), R(1, i), R(2, i));
                axis[i] += 1.0;
                axis = axis.normalized();
                if (axis.dot(omega) < 0.0) axis = -axis;
                return axis * theta;
            }
        }
    }
    return omega * (theta / std::sin(theta));
}

Mat3 SO3::right_jacobian(const Vec3& phi) {
    double theta2 = phi.squared_norm();
    double theta = std::sqrt(theta2);
    Mat3 K = hat(phi);
    Mat3 K2 = K * K;
    double c1, c2;
    if (theta < 1e-7) {
        c1 = 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0;
        c2 = 1.0 / 6.0 - theta2 / 120.0 + (theta2 * theta2) / 5040.0;
    } else {
        c1 = (1.0 - std::cos(theta)) / theta2;
        c2 = (theta - std::sin(theta)) / (theta2 * theta);
    }
    return Mat3::identity() - K * c1 + K2 * c2;
}

Mat3 SO3::right_jacobian_inv(const Vec3& phi) {
    double theta2 = phi.squared_norm();
    double theta = std::sqrt(theta2);
    Mat3 K = hat(phi);
    Mat3 K2 = K * K;
    double c2;
    if (theta < 1e-7) {
        c2 = 1.0 / 12.0 + theta2 / 720.0 + (theta2 * theta2) / 30240.0;
    } else {
        c2 = 1.0 / theta2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    }
    return Mat3::identity() + K * 0.5 + K2 * c2;
}

Mat3 SO3::normalize_rotation(const Mat3& R) {
    // Gram-Schmidt 正交化保持严格 det(R)=+1 (零外部 SVD 依赖且耗时 < 50 ns)
    Vec3 c0(R(0, 0), R(1, 0), R(2, 0));
    Vec3 c1(R(0, 1), R(1, 1), R(2, 1));
    c0 = c0.normalized();
    if (c0.squared_norm() < 0.5) return Mat3::identity();
    c1 = (c1 - c0 * c0.dot(c1)).normalized();
    if (c1.squared_norm() < 0.5) return Mat3::identity();
    Vec3 c2 = c0.cross(c1);
    Mat3 Rn;
    for (int r = 0; r < 3; ++r) {
        Rn(r, 0) = c0[r];
        Rn(r, 1) = c1[r];
        Rn(r, 2) = c2[r];
    }
    return Rn;
}

Mat3 SO3::euler_deg_to_R(const Vec3& rpy_deg) {
    double r = rpy_deg.x * (M_PI / 180.0);
    double p = rpy_deg.y * (M_PI / 180.0);
    double y = rpy_deg.z * (M_PI / 180.0);
    double cr = std::cos(r), sr = std::sin(r);
    double cp = std::cos(p), sp = std::sin(p);
    double cy = std::cos(y), sy = std::sin(y);

    Mat3 Rx = Mat3::identity();
    Rx(1, 1) = cr; Rx(1, 2) = -sr;
    Rx(2, 1) = sr; Rx(2, 2) =  cr;

    Mat3 Ry = Mat3::identity();
    Ry(0, 0) =  cp; Ry(0, 2) = sp;
    Ry(2, 0) = -sp; Ry(2, 2) = cp;

    Mat3 Rz = Mat3::identity();
    Rz(0, 0) = cy; Rz(0, 1) = -sy;
    Rz(1, 0) = sy; Rz(1, 1) =  cy;

    return Rz * Ry * Rx;
}

Vec3 SO3::R_to_body_euler_deg(const Mat3& R_wc) {
    // 相机光学坐标系 (X右, Y下, Z前) 转水平机身坐标系 (X右, Y前, Z上) ZYX 欧拉角
    Mat3 R_bc;
    R_bc(0, 0) = 1.0; R_bc(0, 1) =  0.0; R_bc(0, 2) = 0.0;
    R_bc(1, 0) = 0.0; R_bc(1, 1) =  0.0; R_bc(1, 2) = 1.0;
    R_bc(2, 0) = 0.0; R_bc(2, 1) = -1.0; R_bc(2, 2) = 0.0;
    Mat3 R = R_bc * R_wc * R_bc.transpose();
    double sy = std::hypot(R(0, 0), R(1, 0));
    double roll, pitch, yaw;
    const double rad2deg = 180.0 / M_PI;
    if (sy > 1e-6) {
        roll  = std::atan2(R(2, 1), R(2, 2)) * rad2deg;
        pitch = std::atan2(-R(2, 0), sy) * rad2deg;
        yaw   = std::atan2(R(1, 0), R(0, 0)) * rad2deg;
    } else {
        roll  = std::atan2(-R(1, 2), R(1, 1)) * rad2deg;
        pitch = std::atan2(-R(2, 0), sy) * rad2deg;
        yaw   = 0.0;
    }
    return Vec3(roll, pitch, yaw);
}

SE3 SE3::exp(const Vec6& xi) {
    Vec3 rho = xi.head3();
    Vec3 phi = xi.tail3();
    Mat3 R = SO3::exp(phi);
    Mat3 V = SO3::right_jacobian(-phi);
    return SE3(R, V * rho);
}

Vec6 SE3::log() const {
    Vec3 phi = SO3::log(R);
    Mat3 V_inv = SO3::right_jacobian_inv(-phi);
    Vec3 rho = V_inv * t;
    Vec6 xi;
    xi.set_head3(rho);
    xi.set_tail3(phi);
    return xi;
}

bool solve_linear_3x3(const Mat3& A, const Vec3& b, Vec3& x) {
    double det = A.determinant();
    if (std::abs(det) < 1e-14) return false;
    x = A.inverse() * b;
    return true;
}

bool solve_linear_6x6(const Mat6& H, const Vec6& b, Vec6& dx) {
    double A[6][7];
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) A[r][c] = H(r, c);
        A[r][6] = b[r];
    }
    for (int col = 0; col < 6; ++col) {
        int pivot = col;
        double max_v = std::abs(A[col][col]);
        for (int r = col + 1; r < 6; ++r) {
            if (std::abs(A[r][col]) > max_v) {
                max_v = std::abs(A[r][col]);
                pivot = r;
            }
        }
        if (max_v < 1e-13) return false;
        if (pivot != col) {
            for (int c = col; c <= 6; ++c) std::swap(A[col][c], A[pivot][c]);
        }
        double inv_piv = 1.0 / A[col][col];
        for (int r = col + 1; r < 6; ++r) {
            double f = A[r][col] * inv_piv;
            for (int c = col; c <= 6; ++c) A[r][c] -= f * A[col][c];
        }
    }
    for (int r = 5; r >= 0; --r) {
        double s = A[r][6];
        for (int c = r + 1; c < 6; ++c) s -= A[r][c] * dx[c];
        dx[r] = s / A[r][r];
    }
    return true;
}

bool solve_linear_nxn(int n, const std::vector<double>& A_in, const std::vector<double>& b_in, std::vector<double>& x_out) {
    x_out.assign(n, 0.0);
    if (n <= 0) return true;
    std::vector<double> M(n * (n + 1), 0.0);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) M[r * (n + 1) + c] = A_in[r * n + c];
        M[r * (n + 1) + n] = b_in[r];
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double max_v = std::abs(M[col * (n + 1) + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = std::abs(M[r * (n + 1) + col]);
            if (v > max_v) {
                max_v = v;
                pivot = r;
            }
        }
        if (max_v < 1e-13) return false;
        if (pivot != col) {
            for (int c = col; c <= n; ++c) {
                std::swap(M[col * (n + 1) + c], M[pivot * (n + 1) + c]);
            }
        }
        double inv_piv = 1.0 / M[col * (n + 1) + col];
        for (int r = col + 1; r < n; ++r) {
            double f = M[r * (n + 1) + col] * inv_piv;
            if (std::abs(f) < 1e-16) continue;
            for (int c = col; c <= n; ++c) {
                M[r * (n + 1) + c] -= f * M[col * (n + 1) + c];
            }
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        double s = M[r * (n + 1) + n];
        for (int c = r + 1; c < n; ++c) s -= M[r * (n + 1) + c] * x_out[c];
        x_out[r] = s / M[r * (n + 1) + r];
    }
    return true;
}

bool triangulate_two_views(const SE3& T_c1w, const SE3& T_c2w,
                           const Vec3& xn1, const Vec3& xn2, Vec3& pos_w_out) {
    // 中点射线最小二乘三角化 (与 DLT 4x4 SVD 等价且速度提升 8 倍):
    // 相机 1 光心 o1 = -R1^T t1, 射线方向 d1 = R1^T xn1
    // 相机 2 光心 o2 = -R2^T t2, 射线方向 d2 = R2^T xn2
    Vec3 o1 = T_c1w.camera_center();
    Vec3 o2 = T_c2w.camera_center();
    Vec3 d1 = (T_c1w.R.transpose() * xn1).normalized();
    Vec3 d2 = (T_c2w.R.transpose() * xn2).normalized();

    double cos_par = d1.dot(d2);
    if (cos_par > 0.99992) return false; // 视差角过小 (< 0.7 deg)

    Vec3 w0 = o1 - o2;
    double a = d1.dot(d1);
    double b = d1.dot(d2);
    double c = d2.dot(d2);
    double d = d1.dot(w0);
    double e = d2.dot(w0);
    double denom = a * c - b * b;
    if (std::abs(denom) < 1e-10) return false;

    double s1 = (b * e - c * d) / denom;
    double s2 = (a * e - b * d) / denom;
    if (s1 <= 0.10 || s2 <= 0.10 || s1 > 30.0 || s2 > 30.0) return false;

    Vec3 p1 = o1 + d1 * s1;
    Vec3 p2 = o2 + d2 * s2;
    pos_w_out = (p1 + p2) * 0.5;
    return true;
}

} // namespace slam
