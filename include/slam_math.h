#ifndef HFNET_SLAM_MATH_H
#define HFNET_SLAM_MATH_H

#include <cmath>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace slam {

struct Vec2 {
    double x = 0.0, y = 0.0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& o) const { return Vec2(x + o.x, y + o.y); }
    Vec2 operator-(const Vec2& o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator*(double s) const { return Vec2(x * s, y * s); }
    double norm() const { return std::hypot(x, y); }
    double squared_norm() const { return x * x + y * y; }
};

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double& operator[](int i) { return (&x)[i]; }
    const double& operator[](int i) const { return (&x)[i]; }

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(double s) const { double inv = 1.0 / s; return Vec3(x * inv, y * inv, z * inv); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
    }
    double squared_norm() const { return x * x + y * y + z * z; }
    double norm() const { return std::sqrt(squared_norm()); }
    Vec3 normalized() const {
        double n = norm();
        return (n > 1e-12) ? (*this / n) : Vec3(0.0, 0.0, 0.0);
    }
};

struct Mat3 {
    double m[3][3];

    Mat3() { std::memset(m, 0, sizeof(m)); }

    static Mat3 zero() { return Mat3(); }
    static Mat3 identity() {
        Mat3 I;
        I.m[0][0] = I.m[1][1] = I.m[2][2] = 1.0;
        return I;
    }
    static Mat3 diag(double d0, double d1, double d2) {
        Mat3 D;
        D.m[0][0] = d0; D.m[1][1] = d1; D.m[2][2] = d2;
        return D;
    }
    static Mat3 outer(const Vec3& a, const Vec3& b) {
        Mat3 M;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                M.m[r][c] = a[r] * b[c];
        return M;
    }

    double& operator()(int r, int c) { return m[r][c]; }
    const double& operator()(int r, int c) const { return m[r][c]; }

    Mat3 operator+(const Mat3& o) const {
        Mat3 R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R.m[i][j] = m[i][j] + o.m[i][j];
        return R;
    }
    Mat3 operator-(const Mat3& o) const {
        Mat3 R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R.m[i][j] = m[i][j] - o.m[i][j];
        return R;
    }
    Mat3 operator*(double s) const {
        Mat3 R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R.m[i][j] = m[i][j] * s;
        return R;
    }
    Vec3 operator*(const Vec3& v) const {
        return Vec3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }
    Mat3 operator*(const Mat3& o) const {
        Mat3 R;
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) {
                double aik = m[i][k];
                for (int j = 0; j < 3; ++j)
                    R.m[i][j] += aik * o.m[k][j];
            }
        return R;
    }
    Mat3 transpose() const {
        Mat3 T;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                T.m[i][j] = m[j][i];
        return T;
    }
    double trace() const { return m[0][0] + m[1][1] + m[2][2]; }
    double determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
    Mat3 inverse() const;
};

struct Vec6 {
    double v[6];
    Vec6() { std::memset(v, 0, sizeof(v)); }
    double& operator[](int i) { return v[i]; }
    const double& operator[](int i) const { return v[i]; }
    Vec3 head3() const { return Vec3(v[0], v[1], v[2]); }
    Vec3 tail3() const { return Vec3(v[3], v[4], v[5]); }
    void set_head3(const Vec3& a) { v[0] = a.x; v[1] = a.y; v[2] = a.z; }
    void set_tail3(const Vec3& b) { v[3] = b.x; v[4] = b.y; v[5] = b.z; }
    double squared_norm() const {
        double s = 0.0;
        for (int i = 0; i < 6; ++i) s += v[i] * v[i];
        return s;
    }
    double norm() const { return std::sqrt(squared_norm()); }
};

struct Mat6 {
    double m[6][6];
    Mat6() { std::memset(m, 0, sizeof(m)); }
    static Mat6 identity() {
        Mat6 I;
        for (int i = 0; i < 6; ++i) I.m[i][i] = 1.0;
        return I;
    }
    double& operator()(int r, int c) { return m[r][c]; }
    const double& operator()(int r, int c) const { return m[r][c]; }
};

// 李群 SO(3) 与李代数 so(3) 算子 (1:1 对齐 Python src/core/lie_algebra.py)
class SO3 {
public:
    static Mat3 hat(const Vec3& phi);
    static Vec3 vee(const Mat3& Omega);
    static Mat3 exp(const Vec3& phi);
    static Vec3 log(const Mat3& R);
    static Mat3 right_jacobian(const Vec3& phi);
    static Mat3 right_jacobian_inv(const Vec3& phi);
    static Mat3 normalize_rotation(const Mat3& R);
    static Mat3 euler_deg_to_R(const Vec3& rpy_deg);
    static Vec3 R_to_body_euler_deg(const Mat3& R_wc);
};

// 刚体变换群 SE(3) 与李代数 se(3) (xi = [rho(平移3), phi(旋转3)]^T)
class SE3 {
public:
    Mat3 R;
    Vec3 t;

    SE3() : R(Mat3::identity()), t(0.0, 0.0, 0.0) {}
    SE3(const Mat3& R_, const Vec3& t_) : R(SO3::normalize_rotation(R_)), t(t_) {}

    static SE3 identity() { return SE3(); }
    static SE3 exp(const Vec6& xi);
    Vec6 log() const;

    SE3 inverse() const {
        Mat3 Rt = R.transpose();
        return SE3(Rt, -(Rt * t));
    }
    SE3 operator*(const SE3& other) const {
        return SE3(R * other.R, R * other.t + t);
    }
    Vec3 transform(const Vec3& p) const {
        return R * p + t;
    }
    Vec3 camera_center() const {
        return -(R.transpose() * t);
    }
};

// 稠密线性方程组求解器 (高斯列主元消元 + Cholesky/LDLT 阻尼求解)
bool solve_linear_3x3(const Mat3& A, const Vec3& b, Vec3& x);
bool solve_linear_6x6(const Mat6& H, const Vec6& b, Vec6& dx);
bool solve_linear_nxn(int n, const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x);

// 双视图几何与三角化算子
bool triangulate_two_views(const SE3& T_c1w, const SE3& T_c2w,
                           const Vec3& xn1, const Vec3& xn2, Vec3& pos_w_out);

} // namespace slam

#endif // HFNET_SLAM_MATH_H
