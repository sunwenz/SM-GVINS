#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <cmath>

#include "sophus/se2.hpp"
#include "sophus/se3.hpp"


/// 常用的数学函数
namespace math {

// 常量定义
constexpr double kDEG2RAD = M_PI / 180.0;  // deg->rad
constexpr double kRAD2DEG = 180.0 / M_PI;  // rad -> deg
constexpr double G_m_s2 = 9.81;            // 重力大小

// 非法定义
constexpr size_t kINVALID_ID = std::numeric_limits<size_t>::max();


/**
 * 边缘化
 * @param H
 * @param start
 * @param end
 * @return
 */
inline Eigen::MatrixXd Marginalize(const Eigen::MatrixXd& H, const int& start, const int& end) {
    // Goal
    // a  | ab | ac       a*  | 0 | ac*
    // ba | b  | bc  -->  0   | 0 | 0
    // ca | cb | c        ca* | 0 | c*

    // Size of block before block to marginalize
    const int a = start;
    // Size of block to marginalize
    const int b = end - start + 1;
    // Size of block after block to marginalize
    const int c = H.cols() - (end + 1);

    // Reorder as follows:
    // a  | ab | ac       a  | ac | ab
    // ba | b  | bc  -->  ca | c  | cb
    // ca | cb | c        ba | bc | b

    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    if (a > 0) {
        Hn.block(0, 0, a, a) = H.block(0, 0, a, a);
        Hn.block(0, a + c, a, b) = H.block(0, a, a, b);
        Hn.block(a + c, 0, b, a) = H.block(a, 0, b, a);
    }
    if (a > 0 && c > 0) {
        Hn.block(0, a, a, c) = H.block(0, a + b, a, c);
        Hn.block(a, 0, c, a) = H.block(a + b, 0, c, a);
    }
    if (c > 0) {
        Hn.block(a, a, c, c) = H.block(a + b, a + b, c, c);
        Hn.block(a, a + c, c, b) = H.block(a + b, a, c, b);
        Hn.block(a + c, a, b, c) = H.block(a, a + b, b, c);
    }
    Hn.block(a + c, a + c, b, b) = H.block(a, a, b, b);

    // Perform marginalization (Schur complement)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a + c, a + c, b, b), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv = svd.singularValues();
    for (int i = 0; i < b; ++i) {
        if (singularValues_inv(i) > 1e-6) singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else
            singularValues_inv(i) = 0;
    }
    Eigen::MatrixXd invHb = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();
    Hn.block(0, 0, a + c, a + c) =
        Hn.block(0, 0, a + c, a + c) - Hn.block(0, a + c, a + c, b) * invHb * Hn.block(a + c, 0, b, a + c);
    Hn.block(a + c, a + c, b, b) = Eigen::MatrixXd::Zero(b, b);
    Hn.block(0, a + c, a + c, b) = Eigen::MatrixXd::Zero(a + c, b);
    Hn.block(a + c, 0, b, a + c) = Eigen::MatrixXd::Zero(b, a + c);

    // Inverse reorder
    // a*  | ac* | 0       a*  | 0 | ac*
    // ca* | c*  | 0  -->  0   | 0 | 0
    // 0   | 0   | 0       ca* | 0 | c*
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    if (a > 0) {
        res.block(0, 0, a, a) = Hn.block(0, 0, a, a);
        res.block(0, a, a, b) = Hn.block(0, a + c, a, b);
        res.block(a, 0, b, a) = Hn.block(a + c, 0, b, a);
    }
    if (a > 0 && c > 0) {
        res.block(0, a + b, a, c) = Hn.block(0, a, a, c);
        res.block(a + b, 0, c, a) = Hn.block(a, 0, c, a);
    }
    if (c > 0) {
        res.block(a + b, a + b, c, c) = Hn.block(a, a, c, c);
        res.block(a + b, a, c, b) = Hn.block(a, a + c, c, b);
        res.block(a, a + b, b, c) = Hn.block(a + c, a, b, c);
    }

    res.block(a, a, b, b) = Hn.block(a + c, a + c, b, b);

    return res;
}

inline void triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                                const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw) {
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();

    design_matrix.row(0) = pc0[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(1) = pc0[1] * pose0.row(2) - pose0.row(1);
    design_matrix.row(2) = pc1[0] * pose1.row(2) - pose1.row(0);
    design_matrix.row(3) = pc1[1] * pose1.row(2) - pose1.row(1);

    Eigen::Vector4d point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    pw                    = point.head<3>() / point(3);
}

inline bool triangulatePoint(const std::vector<Sophus::SE3d> &poses,   //Twc
                   const std::vector<Eigen::Vector3d> points,       //Pc 
                   Eigen::Vector3d &pt_world) {
    Eigen::MatrixXd A(2 * poses.size(), 4);
    Eigen::VectorXd b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    // if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-1) {
    //     // 解质量不好，放弃
    //     return true;
    // }
    return true;
}

}