#pragma once

#include "eigen_types.h"
#include <glog/logging.h>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

/// vertex and edges used in g2o ba
/// 位姿顶点
class VertexPose : public g2o::BaseVertex<6, SE3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override { _estimate = SE3(); }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        Vec6d update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4],
            update[5];
        _estimate = SE3::exp(update_eigen) * _estimate;
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
};

/// 路标顶点
class VertexXYZ : public g2o::BaseVertex<3, Vec3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void setToOriginImpl() override { _estimate = Vec3d::Zero(); }

    virtual void oplusImpl(const double *update) override {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
        _estimate[2] += update[2];
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
};

class VertexInverseDepth : public g2o::BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexInverseDepth() {}
    bool read(std::istream & /*is*/) override
    {
        std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
        return false;
    }

    bool write(std::ostream & /*os*/) const override
    {
        std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
        return false;
    }
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const double *update)
    {
        _estimate += update[0];
    }
};

/* 
class EdgeProjection : public g2o::BaseMultiEdge<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjection() : g2o::BaseMultiEdge<2, Eigen::Vector2d>()
    {
        resize(4);
    }

    EdgeProjection(const Eigen::Vector2d &_pts_i, const Eigen::Vector2d &_pts_j, const Eigen::Matrix3d &_K) 
        : g2o::BaseMultiEdge<2, Eigen::Vector2d>(), pts_i(_pts_i), pts_j(_pts_j), K(_K)
    {
        resize(4);
    }

    virtual bool read(std::istream& is) override { return true; }
    virtual bool write(std::ostream& os) const override { return true; }

    virtual void computeError() override
    {
        const auto* v0 = static_cast<const VertexPose*>(_vertices[0]);
        const auto* v1 = static_cast<const VertexPose*>(_vertices[1]);
        const auto* v2 = static_cast<const VertexPose*>(_vertices[2]);
        const auto* v3 = static_cast<const VertexInverseDepth*>(_vertices[3]);

        const SE3& Ti = v0->estimate();
        const SE3& Tj = v1->estimate();
        const SE3& Tic = v2->estimate();
        double rho = v3->estimate();

        if(!Ti.translation().isZero()){
            int tt = 1;
        }

        if (std::abs(rho) < 1e-6) {
            _error.setZero();
            return;
        }

        Eigen::Vector3d pts_camera_i_norm;
        pts_camera_i_norm << 
            (pts_i[0] - K(0,2)) / K(0,0),
            (pts_i[1] - K(1,2)) / K(1,1),
            1.0;

        Eigen::Vector3d pts_camera_i = pts_camera_i_norm / rho;

        Eigen::Vector3d pts_imu_i = Tic.inverse() * pts_camera_i;
        Eigen::Vector3d pts_w = Ti.inverse() * pts_imu_i;
        Eigen::Vector3d pts_imu_j = Tj * pts_w;
        Eigen::Vector3d pts_camera_j = Tic * pts_imu_j;

        if (std::abs(pts_camera_j.z()) < 1e-6) {
            _error.setZero();
            return;
        }

        Eigen::Vector3d pixel_projection = K * pts_camera_j;
        pixel_projection /= pixel_projection.z();
        _error = pixel_projection.head<2>() - pts_j;
    }

    virtual void linearizeOplus() override
    {
        const auto* v0 = static_cast<const VertexPose*>(_vertices[0]); // Ti
        const auto* v1 = static_cast<const VertexPose*>(_vertices[1]); // Tj
        const auto* v2 = static_cast<const VertexPose*>(_vertices[2]); // Tic
        const auto* v3 = static_cast<const VertexInverseDepth*>(_vertices[3]); // rho

        const SE3& Ti = v0->estimate();
        const SE3& Tj = v1->estimate();
        const SE3& Tic = v2->estimate();
        double rho = v3->estimate();

        // Intermediate points (calculated in computeError as well)
        Eigen::Vector3d pts_camera_i_norm;
        pts_camera_i_norm <<
            (pts_i[0] - K(0,2)) / K(0,0),
            (pts_i[1] - K(1,2)) / K(1,1),
            1.0;

        Eigen::Vector3d pts_camera_i = pts_camera_i_norm / rho;
        Eigen::Vector3d pts_imu_i = Tic.inverse() * pts_camera_i;
        Eigen::Vector3d pts_w = Ti.inverse() * pts_imu_i;
        Eigen::Vector3d pts_imu_j = Tj * pts_w;
        Eigen::Vector3d pts_camera_j = Tic * pts_imu_j;

        // Check for division by zero (can happen if point goes behind camera or inverse depth is zero)
        if (std::abs(pts_camera_j.z()) < 1e-6 || std::abs(rho) < 1e-6) {
            _jacobianOplus[0].setZero();
            _jacobianOplus[1].setZero();
            _jacobianOplus[2].setZero();
            _jacobianOplus[3].setZero();
            return;
        }

        // Jacobian of projection (pi(P_cj)) w.r.t. P_cj
        // pi(P_cj) = [ K(0,0)*Xc/Zc + K(0,2); K(1,1)*Yc/Zc + K(1,2) ]
        // We want d(pi(P_cj))/d(P_cj) where P_cj = [Xc, Yc, Zc]^T
        double Xc = pts_camera_j.x();
        double Yc = pts_camera_j.y();
        double Zc = pts_camera_j.z();
        double Zc_inv = 1.0 / Zc;
        double Zc_inv2 = Zc_inv * Zc_inv;

        Eigen::Matrix<double, 2, 3> J_pi_Pcj;
        J_pi_Pcj << K(0,0) * Zc_inv, 0,           -K(0,0) * Xc * Zc_inv2,
                    0,           K(1,1) * Zc_inv, -K(1,1) * Yc * Zc_inv2;

        // Jacobian w.r.t. v0 (Ti) - Pose of IMU i in World Frame
        // P_w = Ti.inverse() * P_ii
        // For SE3::exp(delta_xi) * T updates, d(T.inverse() * X)/d(delta_xi) = [-R^T | R^T * hat(X)]
        Eigen::Matrix<double, 3, 6> J_Pw_Ti;
        J_Pw_Ti.block<3,3>(0,0) = -Ti.rotationMatrix().transpose(); // Jacobian w.r.t. translational part (d(t_i))
        J_Pw_Ti.block<3,3>(0,3) = Ti.rotationMatrix().transpose() * Sophus::SO3d::hat(pts_imu_i); // Jacobian w.r.t. rotational part (d(phi_i))
        // Chain rule: d(error)/d(delta_xi_i) = d(pi)/d(P_cj) * d(P_cj)/d(P_ij) * d(P_ij)/d(P_w) * d(P_w)/d(delta_xi_i)
        // d(P_cj)/d(P_ij) = Tic.rotationMatrix()
        // d(P_ij)/d(P_w) = Tj.rotationMatrix()
        _jacobianOplus[0] = J_pi_Pcj * Tic.rotationMatrix() * Tj.rotationMatrix() * J_Pw_Ti;

        // Jacobian w.r.t. v1 (Tj) - Pose of IMU j in World Frame
        // P_ij = Tj * P_w
        // For SE3::exp(delta_xi) * T updates, d(T * X)/d(delta_xi) = [I | -hat(T*X)]
        Eigen::Matrix<double, 3, 6> J_Pij_Tj;
        J_Pij_Tj.block<3,3>(0,0) = Eigen::Matrix3d::Identity(); // Jacobian w.r.t. translational part (d(t_j))
        J_Pij_Tj.block<3,3>(0,3) = -Sophus::SO3d::hat(pts_imu_j); // Jacobian w.r.t. rotational part (d(phi_j))
        // Chain rule: d(error)/d(delta_xi_j) = d(pi)/d(P_cj) * d(P_cj)/d(P_ij) * d(P_ij)/d(delta_xi_j)
        // d(P_cj)/d(P_ij) = Tic.rotationMatrix()
        _jacobianOplus[1] = J_pi_Pcj * Tic.rotationMatrix() * J_Pij_Tj;

        // Jacobian w.r.t. v2 (Tic) - Pose of Camera in IMU Frame
        // P_cj = Tic * P_ij
        // For SE3::exp(delta_xi) * T updates, d(T * X)/d(delta_xi) = [I | -hat(T*X)]
        Eigen::Matrix<double, 3, 6> J_Pcj_Tic;
        J_Pcj_Tic.block<3,3>(0,0) = Eigen::Matrix3d::Identity(); // Jacobian w.r.t. translational part (d(t_ic))
        J_Pcj_Tic.block<3,3>(0,3) = -Sophus::SO3d::hat(pts_camera_j); // Jacobian w.r.t. rotational part (d(phi_ic))
        // Chain rule: d(error)/d(delta_xi_ic) = d(pi)/d(P_cj) * d(P_cj)/d(delta_xi_ic)
        _jacobianOplus[2] = J_pi_Pcj * J_Pcj_Tic;

        // Jacobian w.r.t. v3 (rho) - Inverse Depth of the Feature
        // P_ci = P_ci_norm / rho
        // d(P_ci)/d(rho) = -1/rho^2 * P_ci_norm = -1/rho * P_ci
        // The total derivative of P_cj w.r.t. rho simplifies to -1/rho * P_cj
        Eigen::Matrix<double, 3, 1> J_Pcj_rho;
        J_Pcj_rho = -1.0 / rho * pts_camera_j;
        // Chain rule: d(error)/d(rho) = d(pi)/d(P_cj) * d(P_cj)/d(rho)
        _jacobianOplus[3] = J_pi_Pcj * J_Pcj_rho;
    }

    Eigen::Vector2d pts_i, pts_j;
    Eigen::Matrix3d K;
    static Eigen::Matrix2d sqrt_info;
};

Eigen::Matrix2d EdgeProjection::sqrt_info = Eigen::Matrix2d::Identity(); // 调整为 1.0
 */

class EdgeProjection : public g2o::BaseMultiEdge<2, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjection() : g2o::BaseMultiEdge<2, Eigen::Vector3d>()
    {
        resize(4);
    }

    EdgeProjection(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) 
        : g2o::BaseMultiEdge<2, Eigen::Vector3d>(), pts_i(_pts_i), pts_j(_pts_j)
    {
        resize(4);
    }

    virtual bool read(std::istream& is) override { return false; }
    virtual bool write(std::ostream& os) const override { return false; }

    virtual void computeError() override
    {
        const auto* v0 = static_cast<const VertexPose*>(_vertices[0]);
        const auto* v1 = static_cast<const VertexPose*>(_vertices[1]);
        const auto* v2 = static_cast<const VertexPose*>(_vertices[2]);
        const auto* v3 = static_cast<const VertexInverseDepth*>(_vertices[3]);

        const SE3& Ti = v0->estimate();
        const SE3& Tj = v1->estimate();
        const SE3& Tic = v2->estimate();
        double inv_dep_i = v3->estimate();

        if(!Ti.translation().isZero()){
            int tt = 1;
        }

        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
        Eigen::Vector3d pts_imu_i = Tic.inverse() * pts_camera_i;
        Eigen::Vector3d pts_w = Ti.inverse() * pts_imu_i;
        Eigen::Vector3d pts_imu_j = Tj * pts_w;
        Eigen::Vector3d pts_camera_j = Tic * pts_imu_j;

        if (std::abs(pts_camera_j.z()) < 1e-6) {
            _error.setZero();
            return;
        }

        double dep_j = pts_camera_j.z();
        _error = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
    }

    virtual void linearizeOplus() override
    {
        const auto* v0 = static_cast<const VertexPose*>(_vertices[0]); // Ti
        const auto* v1 = static_cast<const VertexPose*>(_vertices[1]); // Tj
        const auto* v2 = static_cast<const VertexPose*>(_vertices[2]); // Tic
        const auto* v3 = static_cast<const VertexInverseDepth*>(_vertices[3]); // rho

        const SE3& Ti = v0->estimate();
        const SE3& Tj = v1->estimate();
        const SE3& Tic = v2->estimate();
        double inv_dep_i = v3->estimate();

        // Compute intermediate transformations
        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
        Eigen::Vector3d pts_imu_i = Tic.inverse() * pts_camera_i;
        Eigen::Vector3d pts_w = Ti.inverse() * pts_imu_i;
        Eigen::Vector3d pts_imu_j = Tj * pts_w;
        Eigen::Vector3d pts_camera_j = Tic * pts_imu_j;

        double dep_j = pts_camera_j.z();
        if (std::abs(dep_j) < 1e-6) {
            _jacobianOplus[0].setZero();
            _jacobianOplus[1].setZero();
            _jacobianOplus[2].setZero();
            _jacobianOplus[3].setZero();
            return;
        }

        // Compute reduce matrix (∂e/∂pts_camera_j)
        Eigen::Matrix<double, 2, 3> reduce;
        reduce << 1.0 / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                0, 1.0 / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

        // Rotation matrices
        Eigen::Matrix3d Ri = Ti.rotationMatrix();
        Eigen::Matrix3d Rj = Tj.rotationMatrix();
        Eigen::Matrix3d Ric = Tic.rotationMatrix();

        // Jacobian w.r.t. Ti (2x6)
        Eigen::Matrix<double, 3, 6> jaco_i;
        jaco_i.leftCols<3>() = Eigen::Matrix3d::Identity();
        jaco_i.rightCols<3>() = Sophus::SO3d::hat(pts_w);
        _jacobianOplus[0] = -reduce * Ric * Rj * jaco_i;

        // Jacobian w.r.t. Tj (2x6)
        Eigen::Matrix<double, 3, 6> jaco_j;
        jaco_j.leftCols<3>() = Eigen::Matrix3d::Identity();
        jaco_j.rightCols<3>() = -Sophus::SO3d::hat(Rj * pts_w);
        _jacobianOplus[1] = reduce * Ric * jaco_j;

        // Jacobian w.r.t. Tic (2x6)
        Eigen::Matrix<double, 3, 6> jaco_ic;
        jaco_ic.leftCols<3>() = Eigen::Matrix3d::Identity();
        jaco_ic.rightCols<3>() = -Sophus::SO3d::hat(Ric * pts_imu_j);
        Eigen::Matrix<double, 3, 6> jaco_ic_via_i;
        jaco_ic_via_i.setZero();
        jaco_ic_via_i.rightCols<3>() = -Ric * Rj * Ri.transpose() * Sophus::SO3d::hat(Ric.transpose() * pts_camera_i);
        _jacobianOplus[2] = reduce * (jaco_ic + jaco_ic_via_i);

        // Jacobian w.r.t. inv_dep_i (2x1)
        _jacobianOplus[3] = -reduce * Ric * Rj * Ri.transpose() * Ric.transpose() * pts_i / (inv_dep_i * inv_dep_i);
    }


    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix3d K;
    static Eigen::Matrix2d sqrt_info;
};

Eigen::Matrix2d EdgeProjection::sqrt_info = Eigen::Matrix2d::Identity(); // 调整为 1.0