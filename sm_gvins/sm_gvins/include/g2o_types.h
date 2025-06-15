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

class VertexRot : public g2o::BaseVertex<3, SO3> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexRot() {}

    bool read(std::istream& is) override {
        double data[4];
        for (int i = 0; i < 4; i++) {
            is >> data[i];
        }
        setEstimate(SO3(Quatd(data[3], data[0], data[1], data[2])));
        return true;
    }

    bool write(std::ostream& os) const override {
        os << "VERTEX_SO3:QUAT ";
        os << id() << " ";
        Quatd q = _estimate.unit_quaternion();
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << std::endl;
        return true;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double* update_) {
        _estimate = _estimate * SO3::exp(Eigen::Map<const Vec3d>(&update_[0]));  // 旋转部分
        updateCache();
    }
};

/**
 * 速度顶点，单纯的Vec3d
 */
class VertexVelocity : public g2o::BaseVertex<3, Vec3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity() {}

    virtual bool read(std::istream& is) { return false; }
    virtual bool write(std::ostream& os) const { return false; }

    virtual void setToOriginImpl() { _estimate.setZero(); }

    virtual void oplusImpl(const double* update_) { _estimate += Eigen::Map<const Vec3d>(update_); }
};


/* 
    位置顶点
 */
class VertexPosition : public VertexVelocity {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPosition() {}
};

/**
 * 陀螺零偏顶点，亦为Vec3d，从速度顶点继承
 */
class VertexGyroBias : public VertexVelocity {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias() {}
};

/**
 * 加计零偏顶点，Vec3d，亦从速度顶点继承
 */
class VertexAccBias : public VertexVelocity {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias() {}
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
        resize(7);
    }

    EdgeProjection(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) 
        : g2o::BaseMultiEdge<2, Eigen::Vector3d>(), pts_i(_pts_i), pts_j(_pts_j)
    {
        resize(7);
    }

    virtual bool read(std::istream& is) override { return false; }
    virtual bool write(std::ostream& os) const override { return false; }

    virtual void computeError() override
    {
        auto* r1 = dynamic_cast<const VertexRot*>(_vertices[0]);
        auto* p1 = dynamic_cast<const VertexPosition*>(_vertices[1]);
        auto* r2 = dynamic_cast<const VertexRot*>(_vertices[2]);
        auto* p2 = dynamic_cast<const VertexPosition*>(_vertices[3]);
        auto* rbc = static_cast<const VertexRot*>(_vertices[4]);
        auto* pbc = static_cast<const VertexPosition*>(_vertices[5]);
        auto* v = static_cast<const VertexInverseDepth*>(_vertices[6]);

        const SO3& Rwbi  = r1->estimate();
        const SO3& Rwbj  = r2->estimate();
        const SO3& Rbc = rbc->estimate();

        const Vec3d&  twbi  = p1->estimate();
        const Vec3d&  twbj  = p2->estimate();  
        const Vec3d&  tbc = pbc->estimate();

        double inv_dep_i = v->estimate();

        if(!twbi.isZero()){
            int tt = 1;
        }

        Vec3d pts_camera_i = pts_i / inv_dep_i;
        Vec3d pts_imu_i = Rbc.inverse() * pts_camera_i + tbc;
        Vec3d pts_w = Rwbi.inverse() * pts_imu_i + twbi;
        Vec3d pts_imu_j = Rwbj * (pts_w - twbj);
        Vec3d pts_camera_j = Rbc * (pts_imu_j - tbc);

        if (std::abs(pts_camera_j.z()) < 1e-6) {
            _error.setZero();
            return;
        }

        double dep_j = pts_camera_j.z();
        Vec3d mes = (pts_camera_j / dep_j);
        _error = mes.head<2>() - pts_j.head<2>();
    }

    virtual void linearizeOplus() override
    {
        auto* r1 = dynamic_cast<const VertexRot*>(_vertices[0]);
        auto* p1 = dynamic_cast<const VertexPosition*>(_vertices[1]);
        auto* r2 = dynamic_cast<const VertexRot*>(_vertices[2]);
        auto* p2 = dynamic_cast<const VertexPosition*>(_vertices[3]);
        auto* rbc = static_cast<const VertexRot*>(_vertices[4]);
        auto* pbc = static_cast<const VertexPosition*>(_vertices[5]);
        auto* v = static_cast<const VertexInverseDepth*>(_vertices[6]);

        const SO3& Rwbi  = r1->estimate();
        const SO3& Rwbj  = r2->estimate();
        const SO3& Rbc = rbc->estimate();

        const Vec3d&  twbi  = p1->estimate();
        const Vec3d&  twbj  = p2->estimate();  
        const Vec3d&  tbc = pbc->estimate();

        double inv_dep_i = v->estimate();

        Vec3d pts_camera_i = pts_i / inv_dep_i;
        Vec3d pts_imu_i = Rbc.inverse() * pts_camera_i + tbc;
        Vec3d pts_w = Rwbi.inverse() * pts_imu_i + twbi;
        Vec3d pts_imu_j = Rwbj * (pts_w - twbj);
        Vec3d pts_camera_j = Rbc * (pts_imu_j - tbc);

        double z = pts_camera_j.z();
        Mat23d reduce;
        reduce << 1. / z, 0, -pts_camera_j.x() / (z * z),
                0, 1. / z, -pts_camera_j.y() / (z * z);

        // common
        Mat3d Rwbi_mat = Rwbi.matrix();
        Mat3d Rwbj_mat = Rwbj.matrix();
        Mat3d Rbc_mat  = Rbc.matrix();
        Mat3d Rbc_inv  = Rbc.inverse().matrix();

        // 1. ∂res / ∂R_wbi (VertexRot 0)
        _jacobianOplus[0] = reduce * Rbc_mat * Rwbj_mat * SO3::hat(Rwbi_mat.inverse() * pts_imu_i);

        // 2. ∂res / ∂P_wbi (VertexPosition 1)
        _jacobianOplus[1] = reduce * Rbc_mat * Rwbj_mat;    

        // 3. ∂res / ∂R_wbj (VertexRot 2)
        _jacobianOplus[2] = -reduce * Rbc_mat * Rwbj_mat * SO3::hat(pts_w - twbj);

        // 4. ∂res / ∂P_wbj (VertexPosition 3)
        _jacobianOplus[3] = -reduce * Rbc_mat * Rwbj_mat;

        // 5. ∂res / ∂R_bc (VertexRot 4)
        _jacobianOplus[4] = reduce * (-Rbc_mat * SO3::hat(pts_imu_j - tbc) 
                                + Rbc_mat * Rwbj_mat * Rwbi_mat.inverse() * SO3::hat(Rbc_inv * pts_camera_i));

        // 6. ∂res / ∂t_bc (VertexPosition 5)
        _jacobianOplus[5] = reduce * (-Rbc_mat + Rbc_mat * Rwbj_mat * Rwbi_mat.inverse());

        // 7. ∂res / ∂inv_depth (VertexInverseDepth 6)
        _jacobianOplus[6] = reduce * (-Rbc_mat * Rwbj_mat * Rwbi_mat.inverse() * Rbc_inv * 
                                pts_i / (inv_dep_i * inv_dep_i));
    }

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix3d K;
    static Eigen::Matrix2d sqrt_info;
};

Eigen::Matrix2d EdgeProjection::sqrt_info = 460 / 1.5 * Eigen::Matrix2d::Identity(); // 调整为 1.0