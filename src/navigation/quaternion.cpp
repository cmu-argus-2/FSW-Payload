#include "navigation/quaternion.hpp"

#include <eigen3/Eigen/QR>
#include <vector>

using casadi::MX;

// Skew-symmetric matrix for cross-product equivalent.
Matrix3x3 SkewSymmetric(const Vector& omega) {
    Matrix3x3 skew;
    skew <<  0, -omega[2], omega[1],
             omega[2], 0, -omega[0],
            -omega[1], omega[0], 0;
    return skew;
}

// Left-multiplication matrix for quaternion multiplication.
Matrix4x4 L(const Quaternion& q) {
    Matrix4x4 L;
    L <<  q[0], -q[1], -q[2], -q[3],
          q[1],  q[0], -q[3],  q[2],
          q[2],  q[3],  q[0], -q[1],
          q[3], -q[2],  q[1],  q[0];
    return L;
}

// Right-multiplication matrix for quaternion multiplication.
Matrix4x4 R(const Quaternion& q) {
    Matrix4x4 R;
    R <<  q[0], -q[1], -q[2], -q[3],
          q[1],  q[0],  q[3], -q[2],
          q[2], -q[3],  q[0],  q[1],
          q[3],  q[2], -q[1],  q[0];
    return R;
}

// Returns the conjugate of a quaternion. Does not normalize.
Quaternion Conj(const Quaternion& q) {
    return Quaternion(q[0], -q[1], -q[2], -q[3]);
}

// Converts quaternion to MRP vector.
Vector MRPFromQuat(const Quaternion& q) {
    // Ensure the quaternion is not zero to prevent division by zero
    assert(q.norm() > 1e-8 && "Input quaternion has zero norm.");

    Quaternion q_normalized = q / q.norm();

    // Prevent division by zero when q_normalized[0] is -1
    double denominator = 1 + q_normalized[0];
    assert(std::abs(denominator) > 1e-8 && "Denominator is zero in MRP computation.");
    Vector mrp = q_normalized.segment(1, 3) / denominator;

    // Check if MRP norm exceeds unit; if so, adjust.
    // Shadow set switching 
    // Karlgaard, C. D., & Schaub, H. (2009). 
    // Nonsingular attitude filtering using modified Rodrigues parameters. 
    double mrp_norm = mrp.norm();
    if (mrp_norm >= 1.0) {
        mrp = -mrp / (mrp_norm * mrp_norm);
    }
    return mrp;
}


// Quaternion product using left-multiplication matrix.
Quaternion QuatProduct(const Quaternion& q1, const Quaternion& q2) {
    return L(q1) * q2;
}


// Spherical Linear Interpolation (SLERP) between two quaternions.
Quaternion Slerp(const Quaternion& q1, const Quaternion& q2, double t) {
    double lambda = q1.dot(q2);
    Quaternion nq2 = lambda < 0 ? -q2 : q2;
    lambda = std::abs(lambda);

    // Linear interpolation for nearly parallel quaternions.
    if (std::abs(1 - lambda) < 1e-2) {
        return (1 - t) * q1 + t * nq2;
    } else {
        // Calculate spherical interpolation factors.
        double alpha = std::acos(lambda);
        double gamma = 1 / std::sin(alpha);
        Quaternion qf = (std::sin((1 - t) * alpha) * gamma) * q1 + (std::sin(t * alpha) * gamma) * nq2;
        qf.normalize();
        return qf;
    }
}

// Quaternion kinematics for angular velocity integration.
Quaternion QuaternionKinematics(const Quaternion& q, const Vector& angvel) {
    return 0.5 * L(q) * Quaternion(0, angvel[0], angvel[1], angvel[2]);
}

MX quat_product_xyzw(const MX& p, const MX& q) {
    MX px = p(0, 0), py = p(1, 0), pz = p(2, 0), pw = p(3, 0);
    MX qx = q(0, 0), qy = q(1, 0), qz = q(2, 0), qw = q(3, 0);
    return MX::vertcat(std::vector<MX>{
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
        pw*qw - px*qx - py*qy - pz*qz
    });
}

MX quat_conjugate_xyzw(const MX& q) {
    return MX::vertcat(std::vector<MX>{-q(0, 0), -q(1, 0), -q(2, 0), q(3, 0)});
}

MX angle_axis_to_quat_xyzw(const MX& angle_axis) {
    MX safe_norm = sqrt(dot(angle_axis, angle_axis) + 1e-20);
    MX half      = safe_norm / 2.0;
    MX sinc_half = sin(half) / safe_norm;
    return MX::vertcat(std::vector<MX>{
        angle_axis(0, 0) * sinc_half,
        angle_axis(1, 0) * sinc_half,
        angle_axis(2, 0) * sinc_half,
        cos(half)
    });
}

MX quat_inv_rotate_xyzw(const MX& q, const MX& v) {
    MX x = q(0, 0), y = q(1, 0), z = q(2, 0), w = q(3, 0);
    MX vx = v(0, 0), vy = v(1, 0), vz = v(2, 0);
    return MX::vertcat(std::vector<MX>{
        (1.0 - 2.0*(y*y + z*z))*vx + 2.0*(x*y + w*z)*vy + 2.0*(x*z - w*y)*vz,
        2.0*(x*y - w*z)*vx + (1.0 - 2.0*(x*x + z*z))*vy + 2.0*(y*z + w*x)*vz,
        2.0*(x*z + w*y)*vx + 2.0*(y*z - w*x)*vy + (1.0 - 2.0*(x*x + y*y))*vz
    });
}

Eigen::Matrix<double, 4, 3> quat_tangent_basis_xyzw(double qx, double qy,
                                                    double qz, double qw) {
    Eigen::Vector4d q(qx, qy, qz, qw);
    const double q_norm = q.norm();
    if (q_norm > 0.0) {
        q /= q_norm;
    } else {
        q = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    }

    Eigen::Matrix4d A;
    A.col(0) = q;
    A.col(1) = Eigen::Vector4d::UnitX();
    A.col(2) = Eigen::Vector4d::UnitY();
    A.col(3) = Eigen::Vector4d::UnitZ();

    const Eigen::HouseholderQR<Eigen::Matrix4d> qr(A);
    const Eigen::Matrix4d Q = qr.householderQ() * Eigen::Matrix4d::Identity();
    return Q.block<4, 3>(0, 1);
}
