#include "navigation/quaternion.hpp"

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

// Returns the conjugate of a quaternion.
Quaternion Conj(const Quaternion& q) {
    return Quaternion(q[0], -q[1], -q[2], -q[3]);
}

// Converts quaternion to MRP vector.
Vector MRPFromQuat(const Quaternion& q) {
    Quaternion q_normalized = q / q.norm();
    Vector mrp = q_normalized.segment(1, 3) / (1 + q_normalized[0]);

    // Check if MRP norm exceeds unit; if so, adjust.
    double mrp_norm = mrp.norm();
    if (mrp_norm >= 1) {
        mrp /= -(mrp_norm * mrp_norm);
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
