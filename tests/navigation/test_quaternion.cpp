#include "navigation/quaternion.hpp"
#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
#include <stdexcept>

// Test for SkewSymmetric 

TEST(QuaternionTest, SkewSymmetric_ZeroVector) {
    Vector omega = Vector::Zero();
    Matrix3x3 expected = Matrix3x3::Zero();
    Matrix3x3 result = SkewSymmetric(omega);
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}

TEST(QuaternionTest, SkewSymmetric_NonZeroVector) {
    Vector omega;
    omega << 1.0, 2.0, 3.0;
    Matrix3x3 expected;
    expected <<  0.0, -3.0,  2.0,
                 3.0,  0.0, -1.0,
                -2.0,  1.0,  0.0;
    Matrix3x3 result = SkewSymmetric(omega);
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}

TEST(QuaternionTest, SkewSymmetric_NegativeValues) {
    Vector omega;
    omega << -1.0, -2.0, -3.0;
    Matrix3x3 expected;
    expected <<  0.0,  3.0, -2.0,
                -3.0,  0.0,  1.0,
                 2.0, -1.0,  0.0;
    Matrix3x3 result = SkewSymmetric(omega);
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}


// Test for L function

TEST(QuaternionTest, L_KnownQuaternion) {
    Quaternion q;
    q <<  0.7517439, 0.1997945, 0.0799178, 0.6233589;
    Matrix4x4 expected;
    expected << 0.7517439, -0.1997945, -0.0799178, -0.6233589,
                0.1997945,  0.7517439, -0.6233589,  0.0799178,
                0.0799178,  0.6233589,  0.7517439, -0.1997945,
                0.6233589, -0.0799178,  0.1997945,  0.7517439;
    Matrix4x4 result = L(q);
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}

// Test for R function
TEST(QuaternionTest, R_KnownQuaternion) {
    Quaternion q;
    q << 0.7517439, 0.1997945, 0.0799178, 0.6233589;
    Matrix4x4 expected;
    expected << 0.7517439, -0.1997945, -0.0799178, -0.6233589,
    0.1997945,  0.7517439,  0.6233589, -0.0799178,
    0.0799178, -0.6233589,  0.7517439,  0.1997945,
    0.6233589,  0.0799178, -0.1997945,  0.7517439;
    Matrix4x4 result = R(q);
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}

// Test for Conj function

TEST(QuaternionTest, Conj_KnownQuaternion) {
    Quaternion q;
    q << 0.7517439, 0.1997945, 0.0799178, 0.6233589;
    Quaternion expected;
    expected << 0.7517439, -0.1997945, -0.0799178, -0.6233589;;
    Quaternion result = Conj(q);
    EXPECT_TRUE(result.isApprox(expected, 1e-6));
}


// shouldn't normalize
TEST(QuaternionTest, Conj_NonUnitQuaternion) {
    Quaternion q;
    q << 2.0, -1.0, 0.5, 3.0;
    Quaternion expected;
    expected << 2.0, 1.0, -0.5, -3.0;
    Quaternion result = Conj(q);
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}



// Test for MRPFromQuat function

TEST(QuaternionTest, MRPFromQuat_NormalizedQuaternion) {
    Quaternion q;
    q << sqrt(2)/2, sqrt(2)/2, 0.0, 0.0;
    Vector expected;
    expected << 0.4142135623730951, 0.0, 0.0;
    Vector result = MRPFromQuat(q);
    EXPECT_TRUE(result.isApprox(expected, 1e-4));
}


TEST(QuaternionTest, MRPFromQuat_ZeroQuaternion) {
    Quaternion q = Quaternion::Zero();
    // Since passing a zero quaternion is invalid, we expect the assertion to fail in debug mode
    #ifndef NDEBUG
    EXPECT_DEATH(MRPFromQuat(q), "Input quaternion has zero norm.");
    #else
    // In release mode, the behavior is undefined; we ensure it does not produce finite values
    Vector mrp = MRPFromQuat(q);
    EXPECT_FALSE(mrp.allFinite());
    #endif
}


// Test for QuatProduct function
TEST(QuaternionTest, QuatProduct_Identity) {
    Quaternion q1;
    q1 << 1.0, 0.0, 0.0, 0.0;
    Quaternion q2;
    q2 << 0.7071, 0.7071, 0.0, 0.0;
    Quaternion expected = q2;
    Quaternion result = QuatProduct(q1, q2);
    EXPECT_TRUE(result.isApprox(expected, 1e-4));
}

TEST(QuaternionTest, QuatProduct_KnownQuaternions) {
    Quaternion q1;
    q1 << 0.7071, 0.0, 0.7071, 0.0;
    Quaternion q2;
    q2 << 0.7071, 0.7071, 0.0, 0.0;
    Quaternion expected;
    expected << 0.5, 0.5, 0.5, -0.5;
    Quaternion result = QuatProduct(q1, q2);
    EXPECT_TRUE(result.isApprox(expected, 1e-4));
}

TEST(QuaternionTest, QuatProduct_Associativity) {
    Quaternion q1, q2, q3;
    q1 << 1.0, 0.0, 0.0, 0.0;
    q2 << 0.0, 1.0, 0.0, 0.0;
    q3 << 0.0, 0.0, 1.0, 0.0;
    Quaternion result1 = QuatProduct(QuatProduct(q1, q2), q3);
    Quaternion result2 = QuatProduct(q1, QuatProduct(q2, q3));
    EXPECT_TRUE(result1.isApprox(result2, 1e-9));
}


// Helper function to compare quaternions considering sign ambiguity
bool QuaternionsAreEquivalent(const Quaternion& q1, const Quaternion& q2, double tolerance) {
    return q1.isApprox(q2, tolerance) || q1.isApprox(-q2, tolerance);
}

// Unit tests for Slerp function
TEST(QuaternionTest, Slerp_tZero) {
    Quaternion q1(1.0, 0.0, 0.0, 0.0); // Identity quaternion
    Quaternion q2(0.7071, 0.7071, 0.0, 0.0);
    q2.normalize();
    double t = 0.0;
    Quaternion result = Slerp(q1, q2, t);
    EXPECT_TRUE(QuaternionsAreEquivalent(result.normalized(), q1, 1e-6));
}

TEST(QuaternionTest, Slerp_tOne) {
    Quaternion q1(0.7071, 0.0, 0.7071, 0.0);
    q1.normalize();
    Quaternion q2(0.0, 1.0, 0.0, 0.0);
    double t = 1.0;
    q2.normalize();
    Quaternion result = Slerp(q1, q2, t);
    EXPECT_TRUE(QuaternionsAreEquivalent(result.normalized(), q2, 1e-6));
}

TEST(QuaternionTest, Slerp_Halfway) {
    Quaternion q1(1.0, 0.0, 0.0, 0.0); // Identity quaternion
    Quaternion q2(0.0, 1.0, 0.0, 0.0); // 180 degrees around x-axis
    q2.normalize();
    double t = 0.5;
    Quaternion expected(std::sqrt(2)/2, std::sqrt(2)/2, 0.0, 0.0); // 90 degrees around x-axis
    Quaternion result = Slerp(q1, q2, t);
    EXPECT_TRUE(QuaternionsAreEquivalent(result.normalized(), expected.normalized(), 1e-6));
}

TEST(QuaternionTest, Slerp_SameQuaternions) {
    Quaternion q1(0.7071, 0.7071, 0.0, 0.0);
    q1.normalize();
    double t = 0.5;
    Quaternion result = Slerp(q1, q1, t);
    EXPECT_TRUE(QuaternionsAreEquivalent(result.normalized(), q1, 1e-6));
}

TEST(QuaternionTest, Slerp_NegativeQuaternions) {
    Quaternion q1(0.7071, 0.0, 0.7071, 0.0);
    q1.normalize();
    Quaternion q2 = -q1; // Negative of q1
    double t = 0.5;
    Quaternion result = Slerp(q1, q2, t);
    // Since q1 and -q1 represent the same rotation, result should be equivalent to q1
    EXPECT_TRUE(QuaternionsAreEquivalent(result.normalized(), q1, 1e-6));
}

TEST(QuaternionTest, Slerp_NearlyParallelQuaternions) {
    Quaternion q1(1.0, 0.0, 0.0, 0.0);
    Quaternion q2(0.9999, 0.01, 0.0, 0.0);
    q1.normalize();
    q2.normalize();
    double t = 0.5;
    Quaternion expected = ((1.0 - t) * q1 + t * q2).normalized();
    Quaternion result = Slerp(q1, q2, t);
    EXPECT_TRUE(QuaternionsAreEquivalent(result.normalized(), expected, 1e-6));
}

TEST(QuaternionTest, Slerp_NearlyOppositeQuaternions) {
    Quaternion q1(0.0, 1.0, 0.0, 0.0);
    Quaternion q2(0.0, -1.0, 0.0, 0.0);
    q1.normalize();
    q2.normalize();
    double t = 0.5;
    // The function should handle this gracefully
    Quaternion result = Slerp(q1, q2, t);
    // Since q1 and q2 are opposites, any interpolation is undefined, but function should not crash
    EXPECT_TRUE(result.allFinite());
    EXPECT_NEAR(result.norm(), 1.0, 1e-6); // Result should still be a unit quaternion
}

TEST(QuaternionTest, Slerp_InvalidT_Negative) {
    Quaternion q1(1.0, 0.0, 0.0, 0.0);
    Quaternion q2(0.0, 1.0, 0.0, 0.0);
    q1.normalize();
    q2.normalize();
    double t = -0.1;
    Quaternion result = Slerp(q1, q2, t);
    // Extrapolation may occur; check that the result is still a valid unit quaternion
    EXPECT_TRUE(result.allFinite());
    EXPECT_NEAR(result.norm(), 1.0, 1e-6);
}

TEST(QuaternionTest, Slerp_InvalidT_ExceedsOne) {
    Quaternion q1(1.0, 0.0, 0.0, 0.0);
    Quaternion q2(0.0, 1.0, 0.0, 0.0);
    q1.normalize();
    q2.normalize();
    double t = 1.1;
    Quaternion result = Slerp(q1, q2, t);
    // Extrapolation may occur; check that the result is still a valid unit quaternion
    EXPECT_TRUE(result.allFinite());
    EXPECT_NEAR(result.norm(), 1.0, 1e-6);
}


TEST(QuaternionTest, Slerp_DegenerateCase_LambdaEqualsOne) {
    Quaternion q1(1.0, 0.0, 0.0, 0.0);
    q1.normalize();
    Quaternion q2 = q1;
    double t = 0.5;
    Quaternion result = Slerp(q1, q2, t);
    EXPECT_TRUE(QuaternionsAreEquivalent(result.normalized(), q1, 1e-6));
}

TEST(QuaternionTest, Slerp_DegenerateCase_LambdaEqualsMinusOne) {
    Quaternion q1(1.0, 0.0, 0.0, 0.0);
    q1.normalize();
    Quaternion q2 = -q1;
    double t = 0.5;
    Quaternion result = Slerp(q1, q2, t);
    // The result is undefined, but function should not crash
    EXPECT_TRUE(result.allFinite());
    EXPECT_NEAR(result.norm(), 1.0, 1e-6);
}



// Unit tests for QuaternionKinematics function

TEST(QuaternionTest, QuaternionKinematics_ZeroAngularVelocity) {
    Quaternion q(1.0, 0.0, 0.0, 0.0);
    q.normalize();
    Vector angvel = Vector::Zero();
    Quaternion expected = Quaternion::Zero();
    Quaternion result = QuaternionKinematics(q, angvel);
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}

TEST(QuaternionTest, QuaternionKinematics_NonZeroAngularVelocity) {
    Quaternion q(std::sqrt(2)/2, std::sqrt(2)/2, 0.0, 0.0); // 90 degrees around x-axis
    q.normalize();
    Vector angvel(0.0, 0.0, 1.0); // Angular velocity around z-axis
    Quaternion omega_quat(0.0, angvel[0], angvel[1], angvel[2]);
    Quaternion expected = 0.5 * L(q) * omega_quat;
    Quaternion result = QuaternionKinematics(q, angvel);
    EXPECT_TRUE(result.isApprox(expected, 1e-6));
}


TEST(QuaternionTest, QuaternionKinematics_ZeroQuaternion) {
    Quaternion q = Quaternion::Zero(); // Zero quaternion
    Vector angvel = Vector::Zero();
    // The function assumes unit quaternions; behavior is undefined
    Quaternion result = QuaternionKinematics(q, angvel);
    // Ensure that the result does not contain NaNs or Infs
    EXPECT_TRUE(result.allFinite());
}

