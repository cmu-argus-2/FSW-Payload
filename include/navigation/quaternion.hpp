#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include <eigen3/Eigen/Dense>

typedef Eigen::Vector3d Vector;
typedef Eigen::Vector4d Quaternion;
typedef Eigen::Matrix3d Matrix3x3;
typedef Eigen::Matrix4d Matrix4x4;

/**
 * @brief Generates a skew-symmetric matrix from a 3D vector.
 * 
 * @param omega Input 3D vector representing angular velocity.
 * @return Matrix3x3 Skew-symmetric matrix.
 */
Matrix3x3 SkewSymmetric(const Vector& omega);

/**
 * @brief Constructs the left quaternion multiplication matrix.
 * 
 * @param q Input quaternion.
 * @return Matrix4x4 Left-multiplication matrix.
 */
Matrix4x4 L(const Quaternion& q);

/**
 * @brief Constructs the right quaternion multiplication matrix.
 * 
 * @param q Input quaternion.
 * @return Matrix4x4 Right-multiplication matrix.
 */
Matrix4x4 R(const Quaternion& q);

/**
 * @brief Computes the conjugate of a quaternion.
 * 
 * @param q Input quaternion.
 * @return Quaternion Conjugate quaternion.
 */
Quaternion Conj(const Quaternion& q);

/**
 * @brief Converts a quaternion to a Modified Rodrigues Parameter (MRP) vector.
 * 
 * @param q Input quaternion, expected to be normalized.
 * @return Vector MRP vector representing the same rotation.
 */
Vector MRPFromQuat(const Quaternion& q);

/**
 * @brief Multiplies two quaternions.
 * 
 * @param q1 First quaternion.
 * @param q2 Second quaternion.
 * @return Quaternion Resulting quaternion from the product.
 */
Quaternion QuatProduct(const Quaternion& q1, const Quaternion& q2);

/**
 * @brief Performs Spherical Linear Interpolation (SLERP) between two quaternions.
 * 
 * @param q1 Starting quaternion.
 * @param q2 Ending quaternion.
 * @param t Interpolation factor (0 <= t <= 1).
 * @return Quaternion Interpolated quaternion.
 */
Quaternion Slerp(const Quaternion& q1, const Quaternion& q2, double t);

/**
 * @brief Computes the quaternion derivative for kinematic equations.
 * 
 * @param q Input quaternion representing orientation.
 * @param angvel Angular velocity vector.
 * @return Quaternion Quaternion derivative.
 */
Quaternion QuaternionKinematics(const Quaternion& q, const Vector& angvel);

#endif
