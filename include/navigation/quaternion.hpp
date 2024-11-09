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
 * @brief Computes the conjugate of a quaternion. Does not normalize the input.
 * 
 * @param q Input quaternion.
 * @return Quaternion Conjugate quaternion.
 */
Quaternion Conj(const Quaternion& q);

/**
 * @brief Converts a quaternion to a Modified Rodrigues Parameter (MRP) vector.
 * 
 * This function normalizes the input quaternion before performing the conversion.
 * The input quaternion must be non-zero to avoid division by zero during normalization.
 * It is the caller's responsibility to ensure that the input quaternion is non-zero.
 * 
 * @param q Input quaternion, must be non-zero (can be unnormalized).
 * @return Vector MRP vector representing the same rotation.
 * 
 * @note Passing a zero quaternion will result in undefined behavior. The function uses assertions
 *       to check for a zero quaternion during development but does not throw exceptions.
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
 * This function computes the interpolated quaternion between two given quaternions `q1` and `q2` at
 * a parameter `t`, where `t` ranges from 0 to 1. The interpolation is done along the shortest path
 * on the 4D hypersphere to ensure smooth rotation transitions.
 * 
 * The function adjusts for the sign of the dot product to ensure the shortest path is taken.
 * For quaternions that are nearly parallel, linear interpolation is used to avoid numerical issues.
 * 
 * @param q1 Starting quaternion (must be a unit quaternion).
 * @param q2 Ending quaternion (must be a unit quaternion).
 * @param t Interpolation factor (0 <= t <= 1).
 * @return Quaternion Interpolated unit quaternion representing the rotation at parameter `t`.
 * 
 * @note Both input quaternions must be normalized (unit quaternions). The function normalizes the
 *       output quaternion before returning it. It is the caller's responsibility to ensure that `t`
 *       is within the range [0, 1].
 * @note If `t` is outside the range [0, 1], the function will extrapolate, which may not be desired.
 */
Quaternion Slerp(const Quaternion& q1, const Quaternion& q2, double t);


/**
 * @brief Computes the quaternion derivative for kinematic equations given angular velocity.
 * 
 * This function calculates the time derivative of a quaternion `q` representing orientation,
 * based on the provided angular velocity vector `angvel`. The angular velocity should be expressed
 * in the same frame as the quaternion.
 * 
 * The derivative is computed using the quaternion kinematic equation:
 * 
 *     dq/dt = 0.5 * L(q) * omega_quat
 * 
 * where `L(q)` is the left quaternion multiplication matrix of `q`, and `omega_quat` is a pure
 * quaternion (scalar part zero) constructed from the angular velocity vector.
 * 
 * @param q Input quaternion representing orientation (must be a unit quaternion).
 * @param angvel Angular velocity vector [rad/s].
 * @return Quaternion Time derivative of the quaternion.
 * 
 * @note The input quaternion must be normalized. The function assumes that the angular velocity is
 *       expressed in the same coordinate frame as the quaternion.
 */
Quaternion QuaternionKinematics(const Quaternion& q, const Vector& angvel);

#endif
