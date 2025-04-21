#ifndef BATCH_OPTIMIZATION_HPP
#define BATCH_OPTIMIZATION_HPP

#include <cstdint>
#include <utility>

#include <eigen3/Eigen/Dense>

enum class BatchOptimizationState {
    NOT_STARTED = 0,
    RUNNING = 1,
    COMPLETED = 2,
    FAILED = 3
};

enum class LandmarkMeasurementIdx : uint8_t {
    TIMESTAMP = 0,
    BEARING_VEC_X = 1,
    BEARING_VEC_Y = 2,
    BEARING_VEC_Z = 3,
    LANDMARK_POS_X = 4,
    LANDMARK_POS_Y = 5,
    LANDMARK_POS_Z = 6,
    COUNT = 7
};

enum class GyroMeasurementIdx : uint8_t {
    TIMESTAMP = 0,
    ANG_VEL_X = 1,
    ANG_VEL_Y = 2,
    ANG_VEL_Z = 3,
    COUNT = 4
};

enum class StateEstimateIdx : uint8_t {
    TIMESTAMP = 0,
    POS_X = 1,
    POS_Y = 2,
    POS_Z = 3,
    VEL_X = 4,
    VEL_Y = 5,
    VEL_Z = 6,
    QUAT_X = 7,
    QUAT_Y = 8,
    QUAT_Z = 9,
    QUAT_W = 10,
    ANG_VEL_X = 11,
    ANG_VEL_Y = 12,
    ANG_VEL_Z = 13,
    COUNT = 14
};

using LandmarkMeasurements = Eigen::Matrix<double, Eigen::Dynamic, LandmarkMeasurementIdx::COUNT, Eigen::RowMajor>;
using LandmarkGroupStarts = Eigen::Matrix<bool, Eigen::Dynamic, 1, Eigen::RowMajor>;
using GyroMeasurements = Eigen::Matrix<double, Eigen::Dynamic, GyroMeasurementIdx::COUNT, Eigen::RowMajor>;
using StateEstimates = Eigen::Matrix<double, Eigen::Dynamic, StateEstimateIdx::COUNT, Eigen::RowMajor>;

/**
 * @brief Solves the batched nonlinear least squares optimization problem for orbit determination using Ceres Solver.
 *
 * For any given landmark group, only the first measurement in that group is used to read the timestamp. The rest of the
 * timestamps in that group are ignored and assumed to be the same as the first one.
 *
 * @param landmark_measurements An Eigen matrix where each row contains a landmark bearing measurement. Each row
 *                              contains 7 elements: the timestamp, the bearing unit vector to the landmark in the body
 *                              frame, and the ECI position of the landmark. The timestamps must be non-strictly
 *                              monotonically increasing.
 * @param landmark_group_starts An Eigen matrix that contains the same number of rows as in
 *                              landmark_bearing_measurements where each row contains a single bool indicating whether
 *                              or not the corresponding measurement is the start of a new group. Measurements from the
 *                              same group are captured from the same image and are assumed to have the same timestamp.
 * @param gyro_measurements An Eigen matrix where each row contains a gyro measurement. Each row contains 4 elements:
 *                          the timestamp and the angular velocity vector in the body frame. The timestamps must be
 *                          strictly monotonically increasing.
 * @param max_dt The maximum allowed time step between adjacent rows in the optimized states.
 * @return An std::pair containing:
 *         - An Eigen matrix containing the state estimates. Each row contains 14 elements: the timestamp, the ECI
 *           position of the cubesat, the ECI velocity of the cubesat, the quaternion representing the transformation
 *           from the body frame to the ECI frame, and the gyro bias in the body frame.
 *         - An Eigen vector containing the estimated gyro bias in the body frame.
 */

std::pair <StateEstimates, Eigen::Vector3d>
solve_ceres_batch_opt(const LandmarkMeasurements& landmark_measurements,
                      const LandmarkGroupStarts& landmark_group_starts,
                      const GyroMeasurements& gyro_measurements,
                      const double max_dt);

#endif // BATCH_OPTIMIZATION_HPP
