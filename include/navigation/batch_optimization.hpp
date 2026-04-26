#ifndef BATCH_OPTIMIZATION_HPP
#define BATCH_OPTIMIZATION_HPP

#include "navigation/od_measurements.hpp"
#include "navigation/od.hpp"
#include "navigation/pose_dynamics.hpp"
#include <cstdint>
#include <string>
#include <utility>

#include <eigen3/Eigen/Dense>

enum class BatchOptimizationState {
    NOT_STARTED = 0,
    RUNNING = 1,
    COMPLETED = 2,
    FAILED = 3
};

enum StateEstimateIdx {
    STATE_ESTIMATE_TIMESTAMP = 0,
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
    GYRO_BIAS_X = 11,
    GYRO_BIAS_Y = 12,
    GYRO_BIAS_Z = 13,
    STATE_ESTIMATE_COUNT = 14
};

// Column indexes shared by the covariance matrix and the dynamics residuals matrix.
// Both are (N-1)×13 (or N×13 for covariance) with one column per tangent-space component.
enum StateResIdx {
    RES_TIMESTAMP      = 0,
    RES_POS_X          = 1,
    RES_POS_Y          = 2,
    RES_POS_Z          = 3,
    RES_VEL_X          = 4,
    RES_VEL_Y          = 5,
    RES_VEL_Z          = 6,
    RES_ROT_X          = 7,
    RES_ROT_Y          = 8,
    RES_ROT_Z          = 9,
    RES_GYRO_BIAS_X    = 10,
    RES_GYRO_BIAS_Y    = 11,
    RES_GYRO_BIAS_Z    = 12,
    STATE_RES_COUNT    = 13
};

// Column indexes for the landmark measurement residuals matrix (M rows, one per measurement)
enum LandmarkResIdx {
    LANDMARK_RES_X     = 0,
    LANDMARK_RES_Y     = 1,
    LANDMARK_RES_Z     = 2,
    LANDMARK_RES_COUNT = 3
};

using LandmarkMeasurements   = Eigen::Matrix<double, Eigen::Dynamic, LandmarkMeasurementIdx::LANDMARK_COUNT, Eigen::RowMajor>;
using LandmarkGroupStarts    = Eigen::Matrix<bool,   Eigen::Dynamic, 1, Eigen::ColMajor>;
using GyroMeasurements       = Eigen::Matrix<double, Eigen::Dynamic, GyroMeasurementIdx::GYRO_MEAS_COUNT, Eigen::RowMajor>;
using StateEstimates         = Eigen::Matrix<double, Eigen::Dynamic, StateEstimateIdx::STATE_ESTIMATE_COUNT, Eigen::RowMajor>;
using ResidualsOrCovariances = Eigen::Matrix<double, Eigen::Dynamic, StateResIdx::STATE_RES_COUNT, Eigen::RowMajor>;
using DynamicsResiduals      = Eigen::Matrix<double, Eigen::Dynamic, StateResIdx::STATE_RES_COUNT, Eigen::RowMajor>;
using LandmarkResiduals      = Eigen::Matrix<double, Eigen::Dynamic, LandmarkResIdx::LANDMARK_RES_COUNT, Eigen::RowMajor>;
using idx_t = Eigen::Index;

struct StateTimestampsResult {
    ErrorCode            code = ErrorCode::OK;
    std::vector<double>  state_timestamps;
    std::vector<idx_t>   landmark_group_indices;
};

struct SolverSummaryInfo {
    int    termination_type = 0;  // ceres::TerminationType: 0=CONVERGENCE, 1=NO_CONVERGENCE, 2=FAILURE, 3=USER_SUCCESS, 4=USER_FAILURE
    int    num_iterations   = 0;
    double initial_cost     = 0.0;
    double final_cost       = 0.0;
};

struct BatchOptResult {
    ErrorCode              code = ErrorCode::OK;
    StateEstimates         initial_trajectory;  // Nx14 pre-solve initial guess
    StateEstimates         state_estimates;
    ResidualsOrCovariances covariance;          // Nx13 per StateResIdx; 0 rows if unavailable
    DynamicsResiduals      dynamics_residuals;  // (N-1)x13 per StateResIdx
    LandmarkResiduals      landmark_residuals;  // Mx3 per LandmarkResIdx
    SolverSummaryInfo      solver_summary;
};

StateTimestampsResult
get_state_timestamps(const LandmarkMeasurements& landmark_measurements,
                     const LandmarkGroupStarts& landmark_group_starts,
                     const GyroMeasurements& gyro_measurements,
                     const idx_t num_groups);

// Runs the batch NLP orbit determination optimizer (IPOPT).
// Returns BatchOptResult::code == ErrorCode::OK on success. On failure the code
// is set to ODMEAS_NOT_VALID, BATCH_OPT_NO_CONVERGENCE, BATCH_OPT_SOLVER_FAILED,
// or BATCH_OPT_INVALID_OUTPUT. The covariance field is always empty (0 rows).
BatchOptResult solve_batch_opt(const ODMeasurements& measurements,
                               BATCH_OPT_config bo_config);

#endif // BATCH_OPTIMIZATION_HPP
