#ifndef BATCH_OPTIMIZATION_HPP
#define BATCH_OPTIMIZATION_HPP

#include "navigation/od_measurements.hpp"
#include "navigation/od.hpp"
#include "navigation/pose_dynamics.hpp"
#include <cstdint>
#include <utility>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>

// ── Ceres cost functor for landmark bearing residuals ──────────────────────────
struct LandmarkCostFunctor {
public:
    LandmarkCostFunctor(const double* const landmark_row, const double landmark_std_dev)
            : bearing_vec(landmark_row + LandmarkMeasurementIdx::BEARING_VEC_X),
              landmark_pos(landmark_row + LandmarkMeasurementIdx::LANDMARK_POS_X),
              landmark_std_dev(landmark_std_dev) {}

    template<typename T>
    bool operator()(const T* const pos,
                    const T* const quat,
                    T* const residuals) const {
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r(pos);
        const Eigen::Map<const Eigen::Quaternion <T>> q(quat);

        const Eigen::Matrix<T, 3, 1> landmark_pos_T = landmark_pos.template cast<T>();
        const Eigen::Matrix<T, 3, 1> bearing_vec_T  = bearing_vec.template cast<T>();

        Eigen::Map<Eigen::Matrix<T, 3, 1>> r_res(residuals);

        const Eigen::Matrix<T, 3, 1> diff = (landmark_pos_T - r);
        const T norm_sq  = diff.squaredNorm();
        const T eps      = T(1e-6);
        const T inv_norm = T(1.0) / ceres::sqrt(norm_sq + eps);
        const Eigen::Matrix<T, 3, 1> predicted_bearing = diff * inv_norm;

        r_res = (q.inverse() * predicted_bearing - bearing_vec_T) / T(landmark_std_dev);

        return true;
    }

private:
    const Eigen::Map<const Eigen::Vector3d> bearing_vec;
    const Eigen::Map<const Eigen::Vector3d> landmark_pos;
    const double landmark_std_dev;
};

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

// Indexes for the residuals/covariance vector in the tangent space of the manifold
enum StateResCovIdx {
    RES_COV_TIMESTAMP = 0,
    POS_COV_X = 1,
    POS_COV_Y = 2,
    POS_COV_Z = 3,
    VEL_COV_X = 4,
    VEL_COV_Y = 5,
    VEL_COV_Z = 6,
    ROT_COV_X = 7,
    ROT_COV_Y = 8,
    ROT_COV_Z = 9,
    GYRO_BIAS_COV_X = 10,
    GYRO_BIAS_COV_Y = 11,
    GYRO_BIAS_COV_Z = 12,
    STATE_RES_COV_COUNT = 13
};

using LandmarkMeasurements = Eigen::Matrix<double, Eigen::Dynamic, LandmarkMeasurementIdx::LANDMARK_COUNT, Eigen::RowMajor>;
using LandmarkGroupStarts = Eigen::Matrix<bool, Eigen::Dynamic, 1, Eigen::ColMajor>;
using GyroMeasurements = Eigen::Matrix<double, Eigen::Dynamic, GyroMeasurementIdx::GYRO_MEAS_COUNT, Eigen::RowMajor>;
using StateEstimates = Eigen::Matrix<double, Eigen::Dynamic, StateEstimateIdx::STATE_ESTIMATE_COUNT, Eigen::RowMajor>;
using ResidualsOrCovariances = Eigen::Matrix<double, Eigen::Dynamic, StateResCovIdx::STATE_RES_COV_COUNT, Eigen::RowMajor>;
using idx_t = Eigen::Index;

struct StateTimestampsResult {
    ErrorCode            code = ErrorCode::OK;
    std::vector<double>  state_timestamps;
    std::vector<idx_t>   landmark_group_indices;
};

struct BatchOptResult {
    ErrorCode           code = ErrorCode::OK;
    StateEstimates      state_estimates;
    std::vector<double> covariance;
    std::vector<double> residuals;
};

StateTimestampsResult
get_state_timestamps(const LandmarkMeasurements& landmark_measurements,
                     const LandmarkGroupStarts& landmark_group_starts,
                     const GyroMeasurements& gyro_measurements,
                     const idx_t num_groups);

std::vector<double> compute_covariance(ceres::Problem& problem,
                                       StateEstimates& state_estimates,
                                       BIAS_MODE bias_mode);

ErrorCode build_ceres_problem(StateEstimates& state_estimates,
                              const std::vector<double> state_timestamps,
                              const std::vector<idx_t> landmark_group_indices,
                              const LandmarkMeasurements& landmark_measurements,
                              const LandmarkGroupStarts& landmark_group_starts,
                              const GyroMeasurements& gyro_measurements,
                              BIAS_MODE bias_mode,
                              double uma_std_dev,
                              double gyro_wn_std_dev_rad_s,
                              double gyro_bias_instability,
                              const Eigen::VectorXd& landmark_uncertainties,
                              ceres::EigenQuaternionManifold* quaternion_manifold,
                              ceres::Problem* problem);

// Runs the batch nonlinear least-squares orbit determination optimizer.
// Returns BatchOptResult::code == ErrorCode::OK on success. On failure the code
// is set to the appropriate ErrorCode (ODMEAS_NOT_VALID, BATCH_OPT_BUILD_FAILED,
// BATCH_OPT_NO_CONVERGENCE, BATCH_OPT_SOLVER_FAILED, or BATCH_OPT_INVALID_OUTPUT)
// and the remaining fields are empty. Covariance failure is non-fatal (code stays
// OK, covariance field is empty).
BatchOptResult solve_ceres_batch_opt(const ODMeasurements& measurements,
                                     BATCH_OPT_config bo_config);

#endif // BATCH_OPTIMIZATION_HPP
