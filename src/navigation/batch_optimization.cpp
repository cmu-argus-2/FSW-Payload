#include "navigation/batch_optimization.hpp"

#include <ceres/ceres.h>
#include <ceres/manifold.h>
// Is there some symlink in place that makes ceres/eigen work by default?
#include <ceres/internal/eigen.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

using idx_t = Eigen::Index;

static constexpr double GM_EARTH = 3.9860044188e14;  // m^3/s^2

struct LinearDynamicsCostFunctor {
public:
    LinearDynamicsCostFunctor(const double& dt) : dt(dt) {}

    template<typename T>
    bool operator()(const T* const pos_curr,
                    const T* const vel_curr,
                    const T* const pos_next,
                    const T* const vel_next,
                    T* const residuals) const {
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r0(pos_curr);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> v0(vel_curr);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r1(pos_next);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> v1(vel_next);

        Eigen::Map <Eigen::Matrix<T, 3, 1>> r_res(residuals);
        Eigen::Map <Eigen::Matrix<T, 3, 1>> v_res(residuals + 3);

        const T r0_norm_squared = r0.squaredNorm();
        const T r0_norm_cubed = r0_norm_squared * ceres::sqrt(r0_norm_squared);
        r_res = r1 - (r0 + v0 * dt);
        v_res = v1 - (v0 - GM_EARTH * r0 / r0_norm_cubed * dt);
        return true;
    }

private:
    const double& dt;
};

struct AngularDynamicsCostFunctor {
public:
    AngularDynamicsCostFunctor(const double& dt) : dt(dt) {}

    template<typename T>
    bool operator()(const T* const quat_curr,
                    const T* const ang_vel_curr,
                    const T* const quat_next,
                    const T* const ang_vel_next,
                    T* const residuals) const {
        const Eigen::Map<const Eigen::Quaternion <T>> q0(quat_curr);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> w0(ang_vel_curr);
        const Eigen::Map<const Eigen::Quaternion <T>> q1(quat_next);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> w1(ang_vel_next);

        Eigen::Map <Eigen::Matrix<T, 3, 1>> q_res(residuals);  // in axis-angle form
        Eigen::Map <Eigen::Matrix<T, 3, 1>> w_res(residuals + 3);

        const T half_dt = T(0.5) * T(dt);
        const Eigen::Quaternion <T> dq = Eigen::Quaternion<T>(
                Eigen::AngleAxis<T>(w0.norm() * half_dt, w0.normalized()));
        const Eigen::Quaternion <T> q_pred = q0 * dq;
        const Eigen::AngleAxis <T> q_error = Eigen::AngleAxis<T>(q_pred.conjugate() * q1);
        q_res = q_error.angle() * q_error.axis();
        w_res = w1 - w0;
        return true;
    }

private:
    const double& dt;
};

struct GyroCostFunctor {
public:
    GyroCostFunctor(const double* const gyro_row) :
            gyro_ang_vel(gyro_row + GyroMeasurementIdx::ANG_VEL_X) {}

    template<typename T>
    bool operator()(const T* const ang_vel,
                    const T* const gyro_bias,
                    T* const residuals) const {
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> w(ang_vel);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> b(gyro_bias);

        Eigen::Map <Eigen::Matrix<T, 3, 1>> w_res(residuals);

        w_res = w - gyro_ang_vel.cast<T>() - b;
        return true;
    }

private:
    const Eigen::Map<const Eigen::Vector3d> gyro_ang_vel;
};

struct LandmarkCostFunctor {
public:
    LandmarkCostFunctor(const double* const landmark_row)
            : bearing_vec(landmark_row + LandmarkMeasurementIdx::BEARING_VEC_X),
              landmark_pos(landmark_row + LandmarkMeasurementIdx::LANDMARK_POS_X) {}

    template<typename T>
    bool operator()(const T* const pos,
                    const T* const quat,
                    T* const residuals) const {
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r(pos);
        const Eigen::Map<const Eigen::Quaternion <T>> q(quat);

        const Eigen::Matrix<T, 3, 1> landmark_pos_T   = landmark_pos.template cast<T>();
        const Eigen::Matrix<T, 3, 1> bearing_vec_T    = bearing_vec.template cast<T>();

        Eigen::Map <Eigen::Matrix<T, 3, 1>> r_res(residuals);

        const Eigen::Matrix<T, 3, 1> predicted_bearing = (landmark_pos_T - r).normalized();
        r_res = predicted_bearing - q * bearing_vec_T;
        return true;
    }

private:
    const Eigen::Map<const Eigen::Vector3d> bearing_vec;
    const Eigen::Map<const Eigen::Vector3d> landmark_pos;
};

/**
 * @brief Generates timestamps for the state estimates and the corresponding indices for the landmark groups and gyro
 *        measurements.
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
 * @param num_groups The number of landmark groups. Must be the same as the number of true values in
 *                   landmark_group_starts.
 * @return A std::tuple containing:
 *         - A vector of timestamps for the state estimates.
 *         - A vector of indices into the state_timestamps vector for each landmark group.
 *         - A vector of indices into the state_timestamps vector for each gyro measurement.
 */
std::tuple <std::vector<double>, std::vector<idx_t>, std::vector<idx_t>>
get_state_timestamps(const LandmarkMeasurements& landmark_measurements,
                     const LandmarkGroupStarts& landmark_group_starts,
                     const GyroMeasurements& gyro_measurements,
                     const double max_dt,
                     const idx_t num_groups) {
    std::vector<double> state_timestamps;
    // We know that we need at least this many timestamps, but we may need more
    state_timestamps.reserve(num_groups + gyro_measurements.rows() + 1);

    std::vector <idx_t> landmark_group_indices;
    std::vector <idx_t> gyro_measurement_indices;
    // These reservations are the exact number of indices we need
    landmark_group_indices.reserve(num_groups);
    gyro_measurement_indices.reserve(gyro_measurements.rows());

    idx_t next_landmark_idx = 0;
    idx_t next_gyro_idx = 0;

    const auto append_timestamp = [&](const double timestamp) -> void {
        if (state_timestamps.empty()) {
            state_timestamps.push_back(timestamp);
            return;
        }

        const double& last_timestamp = state_timestamps.back();
        const double dt = timestamp - last_timestamp;
        if (dt <= max_dt) {
            state_timestamps.push_back(timestamp);
            return;
        }

        const double num_steps = std::ceil(dt / max_dt);
        for (idx_t i = 1; static_cast<double>(i) < num_steps + 0.5; ++i) {
            state_timestamps.push_back(last_timestamp + (static_cast<double>(i) / num_steps) * dt);
        }
    };

    const auto append_landmark_timestamp = [&](const double landmark_timestamp) -> void {
        append_timestamp(landmark_timestamp);
        landmark_group_indices.push_back(state_timestamps.size() - 1);

        ++next_landmark_idx;
        while (next_landmark_idx < landmark_group_starts.rows() && !landmark_group_starts(next_landmark_idx, 0)) {
            ++next_landmark_idx;
        }
    };

    const auto append_gyro_timestamp = [&](const double gyro_timestamp) -> void {
        append_timestamp(gyro_timestamp);
        gyro_measurement_indices.push_back(state_timestamps.size() - 1);
        ++next_gyro_idx;
    };

    while (true) {
        const bool next_landmark_idx_valid = next_landmark_idx < landmark_group_starts.rows();
        const bool next_gyro_idx_valid = next_gyro_idx < gyro_measurements.rows();

        if (next_landmark_idx_valid && next_gyro_idx_valid) {
            assert(landmark_group_starts(next_landmark_idx, 0));
            const double& landmark_timestamp = landmark_measurements(next_landmark_idx,
                                                                     LandmarkMeasurementIdx::LANDMARK_TIMESTAMP);
            const double& gyro_timestamp = gyro_measurements(next_gyro_idx, GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP);
            if (landmark_timestamp <= gyro_timestamp) {
                append_landmark_timestamp(landmark_timestamp);
            } else {
                append_gyro_timestamp(gyro_timestamp);
            }
        } else if (next_landmark_idx_valid) {
            assert(landmark_group_starts(next_landmark_idx, 0));
            append_landmark_timestamp(landmark_measurements(next_landmark_idx, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP));
        } else if (next_gyro_idx_valid) {
            append_gyro_timestamp(gyro_measurements(next_gyro_idx, GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP));
        } else {
            break;
        }
    }
    assert(landmark_group_indices.size() == num_groups);
    assert(gyro_measurement_indices.size() == gyro_measurements.rows());

    // Manually add an extra timestamp so that we can guarantee that every gyro timestamp has another timestamp after it
    state_timestamps.push_back(state_timestamps.back() + max_dt);

    return std::make_tuple(state_timestamps,
                           landmark_group_indices,
                           gyro_measurement_indices);
}

StateEstimates solve_ceres_batch_opt(const LandmarkMeasurements& landmark_measurements,
                                     const LandmarkGroupStarts& landmark_group_starts,
                                     const GyroMeasurements& gyro_measurements,
                                     const double max_dt) {
    assert(landmark_measurements.rows() == landmark_group_starts.rows() &&
           "landmark_measurements and landmark_group_starts must have the same number of rows.");
    assert(landmark_measurements.rows() > 0 &&
           "landmark_measurements must have at least one row.");
    assert(std::adjacent_find(landmark_measurements.col(0).begin(),
                              landmark_measurements.col(0).end(),
                              [](const double timestamp, const double next_timestamp) -> bool {
                                  return next_timestamp < timestamp;
                              })
           == landmark_measurements.col(0).end() &&
           "landmark_measurements timestamps must be non-strictly monotonically increasing.");
    assert(landmark_group_starts(0, 0) == true &&
           "landmark_group_starts must start with a true value.");
    assert(gyro_measurements.rows() > 0 &&
           "gyro_measurements must have at least one row.");
    assert(std::adjacent_find(gyro_measurements.col(0).begin(),
                              gyro_measurements.col(0).end(),
                              [](const double timestamp, const double next_timestamp) -> bool {
                                  return next_timestamp <= timestamp;
                              })
           == gyro_measurements.col(0).end() &&
           "gyro_measurements timestamps must be strictly monotonically increasing.");
    assert(max_dt > 0.0 &&
           "max_dt must be greater than 0.0.");

    const idx_t num_groups = std::count(landmark_group_starts.col(0).begin(),
                                        landmark_group_starts.col(0).end(),
                                        true);
    const auto [state_timestamps, landmark_group_indices, gyro_measurement_indices] = get_state_timestamps(
            landmark_measurements,
            landmark_group_starts,
            gyro_measurements,
            max_dt,
            num_groups);

    StateEstimates state_estimates(state_timestamps.size(), StateEstimateIdx::STATE_ESTIMATE_COUNT);
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    ceres::EigenQuaternionManifold quaternion_manifold = ceres::EigenQuaternionManifold{};

    ceres::Problem::Options problem_options;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(gyro_bias.data(), 3);
    for (idx_t i = 0; i < state_timestamps.size(); ++i) {
        state_estimates(i, StateEstimateIdx::STATE_ESTIMATE_TIMESTAMP) = state_timestamps[i];
        double* const row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;

        problem.AddParameterBlock(row_start + StateEstimateIdx::POS_X,
                                  StateEstimateIdx::VEL_X - StateEstimateIdx::POS_X);
        problem.AddParameterBlock(row_start + StateEstimateIdx::VEL_X,
                                  StateEstimateIdx::QUAT_X - StateEstimateIdx::VEL_X);
        problem.AddParameterBlock(row_start + StateEstimateIdx::QUAT_X,
                                  GyroMeasurementIdx::ANG_VEL_X - StateEstimateIdx::QUAT_X,
                                  &quaternion_manifold);
        problem.AddParameterBlock(row_start + GyroMeasurementIdx::ANG_VEL_X,
                                  StateEstimateIdx::GYRO_BIAS_X - GyroMeasurementIdx::ANG_VEL_X);
        problem.AddParameterBlock(row_start + StateEstimateIdx::GYRO_BIAS_X,
                                  StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X);
    }

    // Dynamics costs
    for (idx_t i = 0; i < state_timestamps.size() - 1; ++i) {
        const double dt = state_timestamps[i + 1] - state_timestamps[i];
        double* const curr_row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        double* const next_row_start = state_estimates.data() + (i + 1) * StateEstimateIdx::STATE_ESTIMATE_COUNT;

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<LinearDynamicsCostFunctor, 6, 3, 3, 3, 3>(
                        new LinearDynamicsCostFunctor{dt}),
                nullptr,
                curr_row_start + StateEstimateIdx::POS_X,
                curr_row_start + StateEstimateIdx::VEL_X,
                next_row_start + StateEstimateIdx::POS_X,
                next_row_start + StateEstimateIdx::VEL_X);
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctor, 6, 4, 3, 4, 3>(
                        new AngularDynamicsCostFunctor{dt}),
                nullptr,
                curr_row_start + StateEstimateIdx::QUAT_X,
                curr_row_start + GyroMeasurementIdx::ANG_VEL_X,
                next_row_start + StateEstimateIdx::QUAT_X,
                next_row_start + GyroMeasurementIdx::ANG_VEL_X);
    }

    idx_t gyro_idx = 0;
    for (idx_t i = 0; i < gyro_measurement_indices.size(); ++i) {
        const idx_t& gyro_measurement_idx = gyro_measurement_indices[i];
        assert(gyro_measurement_idx < state_timestamps.size());

        auto* gyro_row = gyro_measurements.data() + gyro_measurement_idx * GyroMeasurementIdx::GYRO_MEAS_COUNT;
        auto* state_estimate_row = state_estimates.data() + gyro_measurement_idx * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GyroCostFunctor, 3, 3, 3>(new GyroCostFunctor{gyro_row}),
                nullptr,
                state_estimate_row + GyroMeasurementIdx::ANG_VEL_X,
                gyro_bias.data());
    }

    // Landmark costs
    idx_t landmark_idx = 0;
    for (const auto& landmark_group_index : landmark_group_indices) {
        assert(landmark_idx < landmark_group_starts.rows());

        double* const state_estimate_row =
                state_estimates.data() + landmark_group_index * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        double* const pos_estimate = state_estimate_row + StateEstimateIdx::POS_X;
        double* const quat_estimate = state_estimate_row + StateEstimateIdx::QUAT_X;

        do {
            auto* landmark_row =
                    landmark_measurements.data() + landmark_idx * LandmarkMeasurementIdx::LANDMARK_COUNT;
            problem.AddResidualBlock(
                    // Ceres will automatically take ownership of the cost function and cost functor
                    new ceres::AutoDiffCostFunction<LandmarkCostFunctor, 3, 3, 4>(
                            new LandmarkCostFunctor{landmark_row}),
                    nullptr,
                    pos_estimate,
                    quat_estimate);
            ++landmark_idx;
        } while (landmark_idx < landmark_group_starts.rows() && !landmark_group_starts(landmark_idx, 0));
    }
    assert(landmark_idx == landmark_group_starts.rows());

    return state_estimates;
}
