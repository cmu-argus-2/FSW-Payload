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

StateEstimates init_state_estimate(std::vector<double> state_timestamps) {
    StateEstimates state_estimates(state_timestamps.size(), StateEstimateIdx::STATE_ESTIMATE_COUNT);
    for (int i = 0; i < state_timestamps.size(); ++i) {
        state_estimates(i, StateEstimateIdx::STATE_ESTIMATE_TIMESTAMP) = state_timestamps[i];
        state_estimates(i, StateEstimateIdx::POS_X) = 0.0; // km
        state_estimates(i, StateEstimateIdx::POS_Y) = 0.0;
        state_estimates(i, StateEstimateIdx::POS_Z) = 7000.0;
        state_estimates(i, StateEstimateIdx::VEL_X) = 0.0;
        state_estimates(i, StateEstimateIdx::VEL_Y) = 8.0; // km/s
        state_estimates(i, StateEstimateIdx::VEL_Z) = 0.0; // km/s
        state_estimates(i, StateEstimateIdx::QUAT_X) = 0.0;
        state_estimates(i, StateEstimateIdx::QUAT_Y) = 0.0;
        state_estimates(i, StateEstimateIdx::QUAT_Z) = 0.0;
        state_estimates(i, StateEstimateIdx::QUAT_W) = 1.0;
        state_estimates(i, StateEstimateIdx::GYRO_BIAS_X) = 0.0;
        state_estimates(i, StateEstimateIdx::GYRO_BIAS_Y) = 0.0;
        state_estimates(i, StateEstimateIdx::GYRO_BIAS_Z) = 0.0;
    }
    return state_estimates;
}

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
            if (dt <= 1e-9) {
                for (idx_t i = 0; i < state_timestamps.size(); ++i) {
                    if (abs(state_timestamps[i] - timestamp) < 1e-9) {
                        return;
                    }
                }
            }
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
    std::cout << "Final timestamp: " << state_timestamps.back() << std::endl;
    // state_timestamps.push_back(state_timestamps.back() + dt); // + max_dt);

    return std::make_tuple(state_timestamps,
                           landmark_group_indices,
                           gyro_measurement_indices);
}

StateEstimates solve_ceres_batch_opt(const LandmarkMeasurements& landmark_measurements,
                                     const LandmarkGroupStarts& landmark_group_starts,
                                     const GyroMeasurements& gyro_measurements,
                                     const double max_dt,
                                     const std::string bias_mode) {
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
    // give an initial guess
    StateEstimates state_estimates = init_state_estimate(state_timestamps);
    
    // ceres::EigenQuaternionManifold quaternion_manifold = ceres::EigenQuaternionManifold{};

    ceres::Problem::Options problem_options;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problem_options);
    for (idx_t i = 0; i < state_timestamps.size(); ++i) {
        state_estimates(i, StateEstimateIdx::STATE_ESTIMATE_TIMESTAMP) = state_timestamps[i];
        double* const row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;

        // Position parameter block
        problem.AddParameterBlock(row_start + StateEstimateIdx::POS_X,
                                    StateEstimateIdx::VEL_X - StateEstimateIdx::POS_X);
        // Velocity parameter block
        problem.AddParameterBlock(row_start + StateEstimateIdx::VEL_X,
                                    StateEstimateIdx::QUAT_X - StateEstimateIdx::VEL_X);
        // Quaternion parameter block
        problem.AddParameterBlock(row_start + StateEstimateIdx::QUAT_X,
                                    StateEstimateIdx::GYRO_BIAS_X - StateEstimateIdx::QUAT_X,
                                    new ceres::EigenQuaternionManifold());
        // problem.SetParameterLowerBound(row_start + StateEstimateIdx::QUAT_X, 3, 0.0);
        // Gyro bias parameter block
        if (bias_mode == "tv_bias") {
            problem.AddParameterBlock(row_start + StateEstimateIdx::GYRO_BIAS_X,
                StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X);
        } else if (bias_mode == "fix_bias") {
            if (i ==0) {
                problem.AddParameterBlock(row_start + StateEstimateIdx::GYRO_BIAS_X,
                    StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X);
            }
        } else if (bias_mode == "no_bias") {
            // Do not add parameter block for gyro bias
        } else {
            throw std::invalid_argument("Invalid bias_mode: " + bias_mode +
                                        ". Must be 'tv_bias', 'fix_bias', or 'no_bias'.");
        }

    }

    // TODO: Figure out why the gradient check fails for the Linear Dynamics cost function
    // Linear Dynamics costs
    for (idx_t i = 0; i+1 < state_timestamps.size(); ++i) {
        // In the last timestep it seems to show dt 60 
        // seems to be because of the condition of requiring every timestamp to have an additional timestamp after it
        const double dt = state_timestamps[i + 1] - state_timestamps[i];
        // double* const curr_row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        // double* const next_row_start = state_estimates.data() + (i + 1) * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        
        double* p_r0 = &state_estimates(i, StateEstimateIdx::POS_X);
        double* p_v0 = &state_estimates(i, StateEstimateIdx::VEL_X);
        double* p_r1 = &state_estimates(i+1, StateEstimateIdx::POS_X);
        double* p_v1 = &state_estimates(i+1, StateEstimateIdx::VEL_X);
        problem.AddResidualBlock(
            new LinearDynamicsAnalytic{dt},
                nullptr,
                p_r0,
                p_v0,
                p_r1,
                p_v1);
        /*
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LinearDynamicsCostFunctor, 6, 3, 3, 3, 3>(
                new LinearDynamicsCostFunctor{dt}),
                nullptr,
                p_r0,
                p_v0,
                p_r1,
                p_v1);
        */
    }

    // Angular Dynamics costs
    for (idx_t i = 0; i+1 < state_timestamps.size(); ++i) {
        // In the last timestep it seems to show dt 60 
        // seems to be because of the condition of requiring every timestamp to have an additional timestamp after it
        const double dt = state_timestamps[i + 1] - state_timestamps[i];
        // double* const curr_row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        // double* const next_row_start = state_estimates.data() + (i + 1) * StateEstimateIdx::STATE_ESTIMATE_COUNT;

        double* p_q0  = &state_estimates(i, StateEstimateIdx::QUAT_X);
        double* p_q1  = &state_estimates(i+1, StateEstimateIdx::QUAT_X);

        // TODO: Find the correct gyro measurement for this time step
        auto* gyro_row = gyro_measurements.data() + GyroMeasurementIdx::GYRO_MEAS_COUNT * i;

        if (bias_mode == "fix_bias") { // fixed bias
            double* p_bw = &state_estimates(1, StateEstimateIdx::GYRO_BIAS_X);

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctorFixBias, 3, 4, 4, 3>(
                    new AngularDynamicsCostFunctorFixBias{gyro_row, dt}),
                nullptr,
                p_q0,
                p_q1,
                p_bw
            );
        } else if (bias_mode == "tv_bias") { // time-varying bias
            double* p_bw0 = &state_estimates(i, StateEstimateIdx::GYRO_BIAS_X);
            double* p_bw1 = &state_estimates(i+1, StateEstimateIdx::GYRO_BIAS_X);

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctor, 6, 4, 3, 4, 3>(
                    new AngularDynamicsCostFunctor{gyro_row, dt}),
                nullptr,
                p_q0,
                p_bw0,
                p_q1,
                p_bw1
            );
        } else if (bias_mode == "no_bias") { // no bias

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctorNoBias, 3, 4, 4>(
                    new AngularDynamicsCostFunctorNoBias{gyro_row, dt}),
                nullptr,
                p_q0,
                p_q1
            );
        }
    }

    // TODO: Add magnetometer measurements for attitude estimation

    // TODO: Figure out why the gradient check fails for the landmark cost function
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
    // std::vector<double*> residual_blocks;


    // int num_blocks = problem.NumResidualBlocks();
    // for (int block_id = 0; block_id < num_blocks; ++block_id) {
    // // const ceres::CostFunction* cost_fn = problem.residual_blocks()[block_id]->cost_function();
    // const ceres::CostFunction* cost_fn =
    //     problem.GetResidualBlocks
    // std::vector<const double*> params;
    // problem.GetParameterBlocksForResidualBlock(block_id, &params);

    // // allocate storage
    // std::vector<double>  residuals(cost_fn->num_residuals());
    // std::vector<double*> jacs(params.size());
    // for (int i = 0; i < (int)params.size(); ++i) {
    //     jacs[i] = new double[cost_fn->num_residuals() *
    //                         cost_fn->parameter_block_sizes()[i]];
    // }

    // bool ok = cost_fn->Evaluate(params.data(),
    //                             residuals.data(),
    //                             jacs.data());

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = 1000;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.check_gradients = false;
    solver_options.gradient_check_relative_precision = 1e-6;
    solver_options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    double* row_start = state_estimates.data() + 0 * StateEstimateIdx::STATE_ESTIMATE_COUNT;
    double* q_ptr = row_start + StateEstimateIdx::QUAT_X;

    std::cout << "HasManifold: " << problem.HasManifold(q_ptr) << "\n";
    std::cout << "BlockSize: " << problem.ParameterBlockSize(q_ptr) << "\n";
    std::cout << "TangentSize: " << problem.ParameterBlockTangentSize(q_ptr) << "\n";
    std::cout << "IsConstantBlock: " << problem.IsParameterBlockConstant(q_ptr) << "\n";


    std::cout << summary.FullReport() << "\n";

    if (bias_mode == "fix_bias") {
        // Copy the fixed bias to all time steps
        const double* const fixed_bias = state_estimates.data() + StateEstimateIdx::GYRO_BIAS_X;
        for (idx_t i = 1; i < state_estimates.rows(); ++i) {
            double* const row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
            double* const bias_ptr = row_start + StateEstimateIdx::GYRO_BIAS_X;
            std::copy(fixed_bias,
                      fixed_bias + (StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X),
                      bias_ptr);
        }
    }

    // Print state estimates
    for (idx_t i = 0; i < state_estimates.rows(); i+=100){
            double* const state_estimate_row = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
            std::cout << "State estimate " << i << ": ";
            for (idx_t j = 0; j < StateEstimateIdx::STATE_ESTIMATE_COUNT; ++j) {
                std::cout << state_estimate_row[j] << " ";
            }
        std::cout << "\n";
    }

    return state_estimates;
}
