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

void build_ceres_problem(ceres::Problem::Options problem_options,
                         StateEstimates& state_estimates,
                         const std::vector<double> state_timestamps,
                         const std::vector<idx_t> landmark_group_indices,
                         const LandmarkMeasurements& landmark_measurements,
                         const LandmarkGroupStarts& landmark_group_starts,
                         const GyroMeasurements& gyro_measurements,
                         BIAS_MODE bias_mode,
                         double uma_std_dev,
                         double gyro_wn_std_dev_rad_s,
                         double gyro_bias_instability,
                         double landmark_std_dev,
                         ceres::EigenQuaternionManifold* quaternion_manifold,
                         ceres::Problem* problem) {
    // ceres::Problem problem(problem_options);
    // problem->Options(problem_options);
    
    // ceres::EigenQuaternionManifold quaternion_manifold = ceres::EigenQuaternionManifold{};

    for (idx_t i = 0; i < state_timestamps.size(); ++i) {
        state_estimates(i, StateEstimateIdx::STATE_ESTIMATE_TIMESTAMP) = state_timestamps[i];
        double* const row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;

        // Position parameter block
        problem->AddParameterBlock(row_start + StateEstimateIdx::POS_X,
                                    StateEstimateIdx::VEL_X - StateEstimateIdx::POS_X);
        // Velocity parameter block
        problem->AddParameterBlock(row_start + StateEstimateIdx::VEL_X,
                                    StateEstimateIdx::QUAT_X - StateEstimateIdx::VEL_X);
        // Quaternion parameter block
        problem->AddParameterBlock(row_start + StateEstimateIdx::QUAT_X,
                                    StateEstimateIdx::GYRO_BIAS_X - StateEstimateIdx::QUAT_X,
                                    quaternion_manifold);
        // problem.SetParameterLowerBound(row_start + StateEstimateIdx::QUAT_X, 3, 0.0);
        // Gyro bias parameter block
        if (bias_mode == BIAS_MODE::TV_BIAS) {
            problem->AddParameterBlock(row_start + StateEstimateIdx::GYRO_BIAS_X,
                StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X);
        } else if (bias_mode == BIAS_MODE::FIX_BIAS) {
            if (i ==0) {
                problem->AddParameterBlock(row_start + StateEstimateIdx::GYRO_BIAS_X,
                    StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X);
            }
        } else if (bias_mode == BIAS_MODE::NO_BIAS) {
            // Do not add parameter block for gyro bias
        } else {
            throw std::invalid_argument("Invalid bias_mode: " + std::to_string(static_cast<int>(bias_mode)) +
                                        ". Must be 2:'tv_bias', 1:'fix_bias', or 0:'no_bias'.");
        }

    }
    // Linear Dynamics costs
    for (idx_t i = 0; i+1 < state_timestamps.size(); ++i) {
        // In the last timestep it seems to show dt 60 
        // seems to be because of the condition of requiring every timestamp to have an additional timestamp after it
        const double dt = state_timestamps[i + 1] - state_timestamps[i];
        const double pos_std_dev = 0.5*uma_std_dev*dt*std::sqrt(dt); // km
        const double vel_std_dev = uma_std_dev * std::sqrt(dt); // km/s
        double* const curr_row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        double* const next_row_start = state_estimates.data() + (i + 1) * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        
        double* p_r0 = &state_estimates(i, StateEstimateIdx::POS_X);
        double* p_v0 = &state_estimates(i, StateEstimateIdx::VEL_X);
        double* p_r1 = &state_estimates(i+1, StateEstimateIdx::POS_X);
        double* p_v1 = &state_estimates(i+1, StateEstimateIdx::VEL_X);
        // new ceres::LossFunctionWrapper(
        // new ceres::HuberLoss(6.0), ceres::DO_NOT_TAKE_OWNERSHIP),
        problem->AddResidualBlock(
            new LinearDynamicsAnalytic{dt, pos_std_dev, vel_std_dev},
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
        const double quat_std_dev = gyro_wn_std_dev_rad_s * dt; // rad
        const double bias_std_dev = gyro_bias_instability * std::sqrt(dt); // rad/s
        double* const curr_row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        double* const next_row_start = state_estimates.data() + (i + 1) * StateEstimateIdx::STATE_ESTIMATE_COUNT;

        double* p_q0  = &state_estimates(i, StateEstimateIdx::QUAT_X);
        double* p_q1  = &state_estimates(i+1, StateEstimateIdx::QUAT_X);

        auto* gyro_row = gyro_measurements.data() + GyroMeasurementIdx::GYRO_MEAS_COUNT * i;

        if (bias_mode == BIAS_MODE::FIX_BIAS) { // fixed bias
            double* p_bw = &state_estimates(0, StateEstimateIdx::GYRO_BIAS_X);

            problem->AddResidualBlock(
                new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctorFixBias, 3, 4, 4, 3>(
                    new AngularDynamicsCostFunctorFixBias{gyro_row, dt, quat_std_dev}),
                new ceres::LossFunctionWrapper(
                    new ceres::HuberLoss(6.0), ceres::TAKE_OWNERSHIP),
                p_q0,
                p_q1,
                p_bw
            );
        } else if (bias_mode == BIAS_MODE::TV_BIAS) { // time-varying bias
            double* p_bw0 = &state_estimates(i, StateEstimateIdx::GYRO_BIAS_X);
            double* p_bw1 = &state_estimates(i+1, StateEstimateIdx::GYRO_BIAS_X);

            problem->AddResidualBlock(
                new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctor, 6, 4, 3, 4, 3>(
                    new AngularDynamicsCostFunctor{gyro_row, dt, quat_std_dev, bias_std_dev}),
                new ceres::LossFunctionWrapper(
                    new ceres::HuberLoss(6.0), ceres::TAKE_OWNERSHIP),
                p_q0,
                p_bw0,
                p_q1,
                p_bw1
            );
        } else if (bias_mode == BIAS_MODE::NO_BIAS) { // no bias

            problem->AddResidualBlock(
                new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctorNoBias, 3, 4, 4>(
                    new AngularDynamicsCostFunctorNoBias{gyro_row, dt, quat_std_dev}),
                new ceres::LossFunctionWrapper(
                    new ceres::HuberLoss(6.0), ceres::TAKE_OWNERSHIP),
                p_q0,
                p_q1
            );
        }
    }

    // TODO: Add magnetometer measurements for attitude estimation

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
            problem->AddResidualBlock(
                // Ceres will automatically take ownership of the cost function and cost functor
                new ceres::AutoDiffCostFunction<LandmarkCostFunctor, 3, 3, 4>(
                        new LandmarkCostFunctor{landmark_row, landmark_std_dev}),
                new ceres::LossFunctionWrapper(
                    new ceres::HuberLoss(3.0), ceres::TAKE_OWNERSHIP),
                pos_estimate,
                quat_estimate);
            ++landmark_idx;
        } while (landmark_idx < landmark_group_starts.rows() && !landmark_group_starts(landmark_idx, 0));
    }
    assert(landmark_idx == landmark_group_starts.rows());

}


std::vector<double> compute_covariance(ceres::Problem& problem,
                        StateEstimates& state_estimates,
                        BIAS_MODE bias_mode) {
    ceres::Covariance::Options options;
    // options.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
    ceres::Covariance covariance(options);

    std::vector<std::pair<const double*, const double*>> covariance_blocks;
    for (idx_t i = 0; i < state_estimates.rows(); ++i) {
        double* const row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        
        covariance_blocks.push_back(std::make_pair(
            row_start + StateEstimateIdx::POS_X,
            row_start + StateEstimateIdx::POS_X
        ));
        covariance_blocks.push_back(std::make_pair(
            row_start + StateEstimateIdx::VEL_X,
            row_start + StateEstimateIdx::VEL_X
        ));
        covariance_blocks.push_back(std::make_pair(
            row_start + StateEstimateIdx::QUAT_X,
            row_start + StateEstimateIdx::QUAT_X
        ));
        if (bias_mode == BIAS_MODE::TV_BIAS)
        covariance_blocks.push_back(std::make_pair(
            row_start + StateEstimateIdx::GYRO_BIAS_X,
            row_start + StateEstimateIdx::GYRO_BIAS_X
        ));
        else if (bias_mode == BIAS_MODE::FIX_BIAS && i == 0) {
            covariance_blocks.push_back(std::make_pair(
                row_start + StateEstimateIdx::GYRO_BIAS_X,
                row_start + StateEstimateIdx::GYRO_BIAS_X
            ));
        }
    }

    CHECK(covariance.Compute(covariance_blocks, &problem));
    bool compute_check = covariance.Compute(covariance_blocks, &problem);
    if(!compute_check) {
        std::cout << "Covariance computation failed." << std::endl;
    }
    // if (!compute_check) {
    //     std::cerr << "Covariance computation failed." << std::endl;
    //     return {};
    // }

    std::vector<double> covariance_diagonal;
    int j = 0;
    int block_size = 0;
    for (const auto& block : covariance_blocks) {
        if (bias_mode == BIAS_MODE::NO_BIAS) {
            if (j % 3 == 0) {
                int block_size = StateEstimateIdx::VEL_X - StateEstimateIdx::POS_X;
            } else if (j % 3 == 1) {
                int block_size = StateEstimateIdx::QUAT_X - StateEstimateIdx::VEL_X;
            } else {
                int block_size = StateEstimateIdx::GYRO_BIAS_X - StateEstimateIdx::QUAT_X;
            }
        } else if (bias_mode == BIAS_MODE::FIX_BIAS) {
            if (j < 4) {
                if (j % 4 == 0) {
                    block_size = StateEstimateIdx::VEL_X - StateEstimateIdx::POS_X;
                } else if (j % 4 == 1) {
                    block_size = StateEstimateIdx::QUAT_X - StateEstimateIdx::VEL_X;
                } else if (j % 4 == 2) {
                    block_size = StateEstimateIdx::GYRO_BIAS_X - StateEstimateIdx::QUAT_X;
                } else {
                    block_size = StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X;
                }
            } else {
                if ((j - 4) % 3 == 0) {
                    block_size = StateEstimateIdx::VEL_X - StateEstimateIdx::POS_X;
                } else if ((j - 4) % 3 == 1) {
                    block_size = StateEstimateIdx::QUAT_X - StateEstimateIdx::VEL_X;
                } else {
                    block_size = StateEstimateIdx::GYRO_BIAS_X - StateEstimateIdx::QUAT_X;
                }
            }
        } else if (bias_mode == BIAS_MODE::TV_BIAS) { // tv_bias
            if (j % 4 == 0) {
                block_size = StateEstimateIdx::VEL_X - StateEstimateIdx::POS_X;
            } else if (j % 4 == 1) {
                block_size = StateEstimateIdx::QUAT_X - StateEstimateIdx::VEL_X;
            } else if (j % 4 == 2) {
                block_size = StateEstimateIdx::GYRO_BIAS_X - StateEstimateIdx::QUAT_X;
            } else {
                block_size = StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X;
            }
        } else {
            throw std::invalid_argument("Invalid bias_mode: " + std::to_string(static_cast<int>(bias_mode)) +
                                        ". Must be 2:'tv_bias', 1:'fix_bias', or 0:'no_bias'.");
        }
        j += 1;
        // TODO: get attitude residual covariance in rotation vector/angle-axis form
        // This can be done with GetCovarianceBlockInTangentSpace
        std::vector<std::pair<const double*, const double*>> single_block = {block};
        double cov_matrix[block_size * block_size];
        covariance.GetCovarianceBlock(block.first, block.first, cov_matrix);
        
        for (int i = 0; i < block_size; ++i) {
            covariance_diagonal.push_back(cov_matrix[i * block_size + i]);
        }
    }
    
    return covariance_diagonal;
}

std::tuple <StateEstimates, std::vector<double>>
solve_ceres_batch_opt(const LandmarkMeasurements& landmark_measurements,
                      const LandmarkGroupStarts& landmark_group_starts,
                      const GyroMeasurements& gyro_measurements,
                      BATCH_OPT_config bo_config) {
    // Unpack configuration
    const double max_dt = bo_config.max_dt;
    const BIAS_MODE bias_mode = bo_config.bias_mode;

    // Input validation
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
    
    // TODO: this information should be obtained from the configuration file
    const double uma_std_dev = 1; // km/s^2
    const double gyro_wn_std_dev_rad_s = 0.0008726; //0.001; // rad/s
    const double gyro_bias_instability = 1; // 0.0001; // rad/s^2
    // TODO: this information should be obtained from the output of the LD nets
    const double landmark_std_dev = 0.009; // very conservative assumption of 1/2 degree error
    // landmark std dev from 1/2 bounding box size (1/12 deg of lat/long) -> earth surface distance -> camera fov

    

    ceres::EigenQuaternionManifold quaternion_manifold = ceres::EigenQuaternionManifold{};

    ceres::Problem::Options problem_options;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problem_options);
    build_ceres_problem(problem_options,
                         state_estimates,
                         state_timestamps,
                         landmark_group_indices,
                         landmark_measurements,
                         landmark_group_starts,
                         gyro_measurements,
                         bias_mode,
                         uma_std_dev,
                         gyro_wn_std_dev_rad_s,
                         gyro_bias_instability,
                         landmark_std_dev,
                         &quaternion_manifold,
                        &problem);

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = bo_config.max_iterations;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.use_explicit_schur_complement = true;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.check_gradients = false;
    solver_options.gradient_check_relative_precision = 1e-6;
    solver_options.function_tolerance = bo_config.solver_function_tolerance;
    // solver_options.preconditioner_type = ceres::SCHUR_POWER_SERIES_EXPANSION;
    solver_options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    if (bias_mode == BIAS_MODE::FIX_BIAS) {
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
    std::vector<double> covariance = compute_covariance(problem, state_estimates, bias_mode);

    // TODO: compute residuals

    // return state_estimates
    return std::make_tuple(state_estimates,
                            covariance);
    //                        residuals);
}