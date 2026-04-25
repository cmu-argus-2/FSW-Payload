#include "navigation/batch_optimization.hpp"
#include <ceres/ceres.h>
#include <ceres/manifold.h>
// Is there some symlink in place that makes ceres/eigen work by default?
#include <ceres/internal/eigen.h>
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>


// Interpolate/extrapolate position from sparse keyframes {t, pos}.
static Eigen::Vector3d interp_kf(
        const std::vector<std::pair<double, Eigen::Vector3d>>& kf,
        double t)
{
    const size_t n = kf.size();
    if (n == 1) return kf[0].second;

    if (t <= kf.front().first) {
        const double dt = kf[1].first - kf[0].first;
        if (std::abs(dt) < 1e-9) return kf[0].second;
        return kf[0].second + (kf[1].second - kf[0].second) / dt * (t - kf[0].first);
    }
    if (t >= kf.back().first) {
        const double dt = kf[n-1].first - kf[n-2].first;
        if (std::abs(dt) < 1e-9) return kf.back().second;
        return kf.back().second + (kf[n-1].second - kf[n-2].second) / dt * (t - kf.back().first);
    }
    size_t lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const size_t mid = (lo + hi) / 2;
        if (kf[mid].first <= t) lo = mid; else hi = mid;
    }
    const double alpha = (t - kf[lo].first) / (kf[hi].first - kf[lo].first);
    return (1.0 - alpha) * kf[lo].second + alpha * kf[hi].second;
}


StateEstimates init_state_estimate(
        const std::vector<double>& state_timestamps,
        const LandmarkMeasurements& landmark_measurements,
        const LandmarkGroupStarts& landmark_group_starts)
{
    // Earth mean radius + 600 km orbit
    constexpr double R_ORBIT_KM = 6371.0 + 600.0;

    // One position keyframe per landmark group: mean landmark ECI position
    // direction normalized to the assumed orbital altitude.
    std::vector<std::pair<double, Eigen::Vector3d>> keyframes;

    const idx_t N = landmark_measurements.rows();
    idx_t row = 0;
    while (row < N) {
        const double t_group =
            landmark_measurements(row, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP);
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        do {
            sum.x() += landmark_measurements(row, LandmarkMeasurementIdx::LANDMARK_POS_X);
            sum.y() += landmark_measurements(row, LandmarkMeasurementIdx::LANDMARK_POS_Y);
            sum.z() += landmark_measurements(row, LandmarkMeasurementIdx::LANDMARK_POS_Z);
            ++row;
        } while (row < N && !landmark_group_starts(row, 0));

        const double norm = sum.norm();
        if (norm < 1e-6) continue;
        keyframes.emplace_back(t_group, sum / norm * R_ORBIT_KM);
    }
    spdlog::info("init_state_estimate: {} landmark groups → {} position keyframes",
                 keyframes.size(), keyframes.size());

    const idx_t M = static_cast<idx_t>(state_timestamps.size());
    StateEstimates se(M, StateEstimateIdx::STATE_ESTIMATE_COUNT);

    // Fill position by interpolating between group keyframes.
    for (idx_t i = 0; i < M; ++i) {
        const double t = state_timestamps[static_cast<size_t>(i)];
        const Eigen::Vector3d pos = keyframes.empty()
            ? Eigen::Vector3d(0.0, 0.0, R_ORBIT_KM)
            : interp_kf(keyframes, t);
        se(i, StateEstimateIdx::STATE_ESTIMATE_TIMESTAMP) = t;
        se(i, StateEstimateIdx::POS_X)       = pos.x();
        se(i, StateEstimateIdx::POS_Y)       = pos.y();
        se(i, StateEstimateIdx::POS_Z)       = pos.z();
        se(i, StateEstimateIdx::QUAT_X)      = 0.0;
        se(i, StateEstimateIdx::QUAT_Y)      = 0.0;
        se(i, StateEstimateIdx::QUAT_Z)      = 0.0;
        se(i, StateEstimateIdx::QUAT_W)      = 1.0;
        se(i, StateEstimateIdx::GYRO_BIAS_X) = 0.0;
        se(i, StateEstimateIdx::GYRO_BIAS_Y) = 0.0;
        se(i, StateEstimateIdx::GYRO_BIAS_Z) = 0.0;
    }

    // Velocity as finite differences: central at interior, one-sided at endpoints.
    for (idx_t i = 0; i < M; ++i) {
        const idx_t i0 = (i == 0)     ? 0     : i - 1;
        const idx_t i1 = (i == M - 1) ? M - 1 : i + 1;
        if (i0 == i1) {
            se(i, StateEstimateIdx::VEL_X) = 0.0;
            se(i, StateEstimateIdx::VEL_Y) = 0.0;
            se(i, StateEstimateIdx::VEL_Z) = 0.0;
            continue;
        }
        const double dt = state_timestamps[static_cast<size_t>(i1)]
                        - state_timestamps[static_cast<size_t>(i0)];
        se(i, StateEstimateIdx::VEL_X) =
            (se(i1, StateEstimateIdx::POS_X) - se(i0, StateEstimateIdx::POS_X)) / dt;
        se(i, StateEstimateIdx::VEL_Y) =
            (se(i1, StateEstimateIdx::POS_Y) - se(i0, StateEstimateIdx::POS_Y)) / dt;
        se(i, StateEstimateIdx::VEL_Z) =
            (se(i1, StateEstimateIdx::POS_Z) - se(i0, StateEstimateIdx::POS_Z)) / dt;
    }

    return se;
}


StateTimestampsResult
get_state_timestamps(const LandmarkMeasurements& landmark_measurements,
                     const LandmarkGroupStarts& landmark_group_starts,
                     const GyroMeasurements& gyro_measurements,
                     const idx_t num_groups) {
    StateTimestampsResult result;
    const idx_t M = gyro_measurements.rows();

    // State timestamps are exactly the gyro measurement timestamps.
    result.state_timestamps.reserve(M);
    for (idx_t k = 0; k < M; ++k)
        result.state_timestamps.push_back(gyro_measurements(k, GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP));

    // For each landmark group, snap to the nearest gyro timestamp.
    result.landmark_group_indices.reserve(num_groups);

    for (idx_t i = 0; i < landmark_group_starts.rows(); ++i) {
        if (!landmark_group_starts(i, 0)) continue;

        const double t_lm = landmark_measurements(i, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP);

        // Binary search for the nearest gyro timestamp.
        auto it = std::lower_bound(result.state_timestamps.begin(), result.state_timestamps.end(), t_lm);
        idx_t best_k;
        if (it == result.state_timestamps.end()) {
            best_k = M - 1;
        } else if (it == result.state_timestamps.begin()) {
            best_k = 0;
        } else {
            const auto prev = std::prev(it);
            best_k = (*it - t_lm < t_lm - *prev)
                     ? static_cast<idx_t>(it   - result.state_timestamps.begin())
                     : static_cast<idx_t>(prev - result.state_timestamps.begin());
        }
        result.landmark_group_indices.push_back(best_k);
    }

    if (static_cast<idx_t>(result.landmark_group_indices.size()) != num_groups) {
        spdlog::error("get_state_timestamps: counted {} landmark group starts but expected {}; "
                      "landmark_group_starts may be malformed.",
                      result.landmark_group_indices.size(), num_groups);
        LogError(ErrorCode::BATCH_OPT_BUILD_FAILED);
        result.code = ErrorCode::BATCH_OPT_BUILD_FAILED;
        return result;
    }

    spdlog::info("State timestamps: {} gyro measurements over {:.3f} s; "
                 "{} landmark groups snapped to nearest gyro timestamp.",
                 M, result.state_timestamps.back() - result.state_timestamps.front(), num_groups);

    return result;
}

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
                              ceres::Problem* problem) {

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
            spdlog::error("build_ceres_problem: invalid bias_mode {}; "
                          "must be 2:'tv_bias', 1:'fix_bias', or 0:'no_bias'.",
                          static_cast<int>(bias_mode));
            LogError(ErrorCode::BATCH_OPT_BUILD_FAILED);
            return ErrorCode::BATCH_OPT_BUILD_FAILED;
        }

    }
    // Linear Dynamics costs
    for (idx_t i = 0; i+1 < state_timestamps.size(); ++i) {
        // In the last timestep it seems to show dt 60
        // seems to be because of the condition of requiring every timestamp to have an additional timestamp after it
        const double dt = state_timestamps[i + 1] - state_timestamps[i];
        const double pos_std_dev = 0.5*uma_std_dev*dt*std::sqrt(dt); // km
        const double vel_std_dev = uma_std_dev * std::sqrt(dt); // km/s

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
    }

    // Angular Dynamics costs.
    // State timestamps == gyro timestamps (one-to-one), so index i directly.
    for (idx_t i = 0; i+1 < static_cast<idx_t>(state_timestamps.size()); ++i) {
        const double dt = state_timestamps[i + 1] - state_timestamps[i];
        const double quat_std_dev = gyro_wn_std_dev_rad_s * dt; // rad
        const double bias_std_dev = gyro_bias_instability * std::sqrt(dt); // rad/s

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

    // TODO (Optional): Add magnetometer measurements for attitude estimation

    // Landmark costs
    idx_t landmark_idx = 0;
    for (const auto& landmark_group_index : landmark_group_indices) {
        if (landmark_idx >= landmark_group_starts.rows()) {
            spdlog::error("build_ceres_problem: landmark_idx {} is out of bounds (rows={}); "
                          "landmark_group_indices contains more groups than group_starts rows.",
                          landmark_idx, landmark_group_starts.rows());
            LogError(ErrorCode::BATCH_OPT_BUILD_FAILED);
            return ErrorCode::BATCH_OPT_BUILD_FAILED;
        }
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
                        new LandmarkCostFunctor{landmark_row, landmark_uncertainties(landmark_idx)}),
                new ceres::LossFunctionWrapper(
                    new ceres::HuberLoss(3.0), ceres::TAKE_OWNERSHIP),
                pos_estimate,
                quat_estimate);
            ++landmark_idx;
        } while (landmark_idx < landmark_group_starts.rows() && !landmark_group_starts(landmark_idx, 0));
    }
    if (landmark_idx != landmark_group_starts.rows()) {
        spdlog::error("build_ceres_problem: consumed {} landmark rows but expected {}; "
                      "group boundary accounting is inconsistent.",
                      landmark_idx, landmark_group_starts.rows());
        LogError(ErrorCode::BATCH_OPT_BUILD_FAILED);
        return ErrorCode::BATCH_OPT_BUILD_FAILED;
    }

    return ErrorCode::OK;
}

ResidualsOrCovariances compute_covariance(ceres::Problem& problem,
                                          StateEstimates& state_estimates,
                                          BIAS_MODE bias_mode) {
    const idx_t N = state_estimates.rows();
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);

    std::vector<std::pair<const double*, const double*>> covariance_blocks;
    for (idx_t i = 0; i < N; ++i) {
        const double* const row = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        covariance_blocks.push_back({row + StateEstimateIdx::POS_X,        row + StateEstimateIdx::POS_X});
        covariance_blocks.push_back({row + StateEstimateIdx::VEL_X,        row + StateEstimateIdx::VEL_X});
        covariance_blocks.push_back({row + StateEstimateIdx::QUAT_X,       row + StateEstimateIdx::QUAT_X});
        if (bias_mode == BIAS_MODE::TV_BIAS) {
            covariance_blocks.push_back({row + StateEstimateIdx::GYRO_BIAS_X, row + StateEstimateIdx::GYRO_BIAS_X});
        } else if (bias_mode == BIAS_MODE::FIX_BIAS && i == 0) {
            covariance_blocks.push_back({row + StateEstimateIdx::GYRO_BIAS_X, row + StateEstimateIdx::GYRO_BIAS_X});
        }
    }

    if (!covariance.Compute(covariance_blocks, &problem)) {
        spdlog::error("Covariance computation failed.");
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }

    ResidualsOrCovariances result(N, StateResIdx::STATE_RES_COUNT);
    result.setZero();

    // For FIX_BIAS: extract shared bias covariance from row 0 and replicate
    double fix_bias_cov[3] = {0.0, 0.0, 0.0};
    if (bias_mode == BIAS_MODE::FIX_BIAS) {
        const double* bias_ptr = state_estimates.data() + StateEstimateIdx::GYRO_BIAS_X;
        double cov9[9];
        covariance.GetCovarianceBlockInTangentSpace(bias_ptr, bias_ptr, cov9);
        fix_bias_cov[0] = cov9[0];
        fix_bias_cov[1] = cov9[4];
        fix_bias_cov[2] = cov9[8];
    }

    double cov9[9];
    for (idx_t i = 0; i < N; ++i) {
        const double* const row = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
        result(i, StateResIdx::RES_TIMESTAMP) =
                state_estimates(i, StateEstimateIdx::STATE_ESTIMATE_TIMESTAMP);

        covariance.GetCovarianceBlockInTangentSpace(
                row + StateEstimateIdx::POS_X, row + StateEstimateIdx::POS_X, cov9);
        result(i, StateResIdx::RES_POS_X) = cov9[0];
        result(i, StateResIdx::RES_POS_Y) = cov9[4];
        result(i, StateResIdx::RES_POS_Z) = cov9[8];

        covariance.GetCovarianceBlockInTangentSpace(
                row + StateEstimateIdx::VEL_X, row + StateEstimateIdx::VEL_X, cov9);
        result(i, StateResIdx::RES_VEL_X) = cov9[0];
        result(i, StateResIdx::RES_VEL_Y) = cov9[4];
        result(i, StateResIdx::RES_VEL_Z) = cov9[8];

        covariance.GetCovarianceBlockInTangentSpace(
                row + StateEstimateIdx::QUAT_X, row + StateEstimateIdx::QUAT_X, cov9);
        result(i, StateResIdx::RES_ROT_X) = cov9[0];
        result(i, StateResIdx::RES_ROT_Y) = cov9[4];
        result(i, StateResIdx::RES_ROT_Z) = cov9[8];

        if (bias_mode == BIAS_MODE::TV_BIAS) {
            covariance.GetCovarianceBlockInTangentSpace(
                    row + StateEstimateIdx::GYRO_BIAS_X, row + StateEstimateIdx::GYRO_BIAS_X, cov9);
            result(i, StateResIdx::RES_GYRO_BIAS_X) = cov9[0];
            result(i, StateResIdx::RES_GYRO_BIAS_Y) = cov9[4];
            result(i, StateResIdx::RES_GYRO_BIAS_Z) = cov9[8];
        } else if (bias_mode == BIAS_MODE::FIX_BIAS) {
            result(i, StateResIdx::RES_GYRO_BIAS_X) = fix_bias_cov[0];
            result(i, StateResIdx::RES_GYRO_BIAS_Y) = fix_bias_cov[1];
            result(i, StateResIdx::RES_GYRO_BIAS_Z) = fix_bias_cov[2];
        }
        // NO_BIAS: bias columns remain zero
    }

    return result;
}

BatchOptResult solve_ceres_batch_opt(const ODMeasurements& measurements,
                                     BATCH_OPT_config bo_config) {
    BatchOptResult result;

    // ── Validate inputs ───────────────────────────────────────────────────────
    if (measurements.Validate() != ErrorCode::OK) {
        result.code = ErrorCode::ODMEAS_NOT_VALID;
        return result;
    }

    // Convert from dynamic ColMajor storage (ODMeasurements) to the RowMajor typed
    // aliases expected by the internal solver functions.
    const LandmarkMeasurements landmark_measurements = measurements.landmark_measurements;
    const LandmarkGroupStarts  landmark_group_starts = measurements.group_starts;
    const GyroMeasurements     gyro_measurements     = measurements.gyro_measurements;

    const BIAS_MODE bias_mode = bo_config.bias_mode;

    const idx_t num_groups = std::count(landmark_group_starts.col(0).begin(),
                                        landmark_group_starts.col(0).end(),
                                        true);

    // ── Build timestamps and group index map ──────────────────────────────────
    StateTimestampsResult ts = get_state_timestamps(
            landmark_measurements, landmark_group_starts, gyro_measurements, num_groups);
    if (ts.code != ErrorCode::OK) {
        result.code = ts.code;
        return result;
    }

    StateEstimates state_estimates = init_state_estimate(
            ts.state_timestamps, landmark_measurements, landmark_group_starts);

    // TODO: this information should be obtained from the configuration file
    double uma_std_dev = 1; // km/s^2
    const double gyro_wn_std_dev_rad_s = 0.0008726; // rad/s
    const double gyro_bias_instability = 1; // rad/s^2

    ceres::EigenQuaternionManifold quaternion_manifold = ceres::EigenQuaternionManifold{};

    ceres::Problem::Options problem_options;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problem_options);

    // ── Build Ceres problem ───────────────────────────────────────────────────
    const ErrorCode build_ec = build_ceres_problem(
            state_estimates, ts.state_timestamps, ts.landmark_group_indices,
            landmark_measurements, landmark_group_starts, gyro_measurements,
            bias_mode, uma_std_dev, gyro_wn_std_dev_rad_s, gyro_bias_instability,
            measurements.landmark_uncertainties, &quaternion_manifold, &problem);
    if (build_ec != ErrorCode::OK) {
        result.code = build_ec;
        return result;
    }

    // ── Solve ─────────────────────────────────────────────────────────────────
    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = bo_config.max_iterations;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.function_tolerance = bo_config.solver_function_tolerance;
    solver_options.parameter_tolerance = bo_config.solver_parameter_tolerance;
    solver_options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    spdlog::info("Ceres Solver Summary:\n{}", summary.FullReport());

    // Populate summary info on all paths so callers always have it.
    result.solver_summary.termination_type = static_cast<int>(summary.termination_type);
    result.solver_summary.num_iterations   = summary.num_successful_steps + summary.num_unsuccessful_steps;
    result.solver_summary.initial_cost     = summary.initial_cost;
    result.solver_summary.final_cost       = summary.final_cost;

    switch (summary.termination_type) {
        case ceres::CONVERGENCE:
            break;
        case ceres::NO_CONVERGENCE:
            spdlog::error("solve_ceres_batch_opt: solver did not converge within limits.");
            LogError(ErrorCode::BATCH_OPT_NO_CONVERGENCE);
            result.code = ErrorCode::BATCH_OPT_NO_CONVERGENCE;
            return result;
        default:
            spdlog::error("solve_ceres_batch_opt: solver failed — {}.", summary.message);
            LogError(ErrorCode::BATCH_OPT_SOLVER_FAILED);
            result.code = ErrorCode::BATCH_OPT_SOLVER_FAILED;
            return result;
    }

    // ── Post-solve validity check ─────────────────────────────────────────────
    bool output_valid = true;
    for (Eigen::Index j = 0; j < state_estimates.size(); ++j) {
        if (!std::isfinite(state_estimates.data()[j])) { output_valid = false; break; }
    }
    if (output_valid) {
        for (idx_t i = 0; i < state_estimates.rows(); ++i) {
            const double qx = state_estimates(i, StateEstimateIdx::QUAT_X);
            const double qy = state_estimates(i, StateEstimateIdx::QUAT_Y);
            const double qz = state_estimates(i, StateEstimateIdx::QUAT_Z);
            const double qw = state_estimates(i, StateEstimateIdx::QUAT_W);
            if (std::abs(std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw) - 1.0) > 0.05) {
                output_valid = false; break;
            }
        }
    }
    if (!output_valid) {
        spdlog::error("solve_ceres_batch_opt: output contains NaN/inf or denormalized quaternions.");
        LogError(ErrorCode::BATCH_OPT_INVALID_OUTPUT);
        result.code = ErrorCode::BATCH_OPT_INVALID_OUTPUT;
        return result;
    }

    // ── FIX_BIAS: propagate shared bias to all rows ───────────────────────────
    if (bias_mode == BIAS_MODE::FIX_BIAS) {
        const double* const fixed_bias = state_estimates.data() + StateEstimateIdx::GYRO_BIAS_X;
        for (idx_t i = 1; i < state_estimates.rows(); ++i) {
            double* const row_start = state_estimates.data() + i * StateEstimateIdx::STATE_ESTIMATE_COUNT;
            double* const bias_ptr = row_start + StateEstimateIdx::GYRO_BIAS_X;
            std::copy(fixed_bias,
                      fixed_bias + (StateEstimateIdx::STATE_ESTIMATE_COUNT - StateEstimateIdx::GYRO_BIAS_X),
                      bias_ptr);
        }
    }

    // ── Covariance (non-fatal) ────────────────────────────────────────────────
    result.covariance = compute_covariance(problem, state_estimates, bias_mode);

    // ── Residuals ─────────────────────────────────────────────────────────────
    // Ceres appends residual blocks in add order: linear dynamics, angular dynamics,
    // then landmark measurements. Unpack into structured matrices.
    std::vector<double> raw_residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &raw_residuals, nullptr, nullptr);

    const idx_t num_intervals  = state_estimates.rows() - 1;
    const int   ang_res_stride = (bias_mode == BIAS_MODE::TV_BIAS) ? 6 : 3;
    const int   lin_res_stride = 6;
    const idx_t M_lm           = landmark_measurements.rows();

    result.dynamics_residuals.resize(num_intervals, StateResIdx::STATE_RES_COUNT);
    result.dynamics_residuals.setZero();

    for (idx_t i = 0; i < num_intervals; ++i) {
        result.dynamics_residuals(i, StateResIdx::RES_TIMESTAMP) = ts.state_timestamps[static_cast<size_t>(i)];

        const size_t lin_base = static_cast<size_t>(i) * lin_res_stride;
        result.dynamics_residuals(i, StateResIdx::RES_POS_X) = raw_residuals[lin_base + 0];
        result.dynamics_residuals(i, StateResIdx::RES_POS_Y) = raw_residuals[lin_base + 1];
        result.dynamics_residuals(i, StateResIdx::RES_POS_Z) = raw_residuals[lin_base + 2];
        result.dynamics_residuals(i, StateResIdx::RES_VEL_X) = raw_residuals[lin_base + 3];
        result.dynamics_residuals(i, StateResIdx::RES_VEL_Y) = raw_residuals[lin_base + 4];
        result.dynamics_residuals(i, StateResIdx::RES_VEL_Z) = raw_residuals[lin_base + 5];

        const size_t ang_base = static_cast<size_t>(num_intervals) * lin_res_stride
                                + static_cast<size_t>(i) * ang_res_stride;
        result.dynamics_residuals(i, StateResIdx::RES_ROT_X) = raw_residuals[ang_base + 0];
        result.dynamics_residuals(i, StateResIdx::RES_ROT_Y) = raw_residuals[ang_base + 1];
        result.dynamics_residuals(i, StateResIdx::RES_ROT_Z) = raw_residuals[ang_base + 2];
        if (bias_mode == BIAS_MODE::TV_BIAS) {
            result.dynamics_residuals(i, StateResIdx::RES_GYRO_BIAS_X) = raw_residuals[ang_base + 3];
            result.dynamics_residuals(i, StateResIdx::RES_GYRO_BIAS_Y) = raw_residuals[ang_base + 4];
            result.dynamics_residuals(i, StateResIdx::RES_GYRO_BIAS_Z) = raw_residuals[ang_base + 5];
        }
    }

    result.landmark_residuals.resize(M_lm, LandmarkResIdx::LANDMARK_RES_COUNT);
    const size_t ldmk_base = static_cast<size_t>(num_intervals) * (lin_res_stride + ang_res_stride);
    for (idx_t i = 0; i < M_lm; ++i) {
        const size_t base = ldmk_base + static_cast<size_t>(i) * 3;
        result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_X) = raw_residuals[base + 0];
        result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_Y) = raw_residuals[base + 1];
        result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_Z) = raw_residuals[base + 2];
    }

    result.state_estimates = std::move(state_estimates);
    result.code = ErrorCode::OK;
    return result;
}
