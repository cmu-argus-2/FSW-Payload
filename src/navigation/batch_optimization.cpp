#include "navigation/batch_optimization.hpp"
#include "navigation/batch_nlp.hpp"
#include "navigation/trajectory_initializer.hpp"
#include <IpIpoptApplication.hpp>
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>


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

// Evaluate dynamics and landmark residuals at the current state_estimates by
// calling the cost functors directly with double scalars (no solver needed).
static void compute_residuals(
        const StateEstimates&        state_estimates,
        const std::vector<double>&   state_timestamps,
        const std::vector<idx_t>&    landmark_group_indices,
        const LandmarkMeasurements&  landmark_measurements,
        const LandmarkGroupStarts&   landmark_group_starts,
        const GyroMeasurements&      gyro_measurements,
        BIAS_MODE                    bias_mode,
        double                       uma_std_dev,
        double                       gyro_wn_std_dev_rad_s,
        double                       gyro_bias_instability,
        const Eigen::VectorXd&       landmark_uncertainties,
        DynamicsResiduals&           dynamics_out,
        LandmarkResiduals&           landmark_out)
{
    const idx_t N  = state_estimates.rows();
    const idx_t M  = N - 1;
    const idx_t Ml = landmark_measurements.rows();

    dynamics_out.resize(M, StateResIdx::STATE_RES_COUNT);
    dynamics_out.setZero();

    for (idx_t i = 0; i < M; ++i) {
        const double dt      = state_timestamps[static_cast<size_t>(i + 1)]
                             - state_timestamps[static_cast<size_t>(i)];
        const double pos_std = 0.5 * uma_std_dev * dt * std::sqrt(dt);
        const double vel_std = uma_std_dev * std::sqrt(dt);

        dynamics_out(i, StateResIdx::RES_TIMESTAMP) = state_timestamps[static_cast<size_t>(i)];

        // ── Linear dynamics ──────────────────────────────────────────────────
        const double* r0 = &state_estimates(i,   StateEstimateIdx::POS_X);
        const double* v0 = &state_estimates(i,   StateEstimateIdx::VEL_X);
        const double* r1 = &state_estimates(i+1, StateEstimateIdx::POS_X);
        const double* v1 = &state_estimates(i+1, StateEstimateIdx::VEL_X);
        LinearDynamicsAnalytic lin_dyn(dt, pos_std, vel_std);
        double lin_res[6] = {};
        const double* lin_params[4] = {r0, v0, r1, v1};
        lin_dyn.Evaluate(lin_params, lin_res, nullptr);
        dynamics_out(i, StateResIdx::RES_POS_X) = lin_res[0];
        dynamics_out(i, StateResIdx::RES_POS_Y) = lin_res[1];
        dynamics_out(i, StateResIdx::RES_POS_Z) = lin_res[2];
        dynamics_out(i, StateResIdx::RES_VEL_X) = lin_res[3];
        dynamics_out(i, StateResIdx::RES_VEL_Y) = lin_res[4];
        dynamics_out(i, StateResIdx::RES_VEL_Z) = lin_res[5];

        // ── Angular dynamics ─────────────────────────────────────────────────
        const double quat_std = gyro_wn_std_dev_rad_s * dt;
        const double bias_std = gyro_bias_instability * std::sqrt(dt);
        const double* q0      = &state_estimates(i,   StateEstimateIdx::QUAT_X);
        const double* q1      = &state_estimates(i+1, StateEstimateIdx::QUAT_X);
        const double* gyro_row = gyro_measurements.data()
                               + static_cast<size_t>(i) * GyroMeasurementIdx::GYRO_MEAS_COUNT;
        double ang_res[6] = {};

        if (bias_mode == BIAS_MODE::FIX_BIAS) {
            const double* bw = &state_estimates(0, StateEstimateIdx::GYRO_BIAS_X);
            AngularDynamicsCostFunctorFixBias{gyro_row, dt, quat_std}(q0, q1, bw, ang_res);
        } else if (bias_mode == BIAS_MODE::TV_BIAS) {
            const double* bw0 = &state_estimates(i,   StateEstimateIdx::GYRO_BIAS_X);
            const double* bw1 = &state_estimates(i+1, StateEstimateIdx::GYRO_BIAS_X);
            AngularDynamicsCostFunctor{gyro_row, dt, quat_std, bias_std}(q0, bw0, q1, bw1, ang_res);
        } else {
            AngularDynamicsCostFunctorNoBias{gyro_row, dt, quat_std}(q0, q1, ang_res);
        }
        dynamics_out(i, StateResIdx::RES_ROT_X) = ang_res[0];
        dynamics_out(i, StateResIdx::RES_ROT_Y) = ang_res[1];
        dynamics_out(i, StateResIdx::RES_ROT_Z) = ang_res[2];
        if (bias_mode == BIAS_MODE::TV_BIAS) {
            dynamics_out(i, StateResIdx::RES_GYRO_BIAS_X) = ang_res[3];
            dynamics_out(i, StateResIdx::RES_GYRO_BIAS_Y) = ang_res[4];
            dynamics_out(i, StateResIdx::RES_GYRO_BIAS_Z) = ang_res[5];
        }
    }

    // ── Landmark residuals ───────────────────────────────────────────────────
    landmark_out.resize(Ml, LandmarkResIdx::LANDMARK_RES_COUNT);
    idx_t lm_idx = 0;
    for (const auto& grp_state_idx : landmark_group_indices) {
        const double* pos  = &state_estimates(grp_state_idx, StateEstimateIdx::POS_X);
        const double* quat = &state_estimates(grp_state_idx, StateEstimateIdx::QUAT_X);
        do {
            if (lm_idx >= Ml) break;
            const double* lm_row = landmark_measurements.data()
                                 + static_cast<size_t>(lm_idx) * LandmarkMeasurementIdx::LANDMARK_COUNT;
            double lm_res[3] = {};
            LandmarkCostFunctor{lm_row, landmark_uncertainties(lm_idx)}(pos, quat, lm_res);
            landmark_out(lm_idx, LandmarkResIdx::LANDMARK_RES_X) = lm_res[0];
            landmark_out(lm_idx, LandmarkResIdx::LANDMARK_RES_Y) = lm_res[1];
            landmark_out(lm_idx, LandmarkResIdx::LANDMARK_RES_Z) = lm_res[2];
            ++lm_idx;
        } while (lm_idx < landmark_group_starts.rows() && !landmark_group_starts(lm_idx, 0));
    }
}

BatchOptResult solve_batch_opt(const ODMeasurements& measurements,
                               BATCH_OPT_config bo_config) {
    BatchOptResult result;

    // ── Validate inputs ───────────────────────────────────────────────────────
    if (measurements.Validate() != ErrorCode::OK) {
        result.code = ErrorCode::ODMEAS_NOT_VALID;
        return result;
    }

    const LandmarkMeasurements landmark_measurements = measurements.landmark_measurements;
    const LandmarkGroupStarts  landmark_group_starts = measurements.group_starts;
    const GyroMeasurements     gyro_measurements     = measurements.gyro_measurements;
    const BIAS_MODE            bias_mode             = bo_config.bias_mode;

    const idx_t num_groups = std::count(landmark_group_starts.col(0).begin(),
                                        landmark_group_starts.col(0).end(), true);

    // ── Timestamps & group index map ─────────────────────────────────────────
    StateTimestampsResult ts = get_state_timestamps(
            landmark_measurements, landmark_group_starts, gyro_measurements, num_groups);
    if (ts.code != ErrorCode::OK) { result.code = ts.code; return result; }

    TrajectoryInitializer initializer(
            ts.state_timestamps, landmark_measurements, landmark_group_starts,
            bias_mode, gyro_measurements);
    result.initial_trajectory = initializer.state_estimates();
    StateEstimates state_estimates = initializer.state_estimates();

    const double uma_std_dev           = 1.0;        // km/s²
    const double gyro_wn_std_dev_rad_s = 0.0008726;  // rad/s
    const double gyro_bias_instability = 1.0;        // rad/s²

    // ── Build and run IPOPT ───────────────────────────────────────────────────
    Ipopt::SmartPtr<BatchNLP> nlp = new BatchNLP(
            state_estimates, ts.state_timestamps, ts.landmark_group_indices,
            landmark_measurements, landmark_group_starts, gyro_measurements,
            bias_mode, uma_std_dev, gyro_wn_std_dev_rad_s, gyro_bias_instability,
            measurements.landmark_uncertainties);

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetIntegerValue("max_iter",              static_cast<int>(bo_config.max_iterations));
    app->Options()->SetNumericValue("tol",                   bo_config.solver_function_tolerance);
    app->Options()->SetNumericValue("acceptable_tol",        bo_config.solver_function_tolerance * 100.0);
    app->Options()->SetStringValue ("hessian_approximation", "limited-memory");
    app->Options()->SetIntegerValue("print_level",           5);
    app->Options()->SetStringValue ("nlp_scaling_method",    "gradient-based");

    const Ipopt::ApplicationReturnStatus init_status = app->Initialize();
    if (init_status != Ipopt::Solve_Succeeded && init_status != Ipopt::Solved_To_Acceptable_Level) {
        spdlog::error("solve_batch_opt: IPOPT initialization failed (status {}).",
                      static_cast<int>(init_status));
        LogError(ErrorCode::BATCH_OPT_SOLVER_FAILED);
        result.code = ErrorCode::BATCH_OPT_SOLVER_FAILED;
        return result;
    }

    const Ipopt::ApplicationReturnStatus solve_status = app->OptimizeTNLP(nlp);

    result.solver_summary.termination_type = static_cast<int>(solve_status);
    result.solver_summary.num_iterations   = nlp->iter_count;
    result.solver_summary.initial_cost     = nlp->initial_cost;
    result.solver_summary.final_cost       = nlp->final_cost;

    spdlog::info("IPOPT solve finished: status={}, iters={}, cost {:.3f} → {:.3f}",
                 static_cast<int>(solve_status), nlp->iter_count,
                 nlp->initial_cost, nlp->final_cost);

    if (solve_status != Ipopt::Solve_Succeeded &&
        solve_status != Ipopt::Solved_To_Acceptable_Level) {
        spdlog::error("solve_batch_opt: IPOPT did not converge (status {}).",
                      static_cast<int>(solve_status));
        LogError(ErrorCode::BATCH_OPT_NO_CONVERGENCE);
        result.code = ErrorCode::BATCH_OPT_NO_CONVERGENCE;
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
        spdlog::error("solve_batch_opt: IPOPT output contains NaN/inf or denormalized quaternions.");
        LogError(ErrorCode::BATCH_OPT_INVALID_OUTPUT);
        result.code = ErrorCode::BATCH_OPT_INVALID_OUTPUT;
        return result;
    }

    // ── Residuals ─────────────────────────────────────────────────────────────
    compute_residuals(state_estimates, ts.state_timestamps, ts.landmark_group_indices,
                      landmark_measurements, landmark_group_starts, gyro_measurements,
                      bias_mode, uma_std_dev, gyro_wn_std_dev_rad_s, gyro_bias_instability,
                      measurements.landmark_uncertainties,
                      result.dynamics_residuals, result.landmark_residuals);

    // Covariance is not computed (field remains 0 rows).
    result.covariance = ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);

    result.state_estimates = std::move(state_estimates);
    result.code = ErrorCode::OK;
    return result;
}
