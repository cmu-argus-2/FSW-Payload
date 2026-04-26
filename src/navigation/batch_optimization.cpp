// Batch orbit determination using CasADi + IPOPT.
// Mirrors python_od/optimizer.py exactly: same quaternion convention [x,y,z,w],
// same residual formulas, same Forward-Euler dynamics constraint.
#include "navigation/batch_optimization.hpp"
#include "navigation/quaternion.hpp"
#include "navigation/trajectory_initializer.hpp"
#include <casadi/casadi.hpp>
#include "spdlog/spdlog.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Sparse>
#include <limits>
#include <vector>

using casadi::DM;
using casadi::MX;
using casadi::Slice;
using casadi::Dict;
using casadi::Function;
using casadi::Opti;
using casadi::OptiSol;

// ── Constants (mirrors python_od/optimizer.py) ────────────────────────────────
static constexpr double CASADI_GYRO_WN_STD_DEV  = 0.0008726;       // rad/s
static constexpr double CASADI_UMA_STD_DEV      = 1e-5;            // km/s²
static constexpr double CASADI_DYN_COV_NORM     = 1e-6;

// ── DM helpers ────────────────────────────────────────────────────────────────

// Build a (3,1) DM column vector
static DM dm3(double x, double y, double z) {
    return DM::vertcat(std::vector<DM>{DM(x), DM(y), DM(z)});
}

// Build a (4,1) DM column vector
static DM dm4(double x, double y, double z, double w) {
    return DM::vertcat(std::vector<DM>{DM(x), DM(y), DM(z), DM(w)});
}

static Eigen::SparseMatrix<double> dm_to_sparse_eigen(const DM& dm) {
    using SparseMatrix = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;

    const std::vector<casadi_int> colind = dm.sparsity().get_colind();
    const std::vector<casadi_int> row = dm.sparsity().get_row();
    const std::vector<double>& values = dm.nonzeros();

    std::vector<Triplet> triplets;
    triplets.reserve(values.size());
    for (casadi_int c = 0; c < dm.size2(); ++c) {
        for (casadi_int k = colind[static_cast<size_t>(c)];
             k < colind[static_cast<size_t>(c + 1)]; ++k) {
            const double value = values[static_cast<size_t>(k)];
            if (value == 0.0) continue;
            triplets.emplace_back(
                static_cast<Eigen::Index>(row[static_cast<size_t>(k)]),
                static_cast<Eigen::Index>(c),
                value);
        }
    }

    SparseMatrix sparse(static_cast<Eigen::Index>(dm.size1()),
                        static_cast<Eigen::Index>(dm.size2()));
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    sparse.makeCompressed();
    return sparse;
}

static ResidualsOrCovariances compute_covariance(
    const std::vector<MX>& uma_residuals,
    const std::vector<MX>& ang_residuals,
    const std::vector<MX>& lmk_residuals,
    const std::vector<MX>& dyn_constraints,
    const std::vector<MX>& drag_residuals,
    const OptiSol& sol,
    const MX& r,
    const MX& v,
    const MX& q,
    const MX& a,
    const MX& b,
    const MX& d,
    const DM& r_val,
    const DM& v_val,
    const DM& q_val,
    const DM& a_val,
    const DM& b_val,
    const DM& d_val,
    bool use_drag,
    const std::vector<double>& state_timestamps,
    double* bc_inv_var_out = nullptr)
{
    const int N = static_cast<int>(state_timestamps.size());
    const int N_uma = N - 1;
    if (N <= 0 || N_uma <= 0) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }

    using Clock = std::chrono::steady_clock;
    auto stage_start = Clock::now();
    auto log_stage = [&stage_start](const char* stage) {
        const auto now = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(now - stage_start).count();
        spdlog::debug("solve_batch_opt: covariance stage '{}' took {:.1f} ms.", stage, elapsed_ms);
        stage_start = now;
    };

    spdlog::info("solve_batch_opt: building covariance Jacobian.");

    std::vector<MX> residual_terms;
    residual_terms.reserve(uma_residuals.size() + ang_residuals.size() +
                           lmk_residuals.size() + dyn_constraints.size() +
                           drag_residuals.size());
    residual_terms.insert(residual_terms.end(), uma_residuals.begin(), uma_residuals.end());
    residual_terms.insert(residual_terms.end(), ang_residuals.begin(), ang_residuals.end());
    residual_terms.insert(residual_terms.end(), lmk_residuals.begin(), lmk_residuals.end());
    for (const MX& c : dyn_constraints) {
        residual_terms.push_back(c / CASADI_DYN_COV_NORM);
    }
    residual_terms.insert(residual_terms.end(), drag_residuals.begin(), drag_residuals.end());

    MX residual_sym = MX::vertcat(residual_terms);
    log_stage("residual assembly");

    MX x_sym;
    Function J_fn;
    std::vector<DM> jac_inputs;
    if (use_drag) {
        x_sym = MX::vertcat(std::vector<MX>{vec(r), vec(v), vec(q), vec(a), b, d});
        MX J_sym = jacobian(residual_sym, x_sym);
        log_stage("symbolic jacobian");
        J_fn = Function("batch_covariance_jacobian", {r, v, q, a, b, d}, {J_sym});
        log_stage("function construction");
        jac_inputs = {r_val, v_val, q_val, a_val, b_val, d_val};
    } else {
        x_sym = MX::vertcat(std::vector<MX>{vec(r), vec(v), vec(q), vec(a), b});
        MX J_sym = jacobian(residual_sym, x_sym);
        log_stage("symbolic jacobian");
        J_fn = Function("batch_covariance_jacobian", {r, v, q, a, b}, {J_sym});
        log_stage("function construction");
        jac_inputs = {r_val, v_val, q_val, a_val, b_val};
    }

    const std::vector<DM> jac_outputs = J_fn(jac_inputs);
    log_stage("jacobian evaluation");
    Eigen::SparseMatrix<double> J = dm_to_sparse_eigen(jac_outputs.at(0));
    log_stage("dm to sparse eigen");

    const int n_drag    = use_drag ? 1 : 0;
    const int n_full    = 10 * N + 3 * N_uma + 3 + n_drag;
    const int n_reduced = 9  * N + 3 * N_uma + 3 + n_drag;
    std::vector<Eigen::Triplet<double>> t_triplets;
    t_triplets.reserve(static_cast<size_t>(6 * N + 12 * N + 3 * N_uma + 3 + n_drag));
    for (int i = 0; i < 3 * N; ++i) {
        t_triplets.emplace_back(i, i, 1.0);
    }
    for (int i = 0; i < 3 * N; ++i) {
        t_triplets.emplace_back(3 * N + i, 3 * N + i, 1.0);
    }

    for (int i = 0; i < N; ++i) {
        const auto B = quat_tangent_basis_xyzw(
            static_cast<double>(q_val(0, i)),
            static_cast<double>(q_val(1, i)),
            static_cast<double>(q_val(2, i)),
            static_cast<double>(q_val(3, i)));
        for (int rr = 0; rr < 4; ++rr) {
            for (int cc = 0; cc < 3; ++cc) {
                t_triplets.emplace_back(
                    6 * N + 4 * i + rr,
                    6 * N + 3 * i + cc,
                    B(rr, cc));
            }
        }
    }

    for (int i = 0; i < 3 * N_uma; ++i) {
        t_triplets.emplace_back(10 * N + i, 9 * N + i, 1.0);
    }
    for (int i = 0; i < 3; ++i) {
        t_triplets.emplace_back(10 * N + 3 * N_uma + i, 9 * N + 3 * N_uma + i, 1.0);
    }
    if (use_drag) {
        t_triplets.emplace_back(10 * N + 3 * N_uma + 3, 9 * N + 3 * N_uma + 3, 1.0);
    }
    Eigen::SparseMatrix<double> T(n_full, n_reduced);
    T.setFromTriplets(t_triplets.begin(), t_triplets.end());
    T.makeCompressed();
    log_stage("sparse tangent projection build");

    spdlog::info("solve_batch_opt: solving covariance normal equations ({}x{} reduced Jacobian).",
                 J.rows(), n_reduced);
    const Eigen::SparseMatrix<double> J_reduced = J * T;
    log_stage("sparse projection multiply");
    Eigen::SparseMatrix<double> normal_sparse = J_reduced.transpose() * J_reduced;
    log_stage("sparse normal matrix");
    const double damping = 1e-18;
    normal_sparse.makeCompressed();
    for (int k = 0; k < normal_sparse.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(normal_sparse, k); it; ++it) {
            if (it.row() == it.col()) {
                it.valueRef() += damping;
                break;
            }
        }
    }

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    ldlt.compute(normal_sparse);
    log_stage("ldlt factorization");
    if (ldlt.info() != Eigen::Success) {
        spdlog::warn("solve_batch_opt: covariance LDLT factorization failed.");
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }

    std::vector<int> covariance_indices;
    covariance_indices.reserve(static_cast<size_t>(9 * N + 3 + n_drag));
    for (int i = 0; i < 9 * N; ++i) {
        covariance_indices.push_back(i);
    }
    for (int i = 0; i < 3; ++i) {
        covariance_indices.push_back(9 * N + 3 * N_uma + i);
    }
    if (use_drag) {
        covariance_indices.push_back(9 * N + 3 * N_uma + 3);
    }

    Eigen::VectorXd cov_diag = Eigen::VectorXd::Zero(n_reduced);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_reduced);
    for (const int idx : covariance_indices) {
        rhs(idx) = 1.0;
        const Eigen::VectorXd column = ldlt.solve(rhs);
        if (ldlt.info() != Eigen::Success) {
            spdlog::warn("solve_batch_opt: covariance solve failed for column {}.", idx);
            return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
        cov_diag(idx) = column(idx);
        rhs(idx) = 0.0;
    }
    log_stage("selected inverse diagonal solve");

    ResidualsOrCovariances covariance(N, StateResIdx::STATE_RES_COUNT);
    covariance.setZero();
    for (int i = 0; i < N; ++i) {
        covariance(i, RES_TIMESTAMP) = state_timestamps[static_cast<size_t>(i)];
        for (int c = 0; c < 3; ++c) {
            covariance(i, RES_POS_X + c) = cov_diag(3 * i + c);
            covariance(i, RES_VEL_X + c) = cov_diag(3 * N + 3 * i + c);
            covariance(i, RES_ROT_X + c) = cov_diag(6 * N + 3 * i + c);
        }
    }
    for (int c = 0; c < 3; ++c) {
        covariance(0, RES_GYRO_BIAS_X + c) = cov_diag(9 * N + 3 * N_uma + c);
    }
    if (use_drag) {
        double bc_inv_var = cov_diag(9 * N + 3 * N_uma + 3);
        spdlog::info("solve_batch_opt: bc_inv variance = {:.4e}", bc_inv_var);
        if (bc_inv_var_out) *bc_inv_var_out = bc_inv_var;
    }
    log_stage("output pack");

    (void)sol;
    spdlog::info("solve_batch_opt: covariance computed.");
    return covariance;
}

// Fixed-bias angular dynamics residual (3-vec) 
static MX angular_dynamics_residual_fix_bias(
    const MX& q0, const MX& q1,
    const DM& gyro_w, const MX& bias,
    double dt, double quat_std)
{
    MX omega  = MX(gyro_w) - bias;
    MX dq     = angle_axis_to_quat_xyzw(omega * dt);
    MX q_pred = quat_product_xyzw(q0, dq);
    MX q_err  = quat_product_xyzw(quat_conjugate_xyzw(q_pred), q1);
    return MX::vertcat(std::vector<MX>{q_err(0, 0), q_err(1, 0), q_err(2, 0)}) / quat_std;
}


// ── Timestamp mapping  ──────────────────────────────────────

StateTimestampsResult
get_state_timestamps(const LandmarkMeasurements& landmark_measurements,
                     const LandmarkGroupStarts& landmark_group_starts,
                     const GyroMeasurements& gyro_measurements,
                     const idx_t num_groups) {
    StateTimestampsResult result;
    const idx_t M = gyro_measurements.rows();

    result.state_timestamps.reserve(static_cast<size_t>(M));
    for (idx_t k = 0; k < M; ++k)
        result.state_timestamps.push_back(gyro_measurements(k, GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP));

    result.landmark_group_indices.reserve(static_cast<size_t>(num_groups));

    for (idx_t i = 0; i < landmark_group_starts.rows(); ++i) {
        if (!landmark_group_starts(i, 0)) continue;

        const double t_lm = landmark_measurements(i, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP);

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
        spdlog::error("get_state_timestamps: counted {} landmark group starts but expected {}.",
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


// ── Main solver ───────────────────────────────────────────────────────────────

BatchOptResult solve_batch_opt(const ODMeasurements& measurements,
                               BATCH_OPT_config bo_config) {
    BatchOptResult result;

    if (measurements.Validate() != ErrorCode::OK) {
        result.code = ErrorCode::ODMEAS_NOT_VALID;
        return result;
    }

    const Eigen::MatrixXd& lm  = measurements.landmark_measurements;
    const auto&            gs  = measurements.group_starts;
    const Eigen::MatrixXd& gm  = measurements.gyro_measurements;
    const BIAS_MODE bias_mode    = bo_config.bias_mode;
    const bool use_j2            = bo_config.use_j2;
    const bool use_drag            = bo_config.use_drag;
    const double bc_inv_nominal    = bo_config.bc_inv_nominal;
    const double bc_inv_std        = bo_config.bc_inv_std;
    const Integrator integrator    = bo_config.integrator;

    if (bias_mode != BIAS_MODE::FIX_BIAS) {
        spdlog::error("solve_batch_opt: only FIX_BIAS mode is supported in the CasADi backend.");
        result.code = ErrorCode::BATCH_OPT_BUILD_FAILED;
        return result;
    }

    const idx_t num_groups = std::count(gs.col(0).begin(), gs.col(0).end(), true);

    StateTimestampsResult ts = get_state_timestamps(
        measurements.landmark_measurements, measurements.group_starts,
        measurements.gyro_measurements, num_groups);
    if (ts.code != ErrorCode::OK) { result.code = ts.code; return result; }

    const int N     = static_cast<int>(ts.state_timestamps.size());
    const int N_uma = N - 1;
    const int Ml    = static_cast<int>(lm.rows());

    // ── Initial guess ─────────────────────────────────────────────────────────
    TrajectoryInitializer initializer(
        ts.state_timestamps, measurements.landmark_measurements,
        measurements.group_starts, bias_mode, measurements.gyro_measurements);
    result.initial_trajectory = initializer.state_estimates();
    const StateEstimates& init_se = initializer.state_estimates();

    spdlog::info("solve_batch_opt: {} states, {} landmark groups; building CasADi problem (J2: {}, drag: {}, integrator: {}) …",
                 N, ts.landmark_group_indices.size(), use_j2 ? "on" : "off", use_drag ? "on" : "off",
                 integrator == Integrator::RK4 ? "RK4" : "Euler");

    // ── Decision variables ────────────────────────────────────────────────────
    Opti opti;
    MX r = opti.variable(3, N);      // ECI positions   [km]
    MX v = opti.variable(3, N);      // ECI velocities  [km/s]
    MX q = opti.variable(4, N);      // quaternions     [x,y,z,w]
    MX a = opti.variable(3, N_uma);  // unmodelled accel [km/s²]
    MX b = opti.variable(3);         // fixed gyro bias  [rad/s]
    MX d = use_drag ? opti.variable(1) : MX(bc_inv_nominal);  // bc_inv [km²/kg]

    // ── Quaternion unit-norm constraints ──────────────────────────────────────
    for (int i = 0; i < N; ++i) {
        MX qi = q(Slice(), i);
        opti.subject_to(dot(qi, qi) == 1.0);   // ADL
    }

    // ── Linear dynamics (equality) + UMA prior (soft) ─────────────────────────
    MX obj = MX(0.0);
    std::vector<MX> dyn_constraints;
    std::vector<MX> uma_residuals;
    std::vector<MX> drag_residuals;
    dyn_constraints.reserve(static_cast<size_t>(N_uma));
    uma_residuals.reserve(static_cast<size_t>(N_uma));

    if (use_drag) {
        MX drag_res = (d - bc_inv_nominal) / bc_inv_std;
        drag_residuals.push_back(drag_res);
        obj = obj + drag_res * drag_res;
    }

    for (int i = 0; i < N_uma; ++i) {
        double dt = ts.state_timestamps[static_cast<size_t>(i + 1)]
                  - ts.state_timestamps[static_cast<size_t>(i)];
        MX c = linear_dynamics_constraint(
            r(Slice(), i), v(Slice(), i),
            r(Slice(), i + 1), v(Slice(), i + 1),
            a(Slice(), i), dt, integrator, use_j2, use_drag, d);
        opti.subject_to(c == 0.0);
        dyn_constraints.push_back(c);

        MX uma_res = a(Slice(), i) / CASADI_UMA_STD_DEV;
        uma_residuals.push_back(uma_res);
        obj = obj + dot(uma_res, uma_res);   // ADL
    }

    // ── Angular dynamics (soft) ───────────────────────────────────────────────
    std::vector<MX> ang_residuals;
    ang_residuals.reserve(static_cast<size_t>(N_uma));

    for (int i = 0; i < N_uma; ++i) {
        double dt       = ts.state_timestamps[static_cast<size_t>(i + 1)]
                        - ts.state_timestamps[static_cast<size_t>(i)];
        double quat_std = CASADI_GYRO_WN_STD_DEV * dt;
        DM gyro_w = dm3(gm(i, GyroMeasurementIdx::ANG_VEL_X),
                        gm(i, GyroMeasurementIdx::ANG_VEL_Y),
                        gm(i, GyroMeasurementIdx::ANG_VEL_Z));
        MX ang_res = angular_dynamics_residual_fix_bias(
            q(Slice(), i), q(Slice(), i + 1), gyro_w, b, dt, quat_std);
        ang_residuals.push_back(ang_res);
        obj = obj + dot(ang_res, ang_res);   // ADL
    }

    // ── Landmark bearing (soft) ───────────────────────────────────────────────
    std::vector<MX> lmk_residuals;
    lmk_residuals.reserve(static_cast<size_t>(Ml));

    {
        idx_t grp_k = 0;
        idx_t row   = 0;
        while (row < static_cast<idx_t>(Ml)) {
            if (!gs(row, 0)) { ++row; continue; }
            int state_idx = static_cast<int>(
                ts.landmark_group_indices[static_cast<size_t>(grp_k++)]);
            MX ri = r(Slice(), state_idx);
            MX qi = q(Slice(), state_idx);
            do {
                DM lmk_pos = dm3(lm(row, LandmarkMeasurementIdx::LANDMARK_POS_X),
                                 lm(row, LandmarkMeasurementIdx::LANDMARK_POS_Y),
                                 lm(row, LandmarkMeasurementIdx::LANDMARK_POS_Z));
                DM bearing  = dm3(lm(row, LandmarkMeasurementIdx::BEARING_VEC_X),
                                  lm(row, LandmarkMeasurementIdx::BEARING_VEC_Y),
                                  lm(row, LandmarkMeasurementIdx::BEARING_VEC_Z));
                double sigma = measurements.landmark_uncertainties(row);
                MX lmk_res = landmark_residual_casadi(ri, qi, lmk_pos, bearing, sigma);
                lmk_residuals.push_back(lmk_res);
                obj = obj + dot(lmk_res, lmk_res);   // ADL
                ++row;
            } while (row < static_cast<idx_t>(Ml) && !gs(row, 0));
        }
    }

    opti.minimize(obj);

    // ── Initial guess (full-matrix assignment, unambiguous in C++ API) ─────────
    {
        std::vector<DM> r_cols, v_cols, q_cols;
        r_cols.reserve(static_cast<size_t>(N));
        v_cols.reserve(static_cast<size_t>(N));
        q_cols.reserve(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) {
            r_cols.push_back(dm3(init_se(i, POS_X), init_se(i, POS_Y), init_se(i, POS_Z)));
            v_cols.push_back(dm3(init_se(i, VEL_X), init_se(i, VEL_Y), init_se(i, VEL_Z)));
            q_cols.push_back(dm4(init_se(i, QUAT_X), init_se(i, QUAT_Y),
                                 init_se(i, QUAT_Z), init_se(i, QUAT_W)));
        }
        opti.set_initial(r, DM::horzcat(r_cols));
        opti.set_initial(v, DM::horzcat(v_cols));
        opti.set_initial(q, DM::horzcat(q_cols));
        opti.set_initial(a, DM::zeros(3, N_uma));
        opti.set_initial(b, dm3(init_se(0, GYRO_BIAS_X),
                                init_se(0, GYRO_BIAS_Y),
                                init_se(0, GYRO_BIAS_Z)));
        if (use_drag) {
            opti.set_initial(d, DM(bc_inv_nominal));
        }
    }

    // ── IPOPT ─────────────────────────────────────────────────────────────────
    Dict ipopt_opts;
    ipopt_opts["max_iter"]       = static_cast<int>(bo_config.max_iterations);
    ipopt_opts["tol"]            = bo_config.solver_function_tolerance;
    ipopt_opts["acceptable_tol"] = bo_config.solver_function_tolerance * 100.0;
    ipopt_opts["print_level"]    = 5;
    opti.solver("ipopt", Dict{}, ipopt_opts);

    spdlog::info("solve_batch_opt: starting IPOPT solve …");

    // Solve — IPOPT failure raises std::exception; "Solved_To_Acceptable_Level"
    // does not throw and is treated as success.
    try {
        OptiSol sol = opti.solve();

        // ── Solver summary ──────────────────────────────────────────────────
        {
            casadi::Dict stats = sol.stats();
            result.solver_summary.return_status =
                static_cast<std::string>(stats.at("return_status"));
            result.solver_summary.iter_count =
                static_cast<int>(stats.at("iter_count"));
            result.solver_summary.final_cost =
                static_cast<double>(sol.value(obj));
        }

        // ── Extract solution ────────────────────────────────────────────────
        DM r_val = sol.value(r);   // (3, N)
        DM v_val = sol.value(v);   // (3, N)
        DM q_val = sol.value(q);   // (4, N)
        DM a_val = sol.value(a);   // (3, N_uma)
        DM b_val = sol.value(b);   // (3, 1)
        DM d_val = use_drag ? sol.value(d) : DM(bc_inv_nominal);

        result.gyro_bias_fixed = {
            static_cast<double>(b_val(0, 0)),
            static_cast<double>(b_val(1, 0)),
            static_cast<double>(b_val(2, 0))
        };
        result.bc_inv           = static_cast<double>(d_val(0, 0));
        result.bc_inv_estimated = use_drag;

        if (use_drag) {
            spdlog::info("solve_batch_opt: bc_inv = {:.4e} km²/kg", result.bc_inv);
        }
        spdlog::info("solve_batch_opt: gyro_bias = [{:.4e}, {:.4e}, {:.4e}] rad/s",
                     result.gyro_bias_fixed[0], result.gyro_bias_fixed[1], result.gyro_bias_fixed[2]);

        StateEstimates state_estimates(N, StateEstimateIdx::STATE_ESTIMATE_COUNT);
        for (int i = 0; i < N; ++i) {
            state_estimates(i, STATE_ESTIMATE_TIMESTAMP) = ts.state_timestamps[static_cast<size_t>(i)];
            state_estimates(i, POS_X)  = static_cast<double>(r_val(0, i));
            state_estimates(i, POS_Y)  = static_cast<double>(r_val(1, i));
            state_estimates(i, POS_Z)  = static_cast<double>(r_val(2, i));
            state_estimates(i, VEL_X)  = static_cast<double>(v_val(0, i));
            state_estimates(i, VEL_Y)  = static_cast<double>(v_val(1, i));
            state_estimates(i, VEL_Z)  = static_cast<double>(v_val(2, i));
            state_estimates(i, QUAT_X) = static_cast<double>(q_val(0, i));
            state_estimates(i, QUAT_Y) = static_cast<double>(q_val(1, i));
            state_estimates(i, QUAT_Z) = static_cast<double>(q_val(2, i));
            state_estimates(i, QUAT_W) = static_cast<double>(q_val(3, i));
            state_estimates(i, GYRO_BIAS_X) = static_cast<double>(b_val(0, 0));
            state_estimates(i, GYRO_BIAS_Y) = static_cast<double>(b_val(1, 0));
            state_estimates(i, GYRO_BIAS_Z) = static_cast<double>(b_val(2, 0));
        }

        // ── Validity check ──────────────────────────────────────────────────
        for (Eigen::Index j = 0; j < state_estimates.size(); ++j) {
            if (!std::isfinite(state_estimates.data()[j])) {
                spdlog::error("solve_batch_opt: solution contains NaN/Inf.");
                result.code = ErrorCode::BATCH_OPT_INVALID_OUTPUT;
                return result;
            }
        }
        for (int i = 0; i < N; ++i) {
            double qx = state_estimates(i, QUAT_X), qy = state_estimates(i, QUAT_Y);
            double qz = state_estimates(i, QUAT_Z), qw = state_estimates(i, QUAT_W);
            if (std::abs(std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw) - 1.0) > 0.05) {
                spdlog::error("solve_batch_opt: denormalised quaternion at state {}.", i);
                result.code = ErrorCode::BATCH_OPT_INVALID_OUTPUT;
                return result;
            }
        }

        // ── Dynamics residuals ──────────────────────────────────────────────
        result.dynamics_residuals.resize(N_uma, StateResIdx::STATE_RES_COUNT);
        result.dynamics_residuals.setZero();
        for (int i = 0; i < N_uma; ++i) {
            result.dynamics_residuals(i, RES_TIMESTAMP) = ts.state_timestamps[static_cast<size_t>(i)];

            DM lc = sol.value(dyn_constraints[static_cast<size_t>(i)]);
            result.dynamics_residuals(i, RES_POS_X) = static_cast<double>(lc(0, 0));
            result.dynamics_residuals(i, RES_POS_Y) = static_cast<double>(lc(1, 0));
            result.dynamics_residuals(i, RES_POS_Z) = static_cast<double>(lc(2, 0));
            result.dynamics_residuals(i, RES_VEL_X) = static_cast<double>(lc(3, 0));
            result.dynamics_residuals(i, RES_VEL_Y) = static_cast<double>(lc(4, 0));
            result.dynamics_residuals(i, RES_VEL_Z) = static_cast<double>(lc(5, 0));

            DM ar = sol.value(ang_residuals[static_cast<size_t>(i)]);
            result.dynamics_residuals(i, RES_ROT_X) = static_cast<double>(ar(0, 0));
            result.dynamics_residuals(i, RES_ROT_Y) = static_cast<double>(ar(1, 0));
            result.dynamics_residuals(i, RES_ROT_Z) = static_cast<double>(ar(2, 0));
        }

        // ── Landmark residuals ──────────────────────────────────────────────
        result.landmark_residuals.resize(Ml, LandmarkResIdx::LANDMARK_RES_COUNT);
        for (size_t k = 0; k < lmk_residuals.size(); ++k) {
            DM lr = sol.value(lmk_residuals[k]);
            result.landmark_residuals(static_cast<idx_t>(k), LANDMARK_RES_X) = static_cast<double>(lr(0, 0));
            result.landmark_residuals(static_cast<idx_t>(k), LANDMARK_RES_Y) = static_cast<double>(lr(1, 0));
            result.landmark_residuals(static_cast<idx_t>(k), LANDMARK_RES_Z) = static_cast<double>(lr(2, 0));
        }

        if (bo_config.compute_covariance) {
            const auto cov_start = std::chrono::steady_clock::now();
            double bc_inv_var_out = 0.0;
            result.covariance = compute_covariance(
                uma_residuals, ang_residuals, lmk_residuals, dyn_constraints,
                drag_residuals,
                sol, r, v, q, a, b, d,
                r_val, v_val, q_val, a_val, b_val, d_val,
                use_drag, ts.state_timestamps, &bc_inv_var_out);
            const auto cov_end = std::chrono::steady_clock::now();
            result.covariance_time_ms =
                std::chrono::duration<double, std::milli>(cov_end - cov_start).count();
            spdlog::info("solve_batch_opt: covariance computation took {:.1f} ms.",
                         result.covariance_time_ms);
            if (result.covariance.rows() > 0) {
                result.covariance_computed = true;
                result.gyro_bias_var = {
                    result.covariance(0, RES_GYRO_BIAS_X),
                    result.covariance(0, RES_GYRO_BIAS_Y),
                    result.covariance(0, RES_GYRO_BIAS_Z)
                };
                result.bc_inv_var = bc_inv_var_out;
            }
        } else {
            result.covariance = ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
        result.state_estimates = std::move(state_estimates);
        result.code            = ErrorCode::OK;

        spdlog::info("solve_batch_opt: IPOPT converged. Final pos [{:.1f}, {:.1f}, {:.1f}] km.",
                     static_cast<double>(r_val(0, N - 1)),
                     static_cast<double>(r_val(1, N - 1)),
                     static_cast<double>(r_val(2, N - 1)));

    } catch (const std::exception& e) {
        spdlog::error("solve_batch_opt: IPOPT failed — {}", e.what());
        LogError(ErrorCode::BATCH_OPT_NO_CONVERGENCE);
        result.code = ErrorCode::BATCH_OPT_NO_CONVERGENCE;
    }

    return result;
}
