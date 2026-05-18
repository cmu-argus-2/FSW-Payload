// Batch orbit determination using CasADi + IPOPT.
// Mirrors python_od/optimizer.py exactly: same quaternion convention [x,y,z,w],
// same residual formulas, same Forward-Euler dynamics constraint.
#include "navigation/batch_optimization.hpp"
#include "navigation/od_measurements.hpp"
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

using SteadyClock = std::chrono::steady_clock;

// ── Constants (mirrors python_od/optimizer.py) ────────────────────────────────
static constexpr double CASADI_GYRO_WN_STD_DEV  = 0.0008726;       // rad/s
static constexpr double CASADI_UMA_STD_DEV      = 1e-5;            // km/s²
static constexpr double CASADI_DYN_COV_NORM     = 1e-6;

// ── DM helpers ────────────────────────────────────────────────────────────────

static DM dm3(double x, double y, double z) {
    return DM::vertcat(std::vector<DM>{DM(x), DM(y), DM(z)});
}

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

// ── SolveContext — carries CasADi symbolic graph + numeric values from a solve ──
// Kept in .cpp to avoid exposing CasADi types in the public header.
struct SolveContext {
    std::vector<MX> uma_residuals;
    std::vector<MX> ang_residuals;
    std::vector<MX> lmk_residuals;     // raw σ-normalised 3-vectors (used for covariance Jacobian)
    std::vector<MX> dyn_constraints;
    std::vector<MX> drag_residuals;
    MX r, v, q, a, b, d;               // symbolic decision variables
    DM r_val, v_val, q_val, a_val, b_val, d_val;  // solution values
    bool use_drag = false;
    std::vector<double> state_timestamps;
    Opti opti_ref;                      // keeps symbolic graph alive after solve_batch_opt_internal returns
};

// ── compute_covariance ────────────────────────────────────────────────────────

static ResidualsOrCovariances compute_covariance(
    const SolveContext& ctx,
    double* cd_var_out = nullptr,
    const SteadyClock::time_point* deadline = nullptr,
    bool* covariance_timed_out = nullptr)
{
    const int N = static_cast<int>(ctx.state_timestamps.size());
    const int N_uma = N - 1;
    if (N <= 0 || N_uma <= 0) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }

    auto mark_timed_out = [&covariance_timed_out](const char* stage) {
        if (covariance_timed_out) *covariance_timed_out = true;
        spdlog::warn("solve_batch_opt: covariance timed out during {}.", stage);
    };
    auto deadline_expired = [&deadline, &mark_timed_out](const char* stage) {
        if (!deadline) return false;
        if (SteadyClock::now() <= *deadline) return false;
        mark_timed_out(stage);
        return true;
    };

    auto stage_start = SteadyClock::now();
    auto log_stage = [&stage_start](const char* stage) {
        const auto now = SteadyClock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(now - stage_start).count();
        spdlog::debug("solve_batch_opt: covariance stage '{}' took {:.1f} ms.", stage, elapsed_ms);
        stage_start = now;
    };

    spdlog::info("solve_batch_opt: building covariance Jacobian.");

    std::vector<MX> residual_terms;
    residual_terms.reserve(ctx.uma_residuals.size() + ctx.ang_residuals.size() +
                           ctx.lmk_residuals.size() + ctx.dyn_constraints.size() +
                           ctx.drag_residuals.size());
    residual_terms.insert(residual_terms.end(), ctx.uma_residuals.begin(), ctx.uma_residuals.end());
    residual_terms.insert(residual_terms.end(), ctx.ang_residuals.begin(), ctx.ang_residuals.end());
    residual_terms.insert(residual_terms.end(), ctx.lmk_residuals.begin(), ctx.lmk_residuals.end());
    for (const MX& c : ctx.dyn_constraints) {
        residual_terms.push_back(c / CASADI_DYN_COV_NORM);
    }
    residual_terms.insert(residual_terms.end(), ctx.drag_residuals.begin(), ctx.drag_residuals.end());

    MX residual_sym = MX::vertcat(residual_terms);
    log_stage("residual assembly");
    if (deadline_expired("residual assembly")) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }

    MX x_sym;
    Function J_fn;
    std::vector<DM> jac_inputs;
    if (ctx.use_drag) {
        x_sym = MX::vertcat(std::vector<MX>{vec(ctx.r), vec(ctx.v), vec(ctx.q),
                                            vec(ctx.a), ctx.b, ctx.d});
        MX J_sym = jacobian(residual_sym, x_sym);
        log_stage("symbolic jacobian");
        if (deadline_expired("symbolic jacobian")) {
            return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
        J_fn = Function("batch_covariance_jacobian",
                        {ctx.r, ctx.v, ctx.q, ctx.a, ctx.b, ctx.d}, {J_sym});
        log_stage("function construction");
        if (deadline_expired("function construction")) {
            return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
        jac_inputs = {ctx.r_val, ctx.v_val, ctx.q_val, ctx.a_val, ctx.b_val, ctx.d_val};
    } else {
        x_sym = MX::vertcat(std::vector<MX>{vec(ctx.r), vec(ctx.v), vec(ctx.q),
                                            vec(ctx.a), ctx.b});
        MX J_sym = jacobian(residual_sym, x_sym);
        log_stage("symbolic jacobian");
        if (deadline_expired("symbolic jacobian")) {
            return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
        J_fn = Function("batch_covariance_jacobian",
                        {ctx.r, ctx.v, ctx.q, ctx.a, ctx.b}, {J_sym});
        log_stage("function construction");
        if (deadline_expired("function construction")) {
            return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
        jac_inputs = {ctx.r_val, ctx.v_val, ctx.q_val, ctx.a_val, ctx.b_val};
    }

    const std::vector<DM> jac_outputs = J_fn(jac_inputs);
    log_stage("jacobian evaluation");
    if (deadline_expired("jacobian evaluation")) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }
    Eigen::SparseMatrix<double> J = dm_to_sparse_eigen(jac_outputs.at(0));
    log_stage("dm to sparse eigen");
    if (deadline_expired("dm to sparse eigen")) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }

    const int n_drag    = ctx.use_drag ? 1 : 0;
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
            static_cast<double>(ctx.q_val(0, i)),
            static_cast<double>(ctx.q_val(1, i)),
            static_cast<double>(ctx.q_val(2, i)),
            static_cast<double>(ctx.q_val(3, i)));
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
    if (ctx.use_drag) {
        t_triplets.emplace_back(10 * N + 3 * N_uma + 3, 9 * N + 3 * N_uma + 3, 1.0);
    }
    Eigen::SparseMatrix<double> T(n_full, n_reduced);
    T.setFromTriplets(t_triplets.begin(), t_triplets.end());
    T.makeCompressed();
    log_stage("sparse tangent projection build");
    if (deadline_expired("sparse tangent projection build")) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }

    spdlog::info("solve_batch_opt: solving covariance normal equations ({}x{} reduced Jacobian).",
                 J.rows(), n_reduced);
    const Eigen::SparseMatrix<double> J_reduced = J * T;
    log_stage("sparse projection multiply");
    if (deadline_expired("sparse projection multiply")) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }
    Eigen::SparseMatrix<double> normal_sparse = J_reduced.transpose() * J_reduced;
    log_stage("sparse normal matrix");
    if (deadline_expired("sparse normal matrix")) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }
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
    if (deadline_expired("ldlt factorization")) {
        return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }
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
    if (ctx.use_drag) {
        covariance_indices.push_back(9 * N + 3 * N_uma + 3);
    }

    Eigen::VectorXd cov_diag = Eigen::VectorXd::Zero(n_reduced);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_reduced);
    for (const int idx : covariance_indices) {
        if (deadline_expired("selected inverse diagonal solve")) {
            return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
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

    constexpr double kCovarianceNegativeTolerance = -1e-10;
    for (const int idx : covariance_indices) {
        const double value = cov_diag(idx);
        if (!std::isfinite(value)) {
            spdlog::warn("solve_batch_opt: covariance diagonal {} is not finite ({}).",
                         idx, value);
            return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
        if (value < kCovarianceNegativeTolerance) {
            spdlog::warn("solve_batch_opt: covariance diagonal {} is negative ({}); "
                         "covariance is not numerically valid.",
                         idx, value);
            return ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        }
        if (value < 0.0) {
            cov_diag(idx) = 0.0;
        }
    }

    ResidualsOrCovariances covariance(N, StateResIdx::STATE_RES_COUNT);
    covariance.setZero();
    for (int i = 0; i < N; ++i) {
        covariance(i, RES_TIMESTAMP) = ctx.state_timestamps[static_cast<size_t>(i)];
        for (int c = 0; c < 3; ++c) {
            covariance(i, RES_POS_X + c) = cov_diag(3 * i + c);
            covariance(i, RES_VEL_X + c) = cov_diag(3 * N + 3 * i + c);
            covariance(i, RES_ROT_X + c) = cov_diag(6 * N + 3 * i + c);
        }
    }
    for (int c = 0; c < 3; ++c) {
        covariance(0, RES_GYRO_BIAS_X + c) = cov_diag(9 * N + 3 * N_uma + c);
    }
    if (ctx.use_drag) {
        double cd_var = cov_diag(9 * N + 3 * N_uma + 3);
        spdlog::info("solve_batch_opt: Cd variance = {:.4e}", cd_var);
        if (cd_var_out) *cd_var_out = cd_var;
    }
    log_stage("output pack");

    spdlog::info("solve_batch_opt: covariance computed.");
    return covariance;
}

// ── Timestamp mapping ─────────────────────────────────────────────────────────

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

// ── filter_landmark_measurements ─────────────────────────────────────────────
// Returns a copy of meas keeping only rows where keep_mask[i] == true.
// Reconstructs group_starts so the first surviving row of each original group
// opens a new group.  Gyro measurements are copied unchanged.
static ODMeasurements filter_landmark_measurements(
    const ODMeasurements& meas,
    const std::vector<bool>& keep_mask)
{
    const idx_t N = meas.landmark_measurements.rows();

    // Compute 0-based group index per row
    std::vector<int> group_ids(static_cast<size_t>(N));
    int g = -1;
    for (idx_t i = 0; i < N; ++i) {
        if (meas.group_starts(i)) ++g;
        group_ids[static_cast<size_t>(i)] = g;
    }

    int n_kept = 0;
    for (idx_t i = 0; i < N; ++i)
        if (keep_mask[static_cast<size_t>(i)]) ++n_kept;

    ODMeasurements out;
    out.landmark_measurements.resize(n_kept, LandmarkMeasurementIdx::LANDMARK_COUNT);
    out.group_starts.resize(n_kept);
    out.landmark_uncertainties.resize(n_kept);
    out.gyro_measurements = meas.gyro_measurements;

    idx_t row_out = 0;
    int prev_gid = -1;
    for (idx_t i = 0; i < N; ++i) {
        if (!keep_mask[static_cast<size_t>(i)]) continue;
        out.landmark_measurements.row(row_out) = meas.landmark_measurements.row(i);
        out.landmark_uncertainties(row_out)    = meas.landmark_uncertainties(i);
        const int gid = group_ids[static_cast<size_t>(i)];
        out.group_starts(row_out) = (gid != prev_gid);
        prev_gid = gid;
        ++row_out;
    }
    return out;
}

// ── eval_landmark_residual_numeric ────────────────────────────────────────────
// Pure Eigen computation of the σ-normalised bearing residual at a concrete state.
// q_xyzw = body-to-ECI quaternion [x, y, z, w].
static Eigen::Vector3d eval_landmark_residual_numeric(
    const Eigen::Vector3d& r,
    const Eigen::Vector4d& q_xyzw,
    const Eigen::Vector3d& lmk_pos,
    const Eigen::Vector3d& bearing_meas,
    double sigma)
{
    const Eigen::Vector3d diff = lmk_pos - r;
    const double dist = std::max(diff.norm(), 1e-3);
    const Eigen::Vector3d bearing_eci = diff / dist;

    // Rotate bearing from ECI to body frame using the conjugate quaternion.
    // Eigen Quaterniond is (w, x, y, z); our convention is [x, y, z, w].
    const Eigen::Quaterniond q_eig(q_xyzw(3), q_xyzw(0), q_xyzw(1), q_xyzw(2));
    const Eigen::Vector3d bearing_body = q_eig.conjugate() * bearing_eci;

    return (bearing_body - bearing_meas) / sigma;
}

// ── solve_batch_opt_internal ──────────────────────────────────────────────────
// Builds the CasADi NLP, runs IPOPT, and extracts residuals.
// Does NOT compute covariance (caller decides based on context).
// Returns {BatchOptResult, SolveContext}; SolveContext is empty on failure.

static std::pair<BatchOptResult, SolveContext>
solve_batch_opt_internal(const ODMeasurements& measurements,
                         const BATCH_OPT_config& bo_config,
                         const WarmStartData* warm_start)
{
    BatchOptResult result;
    SolveContext ctx;

    if (measurements.Validate() != ErrorCode::OK) {
        result.code = ErrorCode::ODMEAS_NOT_VALID;
        return {result, ctx};
    }

    const Eigen::MatrixXd& lm  = measurements.landmark_measurements;
    const auto&            gs  = measurements.group_starts;
    const Eigen::MatrixXd& gm  = measurements.gyro_measurements;
    const bool use_j2            = bo_config.use_j2;
    const bool use_drag          = bo_config.use_drag;
    const double cd_nominal      = bo_config.cd_nominal;
    const double cd_std          = bo_config.cd_std;
    const Integrator integrator  = bo_config.integrator;
    const double huber_M         = bo_config.landmark_huber_delta;

    if (bo_config.bias_mode != BIAS_MODE::FIX_BIAS) {
        spdlog::error("solve_batch_opt: only FIX_BIAS mode is supported in the CasADi backend.");
        result.code = ErrorCode::BATCH_OPT_BUILD_FAILED;
        return {result, ctx};
    }

    const idx_t num_groups = std::count(gs.col(0).begin(), gs.col(0).end(), true);

    StateTimestampsResult ts = get_state_timestamps(
        lm, gs, gm, num_groups);
    if (ts.code != ErrorCode::OK) { result.code = ts.code; return {result, ctx}; }

    const int N     = static_cast<int>(ts.state_timestamps.size());
    const int N_uma = N - 1;
    const int Ml    = static_cast<int>(lm.rows());

    // ── Initial guess ─────────────────────────────────────────────────────────
    if (warm_start == nullptr) {
        TrajectoryInitializer initializer(
            ts.state_timestamps, lm, gs, bo_config.bias_mode, gm);
        result.initial_trajectory = initializer.state_estimates();
    } else {
        result.initial_trajectory = warm_start->state_estimates;
    }
    const StateEstimates& init_se = result.initial_trajectory;

    spdlog::info("solve_batch_opt: {} states, {} landmark groups; building CasADi problem "
                 "(J2: {}, drag: {}, integrator: {}, huber_M: {:.2f}) …",
                 N, ts.landmark_group_indices.size(), use_j2 ? "on" : "off",
                 use_drag ? "on" : "off",
                 integrator == Integrator::RK4 ? "RK4" : "Euler", huber_M);

    // ── Decision variables ────────────────────────────────────────────────────
    Opti opti;
    MX r = opti.variable(3, N);
    MX v = opti.variable(3, N);
    MX q = opti.variable(4, N);
    MX a = opti.variable(3, N_uma);
    MX b = opti.variable(3);
    MX d = use_drag ? opti.variable(1) : MX(cd_nominal);

    // ── Quaternion unit-norm constraints ──────────────────────────────────────
    for (int i = 0; i < N; ++i) {
        MX qi = q(Slice(), i);
        opti.subject_to(dot(qi, qi) == 1.0);
    }

    // ── Linear dynamics (equality) + UMA prior (soft) ─────────────────────────
    MX obj = MX(0.0);
    std::vector<MX> dyn_constraints;
    std::vector<MX> uma_residuals;
    std::vector<MX> drag_residuals;
    dyn_constraints.reserve(static_cast<size_t>(N_uma));
    uma_residuals.reserve(static_cast<size_t>(N_uma));

    if (use_drag) {
        MX drag_res = (d - cd_nominal) / cd_std;
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
        obj = obj + dot(uma_res, uma_res);
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
        obj = obj + dot(ang_res, ang_res);
    }

    // ── Landmark bearing (soft, Pseudo-Huber) ────────────────────────────────
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
                obj = obj + pseudo_huber_cost(lmk_res, huber_M);
                ++row;
            } while (row < static_cast<idx_t>(Ml) && !gs(row, 0));
        }
    }

    opti.minimize(obj);

    // ── Initial guess ─────────────────────────────────────────────────────────
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

        if (warm_start != nullptr && warm_start->uma_accelerations.rows() == N_uma) {
            std::vector<DM> a_cols;
            a_cols.reserve(static_cast<size_t>(N_uma));
            for (int i = 0; i < N_uma; ++i) {
                a_cols.push_back(dm3(warm_start->uma_accelerations(i, 0),
                                     warm_start->uma_accelerations(i, 1),
                                     warm_start->uma_accelerations(i, 2)));
            }
            opti.set_initial(a, DM::horzcat(a_cols));
        } else {
            opti.set_initial(a, DM::zeros(3, N_uma));
        }

        opti.set_initial(b, dm3(init_se(0, GYRO_BIAS_X),
                                init_se(0, GYRO_BIAS_Y),
                                init_se(0, GYRO_BIAS_Z)));
        if (use_drag) {
            opti.set_initial(d, DM(warm_start ? warm_start->cd : cd_nominal));
        }
    }

    // ── IPOPT ─────────────────────────────────────────────────────────────────
    Dict ipopt_opts;
    ipopt_opts["max_iter"]       = static_cast<int>(bo_config.max_iterations);
    ipopt_opts["max_cpu_time"]   = bo_config.max_run_time_sec;
    ipopt_opts["tol"]            = bo_config.solver_function_tolerance;
    ipopt_opts["acceptable_tol"] = bo_config.solver_function_tolerance * 100.0;
    ipopt_opts["print_level"]    = 5;
    opti.solver("ipopt", Dict{}, ipopt_opts);

    spdlog::info("solve_batch_opt: starting IPOPT solve …");

    try {
        const auto solve_start = SteadyClock::now();
        OptiSol sol = opti.solve();
        const auto solve_end = SteadyClock::now();
        const double solve_time_sec =
            std::chrono::duration<double>(solve_end - solve_start).count();

        // ── Solver summary ──────────────────────────────────────────────────
        {
            casadi::Dict stats = sol.stats();
            result.solver_summary.return_status =
                static_cast<std::string>(stats.at("return_status"));
            result.solver_summary.iter_count =
                static_cast<int>(stats.at("iter_count"));
            result.solver_summary.final_cost =
                static_cast<double>(sol.value(obj));
            result.solver_summary.solve_time_ms = solve_time_sec * 1000.0;
        }

        // ── Extract solution ────────────────────────────────────────────────
        DM r_val = sol.value(r);
        DM v_val = sol.value(v);
        DM q_val = sol.value(q);
        DM a_val = sol.value(a);
        DM b_val = sol.value(b);
        DM d_val = use_drag ? sol.value(d) : DM(cd_nominal);

        result.gyro_bias_fixed = {
            static_cast<double>(b_val(0, 0)),
            static_cast<double>(b_val(1, 0)),
            static_cast<double>(b_val(2, 0))
        };
        result.cd           = static_cast<double>(d_val(0, 0));
        result.cd_estimated = use_drag;

        if (use_drag) {
            spdlog::info("solve_batch_opt: Cd = {:.4e}", result.cd);
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
                return {result, ctx};
            }
        }
        for (int i = 0; i < N; ++i) {
            double qx = state_estimates(i, QUAT_X), qy = state_estimates(i, QUAT_Y);
            double qz = state_estimates(i, QUAT_Z), qw = state_estimates(i, QUAT_W);
            if (std::abs(std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw) - 1.0) > 0.05) {
                spdlog::error("solve_batch_opt: denormalised quaternion at state {}.", i);
                result.code = ErrorCode::BATCH_OPT_INVALID_OUTPUT;
                return {result, ctx};
            }
        }

        result.state_estimates = std::move(state_estimates);

        // ── UMA accelerations (needed for warm-starting next iteration) ──────
        result.uma_accelerations.resize(N_uma, 3);
        for (int i = 0; i < N_uma; ++i) {
            result.uma_accelerations(i, 0) = static_cast<double>(a_val(0, i));
            result.uma_accelerations(i, 1) = static_cast<double>(a_val(1, i));
            result.uma_accelerations(i, 2) = static_cast<double>(a_val(2, i));
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

        result.code = ErrorCode::OK;

        spdlog::info("solve_batch_opt: IPOPT converged. Final pos [{:.1f}, {:.1f}, {:.1f}] km.",
                     static_cast<double>(r_val(0, N - 1)),
                     static_cast<double>(r_val(1, N - 1)),
                     static_cast<double>(r_val(2, N - 1)));

        // ── Populate SolveContext for optional covariance computation ────────
        ctx.uma_residuals   = std::move(uma_residuals);
        ctx.ang_residuals   = std::move(ang_residuals);
        ctx.lmk_residuals   = std::move(lmk_residuals);
        ctx.dyn_constraints = std::move(dyn_constraints);
        ctx.drag_residuals  = std::move(drag_residuals);
        ctx.r = r; ctx.v = v; ctx.q = q; ctx.a = a; ctx.b = b; ctx.d = d;
        ctx.r_val = r_val; ctx.v_val = v_val; ctx.q_val = q_val;
        ctx.a_val = a_val; ctx.b_val = b_val; ctx.d_val = d_val;
        ctx.use_drag = use_drag;
        ctx.state_timestamps = ts.state_timestamps;
        ctx.opti_ref = opti;   // holds the symbolic graph alive

    } catch (const std::exception& e) {
        spdlog::error("solve_batch_opt: IPOPT failed — {}", e.what());
        LogError(ErrorCode::BATCH_OPT_NO_CONVERGENCE);
        result.code = ErrorCode::BATCH_OPT_NO_CONVERGENCE;
    }

    return {result, ctx};
}

// ── solve_batch_opt (public, single-shot) ─────────────────────────────────────

BatchOptResult solve_batch_opt(const ODMeasurements& measurements,
                               BATCH_OPT_config bo_config,
                               const WarmStartData* warm_start)
{
    auto [result, ctx] = solve_batch_opt_internal(measurements, bo_config, warm_start);

    if (result.code != ErrorCode::OK || !bo_config.compute_covariance) {
        result.covariance = ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        return result;
    }

    const double solve_time_sec = result.solver_summary.solve_time_ms / 1000.0;
    const double remaining_sec  = bo_config.max_run_time_sec - solve_time_sec;

    if (remaining_sec <= 0.0) {
        result.covariance_timed_out = true;
        result.covariance_time_ms   = 0.0;
        result.covariance = ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
        spdlog::warn("solve_batch_opt: skipping covariance because IPOPT used {:.3f}s of {:.3f}s budget.",
                     solve_time_sec, bo_config.max_run_time_sec);
        return result;
    }

    spdlog::info("solve_batch_opt: covariance has {:.3f}s remaining from {:.3f}s budget.",
                 remaining_sec, bo_config.max_run_time_sec);

    const auto cov_start    = SteadyClock::now();
    const auto cov_deadline = cov_start + std::chrono::duration_cast<SteadyClock::duration>(
                                  std::chrono::duration<double>(remaining_sec));
    double cd_var_out = 0.0;
    result.covariance = compute_covariance(ctx, &cd_var_out, &cov_deadline,
                                           &result.covariance_timed_out);
    result.covariance_time_ms =
        std::chrono::duration<double, std::milli>(SteadyClock::now() - cov_start).count();

    spdlog::info("solve_batch_opt: covariance computation took {:.1f} ms.",
                 result.covariance_time_ms);

    if (result.covariance.rows() > 0) {
        result.covariance_computed = true;
        result.gyro_bias_var = {
            result.covariance(0, RES_GYRO_BIAS_X),
            result.covariance(0, RES_GYRO_BIAS_Y),
            result.covariance(0, RES_GYRO_BIAS_Z)
        };
        result.cd_var = cd_var_out;
    }

    return result;
}

// ── solve_batch_opt_with_outlier_rejection (public) ───────────────────────────

BatchOptResult solve_batch_opt_with_outlier_rejection(
    const ODMeasurements& measurements,
    BATCH_OPT_config bo_config)
{
    const idx_t N_total = measurements.landmark_measurements.rows();
    const double mahal_threshold  = bo_config.mahal_threshold;
    const int    max_od_iters     = bo_config.max_od_iterations;

    // Active measurement set, updated each iteration
    ODMeasurements active = measurements;

    // Maps each active row index → original row index in measurements
    std::vector<idx_t> active_original_indices(static_cast<size_t>(N_total));
    for (idx_t i = 0; i < N_total; ++i)
        active_original_indices[static_cast<size_t>(i)] = i;

    // Outlier flag per original row: 0=inlier, 1=outlier
    std::vector<int8_t> outlier_flags(static_cast<size_t>(N_total), 0);

    // Build IPOPT config for inner solves: covariance computed only at the end
    BATCH_OPT_config iter_config = bo_config;
    iter_config.compute_covariance = false;

    StateEstimates cold_start_trajectory;  // preserved from iteration 0
    bool cold_start_captured = false;

    const WarmStartData* warm_ptr = nullptr;
    WarmStartData warm_data;

    BatchOptResult result;
    SolveContext   last_ctx;
    std::vector<SolverSummaryInfo> iteration_summaries;

    const auto loop_start = SteadyClock::now();

    int iteration = 0;
    for (; iteration < max_od_iters; ++iteration) {
        const int n_active_groups = static_cast<int>(
            std::count(active.group_starts.data(),
                       active.group_starts.data() + active.group_starts.size(), true));

        spdlog::info("[outlier_rejection] Iteration {} | {} groups | {} measurements",
                     iteration + 1, n_active_groups,
                     active.landmark_measurements.rows());

        auto [iter_result, iter_ctx] = solve_batch_opt_internal(active, iter_config, warm_ptr);

        // Capture cold-start initial trajectory once
        if (!cold_start_captured) {
            cold_start_trajectory = iter_result.initial_trajectory;
            cold_start_captured   = true;
        }

        if (iter_result.code != ErrorCode::OK) {
            spdlog::error("[outlier_rejection] Solver failed at iteration {}.", iteration + 1);
            iter_result.lmk_outlier_flags  = outlier_flags;
            iter_result.n_od_solver_calls  = iteration + 1;
            iter_result.n_lmk_outliers     = static_cast<int>(
                std::count(outlier_flags.begin(), outlier_flags.end(), int8_t(1)));
            if (cold_start_captured)
                iter_result.initial_trajectory = cold_start_trajectory;
            return iter_result;
        }

        // Compute per-landmark Mahalanobis distance (residuals are already σ-normalised)
        const int n_active = static_cast<int>(iter_result.landmark_residuals.rows());
        std::vector<double> mahal(static_cast<size_t>(n_active));
        std::vector<bool>   keep_mask(static_cast<size_t>(n_active));
        int n_rejected = 0;
        double mahal_max  = 0.0;
        double mahal_sum  = 0.0;
        for (int k = 0; k < n_active; ++k) {
            const double dx = iter_result.landmark_residuals(k, LANDMARK_RES_X);
            const double dy = iter_result.landmark_residuals(k, LANDMARK_RES_Y);
            const double dz = iter_result.landmark_residuals(k, LANDMARK_RES_Z);
            mahal[static_cast<size_t>(k)] = std::sqrt(dx*dx + dy*dy + dz*dz);
            mahal_max = std::max(mahal_max, mahal[static_cast<size_t>(k)]);
            mahal_sum += mahal[static_cast<size_t>(k)];
            keep_mask[static_cast<size_t>(k)] = (mahal[static_cast<size_t>(k)] <= mahal_threshold);
            if (!keep_mask[static_cast<size_t>(k)]) ++n_rejected;
        }

        spdlog::info("[outlier_rejection] Mahalanobis — max={:.3f}  mean={:.3f}  "
                     "rejected={}/{} (threshold={:.2f}σ)",
                     mahal_max, mahal_sum / static_cast<double>(n_active),
                     n_rejected, n_active, mahal_threshold);

        {
            SolverSummaryInfo iter_summary = iter_result.solver_summary;
            iter_summary.n_active_landmarks = n_active;
            iter_summary.n_rejected         = n_rejected;
            iteration_summaries.push_back(iter_summary);
        }

        if (n_rejected == 0) {
            spdlog::info("[outlier_rejection] Converged after {} iteration(s).", iteration + 1);
            result   = std::move(iter_result);
            last_ctx = std::move(iter_ctx);
            break;
        }

        if (iteration == max_od_iters - 1) {
            spdlog::warn("[outlier_rejection] Reached max_od_iterations={}; "
                         "{} outlier(s) remain.", max_od_iters, n_rejected);
            result   = std::move(iter_result);
            last_ctx = SolveContext{};  // no covariance at cap
            break;
        }

        // Mark new outliers in the global flag array
        for (int k = 0; k < n_active; ++k) {
            if (!keep_mask[static_cast<size_t>(k)]) {
                outlier_flags[static_cast<size_t>(
                    active_original_indices[static_cast<size_t>(k)])] = 1;
            }
        }

        // Rebuild active index map
        std::vector<idx_t> new_active_indices;
        new_active_indices.reserve(static_cast<size_t>(n_active - n_rejected));
        for (int k = 0; k < n_active; ++k) {
            if (keep_mask[static_cast<size_t>(k)])
                new_active_indices.push_back(active_original_indices[static_cast<size_t>(k)]);
        }
        active_original_indices = std::move(new_active_indices);

        // Filter active measurement set
        active = filter_landmark_measurements(active, keep_mask);

        // Prepare warm start for next iteration
        warm_data.state_estimates    = iter_result.state_estimates;
        warm_data.uma_accelerations  = iter_result.uma_accelerations;
        warm_data.cd                 = iter_result.cd;
        warm_ptr = &warm_data;
    }

    // ── Compute covariance from the last converged solve context ──────────────
    if (bo_config.compute_covariance && !last_ctx.state_timestamps.empty()) {
        const double elapsed_sec =
            std::chrono::duration<double>(SteadyClock::now() - loop_start).count();
        const double remaining_sec = bo_config.max_run_time_sec - elapsed_sec;

        if (remaining_sec <= 0.0) {
            result.covariance_timed_out = true;
            result.covariance_time_ms   = 0.0;
            result.covariance = ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
            spdlog::warn("[outlier_rejection] Skipping covariance: OD budget exhausted "
                         "({:.3f}s elapsed of {:.3f}s).", elapsed_sec, bo_config.max_run_time_sec);
        } else {
            spdlog::info("[outlier_rejection] Computing covariance ({:.3f}s remaining).",
                         remaining_sec);
            const auto cov_start    = SteadyClock::now();
            const auto cov_deadline = cov_start + std::chrono::duration_cast<SteadyClock::duration>(
                                          std::chrono::duration<double>(remaining_sec));
            double cd_var_out = 0.0;
            result.covariance = compute_covariance(last_ctx, &cd_var_out, &cov_deadline,
                                                   &result.covariance_timed_out);
            result.covariance_time_ms =
                std::chrono::duration<double, std::milli>(SteadyClock::now() - cov_start).count();
            spdlog::info("[outlier_rejection] Covariance took {:.1f} ms.",
                         result.covariance_time_ms);
            if (result.covariance.rows() > 0) {
                result.covariance_computed = true;
                result.gyro_bias_var = {
                    result.covariance(0, RES_GYRO_BIAS_X),
                    result.covariance(0, RES_GYRO_BIAS_Y),
                    result.covariance(0, RES_GYRO_BIAS_Z)
                };
                result.cd_var = cd_var_out;
            }
        }
    } else {
        result.covariance = ResidualsOrCovariances(0, StateResIdx::STATE_RES_COUNT);
    }

    // ── Restore cold-start initial trajectory ─────────────────────────────────
    if (cold_start_captured)
        result.initial_trajectory = cold_start_trajectory;

    // ── Expand landmark_residuals to full N_total rows ────────────────────────
    // Inlier rows: residuals from the final solve placed at their original indices.
    // Outlier rows: residuals re-evaluated numerically at the final solution state.
    const int N_states = static_cast<int>(result.state_estimates.rows());
    std::vector<double> state_ts(static_cast<size_t>(N_states));
    for (int i = 0; i < N_states; ++i)
        state_ts[static_cast<size_t>(i)] = result.state_estimates(i, STATE_ESTIMATE_TIMESTAMP);

    LandmarkResiduals full_lmk_res(N_total, LandmarkResIdx::LANDMARK_RES_COUNT);
    full_lmk_res.setConstant(std::numeric_limits<double>::quiet_NaN());

    // Place inlier residuals
    const int n_inlier = static_cast<int>(result.landmark_residuals.rows());
    for (int k = 0; k < n_inlier; ++k) {
        const idx_t orig = active_original_indices[static_cast<size_t>(k)];
        full_lmk_res(orig, LANDMARK_RES_X) = result.landmark_residuals(k, LANDMARK_RES_X);
        full_lmk_res(orig, LANDMARK_RES_Y) = result.landmark_residuals(k, LANDMARK_RES_Y);
        full_lmk_res(orig, LANDMARK_RES_Z) = result.landmark_residuals(k, LANDMARK_RES_Z);
    }

    // Re-evaluate outlier residuals at the final solution state
    if (N_states > 0) {
        for (idx_t orig_row = 0; orig_row < N_total; ++orig_row) {
            if (outlier_flags[static_cast<size_t>(orig_row)] == 0) continue;

            const double t_lmk = measurements.landmark_measurements(
                orig_row, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP);

            // Nearest state by timestamp
            auto it = std::lower_bound(state_ts.begin(), state_ts.end(), t_lmk);
            int idx;
            if (it == state_ts.end()) {
                idx = N_states - 1;
            } else if (it == state_ts.begin()) {
                idx = 0;
            } else {
                auto prev = std::prev(it);
                idx = (*it - t_lmk < t_lmk - *prev)
                      ? static_cast<int>(it   - state_ts.begin())
                      : static_cast<int>(prev - state_ts.begin());
            }

            Eigen::Vector3d r_s(result.state_estimates(idx, POS_X),
                                result.state_estimates(idx, POS_Y),
                                result.state_estimates(idx, POS_Z));
            Eigen::Vector4d q_s(result.state_estimates(idx, QUAT_X),
                                result.state_estimates(idx, QUAT_Y),
                                result.state_estimates(idx, QUAT_Z),
                                result.state_estimates(idx, QUAT_W));
            Eigen::Vector3d lmk_pos(
                measurements.landmark_measurements(orig_row, LandmarkMeasurementIdx::LANDMARK_POS_X),
                measurements.landmark_measurements(orig_row, LandmarkMeasurementIdx::LANDMARK_POS_Y),
                measurements.landmark_measurements(orig_row, LandmarkMeasurementIdx::LANDMARK_POS_Z));
            Eigen::Vector3d bearing_meas(
                measurements.landmark_measurements(orig_row, LandmarkMeasurementIdx::BEARING_VEC_X),
                measurements.landmark_measurements(orig_row, LandmarkMeasurementIdx::BEARING_VEC_Y),
                measurements.landmark_measurements(orig_row, LandmarkMeasurementIdx::BEARING_VEC_Z));
            const double sigma = measurements.landmark_uncertainties(orig_row);

            const Eigen::Vector3d res = eval_landmark_residual_numeric(r_s, q_s, lmk_pos, bearing_meas, sigma);
            full_lmk_res(orig_row, LANDMARK_RES_X) = res(0);
            full_lmk_res(orig_row, LANDMARK_RES_Y) = res(1);
            full_lmk_res(orig_row, LANDMARK_RES_Z) = res(2);
        }
    }
    result.landmark_residuals = std::move(full_lmk_res);

    // ── Finalise metadata ─────────────────────────────────────────────────────
    result.lmk_outlier_flags    = outlier_flags;
    result.iteration_summaries  = std::move(iteration_summaries);
    result.n_od_solver_calls = iteration + 1;
    result.n_lmk_outliers    = static_cast<int>(
        std::count(outlier_flags.begin(), outlier_flags.end(), int8_t(1)));

    spdlog::info("[outlier_rejection] Done: {} solver call(s), {} outlier(s) rejected.",
                 result.n_od_solver_calls, result.n_lmk_outliers);

    return result;
}
