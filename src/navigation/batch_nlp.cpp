#include "navigation/batch_nlp.hpp"
#include "spdlog/spdlog.h"
#include <unsupported/Eigen/AutoDiff>
#include <IpIpoptData.hpp>
#include <cmath>
#include <cstring>

// ── AutoDiff scalar types for gradient computation ────────────────────────────
// FIX_BIAS angular: q0(4) + q1(4) + b(3) = 11
using AD11 = Eigen::AutoDiffScalar<Eigen::Matrix<double, 11, 1>>;
// TV_BIAS angular:  q0(4) + b0(3) + q1(4) + b1(3) = 14
using AD14 = Eigen::AutoDiffScalar<Eigen::Matrix<double, 14, 1>>;
// NO_BIAS angular:  q0(4) + q1(4) = 8
using AD8  = Eigen::AutoDiffScalar<Eigen::Matrix<double,  8, 1>>;
// Landmark:         pos(3) + quat(4) = 7
using AD7  = Eigen::AutoDiffScalar<Eigen::Matrix<double,  7, 1>>;

// ── Constructor ───────────────────────────────────────────────────────────────

BatchNLP::BatchNLP(StateEstimates&                    state_estimates,
                   const std::vector<double>&         state_timestamps,
                   const std::vector<idx_t>&          landmark_group_indices,
                   const LandmarkMeasurements&        landmark_measurements,
                   const LandmarkGroupStarts&         landmark_group_starts,
                   const GyroMeasurements&            gyro_measurements,
                   BIAS_MODE                          bias_mode,
                   double                             uma_std_dev,
                   double                             gyro_wn_std_dev_rad_s,
                   double                             gyro_bias_instability,
                   const Eigen::VectorXd&             landmark_uncertainties)
    : state_estimates_(state_estimates),
      ts_(state_timestamps),
      lm_group_indices_(landmark_group_indices),
      lm_meas_(landmark_measurements),
      lm_group_starts_(landmark_group_starts),
      gyro_meas_(gyro_measurements),
      bias_mode_(bias_mode),
      uma_std_dev_(uma_std_dev),
      gyro_wn_std_dev_rad_s_(gyro_wn_std_dev_rad_s),
      gyro_bias_instability_(gyro_bias_instability),
      lm_uncertainties_(landmark_uncertainties)
{
    N_ = static_cast<int>(state_estimates.rows());
    M_ = N_ - 1;

    switch (bias_mode_) {
        case BIAS_MODE::NO_BIAS:  b_size_ = 0;       break;
        case BIAS_MODE::FIX_BIAS: b_size_ = 3;       break;
        case BIAS_MODE::TV_BIAS:  b_size_ = 3 * N_;  break;
    }

    // Variable offsets
    r_off_ = 0;
    v_off_ = 3 * N_;
    q_off_ = 6 * N_;
    a_off_ = 10 * N_;
    b_off_ = (b_size_ > 0) ? (10 * N_ + 3 * M_) : -1;

    n_vars_ = 10 * N_ + 3 * M_ + b_size_;

    // Constraint offsets: qnorm, pos dynamics, vel dynamics
    qnorm_off_ = 0;
    rdyn_off_  = N_;
    vdyn_off_  = N_ + 3 * M_;
    n_con_     = N_ + 6 * M_;

    // Sparsity count: 4N (qnorm) + 9M (rdyn) + 18M (vdyn)
    nnz_jac_   = 4 * N_ + 27 * M_;
}

// ── Gravity helpers ───────────────────────────────────────────────────────────

void BatchNLP::gravity(const double* r3, double* a3) const
{
    const double rx = r3[0], ry = r3[1], rz = r3[2];
    const double r2 = rx*rx + ry*ry + rz*rz + 1e-12;
    const double r  = std::sqrt(r2);
    const double c  = -GM_EARTH / (r * r2);
    a3[0] = c * rx;
    a3[1] = c * ry;
    a3[2] = c * rz;
}

// 3×3 row-major Jacobian ∂a_grav/∂r
void BatchNLP::gravity_jac(const double* r3, double Jg[9]) const
{
    const double rx = r3[0], ry = r3[1], rz = r3[2];
    const double r2  = rx*rx + ry*ry + rz*rz + 1e-12;
    const double r   = std::sqrt(r2);
    const double r3_ = r * r2;
    const double r5  = r3_ * r2;

    // ∂a_grav/∂r = -GM*(I/r³ - 3*r*r^T/r^5)
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double rr  = (row == 0 ? rx : (row == 1 ? ry : rz));
            const double cc  = (col == 0 ? rx : (col == 1 ? ry : rz));
            const double eye = (row == col) ? 1.0 : 0.0;
            Jg[row * 3 + col] = -GM_EARTH * (eye / r3_ - 3.0 * rr * cc / r5);
        }
    }
}

// ── get_nlp_info ─────────────────────────────────────────────────────────────

bool BatchNLP::get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                             Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                             Ipopt::TNLP::IndexStyleEnum& index_style)
{
    n            = static_cast<Ipopt::Index>(n_vars_);
    m            = static_cast<Ipopt::Index>(n_con_);
    nnz_jac_g    = static_cast<Ipopt::Index>(nnz_jac_);
    nnz_h_lag    = 0;  // L-BFGS: Hessian not provided
    index_style  = Ipopt::TNLP::C_STYLE;
    return true;
}

// ── get_bounds_info ───────────────────────────────────────────────────────────

bool BatchNLP::get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                                Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u)
{
    // All variables unbounded
    for (Ipopt::Index i = 0; i < n; ++i) {
        x_l[i] = -1e20;
        x_u[i] =  1e20;
    }
    // All constraints are equality: g = 0
    for (Ipopt::Index i = 0; i < m; ++i) {
        g_l[i] = g_u[i] = 0.0;
    }
    return true;
}

// ── get_starting_point ────────────────────────────────────────────────────────

bool BatchNLP::get_starting_point(Ipopt::Index /*n*/, bool init_x, Ipopt::Number* x,
                                   bool /*init_z*/, Ipopt::Number* /*z_L*/,
                                   Ipopt::Number* /*z_U*/,
                                   Ipopt::Index /*m*/, bool /*init_lambda*/,
                                   Ipopt::Number* /*lambda*/)
{
    if (!init_x) return true;

    for (int i = 0; i < N_; ++i) {
        x[r_off_ + 3*i + 0] = state_estimates_(i, StateEstimateIdx::POS_X);
        x[r_off_ + 3*i + 1] = state_estimates_(i, StateEstimateIdx::POS_Y);
        x[r_off_ + 3*i + 2] = state_estimates_(i, StateEstimateIdx::POS_Z);

        x[v_off_ + 3*i + 0] = state_estimates_(i, StateEstimateIdx::VEL_X);
        x[v_off_ + 3*i + 1] = state_estimates_(i, StateEstimateIdx::VEL_Y);
        x[v_off_ + 3*i + 2] = state_estimates_(i, StateEstimateIdx::VEL_Z);

        // Normalize initial quaternion
        double qx = state_estimates_(i, StateEstimateIdx::QUAT_X);
        double qy = state_estimates_(i, StateEstimateIdx::QUAT_Y);
        double qz = state_estimates_(i, StateEstimateIdx::QUAT_Z);
        double qw = state_estimates_(i, StateEstimateIdx::QUAT_W);
        const double qn = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
        if (qn > 1e-10) { qx /= qn; qy /= qn; qz /= qn; qw /= qn; }
        x[q_off_ + 4*i + 0] = qx;
        x[q_off_ + 4*i + 1] = qy;
        x[q_off_ + 4*i + 2] = qz;
        x[q_off_ + 4*i + 3] = qw;
    }

    // UMA initialised to zero
    for (int i = 0; i < 3 * M_; ++i) x[a_off_ + i] = 0.0;

    // Bias
    if (bias_mode_ == BIAS_MODE::FIX_BIAS) {
        x[b_off_ + 0] = state_estimates_(0, StateEstimateIdx::GYRO_BIAS_X);
        x[b_off_ + 1] = state_estimates_(0, StateEstimateIdx::GYRO_BIAS_Y);
        x[b_off_ + 2] = state_estimates_(0, StateEstimateIdx::GYRO_BIAS_Z);
    } else if (bias_mode_ == BIAS_MODE::TV_BIAS) {
        for (int i = 0; i < N_; ++i) {
            x[b_off_ + 3*i + 0] = state_estimates_(i, StateEstimateIdx::GYRO_BIAS_X);
            x[b_off_ + 3*i + 1] = state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Y);
            x[b_off_ + 3*i + 2] = state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Z);
        }
    }

    // Compute initial cost for reporting
    eval_f(static_cast<Ipopt::Index>(n_vars_), x, true, initial_cost);
    return true;
}

// ── eval_f ────────────────────────────────────────────────────────────────────

bool BatchNLP::eval_f(Ipopt::Index /*n*/, const Ipopt::Number* x, bool /*new_x*/,
                      Ipopt::Number& obj_value)
{
    obj_value = 0.0;

    // UMA prior
    for (int i = 0; i < M_; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double ak = x[a_off_ + 3*i + k] / uma_std_dev_;
            obj_value += ak * ak;
        }
    }

    // Angular dynamics residuals
    for (int i = 0; i < M_; ++i) {
        const double* gyro_row = gyro_meas_.data() + i * GyroMeasurementIdx::GYRO_MEAS_COUNT;
        const double dt        = ts_[static_cast<size_t>(i+1)] - ts_[static_cast<size_t>(i)];
        const double qstd      = gyro_wn_std_dev_rad_s_ * dt;
        const double bstd      = gyro_bias_instability_ * std::sqrt(dt);

        const double* q0 = x + q_off_ + 4*i;
        const double* q1 = x + q_off_ + 4*(i+1);

        if (bias_mode_ == BIAS_MODE::FIX_BIAS) {
            const double* b = x + b_off_;
            double res[3];
            AngularDynamicsCostFunctorFixBias f(gyro_row, dt, qstd);
            f(q0, q1, b, res);
            obj_value += res[0]*res[0] + res[1]*res[1] + res[2]*res[2];
        } else if (bias_mode_ == BIAS_MODE::TV_BIAS) {
            const double* b0 = x + b_off_ + 3*i;
            const double* b1 = x + b_off_ + 3*(i+1);
            double res[6];
            AngularDynamicsCostFunctor f(gyro_row, dt, qstd, bstd);
            f(q0, b0, q1, b1, res);
            for (int k = 0; k < 6; ++k) obj_value += res[k]*res[k];
        } else {
            double res[3];
            AngularDynamicsCostFunctorNoBias f(gyro_row, dt, qstd);
            f(q0, q1, res);
            obj_value += res[0]*res[0] + res[1]*res[1] + res[2]*res[2];
        }
    }

    // Landmark bearing residuals
    idx_t lm_idx = 0;
    for (size_t gi = 0; gi < lm_group_indices_.size(); ++gi) {
        const idx_t state_idx = lm_group_indices_[gi];
        const double* pos  = x + r_off_ + 3 * static_cast<int>(state_idx);
        const double* quat = x + q_off_ + 4 * static_cast<int>(state_idx);
        do {
            const double* lm_row = lm_meas_.data() + lm_idx * LandmarkMeasurementIdx::LANDMARK_COUNT;
            const double   sigma = lm_uncertainties_(lm_idx);
            double res[3];
            LandmarkCostFunctor f(lm_row, sigma);
            f(pos, quat, res);
            obj_value += res[0]*res[0] + res[1]*res[1] + res[2]*res[2];
            ++lm_idx;
        } while (lm_idx < lm_group_starts_.rows() && !lm_group_starts_(lm_idx, 0));
    }

    return true;
}

// ── eval_grad_f ───────────────────────────────────────────────────────────────

bool BatchNLP::eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                            Ipopt::Number* grad_f)
{
    std::memset(grad_f, 0, static_cast<size_t>(n) * sizeof(Ipopt::Number));

    // UMA prior: ∂/∂a_i ||a_i/σ||² = 2*a_i/σ²
    for (int i = 0; i < M_; ++i) {
        for (int k = 0; k < 3; ++k) {
            grad_f[a_off_ + 3*i + k] += 2.0 * x[a_off_ + 3*i + k] / (uma_std_dev_ * uma_std_dev_);
        }
    }

    // Angular dynamics gradient using AutoDiff Jets
    for (int i = 0; i < M_; ++i) {
        if (bias_mode_ == BIAS_MODE::FIX_BIAS)     ang_grad_fix_bias(x, i, grad_f);
        else if (bias_mode_ == BIAS_MODE::TV_BIAS)  ang_grad_tv_bias(x, i, grad_f);
        else                                         ang_grad_no_bias(x, i, grad_f);
    }

    // Landmark gradient
    idx_t lm_idx = 0;
    for (size_t gi = 0; gi < lm_group_indices_.size(); ++gi) {
        const int state_idx = static_cast<int>(lm_group_indices_[gi]);
        do {
            lmk_grad(x, state_idx, static_cast<int>(lm_idx), grad_f);
            ++lm_idx;
        } while (lm_idx < lm_group_starts_.rows() && !lm_group_starts_(lm_idx, 0));
    }

    return true;
}

// ── Angular gradient helpers (AutoDiff) ──────────────────────────────────────

void BatchNLP::ang_grad_fix_bias(const Ipopt::Number* x, int i,
                                  Ipopt::Number* grad_f) const
{
    const double* gyro_row = gyro_meas_.data() + i * GyroMeasurementIdx::GYRO_MEAS_COUNT;
    const double dt   = ts_[static_cast<size_t>(i+1)] - ts_[static_cast<size_t>(i)];
    const double qstd = gyro_wn_std_dev_rad_s_ * dt;

    // param layout [q0(0..3), q1(4..7), b(8..10)]
    AD11 q0[4], q1[4], b[3], res[3];
    for (int k = 0; k < 4; ++k) q0[k] = AD11(x[q_off_ + 4*i     + k], Eigen::Matrix<double,11,1>::Unit(k));
    for (int k = 0; k < 4; ++k) q1[k] = AD11(x[q_off_ + 4*(i+1) + k], Eigen::Matrix<double,11,1>::Unit(4+k));
    for (int k = 0; k < 3; ++k) b[k]  = AD11(x[b_off_            + k], Eigen::Matrix<double,11,1>::Unit(8+k));

    AngularDynamicsCostFunctorFixBias f(gyro_row, dt, qstd);
    f(q0, q1, b, res);

    // grad += 2 * J^T * r  (value() = residual scalar, derivatives() = Jacobian row)
    for (int r = 0; r < 3; ++r) {
        for (int k = 0; k < 4; ++k)
            grad_f[q_off_ + 4*i     + k] += 2.0 * res[r].value() * res[r].derivatives()(k);
        for (int k = 0; k < 4; ++k)
            grad_f[q_off_ + 4*(i+1) + k] += 2.0 * res[r].value() * res[r].derivatives()(4+k);
        for (int k = 0; k < 3; ++k)
            grad_f[b_off_            + k] += 2.0 * res[r].value() * res[r].derivatives()(8+k);
    }
}

void BatchNLP::ang_grad_tv_bias(const Ipopt::Number* x, int i,
                                 Ipopt::Number* grad_f) const
{
    const double* gyro_row = gyro_meas_.data() + i * GyroMeasurementIdx::GYRO_MEAS_COUNT;
    const double dt   = ts_[static_cast<size_t>(i+1)] - ts_[static_cast<size_t>(i)];
    const double qstd = gyro_wn_std_dev_rad_s_ * dt;
    const double bstd = gyro_bias_instability_ * std::sqrt(dt);

    // param layout [q0(0..3), b0(4..6), q1(7..10), b1(11..13)]
    AD14 q0[4], b0[3], q1[4], b1[3], res[6];
    for (int k = 0; k < 4; ++k) q0[k] = AD14(x[q_off_ + 4*i     + k], Eigen::Matrix<double,14,1>::Unit(k));
    for (int k = 0; k < 3; ++k) b0[k] = AD14(x[b_off_ + 3*i     + k], Eigen::Matrix<double,14,1>::Unit(4+k));
    for (int k = 0; k < 4; ++k) q1[k] = AD14(x[q_off_ + 4*(i+1) + k], Eigen::Matrix<double,14,1>::Unit(7+k));
    for (int k = 0; k < 3; ++k) b1[k] = AD14(x[b_off_ + 3*(i+1) + k], Eigen::Matrix<double,14,1>::Unit(11+k));

    AngularDynamicsCostFunctor f(gyro_row, dt, qstd, bstd);
    f(q0, b0, q1, b1, res);

    for (int r = 0; r < 6; ++r) {
        for (int k = 0; k < 4; ++k)
            grad_f[q_off_ + 4*i     + k] += 2.0 * res[r].value() * res[r].derivatives()(k);
        for (int k = 0; k < 3; ++k)
            grad_f[b_off_ + 3*i     + k] += 2.0 * res[r].value() * res[r].derivatives()(4+k);
        for (int k = 0; k < 4; ++k)
            grad_f[q_off_ + 4*(i+1) + k] += 2.0 * res[r].value() * res[r].derivatives()(7+k);
        for (int k = 0; k < 3; ++k)
            grad_f[b_off_ + 3*(i+1) + k] += 2.0 * res[r].value() * res[r].derivatives()(11+k);
    }
}

void BatchNLP::ang_grad_no_bias(const Ipopt::Number* x, int i,
                                 Ipopt::Number* grad_f) const
{
    const double* gyro_row = gyro_meas_.data() + i * GyroMeasurementIdx::GYRO_MEAS_COUNT;
    const double dt   = ts_[static_cast<size_t>(i+1)] - ts_[static_cast<size_t>(i)];
    const double qstd = gyro_wn_std_dev_rad_s_ * dt;

    // param layout [q0(0..3), q1(4..7)]
    AD8 q0[4], q1[4], res[3];
    for (int k = 0; k < 4; ++k) q0[k] = AD8(x[q_off_ + 4*i     + k], Eigen::Matrix<double,8,1>::Unit(k));
    for (int k = 0; k < 4; ++k) q1[k] = AD8(x[q_off_ + 4*(i+1) + k], Eigen::Matrix<double,8,1>::Unit(4+k));

    AngularDynamicsCostFunctorNoBias f(gyro_row, dt, qstd);
    f(q0, q1, res);

    for (int r = 0; r < 3; ++r) {
        for (int k = 0; k < 4; ++k)
            grad_f[q_off_ + 4*i     + k] += 2.0 * res[r].value() * res[r].derivatives()(k);
        for (int k = 0; k < 4; ++k)
            grad_f[q_off_ + 4*(i+1) + k] += 2.0 * res[r].value() * res[r].derivatives()(4+k);
    }
}

void BatchNLP::lmk_grad(const Ipopt::Number* x, int state_idx, int lm_idx,
                         Ipopt::Number* grad_f) const
{
    const double* lm_row = lm_meas_.data() + lm_idx * LandmarkMeasurementIdx::LANDMARK_COUNT;
    const double   sigma = lm_uncertainties_(lm_idx);

    // param layout [pos(0..2), quat(3..6)]
    AD7 pos[3], quat[4], res[3];
    for (int k = 0; k < 3; ++k) pos[k]  = AD7(x[r_off_ + 3*state_idx + k], Eigen::Matrix<double,7,1>::Unit(k));
    for (int k = 0; k < 4; ++k) quat[k] = AD7(x[q_off_ + 4*state_idx + k], Eigen::Matrix<double,7,1>::Unit(3+k));

    LandmarkCostFunctor f(lm_row, sigma);
    f(pos, quat, res);

    for (int r = 0; r < 3; ++r) {
        for (int k = 0; k < 3; ++k)
            grad_f[r_off_ + 3*state_idx + k] += 2.0 * res[r].value() * res[r].derivatives()(k);
        for (int k = 0; k < 4; ++k)
            grad_f[q_off_ + 4*state_idx + k] += 2.0 * res[r].value() * res[r].derivatives()(3+k);
    }
}

// ── eval_g ────────────────────────────────────────────────────────────────────

bool BatchNLP::eval_g(Ipopt::Index /*n*/, const Ipopt::Number* x, bool /*new_x*/,
                      Ipopt::Index /*m*/, Ipopt::Number* g)
{
    // Quaternion norm constraints: ||q_i||² - 1 = 0
    for (int i = 0; i < N_; ++i) {
        const double* q = x + q_off_ + 4*i;
        g[qnorm_off_ + i] = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] - 1.0;
    }

    // Position and velocity dynamics
    for (int i = 0; i < M_; ++i) {
        const double dt = ts_[static_cast<size_t>(i+1)] - ts_[static_cast<size_t>(i)];

        const double* r0 = x + r_off_ + 3*i;
        const double* v0 = x + v_off_ + 3*i;
        const double* r1 = x + r_off_ + 3*(i+1);
        const double* v1 = x + v_off_ + 3*(i+1);
        const double* a  = x + a_off_ + 3*i;

        // r_{i+1} - r_i - dt * v_i = 0
        double* gr = g + rdyn_off_ + 3*i;
        gr[0] = r1[0] - r0[0] - dt * v0[0];
        gr[1] = r1[1] - r0[1] - dt * v0[1];
        gr[2] = r1[2] - r0[2] - dt * v0[2];

        // v_{i+1} - v_i - dt * (a_grav(r_i) + a_i) = 0
        double ag[3];
        gravity(r0, ag);
        double* gv = g + vdyn_off_ + 3*i;
        gv[0] = v1[0] - v0[0] - dt * (ag[0] + a[0]);
        gv[1] = v1[1] - v0[1] - dt * (ag[1] + a[1]);
        gv[2] = v1[2] - v0[2] - dt * (ag[2] + a[2]);
    }

    return true;
}

// ── eval_jac_g ────────────────────────────────────────────────────────────────

bool BatchNLP::eval_jac_g(Ipopt::Index /*n*/, const Ipopt::Number* x, bool /*new_x*/,
                           Ipopt::Index /*m*/, Ipopt::Index /*nele_jac*/,
                           Ipopt::Index* iRow, Ipopt::Index* jCol,
                           Ipopt::Number* values)
{
    int nz = 0;  // non-zero counter

    if (values == nullptr) {
        // ── Sparsity pattern ─────────────────────────────────────────────────
        // 1. Quaternion norm: row i, cols q_off+4i..q_off+4i+3
        for (int i = 0; i < N_; ++i) {
            for (int k = 0; k < 4; ++k) {
                iRow[nz] = qnorm_off_ + i;
                jCol[nz] = q_off_ + 4*i + k;
                ++nz;
            }
        }
        // 2. Position dynamics: row rdyn_off+3j+k
        //    deps: r_i[k], v_i[k], r_{i+1}[k]  (3 entries per row)
        for (int j = 0; j < M_; ++j) {
            for (int k = 0; k < 3; ++k) {
                const int row = rdyn_off_ + 3*j + k;
                iRow[nz] = row; jCol[nz] = r_off_ + 3*j     + k; ++nz;  // -r_i[k]
                iRow[nz] = row; jCol[nz] = v_off_ + 3*j     + k; ++nz;  // -dt*v_i[k]
                iRow[nz] = row; jCol[nz] = r_off_ + 3*(j+1) + k; ++nz;  // +r_{i+1}[k]
            }
        }
        // 3. Velocity dynamics: row vdyn_off+3j+k
        //    deps: v_i[k], r_i[0..2] (gravity Jac row k), a_i[k], v_{i+1}[k]
        for (int j = 0; j < M_; ++j) {
            for (int k = 0; k < 3; ++k) {
                const int row = vdyn_off_ + 3*j + k;
                iRow[nz] = row; jCol[nz] = v_off_ + 3*j     + k; ++nz;  // -v_i[k]
                iRow[nz] = row; jCol[nz] = r_off_ + 3*j     + 0; ++nz;  // grav Jac col 0
                iRow[nz] = row; jCol[nz] = r_off_ + 3*j     + 1; ++nz;  // grav Jac col 1
                iRow[nz] = row; jCol[nz] = r_off_ + 3*j     + 2; ++nz;  // grav Jac col 2
                iRow[nz] = row; jCol[nz] = a_off_ + 3*j     + k; ++nz;  // -dt*a_i[k]
                iRow[nz] = row; jCol[nz] = v_off_ + 3*(j+1) + k; ++nz;  // +v_{i+1}[k]
            }
        }
    } else {
        // ── Values ───────────────────────────────────────────────────────────
        // 1. Quaternion norm: ∂(||q||²-1)/∂q_k = 2*q_k
        for (int i = 0; i < N_; ++i) {
            const double* q = x + q_off_ + 4*i;
            for (int k = 0; k < 4; ++k) {
                values[nz++] = 2.0 * q[k];
            }
        }
        // 2. Position dynamics Jacobian values
        for (int j = 0; j < M_; ++j) {
            const double dt = ts_[static_cast<size_t>(j+1)] - ts_[static_cast<size_t>(j)];
            for (int k = 0; k < 3; ++k) {
                values[nz++] = -1.0;   // ∂g_r/∂r_i[k]
                values[nz++] = -dt;    // ∂g_r/∂v_i[k]
                values[nz++] =  1.0;   // ∂g_r/∂r_{i+1}[k]
            }
        }
        // 3. Velocity dynamics Jacobian values
        for (int j = 0; j < M_; ++j) {
            const double dt = ts_[static_cast<size_t>(j+1)] - ts_[static_cast<size_t>(j)];
            const double* r0 = x + r_off_ + 3*j;
            double Jg[9];
            gravity_jac(r0, Jg);  // ∂a_grav/∂r, 3×3 row-major

            for (int k = 0; k < 3; ++k) {
                values[nz++] = -1.0;           // ∂g_v/∂v_i[k]
                values[nz++] = -dt * Jg[k*3+0]; // ∂g_v/∂r_i[0] (gravity Jac row k, col 0)
                values[nz++] = -dt * Jg[k*3+1]; // ∂g_v/∂r_i[1]
                values[nz++] = -dt * Jg[k*3+2]; // ∂g_v/∂r_i[2]
                values[nz++] = -dt;            // ∂g_v/∂a_i[k]
                values[nz++] =  1.0;           // ∂g_v/∂v_{i+1}[k]
            }
        }
    }

    return true;
}

// ── eval_h ────────────────────────────────────────────────────────────────────

bool BatchNLP::eval_h(Ipopt::Index /*n*/, const Ipopt::Number* /*x*/, bool /*new_x*/,
                      Ipopt::Number /*obj_factor*/, Ipopt::Index /*m*/,
                      const Ipopt::Number* /*lambda*/, bool /*new_lambda*/,
                      Ipopt::Index /*nele_hess*/, Ipopt::Index* /*iRow*/,
                      Ipopt::Index* /*jCol*/, Ipopt::Number* /*values*/)
{
    // Not called when hessian_approximation = "limited-memory"
    return false;
}

// ── finalize_solution ─────────────────────────────────────────────────────────

void BatchNLP::finalize_solution(Ipopt::SolverReturn status,
                                  Ipopt::Index /*n*/, const Ipopt::Number* x,
                                  const Ipopt::Number* /*z_L*/, const Ipopt::Number* /*z_U*/,
                                  Ipopt::Index /*m*/, const Ipopt::Number* /*g*/,
                                  const Ipopt::Number* /*lambda*/,
                                  Ipopt::Number obj_value,
                                  const Ipopt::IpoptData* ip_data,
                                  Ipopt::IpoptCalculatedQuantities* /*ip_cq*/)
{
    solver_status = static_cast<int>(status);
    final_cost    = obj_value;
    iter_count    = ip_data ? static_cast<int>(ip_data->iter_count()) : 0;

    // Copy solution back to state_estimates
    for (int i = 0; i < N_; ++i) {
        state_estimates_(i, StateEstimateIdx::POS_X) = x[r_off_ + 3*i + 0];
        state_estimates_(i, StateEstimateIdx::POS_Y) = x[r_off_ + 3*i + 1];
        state_estimates_(i, StateEstimateIdx::POS_Z) = x[r_off_ + 3*i + 2];

        state_estimates_(i, StateEstimateIdx::VEL_X) = x[v_off_ + 3*i + 0];
        state_estimates_(i, StateEstimateIdx::VEL_Y) = x[v_off_ + 3*i + 1];
        state_estimates_(i, StateEstimateIdx::VEL_Z) = x[v_off_ + 3*i + 2];

        state_estimates_(i, StateEstimateIdx::QUAT_X) = x[q_off_ + 4*i + 0];
        state_estimates_(i, StateEstimateIdx::QUAT_Y) = x[q_off_ + 4*i + 1];
        state_estimates_(i, StateEstimateIdx::QUAT_Z) = x[q_off_ + 4*i + 2];
        state_estimates_(i, StateEstimateIdx::QUAT_W) = x[q_off_ + 4*i + 3];

        if (bias_mode_ == BIAS_MODE::FIX_BIAS) {
            state_estimates_(i, StateEstimateIdx::GYRO_BIAS_X) = x[b_off_ + 0];
            state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Y) = x[b_off_ + 1];
            state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Z) = x[b_off_ + 2];
        } else if (bias_mode_ == BIAS_MODE::TV_BIAS) {
            state_estimates_(i, StateEstimateIdx::GYRO_BIAS_X) = x[b_off_ + 3*i + 0];
            state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Y) = x[b_off_ + 3*i + 1];
            state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Z) = x[b_off_ + 3*i + 2];
        }
        // NO_BIAS: bias columns remain from initializer
    }
}
