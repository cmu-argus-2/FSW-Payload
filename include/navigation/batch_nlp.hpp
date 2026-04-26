#ifndef BATCH_NLP_HPP
#define BATCH_NLP_HPP

#include "navigation/batch_optimization.hpp"
#include "navigation/od_measurements.hpp"
#include "navigation/pose_dynamics.hpp"
#include <IpTNLP.hpp>
#include <Eigen/Core>
#include <vector>

// BatchNLP implements the IPOPT TNLP interface for the batch OD problem.
//
// Decision variables (flat x vector, all in their respective units):
//   [0,       3N)        positions     r_i  (km)
//   [3N,      6N)        velocities    v_i  (km/s)
//   [6N,      10N)       quaternions   q_i  (x,y,z,w, Eigen order)
//   [10N,     10N+3M)    UMA           a_i  (km/s²) per interval  M = N-1
//   [10N+3M,  10N+3M+B)  gyro bias     b    (rad/s)
//                          B = 0  (NO_BIAS), 3 (FIX_BIAS), 3N (TV_BIAS)
//
// Equality constraints g(x) = 0:
//   [0,   N)          quaternion norms   ||q_i||² - 1 = 0
//   [N,   N+3M)       position dynamics  r_{i+1} - r_i - dt_i*v_i = 0
//   [N+3M, N+6M)      velocity dynamics  v_{i+1} - v_i - dt_i*(a_grav(r_i) + a_i) = 0
//
// Objective (soft):
//   Σ_i ||a_i / σ_uma||²           UMA prior
//   Σ_i ||ang_res_i / σ_q||²       angular dynamics (gyro integration error)
//   Σ_j ||lmk_res_j / σ_lmk||²    landmark bearing
//   (TV_BIAS) Σ_i ||(b_{i+1}-b_i)/σ_bias||²
//
class BatchNLP : public Ipopt::TNLP {
public:
    BatchNLP(StateEstimates&                    state_estimates,
             const std::vector<double>&         state_timestamps,
             const std::vector<idx_t>&          landmark_group_indices,
             const LandmarkMeasurements&        landmark_measurements,
             const LandmarkGroupStarts&         landmark_group_starts,
             const GyroMeasurements&            gyro_measurements,
             BIAS_MODE                          bias_mode,
             double                             uma_std_dev,
             double                             gyro_wn_std_dev_rad_s,
             double                             gyro_bias_instability,
             const Eigen::VectorXd&             landmark_uncertainties);

    // ── TNLP interface ────────────────────────────────────────────────────────
    bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                      Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                      Ipopt::TNLP::IndexStyleEnum& index_style) override;

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                         Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u) override;

    bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                            bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                            Ipopt::Index m, bool init_lambda,
                            Ipopt::Number* lambda) override;

    bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                Ipopt::Number& obj_value) override;

    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                     Ipopt::Number* grad_f) override;

    bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                Ipopt::Index m, Ipopt::Number* g) override;

    bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                    Ipopt::Index m, Ipopt::Index nele_jac,
                    Ipopt::Index* iRow, Ipopt::Index* jCol,
                    Ipopt::Number* values) override;

    bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                Ipopt::Number obj_factor, Ipopt::Index m,
                const Ipopt::Number* lambda, bool new_lambda,
                Ipopt::Index nele_hess, Ipopt::Index* iRow, Ipopt::Index* jCol,
                Ipopt::Number* values) override;

    void finalize_solution(Ipopt::SolverReturn status,
                           Ipopt::Index n, const Ipopt::Number* x,
                           const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                           Ipopt::Index m, const Ipopt::Number* g,
                           const Ipopt::Number* lambda,
                           Ipopt::Number obj_value,
                           const Ipopt::IpoptData* ip_data,
                           Ipopt::IpoptCalculatedQuantities* ip_cq) override;

    // ── Results (valid after finalize_solution) ───────────────────────────────
    int    solver_status  = -1;
    double initial_cost   = 0.0;
    double final_cost     = 0.0;
    int    iter_count     = 0;

private:
    // ── Problem data ──────────────────────────────────────────────────────────
    StateEstimates&             state_estimates_;
    const std::vector<double>&  ts_;
    const std::vector<idx_t>&   lm_group_indices_;
    const LandmarkMeasurements& lm_meas_;
    const LandmarkGroupStarts&  lm_group_starts_;
    const GyroMeasurements&     gyro_meas_;
    BIAS_MODE                   bias_mode_;
    double                      uma_std_dev_;
    double                      gyro_wn_std_dev_rad_s_;
    double                      gyro_bias_instability_;
    const Eigen::VectorXd&      lm_uncertainties_;

    // ── Dimensions ────────────────────────────────────────────────────────────
    int N_;       // number of states
    int M_;       // N - 1 (number of intervals)
    int b_size_;  // number of bias scalar variables
    int n_vars_;
    int n_con_;
    int nnz_jac_; // precomputed non-zero count for constraint Jacobian

    // ── Variable offsets ──────────────────────────────────────────────────────
    int r_off_;   // position block start
    int v_off_;   // velocity block start
    int q_off_;   // quaternion block start
    int a_off_;   // UMA block start
    int b_off_;   // bias block start (-1 for NO_BIAS)

    // ── Constraint offsets ────────────────────────────────────────────────────
    int qnorm_off_;  // quaternion norm constraints
    int rdyn_off_;   // position dynamics constraints
    int vdyn_off_;   // velocity dynamics constraints

    // ── Internal helpers ──────────────────────────────────────────────────────
    void gravity(const double* r3, double* a3) const;
    void gravity_jac(const double* r3, double Jg[9]) const;  // 3×3 row-major ∂a_grav/∂r

    // Accumulate angular dynamics gradient for interval i into grad_f.
    void ang_grad_fix_bias(const Ipopt::Number* x, int i, Ipopt::Number* grad_f) const;
    void ang_grad_tv_bias (const Ipopt::Number* x, int i, Ipopt::Number* grad_f) const;
    void ang_grad_no_bias (const Ipopt::Number* x, int i, Ipopt::Number* grad_f) const;

    // Accumulate landmark gradient for landmark measurement lm_idx at state_idx.
    void lmk_grad(const Ipopt::Number* x, int state_idx, int lm_idx,
                  Ipopt::Number* grad_f) const;
};

#endif // BATCH_NLP_HPP
