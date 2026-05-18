#ifndef OD_MEASUREMENTS_HPP
#define OD_MEASUREMENTS_HPP

#include "core/errors.hpp"
#include <eigen3/Eigen/Dense>

inline constexpr int OD_MIN_LANDMARK_MEASUREMENTS = 4;

namespace casadi {
template<typename Scalar>
class Matrix;
using DM = Matrix<double>;
class MX;
}

// ── Landmark column index and cost functor ────────────────────────────────────
enum LandmarkMeasurementIdx {
    LANDMARK_TIMESTAMP = 0,
    BEARING_VEC_X      = 1,
    BEARING_VEC_Y      = 2,
    BEARING_VEC_Z      = 3,
    LANDMARK_POS_X     = 4,
    LANDMARK_POS_Y     = 5,
    LANDMARK_POS_Z     = 6,
    LANDMARK_COUNT     = 7
};

casadi::MX landmark_residual_casadi(
    const casadi::MX& r,
    const casadi::MX& q,
    const casadi::DM& lmk_pos,
    const casadi::DM& bearing_meas,
    double sigma);

// Pseudo-Huber scalar cost for a 3-vector residual (already σ-normalised).
// φ(d) = 2M²(√(1 + d²/M²) − 1)  where d = ‖res‖
// Quadratic near zero, linear for d ≫ M.  Matches L2 gradient at d=0.
casadi::MX pseudo_huber_cost(const casadi::MX& res, double M);

// ── Measurement bundle for solve_batch_opt ────────────────────────────────────
struct ODMeasurements
{
    Eigen::MatrixXd                          landmark_measurements;  // Nx7
    Eigen::Matrix<bool, Eigen::Dynamic, 1>   group_starts;           // Nx1
    Eigen::MatrixXd                          gyro_measurements;      // Mx4
    Eigen::VectorXd                          landmark_uncertainties; // Nx1, one sigma per row

    // Returns ErrorCode::OK on success, ErrorCode::ODMEAS_NOT_VALID on failure.
    ErrorCode Validate() const;
};

#endif // OD_MEASUREMENTS_HPP
