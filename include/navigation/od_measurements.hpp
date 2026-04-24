#ifndef OD_MEASUREMENTS_HPP
#define OD_MEASUREMENTS_HPP

#include "core/errors.hpp"
#include <eigen3/Eigen/Dense>

// ── Landmark column index ──────────────────────────────────────────────────────
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

// ── Measurement bundle for solve_ceres_batch_opt ──────────────────────────────
// Uses raw Eigen dynamic types to avoid a circular include with batch_optimization.hpp
// (which includes od.hpp for BATCH_OPT_config). They are assignment-compatible with the
// RowMajor typedefs in batch_optimization.hpp as long as column counts match at runtime.
struct ODMeasurements
{
    Eigen::MatrixXd                          landmark_measurements;  // Nx7
    Eigen::Matrix<bool, Eigen::Dynamic, 1>   group_starts;           // Nx1
    Eigen::MatrixXd                          gyro_measurements;      // Mx4
    Eigen::VectorXd                          landmark_uncertainties; // Nx1, one sigma per row

    // Validates all fields for use by solve_ceres_batch_opt. Logs each specific
    // problem via spdlog::error and calls LogError on failure.
    // Returns ErrorCode::OK on success, ErrorCode::ODMEAS_NOT_VALID on failure.
    ErrorCode Validate() const;
};

#endif // OD_MEASUREMENTS_HPP
