#ifndef OD_MEASUREMENTS_HPP
#define OD_MEASUREMENTS_HPP

#include <eigen3/Eigen/Dense>

// Measurement matrices produced by OD::DatasetPrepare, ready for solve_ceres_batch_opt.
// Uses raw Eigen dynamic types to avoid a circular include with batch_optimization.hpp
// (which includes od.hpp for BATCH_OPT_config). They are assignment-compatible with the
// RowMajor typedefs in batch_optimization.hpp as long as column counts match at runtime.
struct ODMeasurements
{
    Eigen::MatrixXd                          landmark_measurements;  // Nx7 row-major
    Eigen::Matrix<bool, Eigen::Dynamic, 1>   group_starts;           // Nx1
    Eigen::MatrixXd                          gyro_measurements;      // Mx4 row-major
    Eigen::VectorXd                          landmark_uncertainties; // Nx1, one sigma per row
};

#endif // OD_MEASUREMENTS_HPP
