#ifndef OD_MEASUREMENTS_HPP
#define OD_MEASUREMENTS_HPP

#include "core/errors.hpp"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>

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

// ── Ceres cost functor for landmark bearing residuals ─────────────────────────
struct LandmarkCostFunctor {
public:
    LandmarkCostFunctor(const double* const landmark_row, const double landmark_std_dev)
            : bearing_vec(landmark_row + LandmarkMeasurementIdx::BEARING_VEC_X),
              landmark_pos(landmark_row + LandmarkMeasurementIdx::LANDMARK_POS_X),
              landmark_std_dev(landmark_std_dev) {}

    template<typename T>
    bool operator()(const T* const pos,
                    const T* const quat,
                    T* const residuals) const {
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r(pos);
        const Eigen::Map<const Eigen::Quaternion <T>> q(quat);

        const Eigen::Matrix<T, 3, 1> landmark_pos_T = landmark_pos.template cast<T>();
        const Eigen::Matrix<T, 3, 1> bearing_vec_T  = bearing_vec.template cast<T>();

        Eigen::Map<Eigen::Matrix<T, 3, 1>> r_res(residuals);

        const Eigen::Matrix<T, 3, 1> diff = (landmark_pos_T - r);
        const T norm_sq  = diff.squaredNorm();
        const T eps      = T(1e-6);
        const T inv_norm = T(1.0) / ceres::sqrt(norm_sq + eps);
        const Eigen::Matrix<T, 3, 1> predicted_bearing = diff * inv_norm;

        r_res = (q.inverse() * predicted_bearing - bearing_vec_T) / T(landmark_std_dev);

        return true;
    }

private:
    const Eigen::Map<const Eigen::Vector3d> bearing_vec;
    const Eigen::Map<const Eigen::Vector3d> landmark_pos;
    const double landmark_std_dev;
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
