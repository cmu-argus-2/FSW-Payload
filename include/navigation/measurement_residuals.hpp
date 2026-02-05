#ifndef MEASUREMENT_RESIDUALS_HPP
#define MEASUREMENT_RESIDUALS_HPP
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>


enum LandmarkMeasurementIdx {
    LANDMARK_TIMESTAMP = 0,
    BEARING_VEC_X = 1,
    BEARING_VEC_Y = 2,
    BEARING_VEC_Z = 3,
    LANDMARK_POS_X = 4,
    LANDMARK_POS_Y = 5,
    LANDMARK_POS_Z = 6,
    LANDMARK_COUNT = 7
};

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
    // const Eigen::Map<const Eigen::Quaternion <T>> q(quat);

    // const T x0 = quat[0],
    //         y0 = quat[1],
    //         z0 = quat[2],
    //         w0 = quat[3];
    // const Eigen::Quaternion<T> q(w0, x0, y0, z0);
    const Eigen::Map<const Eigen::Quaternion <T>> q(quat);

    const Eigen::Matrix<T, 3, 1> landmark_pos_T   = landmark_pos.template cast<T>();
    const Eigen::Matrix<T, 3, 1> bearing_vec_T    = bearing_vec.template cast<T>();

    Eigen::Map <Eigen::Matrix<T, 3, 1>> r_res(residuals);

    const Eigen::Matrix<T, 3, 1> diff = (landmark_pos_T - r);
    const T norm_sq = diff.squaredNorm();
    const T eps = T(1e-6);
    const T inv_norm = T(1.0) / ceres::sqrt(norm_sq + eps);
    const Eigen::Matrix<T,3,1> predicted_bearing = diff * inv_norm;

    // r_res = (predicted_bearing - q * bearing_vec_T) / T(landmark_std_dev);

    const Eigen::Matrix<T, 3, 3> bearbear_mat    = bearing_vec_T * bearing_vec_T.transpose();
    const Eigen::Matrix<T, 3, 3> I                 = Eigen::Matrix<T, 3, 3>::Identity();
    
    r_res = ((I-bearbear_mat) / T(landmark_std_dev) + bearbear_mat / T(0.3*landmark_std_dev)) * (q.inverse() * predicted_bearing - bearing_vec_T);
    return true;
};

private:
    const Eigen::Map<const Eigen::Vector3d> bearing_vec;
    const Eigen::Map<const Eigen::Vector3d> landmark_pos;
    const double landmark_std_dev;
};

#endif // MEASUREMENT_RESIDUALS_HPP