#include "navigation/pose_dynamics.hpp"
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <core/timing.hpp>

/*
bool LinearDynamicsAnalytic::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    Eigen::Vector3d r0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d v0(parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Vector3d r1(parameters[0][6], parameters[0][7], parameters[0][8]);
    Eigen::Vector3d v1(parameters[0][9], parameters[0][10], parameters[0][11]);

    double r_norm = ceres::sqrt(r0.squaredNorm());
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
    if (r_norm != 0.0) {
        double denom = r_norm * r_norm * r_norm;
        accel = - GM_EARTH * r0 / denom;
    }
    Eigen::Vector3d r_res = r1 - (r0 + v0 * dt);
    Eigen::Vector3d v_res = v1 - (v0 + accel * dt);

    // pack residuals efficiently using Eigen::Map (avoids element-wise assignment)
    Eigen::Map<Eigen::Matrix<double,6,1>>(residuals).segment<3>(0) = r_res;
    Eigen::Map<Eigen::Matrix<double,6,1>>(residuals).segment<3>(3) = v_res;

    if (jacobians != nullptr && jacobians[0] != nullptr) {
        Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(6, 12);

        jac.block<3,3>(0, 0) = -Eigen::Matrix3d::Identity(); // dr_res/dr0
        jac.block<3,3>(0, 3) = -Eigen::Matrix3d::Identity() * dt; // dr_res/dv0
        jac.block<3,3>(0, 6) = Eigen::Matrix3d::Identity(); // dr_res/dr1
        if (r_norm != 0.0) {
            double denom = r_norm * r_norm * r_norm;
            jac.block<3,3>(3, 0) = GM_EARTH * dt * (Eigen::Matrix3d::Identity() - 3 * (r0 * r0.transpose()) / (r_norm * r_norm)) / denom; // dv_res/dr0
        }
        jac.block<3,3>(3, 3) = -Eigen::Matrix3d::Identity(); // dv_res/dv0
        jac.block<3,3>(3, 9) = Eigen::Matrix3d::Identity(); // dv_res/dv1

        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 12; ++j) {
                jacobians[0][i * 12 + j] = jac(i, j);
            }
        }

    }
    

    return true;
}
*/