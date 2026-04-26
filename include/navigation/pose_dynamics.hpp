#ifndef POSE_DYNAMICS_HPP
#define POSE_DYNAMICS_HPP

#include <casadi/casadi.hpp>
#include "navigation/od.hpp"

// Gyro measurements are treated as inputs to the attitude dynamics
enum GyroMeasurementIdx {
    GYRO_MEAS_TIMESTAMP = 0,
    ANG_VEL_X = 1,
    ANG_VEL_Y = 2,
    ANG_VEL_Z = 3,
    GYRO_MEAS_COUNT = 4
};

casadi::MX keplerian_accel(const casadi::MX& r);
casadi::MX j2_accel(const casadi::MX& r);
casadi::MX drag_accel(const casadi::MX& r, const casadi::MX& v, const casadi::MX& bc_inv);
casadi::MX linear_dynamics(const casadi::MX& r, const casadi::MX& v, const casadi::MX& uma,
                            bool use_j2, bool use_drag, const casadi::MX& bc_inv);
casadi::MX linear_dynamics_constraint(
    const casadi::MX& r0, const casadi::MX& v0,
    const casadi::MX& r1, const casadi::MX& v1,
    const casadi::MX& uma, double dt, Integrator integrator,
    bool use_j2, bool use_drag = false,
    const casadi::MX& bc_inv = casadi::MX(0.0));

#endif // POSE_DYNAMICS_HPP
