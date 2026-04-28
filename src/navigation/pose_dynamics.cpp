#include "navigation/pose_dynamics.hpp"

#include <vector>

using casadi::MX;
using casadi::Slice;

namespace {
constexpr double GM_EARTH = 3.9860044188e5;  // km^3/s^2
constexpr double R_EARTH  = 6378.137;        // km
constexpr double J2       = 1.08262668e-3;
constexpr double OMEGA_E  = 7.2921150e-5;    // rad/s

// Piecewise exponential atmosphere (Vallado Table 8-4): 300–700 km
constexpr double DRAG_H_BOUNDS[]     = {300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0};
constexpr double DRAG_RHO_BASE[]     = {2.418e-2, 9.518e-3, 3.725e-3, 1.585e-3, 6.967e-4, 1.454e-4};
constexpr double DRAG_SCALE_HEIGHT[] = {53.628, 53.298, 58.515, 60.828, 63.822, 71.835};
}

MX keplerian_accel(const MX& r) {
    MX r_sq   = dot(r, r) + 1e-10;
    MX r_norm = sqrt(r_sq);
    MX r_cb   = r_sq * r_norm;
    return (-GM_EARTH / r_cb) * r;
}

MX j2_accel(const MX& r) {
    MX r_sq   = dot(r, r) + 1e-10;
    MX r_norm = sqrt(r_sq);
    MX r5     = r_sq * r_sq * r_norm;
    MX z_sq_over_r_sq = r(2, 0) * r(2, 0) / r_sq;
    MX factor = 1.5 * J2 * GM_EARTH * R_EARTH * R_EARTH / r5;

    return MX::vertcat(std::vector<MX>{
        factor * r(0, 0) * (5.0 * z_sq_over_r_sq - 1.0),
        factor * r(1, 0) * (5.0 * z_sq_over_r_sq - 1.0),
        factor * r(2, 0) * (5.0 * z_sq_over_r_sq - 3.0)
    });
}

static MX atmospheric_density(const MX& r) {
    MX r_norm = sqrt(dot(r, r) + 1e-10);
    MX h_ellp = r_norm - R_EARTH;

    // Initial density: layer 5, anchored at H=600 km (covers 600+ km altitudes)
    MX density = MX(DRAG_RHO_BASE[5]) * exp(-(h_ellp - DRAG_H_BOUNDS[5]) / DRAG_SCALE_HEIGHT[5]);
    for (int i = 4; i >= 0; --i) {
        MX layer = MX(DRAG_RHO_BASE[i]) * exp(-(h_ellp - DRAG_H_BOUNDS[i]) / DRAG_SCALE_HEIGHT[i]);
        density = MX::if_else(h_ellp < DRAG_H_BOUNDS[i + 1], layer, density);
    }
    return density;
}

MX drag_accel(const MX& r, const MX& v, const MX& cd) {
    MX v_atm = MX::vertcat(std::vector<MX>{
        -OMEGA_E * r(1, 0), OMEGA_E * r(0, 0), MX(0.0)
    });
    MX v_rel      = v - v_atm;
    MX rho        = atmospheric_density(r);
    MX A          = 1.0e-8; // km^2
    MX m          = 1.3;    // kg
    MX v_rel_norm = sqrt(dot(v_rel, v_rel) + 1e-20);
    return -0.5 * cd * (A / m) * rho * v_rel_norm * v_rel;
}

MX linear_dynamics(const MX& r, const MX& v, const MX& uma, bool use_j2, bool use_drag, const MX& cd) {
    MX accel = keplerian_accel(r) + uma;
    if (use_j2) {
        accel = accel + j2_accel(r);
    }
    if (use_drag) {
        accel = accel + drag_accel(r, v, cd);
    }
    return MX::vertcat(std::vector<MX>{v, accel});
}

MX linear_dynamics_constraint(
    const MX& r0, const MX& v0,
    const MX& r1, const MX& v1,
    const MX& uma, double dt, Integrator integrator,
    bool use_j2, bool use_drag,
    const MX& cd)
{

    if (integrator == Integrator::RK4) {
        // ADL: RK4 integration
        MX k1 = linear_dynamics(r0, v0, uma, use_j2, use_drag, cd);
        MX k2 = linear_dynamics(r0 + 0.5 * dt * k1(Slice(0, 3), Slice()), v0 + 0.5 * dt * k1(Slice(3, 6), Slice()), uma, use_j2, use_drag, cd);
        MX k3 = linear_dynamics(r0 + 0.5 * dt * k2(Slice(0, 3), Slice()), v0 + 0.5 * dt * k2(Slice(3, 6), Slice()), uma, use_j2, use_drag, cd);
        MX k4 = linear_dynamics(r0 + dt * k3(Slice(0, 3), Slice()), v0 + dt * k3(Slice(3, 6), Slice()), uma, use_j2, use_drag, cd);
        
        MX x1_pred = MX::vertcat(std::vector<MX>{
            r0 + (dt / 6.0) * (k1(Slice(0, 3), Slice()) + 2*k2(Slice(0, 3), Slice()) + 2*k3(Slice(0, 3), Slice()) + k4(Slice(0, 3), Slice())),
            v0 + (dt / 6.0) * (k1(Slice(3, 6), Slice()) + 2*k2(Slice(3, 6), Slice()) + 2*k3(Slice(3, 6), Slice()) + k4(Slice(3, 6), Slice()))
        });
        MX x1 = MX::vertcat(std::vector<MX>{r1, v1});
        return x1 - x1_pred;
    } else {
        // ADL: Euler integration
        MX k1 = linear_dynamics(r0, v0, uma, use_j2, use_drag, cd);
        
        MX x1_pred = MX::vertcat(std::vector<MX>{
            r0 + dt * k1(Slice(0, 3), Slice()),
            v0 + dt * k1(Slice(3, 6), Slice())
        });
        MX x1 = MX::vertcat(std::vector<MX>{r1, v1});
        return x1 - x1_pred;
    }
}
