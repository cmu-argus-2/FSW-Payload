#include "navigation/trajectory_initializer.hpp"
#include "navigation/pose_dynamics.hpp"
#include "spdlog/spdlog.h"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>


// ── Internal helpers ──────────────────────────────────────────────────────────

namespace {

struct PoseKeyframe {
    double             t;
    Eigen::Vector3d    pos;
    Eigen::Quaterniond quat;
};

// Linear interpolation/extrapolation of position between keyframes.
Eigen::Vector3d interp_pos(const std::vector<PoseKeyframe>& kf, double t)
{
    const size_t n = kf.size();
    if (n == 1) return kf[0].pos;

    if (t <= kf.front().t) {
        const double dt = kf[1].t - kf[0].t;
        if (std::abs(dt) < 1e-9) return kf[0].pos;
        return kf[0].pos + (kf[1].pos - kf[0].pos) / dt * (t - kf[0].t);
    }
    if (t >= kf.back().t) {
        const double dt = kf[n-1].t - kf[n-2].t;
        if (std::abs(dt) < 1e-9) return kf.back().pos;
        return kf.back().pos + (kf[n-1].pos - kf[n-2].pos) / dt * (t - kf.back().t);
    }
    size_t lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const size_t mid = (lo + hi) / 2;
        if (kf[mid].t <= t) lo = mid; else hi = mid;
    }
    const double alpha = (t - kf[lo].t) / (kf[hi].t - kf[lo].t);
    return (1.0 - alpha) * kf[lo].pos + alpha * kf[hi].pos;
}

// SLERP between keyframe quaternions; clamps outside the keyframe range.
Eigen::Quaterniond slerp_att(const std::vector<PoseKeyframe>& kf, double t)
{
    const size_t n = kf.size();
    if (n == 1) return kf[0].quat;
    if (t <= kf.front().t) return kf.front().quat;
    if (t >= kf.back().t)  return kf.back().quat;

    size_t lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const size_t mid = (lo + hi) / 2;
        if (kf[mid].t <= t) lo = mid; else hi = mid;
    }
    const double alpha = (t - kf[lo].t) / (kf[hi].t - kf[lo].t);
    return kf[lo].quat.slerp(alpha, kf[hi].quat);
}

// Wahba's problem via SVD: find R minimising sum ||d_eci_j - R * b_body_j||^2.
// B = sum_j  d_eci_j * b_body_j^T.  Returns identity on degenerate input.
Eigen::Quaterniond wahba_svd(const Eigen::Matrix3d& B)
{
    if (B.norm() < 1e-9) return Eigen::Quaterniond::Identity();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d diag   = Eigen::Matrix3d::Identity();
    diag(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() > 0.0 ? 1.0 : -1.0;
    return Eigen::Quaterniond(svd.matrixU() * diag * svd.matrixV().transpose()).normalized();
}

} // namespace


// ── TrajectoryInitializer ─────────────────────────────────────────────────────

TrajectoryInitializer::TrajectoryInitializer(
        const std::vector<double>&  state_timestamps,
        const LandmarkMeasurements& landmark_measurements,
        const LandmarkGroupStarts&  landmark_group_starts,
        BIAS_MODE                   bias_mode,
        const GyroMeasurements&     gyro_measurements)
{
    constexpr double R_ORBIT_KM = 6371.0 + 600.0;

    // ── Build pose keyframes ──────────────────────────────────────────────────
    // One keyframe per landmark group.
    // Position: mean landmark ECI direction scaled to orbital altitude.
    // Attitude: Wahba SVD on (body bearing → satellite-to-landmark ECI) pairs.
    struct LmEntry { Eigen::Vector3d pos, bearing; };
    std::vector<PoseKeyframe> keyframes;

    const idx_t N = landmark_measurements.rows();
    idx_t row = 0;
    while (row < N) {
        const double t_group =
            landmark_measurements(row, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP);

        std::vector<LmEntry> group_lms;
        Eigen::Vector3d pos_sum = Eigen::Vector3d::Zero();
        do {
            const Eigen::Vector3d lm_pos(
                landmark_measurements(row, LandmarkMeasurementIdx::LANDMARK_POS_X),
                landmark_measurements(row, LandmarkMeasurementIdx::LANDMARK_POS_Y),
                landmark_measurements(row, LandmarkMeasurementIdx::LANDMARK_POS_Z));
            const Eigen::Vector3d bearing(
                landmark_measurements(row, LandmarkMeasurementIdx::BEARING_VEC_X),
                landmark_measurements(row, LandmarkMeasurementIdx::BEARING_VEC_Y),
                landmark_measurements(row, LandmarkMeasurementIdx::BEARING_VEC_Z));
            pos_sum += lm_pos;
            group_lms.push_back({lm_pos, bearing});
            ++row;
        } while (row < N && !landmark_group_starts(row, 0));

        const double pos_norm = pos_sum.norm();
        if (pos_norm < 1e-6) continue;
        const Eigen::Vector3d r_group = pos_sum / pos_norm * R_ORBIT_KM;

        Eigen::Matrix3d B = Eigen::Matrix3d::Zero();
        for (const auto& lm : group_lms) {
            const Eigen::Vector3d diff = lm.pos - r_group;
            const double diff_norm = diff.norm();
            if (diff_norm < 1e-6) continue;
            B += (diff / diff_norm) * lm.bearing.transpose();
        }

        keyframes.push_back({t_group, r_group, wahba_svd(B)});
    }
    spdlog::info("TrajectoryInitializer: {} pose keyframes from {} landmark rows",
                 keyframes.size(), N);

    // ── Fill state estimates ──────────────────────────────────────────────────
    const idx_t M = static_cast<idx_t>(state_timestamps.size());
    state_estimates_.resize(M, StateEstimateIdx::STATE_ESTIMATE_COUNT);

    for (idx_t i = 0; i < M; ++i) {
        const double t = state_timestamps[static_cast<size_t>(i)];

        const Eigen::Vector3d    pos = keyframes.empty()
            ? Eigen::Vector3d(0.0, 0.0, R_ORBIT_KM)
            : interp_pos(keyframes, t);
        const Eigen::Quaterniond q   = keyframes.empty()
            ? Eigen::Quaterniond::Identity()
            : slerp_att(keyframes, t);

        state_estimates_(i, StateEstimateIdx::STATE_ESTIMATE_TIMESTAMP) = t;
        state_estimates_(i, StateEstimateIdx::POS_X)       = pos.x();
        state_estimates_(i, StateEstimateIdx::POS_Y)       = pos.y();
        state_estimates_(i, StateEstimateIdx::POS_Z)       = pos.z();
        state_estimates_(i, StateEstimateIdx::QUAT_X)      = q.x();
        state_estimates_(i, StateEstimateIdx::QUAT_Y)      = q.y();
        state_estimates_(i, StateEstimateIdx::QUAT_Z)      = q.z();
        state_estimates_(i, StateEstimateIdx::QUAT_W)      = q.w();
        state_estimates_(i, StateEstimateIdx::GYRO_BIAS_X) = 0.0;
        state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Y) = 0.0;
        state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Z) = 0.0;
    }

    // Velocity: central differences at interior, one-sided at endpoints.
    for (idx_t i = 0; i < M; ++i) {
        const idx_t i0 = (i == 0)     ? 0     : i - 1;
        const idx_t i1 = (i == M - 1) ? M - 1 : i + 1;
        if (i0 == i1) {
            state_estimates_(i, StateEstimateIdx::VEL_X) = 0.0;
            state_estimates_(i, StateEstimateIdx::VEL_Y) = 0.0;
            state_estimates_(i, StateEstimateIdx::VEL_Z) = 0.0;
            continue;
        }
        const double dt = state_timestamps[static_cast<size_t>(i1)]
                        - state_timestamps[static_cast<size_t>(i0)];
        state_estimates_(i, StateEstimateIdx::VEL_X) =
            (state_estimates_(i1, StateEstimateIdx::POS_X) - state_estimates_(i0, StateEstimateIdx::POS_X)) / dt;
        state_estimates_(i, StateEstimateIdx::VEL_Y) =
            (state_estimates_(i1, StateEstimateIdx::POS_Y) - state_estimates_(i0, StateEstimateIdx::POS_Y)) / dt;
        state_estimates_(i, StateEstimateIdx::VEL_Z) =
            (state_estimates_(i1, StateEstimateIdx::POS_Z) - state_estimates_(i0, StateEstimateIdx::POS_Z)) / dt;
    }

    // ── Gyro bias initialization ──────────────────────────────────────────────
    // Differentiate attitude (forward diff q_i^{-1} ⊗ q_{i+1}) to get estimated
    // body-frame angular rate, then bias = ω_gyro − ω_att.
    if (bias_mode != BIAS_MODE::NO_BIAS) {
        const idx_t Mg = gyro_measurements.rows();
        const idx_t Nb = std::min(M, Mg);

        auto quat_at = [&](idx_t i) {
            return Eigen::Quaterniond(
                state_estimates_(i, StateEstimateIdx::QUAT_W),
                state_estimates_(i, StateEstimateIdx::QUAT_X),
                state_estimates_(i, StateEstimateIdx::QUAT_Y),
                state_estimates_(i, StateEstimateIdx::QUAT_Z));
        };

        auto omega_att_at = [&](idx_t i) -> Eigen::Vector3d {
            const idx_t i0 = (i < M - 1) ? i     : i - 1;
            const idx_t i1 = (i < M - 1) ? i + 1 : i;
            const double dt = state_timestamps[static_cast<size_t>(i1)]
                            - state_timestamps[static_cast<size_t>(i0)];
            if (std::abs(dt) < 1e-9) return Eigen::Vector3d::Zero();
            return 2.0 * (quat_at(i0).conjugate() * quat_at(i1)).vec() / dt;
        };

        if (bias_mode == BIAS_MODE::FIX_BIAS) {
            Eigen::Vector3d bias_sum = Eigen::Vector3d::Zero();
            int count = 0;
            for (idx_t i = 0; i < Nb; ++i) {
                const Eigen::Vector3d omega_meas(
                    gyro_measurements(i, GyroMeasurementIdx::ANG_VEL_X),
                    gyro_measurements(i, GyroMeasurementIdx::ANG_VEL_Y),
                    gyro_measurements(i, GyroMeasurementIdx::ANG_VEL_Z));
                bias_sum += omega_meas - omega_att_at(i);
                ++count;
            }
            if (count > 0) {
                const Eigen::Vector3d b = bias_sum / static_cast<double>(count);
                state_estimates_(0, StateEstimateIdx::GYRO_BIAS_X) = b.x();
                state_estimates_(0, StateEstimateIdx::GYRO_BIAS_Y) = b.y();
                state_estimates_(0, StateEstimateIdx::GYRO_BIAS_Z) = b.z();
                spdlog::info("TrajectoryInitializer: fixed bias [{:.4e}, {:.4e}, {:.4e}] rad/s",
                             b.x(), b.y(), b.z());
            }
        } else { // TV_BIAS
            for (idx_t i = 0; i < Nb; ++i) {
                const Eigen::Vector3d omega_meas(
                    gyro_measurements(i, GyroMeasurementIdx::ANG_VEL_X),
                    gyro_measurements(i, GyroMeasurementIdx::ANG_VEL_Y),
                    gyro_measurements(i, GyroMeasurementIdx::ANG_VEL_Z));
                const Eigen::Vector3d b = omega_meas - omega_att_at(i);
                state_estimates_(i, StateEstimateIdx::GYRO_BIAS_X) = b.x();
                state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Y) = b.y();
                state_estimates_(i, StateEstimateIdx::GYRO_BIAS_Z) = b.z();
            }
        }
    }
}

void TrajectoryInitializer::write_csv(const std::string& results_dir) const
{
    const std::string path = results_dir + "/initial_trajectory.csv";
    std::ofstream f(path);
    if (!f.is_open()) {
        spdlog::warn("TrajectoryInitializer::write_csv: cannot open {}", path);
        return;
    }
    f << "timestamp_j2000,pos_x_km,pos_y_km,pos_z_km,"
         "vel_x_kms,vel_y_kms,vel_z_kms,"
         "quat_x,quat_y,quat_z,quat_w,"
         "gyro_bias_x_rads,gyro_bias_y_rads,gyro_bias_z_rads\n";
    f << std::setprecision(12);
    for (idx_t i = 0; i < state_estimates_.rows(); ++i) {
        for (int c = 0; c < StateEstimateIdx::STATE_ESTIMATE_COUNT; ++c) {
            if (c > 0) f << ',';
            f << state_estimates_(i, c);
        }
        f << '\n';
    }
    spdlog::info("TrajectoryInitializer: wrote {} rows to {}",
                 state_estimates_.rows(), path);
}
