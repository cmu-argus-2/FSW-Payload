#include "navigation/od_measurements.hpp"
#include "navigation/quaternion.hpp"
#include "navigation/pose_dynamics.hpp"  // GyroMeasurementIdx
#include <casadi/casadi.hpp>
#include "spdlog/spdlog.h"

using casadi::DM;
using casadi::MX;

MX pseudo_huber_cost(const MX& res, double M)
{
    MX d_sq = dot(res, res);
    return 2.0 * M * M * (sqrt(1.0 + d_sq / (M * M)) - 1.0);
}

MX landmark_residual_casadi(
    const MX& r,
    const MX& q,
    const DM& lmk_pos,
    const DM& bearing_meas,
    double sigma)
{
    MX diff         = MX(lmk_pos) - r;
    MX dist         = sqrt(dot(diff, diff) + 1e-6);
    MX bearing_eci  = diff / dist;
    MX bearing_body = quat_inv_rotate_xyzw(q, bearing_eci);
    return (bearing_body - MX(bearing_meas)) / sigma;
}

ErrorCode ODMeasurements::Validate() const
{
    bool valid = true;
    const auto N = landmark_measurements.rows();
    const auto M = gyro_measurements.rows();

    // ── Emptiness ─────────────────────────────────────────────────────────────
    if (N == 0) {
        spdlog::error("ODMeasurements::Validate: landmark_measurements is empty.");
        valid = false;
    }
    if (M == 0) {
        spdlog::error("ODMeasurements::Validate: gyro_measurements is empty.");
        valid = false;
    }
    if (!valid) {
        SPDLOG_ERROR("ODMeasurements::Validate: emptiness check failed.");
        return ErrorCode::ODMEAS_NOT_VALID;
    }

    if (N < OD_MIN_LANDMARK_MEASUREMENTS) {
        spdlog::error("ODMeasurements::Validate: only {} landmark measurement row(s); need at least {}.",
                      N, OD_MIN_LANDMARK_MEASUREMENTS);
        return ErrorCode::ODMEAS_NOT_VALID;
    }

    // ── Column counts ──────────────────────────────────────────────────────────
    if (landmark_measurements.cols() != LandmarkMeasurementIdx::LANDMARK_COUNT) {
        spdlog::error("ODMeasurements::Validate: landmark_measurements has {} columns, expected {}.",
                      landmark_measurements.cols(), LandmarkMeasurementIdx::LANDMARK_COUNT);
        valid = false;
    }
    if (gyro_measurements.cols() != GyroMeasurementIdx::GYRO_MEAS_COUNT) {
        spdlog::error("ODMeasurements::Validate: gyro_measurements has {} columns, expected {}.",
                      gyro_measurements.cols(), GyroMeasurementIdx::GYRO_MEAS_COUNT);
        valid = false;
    }

    // ── Row count consistency ──────────────────────────────────────────────────
    if (group_starts.rows() != N) {
        spdlog::error("ODMeasurements::Validate: group_starts has {} rows but "
                      "landmark_measurements has {}.",
                      group_starts.rows(), N);
        valid = false;
    }
    if (landmark_uncertainties.size() != N) {
        spdlog::error("ODMeasurements::Validate: landmark_uncertainties has {} elements but "
                      "landmark_measurements has {} rows.",
                      landmark_uncertainties.size(), N);
        valid = false;
    }
    if (!valid) {
        SPDLOG_ERROR("ODMeasurements::Validate: row count consistency check failed.");
        return ErrorCode::ODMEAS_NOT_VALID;
    }

    // ── Group starts ───────────────────────────────────────────────────────────
    if (!group_starts(0)) {
        spdlog::error("ODMeasurements::Validate: group_starts(0) is false; "
                      "the first measurement must open a group.");
        valid = false;
    }

    // ── Sigma sanity: 0 → divide-by-zero in cost functor; NaN/inf → NaN residuals ──
    {
        int bad = 0;
        for (Eigen::Index i = 0; i < N; ++i)
            if (!(landmark_uncertainties(i) > 0.0)) ++bad;
        if (bad == N) {
            spdlog::error("ODMeasurements::Validate: all {} sigma values are invalid "
                          "(zero/nan/inf). Check camera calibration (fx/fy) in the config file.", N);
            valid = false;
        } else if (bad > 0) {
            spdlog::warn("ODMeasurements::Validate: {}/{} sigma values are invalid (zero/nan/inf).",
                         bad, N);
        }
    }

    // ── Temporal range and overlap ─────────────────────────────────────────────
    {
        const double lm_t0 = landmark_measurements(0,   LandmarkMeasurementIdx::LANDMARK_TIMESTAMP);
        const double lm_t1 = landmark_measurements(N-1, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP);
        const double gy_t0 = gyro_measurements(0,   GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP);
        const double gy_t1 = gyro_measurements(M-1, GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP);

        spdlog::info("ODMeasurements: landmarks [{:.3f}, {:.3f}] J2000 s ({:.1f} s span); "
                     "gyro [{:.3f}, {:.3f}] J2000 s ({:.1f} s span).",
                     lm_t0, lm_t1, lm_t1 - lm_t0, gy_t0, gy_t1, gy_t1 - gy_t0);

        if (lm_t0 > gy_t1 || gy_t0 > lm_t1) {
            spdlog::error("ODMeasurements::Validate: no temporal overlap — "
                          "landmarks [{:.3f}, {:.3f}] vs gyro [{:.3f}, {:.3f}]. "
                          "Verify both files are from the same session.",
                          lm_t0, lm_t1, gy_t0, gy_t1);
            valid = false;
        } else if (lm_t0 < gy_t0 || lm_t1 > gy_t1) {
            spdlog::warn("ODMeasurements::Validate: some landmark timestamps fall outside the "
                         "gyro window ([{:.3f},{:.3f}] vs [{:.3f},{:.3f}]) — "
                         "will be snapped to nearest gyro timestamp.",
                         lm_t0, lm_t1, gy_t0, gy_t1);
        }
    }

    // ── Landmark timestamps non-decreasing ────────────────────────────────────
    for (Eigen::Index i = 1; i < N; ++i) {
        if (landmark_measurements(i,   LandmarkMeasurementIdx::LANDMARK_TIMESTAMP) <
            landmark_measurements(i-1, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP)) {
            spdlog::error("ODMeasurements::Validate: landmark timestamps not non-decreasing "
                          "at row {}: {:.6f} < {:.6f}.",
                          i,
                          landmark_measurements(i,   LandmarkMeasurementIdx::LANDMARK_TIMESTAMP),
                          landmark_measurements(i-1, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP));
            valid = false;
            break;
        }
    }

    // ── Gyro timestamps strictly increasing ───────────────────────────────────
    for (Eigen::Index i = 1; i < M; ++i) {
        if (gyro_measurements(i,   GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP) <=
            gyro_measurements(i-1, GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP)) {
            spdlog::error("ODMeasurements::Validate: gyro timestamps not strictly increasing "
                          "at row {}: {:.6f} <= {:.6f}.",
                          i,
                          gyro_measurements(i,   GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP),
                          gyro_measurements(i-1, GyroMeasurementIdx::GYRO_MEAS_TIMESTAMP));
            valid = false;
            break;
        }
    }

    if (!valid) {
        SPDLOG_ERROR("ODMeasurements::Validate: validation failed.");
        return ErrorCode::ODMEAS_NOT_VALID;
    }

    // ── Informational summary (only reached when valid) ────────────────────────
    {
        int num_groups = 0;
        for (Eigen::Index i = 0; i < N; ++i) if (group_starts(i)) ++num_groups;
        spdlog::info("ODMeasurements: {} landmark rows, {} groups, {} gyro rows.",
                     N, num_groups, M);
    }

    return ErrorCode::OK;
}
