/*
  run_batch_opt <dataset_folder> [config_path]

  Runs the batch orbit-determination optimizer on a dataset that has already
  been processed by prepare_ldmeas.

  Inputs (read from <dataset_folder>/):
    landmark_measurements.csv  — written by prepare_ldmeas
      columns: timestamp_ms, bearing_x, bearing_y, bearing_z,
               eci_x_km, eci_y_km, eci_z_km, group, sigma
    imu_data.csv               — raw IMU log
      columns: Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps

  Output (written to <dataset_folder>/):
    state_estimates.h5  — state_estimates, state_estimate_covariance_diagonal,
                          residuals datasets
*/

#include "navigation/od.hpp"
#include "navigation/batch_optimization.hpp"

static constexpr int64_t J2000_EPOCH_UNIX_S = 946727936LL;

#include <Eigen/Eigen>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include <spdlog/spdlog.h>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>

static constexpr const char* kDefaultConfigPath = "config/od.toml";
static constexpr double DPS_TO_RADPS = M_PI / 180.0;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: run_batch_opt <dataset_folder> [config_path]\n";
        return 1;
    }
    spdlog::set_level(spdlog::level::info);

    const std::string dataset_folder = argv[1];
    const std::string config_path    = (argc > 2) ? argv[2] : kDefaultConfigPath;

    OD_Config od_config = ReadODConfig(config_path);
    spdlog::info("Loaded OD config from {}", config_path);

    // ── Read landmark_measurements.csv ────────────────────────────────────
    // columns: timestamp_ms, bearing_x, bearing_y, bearing_z,
    //          eci_x_km, eci_y_km, eci_z_km, group, sigma
    const std::string lm_csv_path = dataset_folder + "/landmark_measurements.csv";

    std::vector<std::array<double, 7>> lm_rows;
    std::vector<bool>   group_starts_vec;
    std::vector<double> uncertainties;

    {
        std::ifstream f(lm_csv_path);
        if (!f.is_open()) {
            spdlog::error("Cannot open {}", lm_csv_path);
            return 1;
        }

        std::string line;
        std::getline(f, line); // skip header

        int prev_group = -1;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string tok;
            std::vector<std::string> tokens;
            while (std::getline(ss, tok, ',')) tokens.push_back(tok);
            if (tokens.size() < 9) continue;
            try {
                const double ts_ms  = std::stod(tokens[0]);
                const double t_j2000 = ts_ms / 1000.0
                                       - static_cast<double>(J2000_EPOCH_UNIX_S);
                const double bx     = std::stod(tokens[1]);
                const double by     = std::stod(tokens[2]);
                const double bz     = std::stod(tokens[3]);
                const double ex     = std::stod(tokens[4]);
                const double ey     = std::stod(tokens[5]);
                const double ez     = std::stod(tokens[6]);
                const int    group  = std::stoi(tokens[7]);
                const double sigma  = std::stod(tokens[8]);

                lm_rows.push_back({t_j2000, bx, by, bz, ex, ey, ez});
                group_starts_vec.push_back(group != prev_group);
                uncertainties.push_back(sigma);
                prev_group = group;
            } catch (const std::exception& e) {
                spdlog::warn("Skipping malformed landmark_measurements row: {}", e.what());
            }
        }
    }

    if (lm_rows.empty()) {
        spdlog::error("No landmark measurements loaded from {}", lm_csv_path);
        return 1;
    }
    spdlog::info("Loaded {} landmark rows from {}", lm_rows.size(), lm_csv_path);

    // ── Read imu_data.csv ─────────────────────────────────────────────────
    // columns: Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps
    const std::string imu_csv_path = dataset_folder + "/imu_data.csv";

    std::vector<std::array<double, 4>> gyro_rows; // [t_j2000, wx, wy, wz] rad/s

    {
        std::ifstream f(imu_csv_path);
        if (!f.is_open()) {
            spdlog::error("Cannot open {}", imu_csv_path);
            return 1;
        }

        std::string line;
        std::getline(f, line); // skip header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string tok;
            std::vector<std::string> tokens;
            while (std::getline(ss, tok, ',')) tokens.push_back(tok);
            if (tokens.size() < 4) continue;
            try {
                const double ts_ms  = std::stod(tokens[0]);
                const double t_j2000 = ts_ms / 1000.0
                                       - static_cast<double>(J2000_EPOCH_UNIX_S);
                const double wx = std::stod(tokens[1]) * DPS_TO_RADPS;
                const double wy = std::stod(tokens[2]) * DPS_TO_RADPS;
                const double wz = std::stod(tokens[3]) * DPS_TO_RADPS;
                gyro_rows.push_back({t_j2000, wx, wy, wz});
            } catch (const std::exception& e) {
                spdlog::warn("Skipping malformed IMU row: {}", e.what());
            }
        }
    }

    if (gyro_rows.empty()) {
        spdlog::error("No IMU data loaded from {}", imu_csv_path);
        return 1;
    }
    spdlog::info("Loaded {} gyro rows from {}", gyro_rows.size(), imu_csv_path);

    // ── Build Eigen matrices ──────────────────────────────────────────────
    const idx_t N = static_cast<idx_t>(lm_rows.size());
    const idx_t M = static_cast<idx_t>(gyro_rows.size());

    LandmarkMeasurements lm(N, LandmarkMeasurementIdx::LANDMARK_COUNT);
    LandmarkGroupStarts  gs(N);
    Eigen::VectorXd      lu(N);
    GyroMeasurements     gm(M, GyroMeasurementIdx::GYRO_MEAS_COUNT);

    for (idx_t i = 0; i < N; ++i) {
        for (int c = 0; c < LandmarkMeasurementIdx::LANDMARK_COUNT; ++c)
            lm(i, c) = lm_rows[static_cast<size_t>(i)][static_cast<size_t>(c)];
        gs(i) = group_starts_vec[static_cast<size_t>(i)];
        lu(i) = uncertainties[static_cast<size_t>(i)];
    }
    for (idx_t i = 0; i < M; ++i) {
        for (int c = 0; c < GyroMeasurementIdx::GYRO_MEAS_COUNT; ++c)
            gm(i, c) = gyro_rows[static_cast<size_t>(i)][static_cast<size_t>(c)];
    }

    assert(lm.cols() == LandmarkMeasurementIdx::LANDMARK_COUNT);
    assert(gm.cols() == GyroMeasurementIdx::GYRO_MEAS_COUNT);
    assert(gs.rows() == lm.rows());

    {
        std::ostringstream oss;
        oss << lm.topRows(std::min(N, idx_t{3}));
        spdlog::info("Landmark measurements (first 3 rows):\n{}", oss.str());
    }
    {
        std::ostringstream oss;
        oss << gm.topRows(std::min(M, idx_t{3}));
        spdlog::info("Gyro measurements     (first 3 rows):\n{}", oss.str());
    }

    // ── Run batch optimization ────────────────────────────────────────────
    auto [state_estimates, covariance, residuals] =
        solve_ceres_batch_opt(lm, gs, gm, lu, od_config.batch_opt);

    spdlog::info("Optimization complete: {} state estimates", state_estimates.rows());

    // ── Save results ──────────────────────────────────────────────────────
    try {
        const std::string out_path = dataset_folder + "/state_estimates.h5";
        H5Easy::File outfile(out_path, HighFive::File::Overwrite);
        H5Easy::dump(outfile, "state_estimates",                     state_estimates);
        H5Easy::dump(outfile, "state_estimate_covariance_diagonal",  covariance);
        H5Easy::dump(outfile, "residuals",                           residuals);
        spdlog::info("Saved state estimates to {}", out_path);
    } catch (const HighFive::Exception& e) {
        spdlog::error("Failed to write state estimates: {}", e.what());
        return 1;
    }

    return 0;
}
