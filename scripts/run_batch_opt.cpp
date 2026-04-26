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

  Output (written to data/results/<dataset_name>_<unix_ms>/):
    od_result.json             — solver metadata and run info
    state_estimates.csv        — Nx14 state estimate matrix
    covariance.csv             — diagonal of tangent-space covariance (absent if failed)
    residuals.csv              — flat residual vector
*/

#include "navigation/od.hpp"
#include "navigation/batch_optimization.hpp"

static constexpr int64_t J2000_EPOCH_UNIX_S = 946727936LL;

#include <Eigen/Eigen>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>

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

    // Capture run timestamp immediately so the results folder name reflects
    // when the run was initiated, not when writing completes.
    const int64_t run_unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    OD_Config od_config;
    try {
        od_config = ReadODConfig(config_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load OD config from {}: {}", config_path, e.what());
        return 1;
    }
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

    // ── Pack into ODMeasurements ──────────────────────────────────────────
    ODMeasurements meas;
    meas.landmark_measurements  = lm;
    meas.group_starts           = gs;
    meas.gyro_measurements      = gm;
    meas.landmark_uncertainties = lu;

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

    // ── Create results folder: data/results/<dataset_name>_<unix_ms> ───────
    const std::string dataset_name =
            std::filesystem::path(dataset_folder).filename().string();
    const std::string results_dir =
            std::string("data/results/") + dataset_name + "_" + std::to_string(run_unix_ms);

    if (!std::filesystem::create_directories(results_dir)) {
        spdlog::error("Failed to create results directory: {}", results_dir);
        return 1;
    }
    spdlog::info("Writing results to {}", results_dir);

    const int num_groups = static_cast<int>(
            std::count(group_starts_vec.begin(), group_starts_vec.end(), true));

    // ── Run batch optimization ────────────────────────────────────────────
    const auto result = solve_batch_opt(meas, od_config.batch_opt);
    const int64_t run_end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    // ── Write od_result.json (always, even on failure) ────────────────────
    {
        nlohmann::json meta;
        meta["dataset_folder"]       = dataset_folder;
        meta["run_unix_ms"]          = run_unix_ms;
        meta["run_time_ms"]          = run_end_ms - run_unix_ms;
        meta["error_code"]           = static_cast<int>(result.code);
        meta["bias_mode"]            = static_cast<int>(od_config.batch_opt.bias_mode);
        meta["solver"]["termination_type"] = result.solver_summary.termination_type;
        meta["solver"]["num_iterations"]   = result.solver_summary.num_iterations;
        meta["solver"]["initial_cost"]     = result.solver_summary.initial_cost;
        meta["solver"]["final_cost"]       = result.solver_summary.final_cost;
        meta["inputs"]["num_landmark_rows"]    = static_cast<int>(lm_rows.size());
        meta["inputs"]["num_landmark_groups"]  = num_groups;
        meta["inputs"]["num_gyro_rows"]        = static_cast<int>(gyro_rows.size());
        meta["outputs"]["num_state_estimates"] = result.state_estimates.rows();
        meta["outputs"]["covariance_available"] = result.covariance.rows() > 0;

        std::ofstream jf(results_dir + "/od_result.json");
        if (jf.is_open()) {
            jf << meta.dump(2) << '\n';
        } else {
            spdlog::warn("Failed to write od_result.json to {}", results_dir);
        }
    }

    if (result.code != ErrorCode::OK) {
        spdlog::error("Batch optimization failed (error code {}); "
                      "partial metadata written to {}.",
                      static_cast<int>(result.code), results_dir);
        return 1;
    }

    spdlog::info("Optimization complete: {} state estimates", result.state_estimates.rows());

    // ── Write initial_trajectory.csv ─────────────────────────────────────
    {
        const std::string path = results_dir + "/initial_trajectory.csv";
        std::ofstream f(path);
        if (!f.is_open()) {
            spdlog::warn("Failed to open initial_trajectory.csv for writing.");
        } else {
            f << "timestamp_j2000,pos_x_km,pos_y_km,pos_z_km,"
                 "vel_x_kms,vel_y_kms,vel_z_kms,"
                 "quat_x,quat_y,quat_z,quat_w,"
                 "gyro_bias_x_rads,gyro_bias_y_rads,gyro_bias_z_rads\n";
            f << std::setprecision(12);
            for (idx_t i = 0; i < result.initial_trajectory.rows(); ++i) {
                for (int c = 0; c < StateEstimateIdx::STATE_ESTIMATE_COUNT; ++c) {
                    if (c > 0) f << ',';
                    f << result.initial_trajectory(i, c);
                }
                f << '\n';
            }
            spdlog::info("Wrote initial_trajectory.csv ({} rows)", result.initial_trajectory.rows());
        }
    }

    // ── Write state_estimates.csv ─────────────────────────────────────────
    {
        std::ofstream f(results_dir + "/state_estimates.csv");
        if (!f.is_open()) {
            spdlog::error("Failed to open state_estimates.csv for writing.");
            return 1;
        }
        f << "timestamp_j2000,pos_x_km,pos_y_km,pos_z_km,"
             "vel_x_kms,vel_y_kms,vel_z_kms,"
             "quat_x,quat_y,quat_z,quat_w,"
             "gyro_bias_x_rads,gyro_bias_y_rads,gyro_bias_z_rads\n";
        f << std::setprecision(12);
        for (idx_t i = 0; i < result.state_estimates.rows(); ++i) {
            for (int c = 0; c < StateEstimateIdx::STATE_ESTIMATE_COUNT; ++c) {
                if (c > 0) f << ',';
                f << result.state_estimates(i, c);
            }
            f << '\n';
        }
        spdlog::info("Wrote state_estimates.csv ({} rows)", result.state_estimates.rows());
    }

    // ── Write covariance.csv (omitted if unavailable) ─────────────────────
    if (result.covariance.rows() > 0) {
        std::ofstream f(results_dir + "/covariance.csv");
        if (!f.is_open()) {
            spdlog::warn("Failed to open covariance.csv for writing.");
        } else {
            f << "timestamp_j2000,pos_cov_x,pos_cov_y,pos_cov_z,"
                 "vel_cov_x,vel_cov_y,vel_cov_z,"
                 "rot_cov_x,rot_cov_y,rot_cov_z,"
                 "gyro_bias_cov_x,gyro_bias_cov_y,gyro_bias_cov_z\n";
            f << std::setprecision(12);
            for (idx_t i = 0; i < result.covariance.rows(); ++i) {
                for (int c = 0; c < StateResIdx::STATE_RES_COUNT; ++c) {
                    if (c > 0) f << ',';
                    f << result.covariance(i, c);
                }
                f << '\n';
            }
            spdlog::info("Wrote covariance.csv ({} rows)", result.covariance.rows());
        }
    } else {
        spdlog::warn("Covariance unavailable — covariance.csv not written.");
    }

    // ── Write dynamics_residuals.csv ──────────────────────────────────────
    {
        std::ofstream f(results_dir + "/dynamics_residuals.csv");
        if (!f.is_open()) {
            spdlog::warn("Failed to open dynamics_residuals.csv for writing.");
        } else {
            f << "timestamp_j2000,pos_res_x,pos_res_y,pos_res_z,"
                 "vel_res_x,vel_res_y,vel_res_z,"
                 "rot_res_x,rot_res_y,rot_res_z,"
                 "gyro_bias_res_x,gyro_bias_res_y,gyro_bias_res_z\n";
            f << std::setprecision(12);
            for (idx_t i = 0; i < result.dynamics_residuals.rows(); ++i) {
                for (int c = 0; c < StateResIdx::STATE_RES_COUNT; ++c) {
                    if (c > 0) f << ',';
                    f << result.dynamics_residuals(i, c);
                }
                f << '\n';
            }
            spdlog::info("Wrote dynamics_residuals.csv ({} rows)", result.dynamics_residuals.rows());
        }
    }

    // ── Write landmark_residuals.csv ───────────────────────────────────────
    {
        std::ofstream f(results_dir + "/landmark_residuals.csv");
        if (!f.is_open()) {
            spdlog::warn("Failed to open landmark_residuals.csv for writing.");
        } else {
            f << "res_x,res_y,res_z\n" << std::setprecision(12);
            for (idx_t i = 0; i < result.landmark_residuals.rows(); ++i) {
                f << result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_X) << ','
                  << result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_Y) << ','
                  << result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_Z) << '\n';
            }
            spdlog::info("Wrote landmark_residuals.csv ({} rows)", result.landmark_residuals.rows());
        }
    }

    spdlog::info("OD complete. Results in {}", results_dir);
    return 0;
}
