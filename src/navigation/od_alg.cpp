#include "navigation/od.hpp"
#include "navigation/batch_optimization.hpp"
#include "navigation/utils.hpp"
#include "configuration.hpp"
#include "vision/regions.hpp"

#include <nlohmann/json.hpp>
#include "spdlog/spdlog.h"
#include <opencv2/core/eigen.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>


void OD::_DoInit()
{
    // Initialize the OD process

    // Might need to read an OD previous run file ~ something that logs the results

    // Check if there is any active AND valid data collection already started (in case came from reboot/or any)
    // -- validity check can also include the amount of time since we did capture
    // Also check that we haven't already used that guy
    // If yes, restart that collection process
    // If No, create that new data process and start it
    // Monitor continuously progress
    // If we need to stop --> stop collection and exit
    // if completed, switch our state to BATCH_OPT



    SwitchState(OD_STATE::BATCH_OPT);
}

void OD::_DoBatchOptimization()
{
    // Perform batch optimization
    if (!measurements_ready_) {
        SPDLOG_ERROR("OD::_DoBatchOptimization: measurements not ready, aborting");
        SwitchState(OD_STATE::IDLE);
        return;
    }

    LandmarkMeasurements lm   = measurements_.landmark_measurements;
    LandmarkGroupStarts  gs   = measurements_.group_starts;
    GyroMeasurements     gm   = measurements_.gyro_measurements;

    auto [state_estimates, covariance, residuals] =
        solve_ceres_batch_opt(lm, gs, gm, measurements_.landmark_uncertainties, config.batch_opt);

    SPDLOG_INFO("OD::_DoBatchOptimization: completed with {} state estimates", state_estimates.rows());

    SwitchState(OD_STATE::IDLE);
}

bool OD::IsODPossible(const std::string& dataset_folder) const
{
    namespace fs = std::filesystem;

    if (!fs::is_directory(dataset_folder)) {
        SPDLOG_WARN("IsODPossible: dataset folder does not exist: {}", dataset_folder);
        return false;
    }

    const std::string imu_csv = dataset_folder + "/imu_data.csv";
    if (!fs::exists(imu_csv)) {
        SPDLOG_WARN("IsODPossible: imu_data.csv not found in {}", dataset_folder);
        return false;
    }

    // Count LDNeted frames with at least one landmark; track their timestamp range
    int ldneted_frame_count = 0;
    double t_min = std::numeric_limits<double>::max();
    double t_max = std::numeric_limits<double>::lowest();

    for (const auto& entry : fs::directory_iterator(dataset_folder)) {
        const std::string fname = entry.path().filename().string();
        if (fname.rfind("frame_", 0) != 0 || entry.path().extension() != ".json") {
            continue;
        }

        std::ifstream f(entry.path());
        if (!f.is_open()) continue;

        try {
            nlohmann::json j = nlohmann::json::parse(f);
            const int stage    = j.value("processing_stage",       0);
            const int lm_count = j.value("detected_landmarks_count", 0);
            if (stage >= 3 && lm_count > 0) {
                ++ldneted_frame_count;
                const uint64_t ts_ms = j.value("timestamp", uint64_t(0));
                const double t_j2000 = static_cast<double>(ts_ms) / 1000.0
                                       - static_cast<double>(J2000_EPOCH_UNIX_S);
                t_min = std::min(t_min, t_j2000);
                t_max = std::max(t_max, t_j2000);
            }
        } catch (const std::exception& e) {
            SPDLOG_WARN("IsODPossible: failed to parse {}: {}", entry.path().string(), e.what());
        }
    }

    if (ldneted_frame_count < 2) {
        SPDLOG_WARN("IsODPossible: only {} LDNeted frame(s) with landmarks (need >= 2)",
                    ldneted_frame_count);
        return false;
    }

    // Verify IMU data spans the landmark window
    std::ifstream imu_f(imu_csv);
    if (!imu_f.is_open()) {
        SPDLOG_WARN("IsODPossible: cannot open imu_data.csv");
        return false;
    }

    double imu_t_first = std::numeric_limits<double>::max();
    double imu_t_last  = std::numeric_limits<double>::lowest();
    int    imu_count   = 0;
    std::string line;
    std::getline(imu_f, line); // skip header
    while (std::getline(imu_f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        if (!std::getline(ss, token, ',')) continue;
        try {
            const double t_j2000 = std::stod(token) / 1000.0
                                   - static_cast<double>(J2000_EPOCH_UNIX_S);
            imu_t_first = std::min(imu_t_first, t_j2000);
            imu_t_last  = std::max(imu_t_last,  t_j2000);
            ++imu_count;
        } catch (...) {}
    }

    if (imu_count == 0) {
        SPDLOG_WARN("IsODPossible: no valid IMU data in {}", imu_csv);
        return false;
    }

    if (imu_t_first > t_min || imu_t_last < t_max) {
        SPDLOG_WARN("IsODPossible: IMU window [{:.1f}, {:.1f}] does not bracket "
                    "landmark window [{:.1f}, {:.1f}]",
                    imu_t_first, imu_t_last, t_min, t_max);
        return false;
    }

    return true;
}

bool OD::DatasetPrepare(const std::string& dataset_folder,
                        const CameraCalibration& calibration,
                        const std::string& ld_model_folder)
{
    namespace fs = std::filesystem;
    measurements_ready_ = false;

    // Build per-camera rotation matrices (cv::Mat → Eigen::Matrix3d, 0-indexed cam_id)
    std::array<Eigen::Matrix3d, NUM_CAMERAS> R_cam_to_body;
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        cv::cv2eigen(calibration.cam_to_body[i], R_cam_to_body[static_cast<size_t>(i)]);
    }

    // Focal length (average of fx and fy) for uncertainty computation
    const double fx = calibration.camera_matrix.at<double>(0, 0);
    const double fy = calibration.camera_matrix.at<double>(1, 1);
    const double focal_length_px = (fx + fy) / 2.0;

    // Lazy-loaded bounding_boxes cache: region_id int → [(lon_deg, lat_deg), ...]
    std::map<int, std::vector<std::pair<double, double>>> bbox_cache;

    auto load_bboxes = [&](int region_id_int) -> bool {
        if (bbox_cache.count(region_id_int)) return true;

        const RegionID rid = static_cast<RegionID>(region_id_int);
        const std::string region_str(GetRegionString(rid));
        if (region_str == "UNKNOWN") {
            SPDLOG_WARN("DatasetPrepare: unknown region_id {}", region_id_int);
            return false;
        }

        const std::string csv_path = ld_model_folder + "/" + region_str + "/bounding_boxes.csv";
        std::ifstream f(csv_path);
        if (!f.is_open()) {
            SPDLOG_WARN("DatasetPrepare: bounding_boxes.csv not found: {}", csv_path);
            return false;
        }

        std::string line;
        std::getline(f, line); // skip header
        std::vector<std::pair<double, double>> rows;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string lon_s, lat_s;
            if (!std::getline(ss, lon_s, ',') || !std::getline(ss, lat_s, ',')) continue;
            try {
                rows.emplace_back(std::stod(lon_s), std::stod(lat_s));
            } catch (const std::exception& e) {
                SPDLOG_WARN("DatasetPrepare: skipping malformed bounding_boxes row: {}", e.what());
            }
        }
        if (rows.empty()) {
            SPDLOG_WARN("DatasetPrepare: bounding_boxes.csv for region {} is empty", region_str);
            return false;
        }
        bbox_cache[region_id_int] = std::move(rows);
        return true;
    };

    // Gather and sort frame JSON files (filename embeds timestamp → lexicographic = chronological)
    std::vector<fs::path> frame_files;
    if (!fs::is_directory(dataset_folder)) {
        SPDLOG_ERROR("DatasetPrepare: dataset folder does not exist: {}", dataset_folder);
        return false;
    }
    for (const auto& entry : fs::directory_iterator(dataset_folder)) {
        const std::string fname = entry.path().filename().string();
        if (fname.rfind("frame_", 0) == 0 && entry.path().extension() == ".json") {
            frame_files.push_back(entry.path());
        }
    }
    std::sort(frame_files.begin(), frame_files.end());

    // Accumulate landmark measurements
    std::vector<std::array<double, 7>> lm_rows;
    std::vector<bool>   group_starts_vec;
    std::vector<double> uncertainties;

    constexpr double DEG_TO_RAD = M_PI / 180.0;

    for (const auto& fpath : frame_files) {
        std::ifstream f(fpath);
        if (!f.is_open()) {
            SPDLOG_WARN("DatasetPrepare: cannot open {}", fpath.string());
            continue;
        }

        nlohmann::json j;
        try {
            j = nlohmann::json::parse(f);
        } catch (const std::exception& e) {
            SPDLOG_WARN("DatasetPrepare: failed to parse {}: {}", fpath.string(), e.what());
            continue;
        }

        const int stage    = j.value("processing_stage",        0);
        const int lm_count = j.value("detected_landmarks_count", 0);
        if (stage < 3 || lm_count == 0) continue;

        const uint64_t ts_ms  = j.value("timestamp", uint64_t(0));
        const int      cam_id = j.value("cam_id",     0);
        if (cam_id < 0 || cam_id >= NUM_CAMERAS) {
            SPDLOG_WARN("DatasetPrepare: cam_id {} out of range in {}", cam_id, fpath.string());
            continue;
        }

        const double t_j2000 = static_cast<double>(ts_ms) / 1000.0
                               - static_cast<double>(J2000_EPOCH_UNIX_S);

        if (!j.contains("landmarks") || !j.at("landmarks").is_array()) continue;

        bool first_in_group = true;
        for (const auto& lm_item : j.at("landmarks")) {
            for (const auto& [key, val] : lm_item.items()) {
                const float    px        = val.value("x",         0.0f);
                const float    py        = val.value("y",         0.0f);
                const float    height    = val.value("height",    0.0f);
                const float    width     = val.value("width",     0.0f);
                const uint16_t class_id  = val.value("class_id",  uint16_t(0));
                const int      region_id = val.value("region_id", 0);

                if (!load_bboxes(region_id)) continue;
                const auto& rows = bbox_cache.at(region_id);
                if (static_cast<size_t>(class_id) >= rows.size()) {
                    SPDLOG_WARN("DatasetPrepare: class_id {} out of range for region {} (size {})",
                                class_id, region_id, rows.size());
                    continue;
                }

                // Bearing in camera frame → body frame
                const Eigen::Vector3d bearing_cam =
                    PixelToBodyBearing(px, py, calibration.camera_matrix, calibration.dist_coeffs);
                const Eigen::Vector3d bearing_body =
                    R_cam_to_body[static_cast<size_t>(cam_id)] * bearing_cam;

                // ECI position of the landmark: geodetic, altitude = 0 m → convert to km
                const double lon_rad = rows[class_id].first  * DEG_TO_RAD;
                const double lat_rad = rows[class_id].second * DEG_TO_RAD;
                const Eigen::Vector3d eci_km =
                    LAT2ECI(Eigen::Vector3d(0.0, lon_rad, lat_rad), t_j2000, false) / 1000.0;

                // Per-landmark uncertainty: 3σ = half the smaller bbox side → σ = min(h,w)/(6f)
                const double sigma =
                    static_cast<double>(std::min(height, width)) / (6.0 * focal_length_px);

                lm_rows.push_back({t_j2000,
                                   bearing_body.x(), bearing_body.y(), bearing_body.z(),
                                   eci_km.x(),       eci_km.y(),       eci_km.z()});
                group_starts_vec.push_back(first_in_group);
                uncertainties.push_back(sigma);
                first_in_group = false;
            }
        }
    }

    if (lm_rows.empty()) {
        SPDLOG_ERROR("DatasetPrepare: no landmark measurements extracted from {}", dataset_folder);
        return false;
    }

    // Parse imu_data.csv: Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps [, ...]
    const std::string imu_csv = dataset_folder + "/imu_data.csv";
    std::ifstream imu_f(imu_csv);
    if (!imu_f.is_open()) {
        SPDLOG_ERROR("DatasetPrepare: cannot open imu_data.csv: {}", imu_csv);
        return false;
    }

    std::vector<std::array<double, 4>> gyro_rows;  // [t_j2000, wx, wy, wz] in rad/s
    std::string line;
    std::getline(imu_f, line); // skip header
    while (std::getline(imu_f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> tokens;
        while (std::getline(ss, tok, ',')) {
            tokens.push_back(tok);
        }
        if (tokens.size() < 4) continue;
        try {
            const double t_j2000 = std::stod(tokens[0]) / 1000.0
                                   - static_cast<double>(J2000_EPOCH_UNIX_S);
            constexpr double DPS_TO_RADPS = M_PI / 180.0;
            const double wx = std::stod(tokens[1]) * DPS_TO_RADPS;
            const double wy = std::stod(tokens[2]) * DPS_TO_RADPS;
            const double wz = std::stod(tokens[3]) * DPS_TO_RADPS;
            gyro_rows.push_back({t_j2000, wx, wy, wz});
        } catch (const std::exception& e) {
            SPDLOG_WARN("DatasetPrepare: skipping malformed IMU row: {}", e.what());
        }
    }

    if (gyro_rows.empty()) {
        SPDLOG_ERROR("DatasetPrepare: no IMU data parsed from {}", imu_csv);
        return false;
    }

    // Fill ODMeasurements struct
    const idx_t N = static_cast<idx_t>(lm_rows.size());
    const idx_t M = static_cast<idx_t>(gyro_rows.size());

    measurements_.landmark_measurements.resize(N, 7);
    measurements_.group_starts.resize(N);
    measurements_.landmark_uncertainties.resize(N);
    measurements_.gyro_measurements.resize(M, 4);

    for (idx_t i = 0; i < N; ++i) {
        for (int c = 0; c < 7; ++c) {
            measurements_.landmark_measurements(i, c) = lm_rows[static_cast<size_t>(i)][static_cast<size_t>(c)];
        }
        measurements_.group_starts(i)           = group_starts_vec[static_cast<size_t>(i)];
        measurements_.landmark_uncertainties(i) = uncertainties[static_cast<size_t>(i)];
    }
    for (idx_t i = 0; i < M; ++i) {
        for (int c = 0; c < 4; ++c) {
            measurements_.gyro_measurements(i, c) = gyro_rows[static_cast<size_t>(i)][static_cast<size_t>(c)];
        }
    }

    measurements_ready_ = true;
    SPDLOG_INFO("DatasetPrepare: {} landmark rows, {} gyro rows extracted from {}",
                N, M, dataset_folder);
    return true;
}
