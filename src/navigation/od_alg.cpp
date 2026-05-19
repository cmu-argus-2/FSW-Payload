#include "navigation/od.hpp"
#include "navigation/batch_optimization.hpp"
#include "navigation/utils.hpp"
#include "configuration.hpp"
#include "core/timing.hpp"
#include "inference/types.hpp"
#include "vision/dataset_manager.hpp"
#include "vision/regions.hpp"

#include <nlohmann/json.hpp>
#include "spdlog/spdlog.h"
#include <opencv2/core/eigen.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <thread>

namespace {

} // namespace

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
    int total_landmark_count = 0;
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
            const nlohmann::json* inf = j.contains("inference_results") && j.at("inference_results").is_object()
                                      ? &j.at("inference_results")
                                      : nullptr;
            const int lm_count = inf ? inf->value("detected_landmarks_count", 0)
                                     : j.value("detected_landmarks_count", 0);
            if (stage >= 3 && lm_count > 0) {
                ++ldneted_frame_count;
                total_landmark_count += lm_count;
                const uint64_t ts_ms = j.value("timestamp", uint64_t(0));
                const double t_j2000 = unixToJ2000(static_cast<double>(ts_ms) / 1000.0);
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

    if (total_landmark_count < OD_MIN_LANDMARK_MEASUREMENTS) {
        SPDLOG_WARN("IsODPossible: only {} total detected landmark(s) (need >= {})",
                    total_landmark_count, OD_MIN_LANDMARK_MEASUREMENTS);
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
            const double t_j2000 = unixToJ2000(std::stod(token) / 1000.0);
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

ErrorCode OD::DatasetPrepare(const std::string& dataset_folder,
                             const CameraCalibration& calibration)
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

    // Gather and sort frame JSON files (filename embeds timestamp → lexicographic = chronological)
    std::vector<fs::path> frame_files;
    if (!fs::is_directory(dataset_folder)) {
        SPDLOG_ERROR("DatasetPrepare: dataset folder does not exist: {}", dataset_folder);
        return ErrorCode::FILE_DOES_NOT_EXIST;
    }
    for (const auto& entry : fs::directory_iterator(dataset_folder)) {
        const std::string fname = entry.path().filename().string();
        if (fname.rfind("frame_", 0) == 0 && entry.path().extension() == ".json") {
            frame_files.push_back(entry.path());
        }
    }
    std::sort(frame_files.begin(), frame_files.end());

    int inferred_ld_version = -1;
    for (const auto& fpath : frame_files) {
        std::ifstream f(fpath);
        if (!f.is_open()) continue;

        try {
            const nlohmann::json j = nlohmann::json::parse(f);
            const int stage = j.value("processing_stage", 0);
            if (stage < static_cast<int>(ProcessingStage::LDNeted)) continue;
            if (!j.contains("inference_results") || !j.at("inference_results").is_object()) continue;

            const nlohmann::json& j_inf = j.at("inference_results");
            if (j_inf.value("detected_landmarks_count", 0) <= 0) continue;

            const int ld_version = j_inf.value("ldnet_version", -1);
            if (ld_version <= 0) {
                SPDLOG_ERROR("DatasetPrepare: invalid or missing ldnet_version in {}", fpath.string());
                return ErrorCode::ODMEAS_NOT_VALID;
            }
            if (inferred_ld_version < 0) {
                inferred_ld_version = ld_version;
            } else if (inferred_ld_version != ld_version) {
                SPDLOG_ERROR("DatasetPrepare: mixed LD model versions in dataset: {} and {}",
                             inferred_ld_version, ld_version);
                return ErrorCode::ODMEAS_NOT_VALID;
            }
        } catch (const std::exception& e) {
            SPDLOG_WARN("DatasetPrepare: failed to inspect {}: {}", fpath.string(), e.what());
        }
    }
    if (inferred_ld_version <= 0) {
        SPDLOG_ERROR("DatasetPrepare: could not infer LD model version from {}", dataset_folder);
        return ErrorCode::ODMEAS_NOT_VALID;
    }
    SPDLOG_INFO("DatasetPrepare: inferred LD model version V{} from {}",
                inferred_ld_version, dataset_folder);

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

        const std::string csv_path = Inference::LDBoundingBoxesPath(inferred_ld_version, region_str);
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

    // Accumulate landmark measurements
    std::vector<std::array<double, 7>> lm_rows;
    std::vector<bool>   group_starts_vec;
    std::vector<double> uncertainties;
    std::vector<uint64_t> lm_timestamps_ms;

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
        if (stage < 3) continue;

        const uint64_t ts_ms  = j.value("timestamp", uint64_t(0));
        const int      cam_id = j.value("cam_id",     0);
        if (cam_id < 0 || cam_id >= NUM_CAMERAS) {
            SPDLOG_WARN("DatasetPrepare: cam_id {} out of range in {}", cam_id, fpath.string());
            continue;
        }

        const double t_j2000 = unixToJ2000(static_cast<double>(ts_ms) / 1000.0);
        
        if (!j.contains("inference_results") || !j.at("inference_results").is_object()) {
            SPDLOG_WARN("DatasetPrepare: inference_results missing or not an object in {}", fpath.string());
            continue;
        }

        nlohmann::json j_inf = j.at("inference_results");

        const int lm_count = j_inf.value("detected_landmarks_count", 0);
        if (lm_count == 0) continue;

        if (!j_inf.contains("landmarks") || !j_inf.at("landmarks").is_array()) {
            SPDLOG_WARN("DatasetPrepare: landmarks missing or not an array in {}", fpath.string());
            continue;
        }

        bool first_in_group = true;
        for (const auto& lm_item : j_inf.at("landmarks")) {
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

                // ECI position of the landmark: geodetic, altitude = 0 km (sea level).
                // LAT2ECI returns km (SPICE bodvrd_c radii are in km), so no conversion needed.
                const double lon_rad = rows[class_id].first  * DEG_TO_RAD;
                const double lat_rad = rows[class_id].second * DEG_TO_RAD;
                const Eigen::Vector3d eci_km =
                    LAT2ECI(Eigen::Vector3d(0.0, lon_rad, lat_rad), t_j2000, false);

                // Per-landmark uncertainty: 3σ = half the smaller bbox side → σ = min(h,w)/(6f)
                const double sigma =
                    static_cast<double>(std::min(height, width)) / (6.0 * focal_length_px);

                lm_rows.push_back({t_j2000,
                                   bearing_body.x(), bearing_body.y(), bearing_body.z(),
                                   eci_km.x(),       eci_km.y(),       eci_km.z()});
                group_starts_vec.push_back(first_in_group);
                uncertainties.push_back(sigma);
                lm_timestamps_ms.push_back(ts_ms);
                first_in_group = false;
            }
        }
    }

    if (lm_rows.empty()) {
        SPDLOG_ERROR("DatasetPrepare: no landmark measurements extracted from {}", dataset_folder);
        return ErrorCode::ODMEAS_NOT_VALID;
    }

    // Write landmark_measurements.csv
    const std::string lm_csv_path = dataset_folder + "/landmark_measurements.csv";
    std::ofstream lm_csv(lm_csv_path);
    if (!lm_csv.is_open()) {
        SPDLOG_WARN("DatasetPrepare: cannot write landmark_measurements.csv to {}", lm_csv_path);
        return ErrorCode::FILE_NOT_AVAILABLE;
    } else {
        lm_csv << "timestamp_ms,bearing_x,bearing_y,bearing_z,eci_x_km,eci_y_km,eci_z_km,group,sigma\n";
        lm_csv << std::fixed << std::setprecision(9);
        int group_idx = -1;
        for (size_t i = 0; i < lm_rows.size(); ++i) {
            if (group_starts_vec[i]) ++group_idx;
            const auto& row = lm_rows[i];
            lm_csv << lm_timestamps_ms[i] << ','
                   << row[1] << ',' << row[2] << ',' << row[3] << ','
                   << row[4] << ',' << row[5] << ',' << row[6] << ','
                   << group_idx << ','
                   << uncertainties[i] << '\n';
        }
        SPDLOG_INFO("DatasetPrepare: wrote {} rows to {}", lm_rows.size(), lm_csv_path);
    }

    // Parse imu_data.csv: Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps [, ...]
    const std::string imu_csv = dataset_folder + "/imu_data.csv";
    std::ifstream imu_f(imu_csv);
    if (!imu_f.is_open()) {
        SPDLOG_ERROR("DatasetPrepare: cannot open imu_data.csv: {}", imu_csv);
        return ErrorCode::FILE_DOES_NOT_EXIST;
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
            const double t_j2000 = unixToJ2000(std::stod(tokens[0]) / 1000.0);
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
        return ErrorCode::ODMEAS_NOT_VALID;
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
    return ErrorCode::OK;
}

ODStage InspectDatasetForOD(const std::string& dataset_folder)
{
    namespace fs = std::filesystem;
    if (!fs::is_directory(dataset_folder)) {
        return ODStage::DATASET_NOT_AVAILABLE;
    }

    // const bool has_measurements = fs::exists(fs::path(dataset_folder) / "landmark_measurements.csv");
    // if (has_measurements) {
    //     return ODStage::MEASUREMENTS_READY;
    // }

    bool saw_frame = false;
    bool saw_ldneted_landmark_frame = false;
    int total_landmark_count = 0;
    for (const auto& entry : fs::directory_iterator(dataset_folder)) {
        const std::string fname = entry.path().filename().string();
        if (fname.rfind("frame_", 0) != 0 || entry.path().extension() != ".json") {
            continue;
        }
        saw_frame = true;
        std::ifstream f(entry.path());
        if (!f.is_open()) continue;
        try {
            const nlohmann::json j = nlohmann::json::parse(f);
            const int stage = j.value("processing_stage", 0);
            const nlohmann::json* inf = j.contains("inference_results") && j.at("inference_results").is_object()
                                      ? &j.at("inference_results")
                                      : nullptr;
            const int lm_count = inf ? inf->value("detected_landmarks_count", 0)
                                     : j.value("detected_landmarks_count", 0);
            if (stage >= static_cast<int>(ProcessingStage::LDNeted) && lm_count > 0) {
                saw_ldneted_landmark_frame = true;
                total_landmark_count += lm_count;
            }
        } catch (const std::exception& e) {
            SPDLOG_WARN("InspectDatasetForOD: failed to parse {}: {}", entry.path().string(), e.what());
        }
    }

    if (!saw_frame || !saw_ldneted_landmark_frame) {
        return ODStage::DATASET_NOT_PROCESSED;
    }
    if (total_landmark_count < OD_MIN_LANDMARK_MEASUREMENTS) {
        SPDLOG_WARN("InspectDatasetForOD: only {} total detected landmark(s) (need >= {})",
                    total_landmark_count, OD_MIN_LANDMARK_MEASUREMENTS);
        return ODStage::DATASET_NOT_PROCESSED;
    }
    return ODStage::DATASET_PROCESSED;
}

ODMeasurementsResult LoadODMeasurementsFromDataset(const std::string& dataset_folder)
{
    constexpr double DPS_TO_RADPS = M_PI / 180.0;
    namespace fs = std::filesystem;
    ODMeasurementsResult result;

    const fs::path lm_csv_path = fs::path(dataset_folder) / "landmark_measurements.csv";
    const fs::path imu_csv_path = fs::path(dataset_folder) / "imu_data.csv";

    std::vector<std::array<double, 7>> lm_rows;
    std::vector<bool> group_starts_vec;
    std::vector<double> uncertainties;

    {
        std::ifstream f(lm_csv_path);
        if (!f.is_open()) {
            SPDLOG_ERROR("LoadODMeasurementsFromDataset: cannot open {}", lm_csv_path.string());
            result.code = ErrorCode::FILE_DOES_NOT_EXIST;
            return result;
        }

        std::string line;
        std::getline(f, line);
        int prev_group = -1;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string tok;
            std::vector<std::string> tokens;
            while (std::getline(ss, tok, ',')) tokens.push_back(tok);
            if (tokens.size() < 9) continue;
            try {
                const double ts_ms = std::stod(tokens[0]);
                const double t_j2000 = unixToJ2000(ts_ms / 1000.0);
                const double bx = std::stod(tokens[1]);
                const double by = std::stod(tokens[2]);
                const double bz = std::stod(tokens[3]);
                const double ex = std::stod(tokens[4]);
                const double ey = std::stod(tokens[5]);
                const double ez = std::stod(tokens[6]);
                const int group = std::stoi(tokens[7]);
                const double sigma = std::stod(tokens[8]);

                lm_rows.push_back({t_j2000, bx, by, bz, ex, ey, ez});
                group_starts_vec.push_back(group != prev_group);
                uncertainties.push_back(sigma);
                prev_group = group;
            } catch (const std::exception& e) {
                SPDLOG_WARN("LoadODMeasurementsFromDataset: skipping malformed landmark row: {}", e.what());
            }
        }
    }
    if (lm_rows.empty()) {
        SPDLOG_ERROR("LoadODMeasurementsFromDataset: no landmark measurements loaded from {}",
                     lm_csv_path.string());
        result.code = ErrorCode::ODMEAS_NOT_VALID;
        return result;
    }

    std::vector<std::array<double, 4>> gyro_rows;
    {
        std::ifstream f(imu_csv_path);
        if (!f.is_open()) {
            SPDLOG_ERROR("LoadODMeasurementsFromDataset: cannot open {}", imu_csv_path.string());
            result.code = ErrorCode::FILE_DOES_NOT_EXIST;
            return result;
        }

        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string tok;
            std::vector<std::string> tokens;
            while (std::getline(ss, tok, ',')) tokens.push_back(tok);
            if (tokens.size() < 4) continue;
            try {
                const double ts_ms = std::stod(tokens[0]);
                const double t_j2000 = unixToJ2000(ts_ms / 1000.0);
                const double wx = std::stod(tokens[1]) * DPS_TO_RADPS;
                const double wy = std::stod(tokens[2]) * DPS_TO_RADPS;
                const double wz = std::stod(tokens[3]) * DPS_TO_RADPS;
                gyro_rows.push_back({t_j2000, wx, wy, wz});
            } catch (const std::exception& e) {
                SPDLOG_WARN("LoadODMeasurementsFromDataset: skipping malformed IMU row: {}", e.what());
            }
        }
    }
    if (gyro_rows.empty()) {
        SPDLOG_ERROR("LoadODMeasurementsFromDataset: no IMU data loaded from {}",
                     imu_csv_path.string());
        result.code = ErrorCode::ODMEAS_NOT_VALID;
        return result;
    }

    const idx_t N = static_cast<idx_t>(lm_rows.size());
    const idx_t M = static_cast<idx_t>(gyro_rows.size());

    ODMeasurements& meas = result.measurements;
    meas.landmark_measurements.resize(N, LandmarkMeasurementIdx::LANDMARK_COUNT);
    meas.group_starts.resize(N);
    meas.landmark_uncertainties.resize(N);
    meas.gyro_measurements.resize(M, GyroMeasurementIdx::GYRO_MEAS_COUNT);

    for (idx_t i = 0; i < N; ++i) {
        for (int c = 0; c < LandmarkMeasurementIdx::LANDMARK_COUNT; ++c) {
            meas.landmark_measurements(i, c) = lm_rows[static_cast<size_t>(i)][static_cast<size_t>(c)];
        }
        meas.group_starts(i) = group_starts_vec[static_cast<size_t>(i)];
        meas.landmark_uncertainties(i) = uncertainties[static_cast<size_t>(i)];
    }
    for (idx_t i = 0; i < M; ++i) {
        for (int c = 0; c < GyroMeasurementIdx::GYRO_MEAS_COUNT; ++c) {
            meas.gyro_measurements(i, c) = gyro_rows[static_cast<size_t>(i)][static_cast<size_t>(c)];
        }
    }

    SPDLOG_INFO("LoadODMeasurementsFromDataset: loaded {} landmark rows and {} gyro rows from {}",
                N, M, dataset_folder);
    result.code = ErrorCode::OK;
    return result;
}

static bool write_state_csv(const std::string& path, const StateEstimates& states)
{
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "timestamp_j2000,pos_x_km,pos_y_km,pos_z_km,"
         "vel_x_kms,vel_y_kms,vel_z_kms,"
         "quat_x,quat_y,quat_z,quat_w\n";
    f << std::setprecision(12);
    for (idx_t i = 0; i < states.rows(); ++i) {
        for (int c = 0; c <= StateEstimateIdx::QUAT_W; ++c) {
            if (c > 0) f << ',';
            f << states(i, c);
        }
        f << '\n';
    }
    return true;
}

static bool WriteBatchODResults(const std::string& dataset_folder,
                                const std::string& results_dir,
                                const BatchOptResult& result,
                                const OD_Config& od_config,
                                int64_t run_unix_ms,
                                int64_t run_end_ms,
                                int num_landmark_rows,
                                int num_landmark_groups,
                                int num_gyro_rows)
{
    namespace fs = std::filesystem;
    fs::create_directories(results_dir);

    const fs::path src_lm = fs::path(dataset_folder) / "landmark_measurements.csv";
    const fs::path dst_lm = fs::path(results_dir)    / "landmark_measurements.csv";
    if (fs::exists(src_lm)) {
        std::error_code ec;
        fs::copy_file(src_lm, dst_lm, fs::copy_options::overwrite_existing, ec);
        if (ec) SPDLOG_WARN("WriteBatchODResults: could not copy landmark_measurements.csv: {}", ec.message());
    }

    nlohmann::json meta;
    meta["dataset_folder"] = dataset_folder;
    meta["run_unix_ms"] = run_unix_ms;
    meta["run_time_ms"] = run_end_ms - run_unix_ms;
    meta["timing"]["covariance_time_ms"] =
        result.covariance_time_ms >= 0.0 ? nlohmann::json(result.covariance_time_ms) : nlohmann::json(nullptr);
    meta["error_code"] = static_cast<int>(result.code);
    meta["bias_mode"] = static_cast<int>(od_config.batch_opt.bias_mode);
    meta["solver"]["return_status"] = result.solver_summary.return_status;
    meta["solver"]["iter_count"] = result.solver_summary.iter_count;
    meta["solver"]["final_cost"] = result.solver_summary.final_cost;
    meta["solver"]["solve_time_ms"] =
        result.solver_summary.solve_time_ms >= 0.0 ? nlohmann::json(result.solver_summary.solve_time_ms) : nlohmann::json(nullptr);
    if (!result.iteration_summaries.empty()) {
        nlohmann::json iters = nlohmann::json::array();
        for (const auto& s : result.iteration_summaries) {
            nlohmann::json entry;
            entry["return_status"]      = s.return_status;
            entry["iter_count"]         = s.iter_count;
            entry["final_cost"]         = s.final_cost;
            entry["solve_time_ms"]      = s.solve_time_ms >= 0.0 ? nlohmann::json(s.solve_time_ms) : nlohmann::json(nullptr);
            entry["n_active_landmarks"] = s.n_active_landmarks;
            entry["n_rejected"]         = s.n_rejected;
            iters.push_back(entry);
        }
        meta["solver_iterations"] = iters;
    }
    meta["inputs"]["num_landmark_rows"] = num_landmark_rows;
    meta["inputs"]["num_landmark_groups"] = num_landmark_groups;
    meta["inputs"]["num_gyro_rows"] = num_gyro_rows;
    meta["outputs"]["num_state_estimates"] = result.state_estimates.rows();
    meta["outputs"]["covariance_available"] = result.covariance.rows() > 0;
    meta["outputs"]["covariance_computed"] = result.covariance_computed;
    meta["outputs"]["covariance_timed_out"] = result.covariance_timed_out;
    meta["outputs"]["od_solver_calls"] = result.n_od_solver_calls;
    meta["outputs"]["landmarks_rejected"] = result.n_lmk_outliers;
    meta["config"]["use_j2"] = od_config.batch_opt.use_j2;
    meta["config"]["use_drag"] = od_config.batch_opt.use_drag;
    meta["config"]["compute_covariance"] = od_config.batch_opt.compute_covariance;
    meta["config"]["max_run_time_sec"] = od_config.batch_opt.max_run_time_sec;
    meta["config"]["integrator"] = static_cast<int>(od_config.batch_opt.integrator);
    if (od_config.batch_opt.bias_mode == BIAS_MODE::FIX_BIAS) {
        meta["estimates"]["gyro_bias_x_rads"] = result.gyro_bias_fixed[0];
        meta["estimates"]["gyro_bias_y_rads"] = result.gyro_bias_fixed[1];
        meta["estimates"]["gyro_bias_z_rads"] = result.gyro_bias_fixed[2];
    }
    if (result.cd_estimated) {
        meta["estimates"]["cd"] = result.cd;
    }
    if (result.state_estimates.rows() > 0) {
        const idx_t i = result.state_estimates.rows() - 1;
        meta["estimates"]["timestamp_j2000"] = result.state_estimates(i, STATE_ESTIMATE_TIMESTAMP);
        meta["estimates"]["pos_x_km"]        = result.state_estimates(i, POS_X);
        meta["estimates"]["pos_y_km"]        = result.state_estimates(i, POS_Y);
        meta["estimates"]["pos_z_km"]        = result.state_estimates(i, POS_Z);
        meta["estimates"]["vel_x_kms"]       = result.state_estimates(i, VEL_X);
        meta["estimates"]["vel_y_kms"]       = result.state_estimates(i, VEL_Y);
        meta["estimates"]["vel_z_kms"]       = result.state_estimates(i, VEL_Z);
        meta["estimates"]["quat_x"]          = result.state_estimates(i, QUAT_X);
        meta["estimates"]["quat_y"]          = result.state_estimates(i, QUAT_Y);
        meta["estimates"]["quat_z"]          = result.state_estimates(i, QUAT_Z);
        meta["estimates"]["quat_w"]          = result.state_estimates(i, QUAT_W);

        if (result.covariance_computed && result.covariance.rows() > i) {
            meta["estimates"]["pos_cov_x_km2"]   = result.covariance(i, RES_POS_X);
            meta["estimates"]["pos_cov_y_km2"]   = result.covariance(i, RES_POS_Y);
            meta["estimates"]["pos_cov_z_km2"]   = result.covariance(i, RES_POS_Z);
            meta["estimates"]["vel_cov_x_km2s2"] = result.covariance(i, RES_VEL_X);
            meta["estimates"]["vel_cov_y_km2s2"] = result.covariance(i, RES_VEL_Y);
            meta["estimates"]["vel_cov_z_km2s2"] = result.covariance(i, RES_VEL_Z);
            meta["estimates"]["rot_cov_x_rad2"]  = result.covariance(i, RES_ROT_X);
            meta["estimates"]["rot_cov_y_rad2"]  = result.covariance(i, RES_ROT_Y);
            meta["estimates"]["rot_cov_z_rad2"]  = result.covariance(i, RES_ROT_Z);
        }
    }
    if (result.covariance_computed) {
        meta["estimates"]["gyro_bias_cov_x_rads2"] = result.gyro_bias_var[0];
        meta["estimates"]["gyro_bias_cov_y_rads2"] = result.gyro_bias_var[1];
        meta["estimates"]["gyro_bias_cov_z_rads2"] = result.gyro_bias_var[2];
        if (result.cd_estimated) {
            meta["estimates"]["cd_var"] = result.cd_var;
        }
    }

    {
        std::ofstream jf(results_dir + "/od_result.json");
        if (!jf.is_open()) return false;
        jf << meta.dump(2) << '\n';
    }

    if (result.code != ErrorCode::OK) return true;

    write_state_csv(results_dir + "/initial_trajectory.csv", result.initial_trajectory);
    write_state_csv(results_dir + "/state_estimates.csv", result.state_estimates);

    if (result.covariance.rows() > 0) {
        std::ofstream f(results_dir + "/covariance.csv");
        if (f.is_open()) {
            f << "timestamp_j2000,pos_cov_x,pos_cov_y,pos_cov_z,"
                 "vel_cov_x,vel_cov_y,vel_cov_z,"
                 "rot_cov_x,rot_cov_y,rot_cov_z\n";
            f << std::setprecision(12);
            for (idx_t i = 0; i < result.covariance.rows(); ++i) {
                for (int c = 0; c <= StateResIdx::RES_ROT_Z; ++c) {
                    if (c > 0) f << ',';
                    f << result.covariance(i, c);
                }
                f << '\n';
            }
        }
    }

    {
        std::ofstream f(results_dir + "/dynamics_residuals.csv");
        if (f.is_open()) {
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
        }
    }

    {
        std::ofstream f(results_dir + "/landmark_residuals.csv");
        if (f.is_open()) {
            f << "res_x,res_y,res_z,outlier\n" << std::setprecision(12);
            for (idx_t i = 0; i < result.landmark_residuals.rows(); ++i) {
                const int8_t flag = (i < static_cast<idx_t>(result.lmk_outlier_flags.size()))
                                    ? result.lmk_outlier_flags[static_cast<size_t>(i)] : 0;
                const double rx = result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_X);
                const double ry = result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_Y);
                const double rz = result.landmark_residuals(i, LandmarkResIdx::LANDMARK_RES_Z);
                if (std::isnan(rx) || std::isnan(ry) || std::isnan(rz)) {
                    f << "nan,nan,nan," << static_cast<int>(flag) << '\n';
                } else {
                    f << rx << ',' << ry << ',' << rz << ',' << static_cast<int>(flag) << '\n';
                }
            }
        }
    }

    return true;
}

ODResult RunODOnDataset(const ODRequest& request)
{
    namespace fs = std::filesystem;
    ODResult out;
    out.dataset_folder = request.dataset_folder;
    out.stage = InspectDatasetForOD(request.dataset_folder);

    if (out.stage == ODStage::DATASET_NOT_AVAILABLE ||
        out.stage == ODStage::DATASET_NOT_PROCESSED) {
        out.code = ErrorCode::ODMEAS_NOT_VALID;
        return out;
    }

    OD_Config od_config;
    if (request.od_config_override.has_value()) {
        od_config = request.od_config_override.value();
    } else {
        const ODConfigResult config_result = ReadODConfig(request.od_config_path);
        if (config_result.code != ErrorCode::OK) {
            out.code = config_result.code;
            out.stage = ODStage::FAILED;
            return out;
        }
        od_config = config_result.config;
    }

    if (out.stage == ODStage::DATASET_PROCESSED) {
        Configuration config;
        config.LoadConfiguration(request.system_config_path);
        OD od(request.od_config_path);
        const ErrorCode prepare_code =
            od.DatasetPrepare(request.dataset_folder, config.GetCameraCalibration());
        if (prepare_code != ErrorCode::OK) {
            out.code = prepare_code;
            out.stage = ODStage::FAILED;
            return out;
        }
        out.stage = ODStage::MEASUREMENTS_READY;
    }

    const ODMeasurementsResult measurements_result =
        LoadODMeasurementsFromDataset(request.dataset_folder);
    if (measurements_result.code != ErrorCode::OK) {
        out.code = measurements_result.code;
        out.stage = ODStage::FAILED;
        return out;
    }
    const ODMeasurements& measurements = measurements_result.measurements;
    out.stage = ODStage::MEASUREMENTS_READY;

    const int64_t run_unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    out.results_dir = std::string("data/results/") + std::to_string(run_unix_ms);

    const int num_groups = static_cast<int>(
        std::count(measurements.group_starts.data(),
                   measurements.group_starts.data() + measurements.group_starts.size(),
                   true));

    BatchOptResult result = solve_batch_opt_with_outlier_rejection(measurements, od_config.batch_opt);
    out.stage = result.initial_trajectory.rows() > 0 ? ODStage::INITIAL_GUESS_CREATED : out.stage;
    const int64_t run_end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    if (!WriteBatchODResults(request.dataset_folder, out.results_dir, result, od_config,
                             run_unix_ms, run_end_ms,
                             static_cast<int>(measurements.landmark_measurements.rows()),
                             num_groups,
                             static_cast<int>(measurements.gyro_measurements.rows()))) {
        SPDLOG_ERROR("RunODOnDataset: failed to write OD results to {}", out.results_dir);
        out.code = ErrorCode::BATCH_OPT_INVALID_OUTPUT;
        out.stage = ODStage::FAILED;
        return out;
    }

    out.code = result.code;
    out.stage = result.code == ErrorCode::OK ? ODStage::OD_COMPLETED : ODStage::FAILED;
    return out;
}

ODResult RunODPipeline(const ODRequest& request,
                       CameraManager& cam_manager,
                       IMUManager& imu_manager,
                       InferenceManager& inference_manager)
{
    ODRequest run_request = request;
    ODResult out;

    try {
        if (run_request.dataset_config.capture_start_time == 0) {
            run_request.dataset_config.capture_start_time = timing::GetCurrentTimeMs();
        }
        auto dataset = DatasetManager::Create(run_request.dataset_config, DATASET_KEY_OD,
                                              cam_manager, imu_manager, inference_manager);
        if (!dataset->StartCollection()) {
            out.code = ErrorCode::ODMEAS_NOT_VALID;
            out.stage = ODStage::FAILED;
            return out;
        }
        while (dataset->Running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        run_request.dataset_folder = dataset->GetDatasetFolder();
        DatasetManager::StopDatasetManager(DATASET_KEY_OD);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("RunODPipeline: failed to capture dataset: {}", e.what());
        out.code = ErrorCode::ODMEAS_NOT_VALID;
        out.stage = ODStage::FAILED;
        return out;
    }

    return RunODOnDataset(run_request);
}
