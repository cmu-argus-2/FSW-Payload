#include "core/data_handling.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>  

namespace DH // Data Handling
{


bool INIT_DATA_FOLDER_TREE = false;

bool make_new_directory(std::string_view directory_path)
{
    bool success = false;
    if (std::filesystem::exists(directory_path)) 
    {
        SPDLOG_INFO("Folder already exists.");
        success = true;
    } 
    else if (std::filesystem::create_directory(directory_path)) 
    {
        SPDLOG_INFO("Data folder created.");
        success = true;
    } else {
        SPDLOG_CRITICAL("Failed to create data folder.");
    }    

    return success;

}


long GetFileSize(std::string_view file_path) 
{
    struct stat stat_buf;
    int rc = stat(std::string(file_path).c_str(), &stat_buf);
    return rc == 0LL ? stat_buf.st_size : -1LL;
}


long GetDirectorySize(std::string_view directory_path) 
{
    long total_size = 0;
    struct stat stat_buf;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
        if (entry.is_regular_file()) 
        {  
            if (stat(entry.path().c_str(), &stat_buf) == 0) {
                total_size += stat_buf.st_size;
            }
        }
    }
    return total_size;
}

int CountFilesInDirectory(std::string_view directory_path) 
{
    int file_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) 
    {
        if (entry.is_regular_file()) {
            ++file_count;
        }
    }
    return file_count;
}


bool InitializeDataStorage()
{
    if (INIT_DATA_FOLDER_TREE) {
        SPDLOG_INFO("Data folder tree already initialized.");
        return true;
    }

    // Construct the full paths using the root and subdirectory names
    std::vector<std::string> directories = {
        ROOT_DATA_FOLDER,
        IMAGES_FOLDER,
        TELEMETRY_FOLDER,
        EXPERIMENTS_FOLDER,
        LOGGING_FOLDER
    };

    bool success = true;
    for (const auto& dir : directories) 
    {
        if (!make_new_directory(dir)) {
            SPDLOG_CRITICAL("Failed to create directory: {}", dir);
            success = false;
        }
    }

    INIT_DATA_FOLDER_TREE = success;

    if (success) 
    {
        SPDLOG_INFO("Data folder tree initialized successfully.");
    }

    return success;
}



void StoreRawImgToDisk(std::uint64_t timestamp, int cam_id, const cv::Mat& img)
{
    std::ostringstream oss;
    oss << IMAGES_FOLDER << "raw" << DELIMITER << timestamp << DELIMITER << cam_id << ".png";
    const std::string& file_path = oss.str();
    cv::imwrite(file_path, img);
    SPDLOG_INFO("Saved img to disk: {}", file_path);
    SPDLOG_INFO("File size: {} bytes", GetFileSize(file_path));
    SPDLOG_DEBUG("Total size of images folder: {} bytes", GetDirectorySize(IMAGES_FOLDER));
    SPDLOG_DEBUG("Number of images in images folder: {}", CountFilesInDirectory(IMAGES_FOLDER));
}


// Returns true if the latest file is found, false otherwise
bool GetLatestRawImgPath(std::filesystem::directory_entry& latest_file)
{

    bool found = false;    
    for (const auto& entry : std::filesystem::directory_iterator(IMAGES_FOLDER)) 
    {
        if (entry.is_regular_file()) 
        {
            
            if (!found || entry.last_write_time() > latest_file.last_write_time()) 
            {
                latest_file = entry;
                found = true;
            }
        }
    }


    SPDLOG_INFO("Latest file: {}", latest_file.path().string());
    return found;
}


// We know the structure of the filename already (raw_timestamp_camid.png)
void SplitRawImgPath(const std::string& input, std::uint64_t& timestamp, int& cam_id) 
{

    size_t start = 4; // after "raw" + delimiter
    size_t end = 0;

    // Extract the timestamp and camera ID
    end = input.find(DELIMITER, start);
    SPDLOG_INFO("Start: {}, End: {}", start, end);
    SPDLOG_DEBUG("Timestamp substring: {}", input.substr(start, end - start));
    timestamp = std::stoull(input.substr(start, end - start));
    //timestamp = std::strtoull(input.substr(start, end - start).c_str(), nullptr, 10);
    //timestamp = static_cast<uint64_t>(atoll(input.substr(start, end - start).c_str()));

    
    start = end + 1;
    
    // Add the last part (before .png)
    end = input.rfind('.');
    cam_id = std::stoi(input.substr(start, end - start));
}



bool ReadLatestStoredRawImg(Frame& frame)
{

    std::filesystem::directory_entry latest_file;

    if (!GetLatestRawImgPath(latest_file)) 
    {
        SPDLOG_WARN("No files found in the directory.");
        return false;
    }

    std::string latest_file_path = latest_file.path().string();
    std::string infolder_latest_file_path = latest_file_path.substr(latest_file_path.find_last_of("/\\") + 1);

    // Extract timestamp and id from filename
    std::uint64_t extracted_timestamp;
    int extracted_cam_id;
    SplitRawImgPath(infolder_latest_file_path, extracted_timestamp, extracted_cam_id);
    SPDLOG_DEBUG("Timestamp: {}, Camera ID: {}", extracted_timestamp, extracted_cam_id);


    // Load the image into memory
    // CV_8UC3 - 8-bit unsigned integer matrix/image with 3 channels
    cv::Mat extracted_img = cv::imread(latest_file_path, cv::IMREAD_COLOR); // Load the image from the file
    if (extracted_img.empty()) 
    {
        SPDLOG_ERROR("Failed to load image from disk.");
        return false;
    }

    frame.Update(extracted_cam_id, extracted_img, extracted_timestamp);

    SPDLOG_INFO("Image loaded successfully.");
    return true;

}


int CountRawImgNumberOnDisk()
{
    return CountFilesInDirectory(IMAGES_FOLDER);
}


int GetTotalDiskUsage() 
{
    // "df -h | grep '/dev/nvme' | awk '{print $5}' | head -n 1"

    std::error_code ec; // capture errors instead of throwing exceptions
    std::filesystem::space_info info = std::filesystem::space(ROOT_DISK, ec);

    if (ec)
    {
        SPDLOG_ERROR("Error accessing disk space for path '{}': {}", ROOT_DISK, ec.message());
        return -1; // Return -1 to indicate an error
    }

    double usage_percentage = (1.0 - static_cast<double>(info.free) / info.capacity) * 100.0;
    return static_cast<int>(std::round(usage_percentage)); // round to nearest integer

}


} // namespace DH
