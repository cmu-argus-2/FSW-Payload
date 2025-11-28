#include "core/data_handling.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>  
#include <fstream>
#include <array>


namespace DH // Data Handling
{


bool INIT_DATA_FOLDER_TREE = false;

bool MakeNewDirectory(std::string_view directory_path)
{
    bool success = false;
    if (fs::exists(directory_path)) 
    {
        SPDLOG_DEBUG("Folder {} already exists.", directory_path);
        success = true;
    } 
    else if (fs::create_directory(directory_path)) 
    {
        SPDLOG_INFO("Folder created: {}.", directory_path);
        success = true;
    } else 
    {
        SPDLOG_CRITICAL("Failed to create folder: {}.", directory_path);
    }    

    return success;

}


long GetFileSize(std::string_view file_path) // should maybe return a pair result / error
{
    struct stat stat_buf;
    int rc = stat(std::string(file_path).c_str(), &stat_buf);

    if (rc == -1) 
    {
        SPDLOG_ERROR("Failed to get file size for {}: {}", file_path, strerror(errno));
        LogError(EC::FILE_DOES_NOT_EXIST);
        return -1LL;
    }

    return stat_buf.st_size;
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
        DATASETS_FOLDER,
        RESULTS_FOLDER,
        LOGGING_FOLDER,
        COMMS_FOLDER
    };

    bool success = true;
    for (const auto& dir : directories) 
    {
        if (!MakeNewDirectory(dir)) {
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
    // TODO retry if failure
}

std::string StoreFrameToDisk(Frame& frame, std::string_view target_folder)
{
    return StoreRawImgToDisk(frame.GetTimestamp(), frame.GetCamID(), frame.GetImg(), target_folder);
}
    

std::string StoreRawImgToDisk(std::uint64_t timestamp, int cam_id, const cv::Mat& img, std::string_view target_folder)
{
    std::ostringstream oss;
    oss << target_folder << "raw" << DELIMITER << timestamp << DELIMITER << cam_id << ".png";
    std::string file_path = oss.str();
    cv::imwrite(file_path, img);
    SPDLOG_INFO("Saved img to disk: {}", file_path);
    SPDLOG_INFO("File size: {} bytes", GetFileSize(file_path));
    SPDLOG_DEBUG("Total size of folder {}: {} bytes", target_folder, GetDirectorySize(target_folder));
    SPDLOG_DEBUG("Number of files in folder {}: {}", target_folder, CountFilesInDirectory(target_folder));

    return file_path; // return value optimized 
}


// Returns true if the latest file is found, false otherwise
bool GetLatestRawFilePath(std::filesystem::directory_entry& latest_file)
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
    SPDLOG_DEBUG("Start: {}, End: {}", start, end);
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

    if (!GetLatestRawFilePath(latest_file)) 
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

bool ReadImageFromDisk(const std::string& file_path, Frame& frame_out)
{
    // Load the image into memory
    cv::Mat img = cv::imread(file_path, cv::IMREAD_COLOR); // Load the image from the file
    if (img.empty()) 
    {
        SPDLOG_ERROR("Failed to load image from disk: {}", file_path);
        LogError(EC::FILE_NOT_FOUND);
        return false;
    }

    // TODO: optional metadata extraction
    frame_out.Update(-1, img, 0);

    SPDLOG_INFO("Image loaded successfully from disk: {}", file_path);
    return true;
}

EC ReadFileChunk(std::string_view file_path, uint32_t start_byte, uint32_t length, std::vector<uint8_t>& data_out)
{
    data_out.resize(length, 0); 

    std::ifstream file(file_path.data(), std::ios::binary);
    if (!file.is_open()) 
    {
        LogError(EC::FILE_NOT_FOUND);
        return EC::FILE_NOT_FOUND;
    }
        
    // TODO: could add a check if start byte is > file size
    if (start_byte >= GetFileSize(file_path)) 
    {
        SPDLOG_ERROR("Start byte {} is out of range for file {}", start_byte, file_path);
        LogError(EC::START_BYTE_OUT_OF_RANGE);
        return EC::START_BYTE_OUT_OF_RANGE;
    }

    file.seekg(start_byte, std::ios::beg);
    file.read(reinterpret_cast<char*>(data_out.data()), length);
    data_out.resize(static_cast<uint32_t>(file.gcount())); // trim to actual bytes read

    return EC::OK;
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

bool WriteFixedPacketFile(const std::string& output_path, const std::vector<std::vector<uint8_t>>& payloads)
{
    static const std::array<uint8_t, DH_FILE_HEADER_SIZE> MAGIC = {'D', 'H', 'G', 'E', 'N'};

    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) 
    {
        SPDLOG_ERROR("Failed to open output file: {}", output_path);
        LogError(EC::FILE_NOT_FOUND);
        return false;
    }

    file.write(reinterpret_cast<const char*>(MAGIC.data()), MAGIC.size());

    std::array<uint8_t, DH_FIXED_PACKET_SIZE> packet_buf{};
    std::size_t idx = 0;
    for (const auto& payload : payloads)
    {
        if (payload.size() > DH_MAX_PAYLOAD_SIZE)
        {
            SPDLOG_ERROR("Payload {} too large for data handler format: {} bytes > {}", idx, payload.size(), DH_MAX_PAYLOAD_SIZE);
            return false;
        }

        packet_buf.fill(0);
        const uint16_t payload_len = static_cast<uint16_t>(payload.size());
        packet_buf[0] = static_cast<uint8_t>((payload_len >> 8) & 0xFF);
        packet_buf[1] = static_cast<uint8_t>(payload_len & 0xFF);
        std::copy(payload.begin(), payload.end(), packet_buf.begin() + DH_PACKET_HEADER_SIZE);

        file.write(reinterpret_cast<const char*>(packet_buf.data()), packet_buf.size());
        ++idx;
    }

    file.flush();
    return true;
}

bool ReadFixedPacketFile(const std::string& input_path, std::vector<std::vector<uint8_t>>& payloads_out)
{
    static const std::array<uint8_t, DH_FILE_HEADER_SIZE> MAGIC = {'D', 'H', 'G', 'E', 'N'};

    payloads_out.clear();

    std::ifstream file(input_path, std::ios::binary);
    if (!file.is_open()) 
    {
        SPDLOG_ERROR("Failed to open input file: {}", input_path);
        LogError(EC::FILE_NOT_FOUND);
        return false;
    }

    std::array<char, DH_FILE_HEADER_SIZE> header{};
    file.read(header.data(), header.size());
    const bool is_magic = (file.gcount() == static_cast<std::streamsize>(header.size()) &&
                           std::equal(header.begin(), header.end(), MAGIC.begin()));

    if (!is_magic)
    {
        // Treat as raw file split into DH_MAX_PAYLOAD_SIZE chunks
        file.clear();
        file.seekg(0, std::ios::beg);
        while (true)
        {
            std::vector<uint8_t> chunk(DH_MAX_PAYLOAD_SIZE, 0);
            file.read(reinterpret_cast<char*>(chunk.data()), DH_MAX_PAYLOAD_SIZE);
            const std::streamsize read_bytes = file.gcount();
            if (read_bytes <= 0) break;

            chunk.resize(static_cast<std::size_t>(read_bytes));
            payloads_out.push_back(std::move(chunk));
        }
        return true;
    }

    std::size_t idx = 0;
    while (true)
    {
        std::array<uint8_t, DH_FIXED_PACKET_SIZE> packet_buf{};
        file.read(reinterpret_cast<char*>(packet_buf.data()), packet_buf.size());
        const std::streamsize read_bytes = file.gcount();
        if (read_bytes == 0) break; // EOF

        if (read_bytes < DH_PACKET_HEADER_SIZE)
        {
            SPDLOG_ERROR("Truncated packet header at index {} in {}", idx, input_path);
            return false;
        }

        const uint16_t payload_len = static_cast<uint16_t>(packet_buf[0]) << 8 | packet_buf[1];
        if (payload_len == 0 || payload_len > DH_MAX_PAYLOAD_SIZE)
        {
            SPDLOG_ERROR("Invalid payload length {} at index {} in {}", payload_len, idx, input_path);
            return false;
        }

        if (payload_len > static_cast<uint16_t>(std::max<std::streamsize>(0, read_bytes - DH_PACKET_HEADER_SIZE)))
        {
            SPDLOG_ERROR("Payload length {} exceeds bytes read {} at index {} in {}", payload_len, read_bytes, idx, input_path);
            return false;
        }

        payloads_out.emplace_back(packet_buf.begin() + DH_PACKET_HEADER_SIZE,
                                  packet_buf.begin() + DH_PACKET_HEADER_SIZE + payload_len);

        if (read_bytes < static_cast<std::streamsize>(DH_FIXED_PACKET_SIZE))
        {
            SPDLOG_WARN("Partial packet read at index {} ({} bytes) in {}", idx, read_bytes, input_path);
            break;
        }

        ++idx;
    }

    return true;
}

void EmptyCommsFolder()
{
    for (const auto& entry : std::filesystem::directory_iterator(COMMS_FOLDER)) 
    {
        std::filesystem::remove_all(entry.path());
    }
}

std::string CopyFrameToCommsFolder(Frame& frame)
{
    if (CountFilesInDirectory(COMMS_FOLDER) > 0)
    {
       SPDLOG_WARN("Overwriting file in comms folder.");
        EmptyCommsFolder();
    }
    // Store the image in the comms folder
    return StoreRawImgToDisk(frame.GetTimestamp(), frame.GetCamID(), frame.GetImg(), COMMS_FOLDER);
}

EC GetCommsFilePath(std::string& path_out) 
{
    std::filesystem::directory_entry latest_file;
    if (!GetLatestRawFilePath(latest_file)) {
        SPDLOG_WARN("No files found in the comms folder.");
        LogError(EC::FILE_NOT_FOUND);
        return EC::FILE_NOT_FOUND;
    }

    path_out = latest_file.path().string();
    return EC::OK;
}

} // namespace DH
