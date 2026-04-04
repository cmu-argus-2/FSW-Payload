#include "commands.hpp"
#include "messages.hpp"
#include "payload.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"
#include "telemetry/telemetry.hpp"
#include "vision/dataset_manager.hpp"
#include "core/errors.hpp"
#include "communication/comms.hpp"
#include "communication/tilepack.hpp"
#include <opencv2/opencv.hpp>

#include <array>
#include <chrono>
#include <thread>
#include <filesystem>

#define DATASET_KEY_CMD "CMD"

// Command functions array definition
std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS =
    {
        ping_ack,                  // PING_ACK
        shutdown,                  // SHUTDOWN
        request_telemetry,         // REQUEST_TELEMETRY
        enable_cameras,            // ENABLE_CAMERAS
        disable_cameras,           // DISABLE_CAMERAS
        capture_images,            // CAPTURE_IMAGES
        start_capture_dataset,     // START_CAPTURE_DATASET
        stop_capture_dataset,      // STOP_CAPTURE_DATASET
        request_storage_info,      // REQUEST_STORAGE_INFO
        request_image,             // REQUEST_IMAGE
        request_next_file_packet,  // REQUEST_NEXT_FILE_PACKET
        clear_storage,             // CLEAR_STORAGE
        ping_od_status,            // PING_OD_STATUS
        run_od,                    // RUN_OD
        request_od_result,         // REQUEST_OD_RESULT
        synchronize_time,          // SYNCHRONIZE_TIME
        full_reset,                // FULL_RESET (no implementation provided)
        debug_display_camera,      // DEBUG_DISPLAY_CAMERA
        debug_stop_display,        // DEBUG_STOP_DISPLAY
        request_next_file_packets  // REQUEST_NEXT_FILE_PACKETS
};

// Define the array of strings mapping CommandID to command names
std::array<std::string_view, COMMAND_NUMBER> COMMAND_NAMES = {
    "PING_ACK",
    "SHUTDOWN",
    "REQUEST_TELEMETRY",
    "ENABLE_CAMERAS",
    "DISABLE_CAMERAS",
    "CAPTURE_IMAGES",
    "START_CAPTURE_DATASET",
    "STOP_CAPTURE_DATASET",
    "REQUEST_STORAGE_INFO",
    "REQUEST_IMAGE",
    "REQUEST_NEXT_FILE_PACKET",
    "CLEAR_STORAGE",
    "PING_OD_STATUS",
    "RUN_OD",
    "REQUEST_OD_RESULT",
    "SYNCHRONIZE_TIME",
    "FULL_RESET",
    "DEBUG_DISPLAY_CAMERA",
    "DEBUG_STOP_DISPLAY",
    "REQUEST_NEXT_FILE_PACKETS"};

void ping_ack([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Received PING_ACK");

    std::vector<uint8_t> transmit_data = {PING_RESP_VALUE};
    std::shared_ptr<Message> msg = CreateMessage(CommandID::PING_ACK, transmit_data);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::PING_ACK);
}

void shutdown([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Initiating Payload shutdown..");
    sys::payload().Stop();

    std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::SHUTDOWN);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::SHUTDOWN);
}

void request_telemetry([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Requesting last telemetry..");

    auto tm = sys::telemetry().GetTmFrame();
    PrintTelemetryFrame(tm);
    std::vector<uint8_t> transmit_data;

    SerializeToBytes(tm.SYSTEM_TIME, transmit_data);
    SerializeToBytes(tm.SYSTEM_UPTIME, transmit_data);
    SerializeToBytes(tm.LAST_EXECUTED_CMD_TIME, transmit_data);
    transmit_data.push_back(tm.LAST_EXECUTED_CMD_ID);
    transmit_data.push_back(tm.PAYLOAD_STATE);
    transmit_data.push_back(tm.ACTIVE_CAMERAS);
    transmit_data.push_back(tm.CAPTURE_MODE);
    for (int i = 0; i < 4; i++)
    {
        transmit_data.push_back(tm.CAM_STATUS[i]);
    }
    transmit_data.push_back(tm.IMU_STATUS);
    SerializeToBytes(tm.IMU_TEMPERATURE, transmit_data);
    transmit_data.push_back(tm.TASKS_IN_EXECUTION);
    transmit_data.push_back(tm.DISK_USAGE);
    transmit_data.push_back(tm.LATEST_ERROR);
    transmit_data.push_back(static_cast<uint8_t>(tm.TEGRASTATS_PROCESS_STATUS));
    transmit_data.push_back(tm.RAM_USAGE);
    transmit_data.push_back(tm.SWAP_USAGE);
    transmit_data.push_back(tm.ACTIVE_CORES);
    for (int i = 0; i < 6; i++)
    {
        transmit_data.push_back(tm.CPU_LOAD[i]);
    }
    transmit_data.push_back(tm.GPU_FREQ);
    transmit_data.push_back(tm.CPU_TEMP);
    transmit_data.push_back(tm.GPU_TEMP);
    SerializeToBytes(tm.VDD_IN, transmit_data);
    SerializeToBytes(tm.VDD_CPU_GPU_CV, transmit_data);
    SerializeToBytes(tm.VDD_SOC, transmit_data);

    std::shared_ptr<Message> msg = CreateMessage(CommandID::REQUEST_TELEMETRY, transmit_data);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_TELEMETRY);
}

void enable_cameras([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Trying to enable all cameras...");
    std::array<bool, NUM_CAMERAS> on_cameras;

    int nb_activated_cams = sys::cameraManager().EnableCameras(on_cameras);
    bool at_least_one_was_enabled = false;
    for (size_t i = 0; i < NUM_CAMERAS; ++i)
    {
        if (on_cameras[i])
        {
            SPDLOG_INFO("Camera {} is enabled.", i);
            at_least_one_was_enabled = true;
        }
    }

    if (!at_least_one_was_enabled)
    {
        SPDLOG_ERROR("No cameras were enabled.");
        // TODO: Get latest error from camera subsystem instead
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::ENABLE_CAMERAS, 0x51);
        sys::payload().TransmitMessage(msg);
        return;
    }

    std::vector<uint8_t> transmit_data;
    transmit_data.push_back(static_cast<uint8_t>(nb_activated_cams));
    for (size_t i = 0; i < NUM_CAMERAS; ++i)
    {
        transmit_data.push_back(static_cast<uint8_t>(on_cameras[i]));
    }

    std::shared_ptr<Message> msg = CreateMessage(CommandID::ENABLE_CAMERAS, transmit_data);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::ENABLE_CAMERAS);
}

void disable_cameras([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Trying to disable all cameras...");
    std::array<bool, NUM_CAMERAS> off_cameras;

    int nb_disabled_cams = sys::cameraManager().DisableCameras(off_cameras);
    bool at_least_one_was_disabled = false;
    for (size_t i = 0; i < NUM_CAMERAS; ++i)
    {
        if (off_cameras[i])
        {
            SPDLOG_INFO("Camera {} is disabled.", i);
            at_least_one_was_disabled = true;
        }
    }

    if (!at_least_one_was_disabled)
    {
        SPDLOG_ERROR("No cameras were disabled.");
        // Get latest error from camera subsystem instead
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::DISABLE_CAMERAS, 0x52); // TODO: define error code
        sys::payload().TransmitMessage(msg);
        return;
    }

    std::vector<uint8_t> transmit_data;
    transmit_data.push_back(static_cast<uint8_t>(nb_disabled_cams));
    for (size_t i = 0; i < NUM_CAMERAS; ++i)
    {
        transmit_data.push_back(static_cast<uint8_t>(off_cameras[i]));
    }

    std::shared_ptr<Message> msg = CreateMessage(CommandID::DISABLE_CAMERAS, transmit_data);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::DISABLE_CAMERAS);
}

void capture_images([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Capturing image now..");

    if (!sys::cameraManager().SendCaptureRequest())
    {
        SPDLOG_ERROR("Failed to prepare cameras for CAPTURE_IMAGES.");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::CAPTURE_IMAGES, 0x51);
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Need to return true or false based on the success of the operations
    // Basically should wait until the camera has captured the image or wait later to send the ocnfirmation and just exit the task?
    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::CAPTURE_IMAGES);
}

void start_capture_dataset(std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Starting dataset capture..");

    // Wire format (13 bytes):
    // [0]     capture_mode          (CAPTURE_MODE enum)
    // [1..2]  max_period            (uint16_t, seconds)
    // [3..4]  target_frame_nb       (uint16_t wire, capped to uint8_t, max MAX_SAMPLES)
    // [5..8]  capture_start_time    (uint32_t, unix timestamp in seconds)
    // [9]     imu_collection_mode   (IMU_COLLECTION_MODE enum)
    // [10]    image_capture_rate    (uint8_t, seconds between captures)
    // [11]    imu_sample_rate       (uint8_t, in 0.1 Hz units, e.g. 250 = 25.0 Hz)
    // [12]    target_processing_stage (ProcessingStage enum)

    if (data.size() < 13)
    {
        SPDLOG_ERROR("START_CAPTURE_DATASET: insufficient data ({} bytes, need 13)", data.size());
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_DATASET, 0x20);
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Validate enum raw bytes before casting — casting an out-of-range byte to an
    // enum class is undefined behaviour.
    if (data[0] < static_cast<uint8_t>(CAPTURE_MODE::PERIODIC) ||
        data[0] > static_cast<uint8_t>(CAPTURE_MODE::PERIODIC_LDMK))
    {
        SPDLOG_ERROR("START_CAPTURE_DATASET: invalid capture_mode byte {}", data[0]);
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_DATASET, 0x20);
        sys::payload().TransmitMessage(msg);
        return;
    }
    if (data[9] > static_cast<uint8_t>(IMU_COLLECTION_MODE::GYRO_MAG_TEMP))
    {
        SPDLOG_ERROR("START_CAPTURE_DATASET: invalid imu_collection_mode byte {}", data[9]);
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_DATASET, 0x20);
        sys::payload().TransmitMessage(msg);
        return;
    }
    if (data[12] > static_cast<uint8_t>(ProcessingStage::LDNeted))
    {
        SPDLOG_ERROR("START_CAPTURE_DATASET: invalid target_processing_stage byte {}", data[12]);
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_DATASET, 0x20);
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Parse fields
    const CAPTURE_MODE       capture_mode             = static_cast<CAPTURE_MODE>(data[0]);
    const double             max_period               = static_cast<double>((data[1] << 8) | data[2]);
    const uint16_t           target_frame_nb_raw      = static_cast<uint16_t>((data[3] << 8) | data[4]);
    const uint32_t           capture_start_time_s     =
        (static_cast<uint32_t>(data[5]) << 24) |
        (static_cast<uint32_t>(data[6]) << 16) |
        (static_cast<uint32_t>(data[7]) << 8)  |
         static_cast<uint32_t>(data[8]);
    const IMU_COLLECTION_MODE imu_collection_mode     = static_cast<IMU_COLLECTION_MODE>(data[9]);
    const uint8_t             image_capture_rate       = data[10];
    const float               imu_sample_rate_hz       = static_cast<float>(data[11]) / 10.0f;
    const ProcessingStage     target_processing_stage  = static_cast<ProcessingStage>(data[12]);

    // target_frame_nb is uint8_t in Dataset (max MAX_SAMPLES = 255); validate before narrowing.
    if (target_frame_nb_raw == 0 || target_frame_nb_raw > MAX_SAMPLES)
    {
        SPDLOG_ERROR("START_CAPTURE_DATASET: target_frame_nb {} out of range [1, {}]",
                     target_frame_nb_raw, MAX_SAMPLES);
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_DATASET, 0x20);
        sys::payload().TransmitMessage(msg);
        return;
    }
    const uint8_t target_frame_nb = static_cast<uint8_t>(target_frame_nb_raw);

    // Clamp start time to now if already in the past
    uint64_t capture_start_time = static_cast<uint64_t>(capture_start_time_s) * 1000ULL;
    const uint64_t now_ms = timing::GetCurrentTimeMs();
    if (capture_start_time < now_ms)
    {
        capture_start_time = now_ms;
    }

    // Semantic validation: period bounds, rate bounds, capture mode consistency
    if (!Dataset::isValidConfiguration(max_period, target_frame_nb, capture_mode, imu_collection_mode,
                                       image_capture_rate, imu_sample_rate_hz, target_processing_stage,
                                       capture_start_time))
    {
        SPDLOG_ERROR("START_CAPTURE_DATASET: parameter validation failed");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_DATASET, 0x20);
        sys::payload().TransmitMessage(msg);
        return;
    }

    auto ds = DatasetManager::GetActiveDatasetManager(DATASET_KEY_CMD);
    if (ds)
    {
        if (ds->Running())
        {
            SPDLOG_ERROR("Dataset already running under key {}, ignoring command", DATASET_KEY_CMD);
            std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_DATASET, 0x21);
            sys::payload().TransmitMessage(msg);
            return;
        }
        DatasetManager::StopDatasetManager(DATASET_KEY_CMD);
    }

    SPDLOG_INFO("Starting dataset collection (mode {}) for {} frames, max period {} s",
                static_cast<uint8_t>(capture_mode), target_frame_nb, max_period);

    try
    {
        ds = DatasetManager::Create(max_period, target_frame_nb, capture_mode, capture_start_time,
                                    imu_collection_mode, image_capture_rate, imu_sample_rate_hz,
                                    target_processing_stage, DATASET_KEY_CMD,
                                    sys::cameraManager(), sys::imuManager(), sys::inferenceManager());
        ds->StartCollection();
    }
    catch (const std::exception &e)
    {
        SPDLOG_ERROR("Failed to start dataset collection: {}", e.what());
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_DATASET, 0x21);
        sys::payload().TransmitMessage(msg);
        return;
    }

    std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::START_CAPTURE_DATASET);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::START_CAPTURE_DATASET);
}

void stop_capture_dataset([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Stopping dataset capture..");

    // sys::cameraManager().SetCaptureMode(CAPTURE_MODE::IDLE);
    // TODO return true or false based on the success of the operations

    auto ds = DatasetManager::GetActiveDatasetManager(DATASET_KEY_CMD);
    if (ds)
    {
        DatasetManager::StopDatasetManager(DATASET_KEY_CMD);

        std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::STOP_CAPTURE_DATASET);
        sys::payload().TransmitMessage(msg);
    }
    else
    {
        // Return (error) ACK telling that no dataset is running
        SPDLOG_ERROR("No dataset collection has been started on the command side");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::STOP_CAPTURE_DATASET, 0x22); // TODO
        sys::payload().TransmitMessage(msg);
    }

    sys::payload().SetLastExecutedCmdID(CommandID::STOP_CAPTURE_DATASET);
}

void request_storage_info([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Requesting storage information...");

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_STORAGE_INFO);
}

void request_image([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Requesting highest value image..");

    // Read the latest stored raw image
    Frame frame;
    bool res = DH::ReadHighestValueStoredRawImg(frame);

    if (!res)
    {
        SPDLOG_ERROR("Failed to read latest stored raw image");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FILE_NOT_AVAILABLE));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Reconstruct the original image path from the frame data
    std::string img_path = std::string(IMAGES_FOLDER) + "raw_" + std::to_string(frame.GetTimestamp()) + "_" + std::to_string(frame.GetCamID()) + ".png";
    std::string abs_img_path = std::filesystem::absolute(img_path).string();
    SPDLOG_INFO("Using image: {}", abs_img_path);

    // Create tilepack encoder and process the image directly from disk
    tilepack::TilepackEncoder encoder(
        1,   // page_id
        640, // target_width
        480, // target_height
        64,  // tile_w
        32,  // tile_h
        30   // jpeg_quality
    );

    if (!encoder.load_image(abs_img_path))
    {
        SPDLOG_ERROR("Failed to load image into tilepack encoder");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FAIL_TO_READ_FILE));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Generate output filename for binary file in comms folder
    std::string bin_file_path = std::string(COMMS_FOLDER) + "img_" + std::to_string(frame.GetTimestamp()) + "_" + std::to_string(frame.GetCamID()) + ".bin";

    // Write the tilepack binary file (data-handler format with 242-byte records)
    if (!encoder.write_radio_file(bin_file_path))
    {
        SPDLOG_ERROR("Failed to write tilepack binary file: {}", bin_file_path);
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FAIL_TO_READ_FILE));
        sys::payload().TransmitMessage(msg);
        return;
    }

    SPDLOG_INFO("Created tilepack binary: {} ({} packets, {} bytes compressed)",
                bin_file_path, encoder.get_total_packets(), encoder.get_compressed_size());

    // Get file size to log info
    long file_size = DH::GetFileSize(bin_file_path);
    if (file_size < 0)
    {
        SPDLOG_ERROR("Failed to get file size for: {}", bin_file_path);
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FILE_NOT_FOUND));
        sys::payload().TransmitMessage(msg);
        return;
    }

    SPDLOG_INFO("Binary image file: {} ({} bytes)", bin_file_path, file_size);

    // Set the file transfer manager to transfer the binary file directly
    // FileTransferManager will read it in 240-byte chunks (which will include headers + payloads from the binary file)
    FileTransferManager::Reset();
    EC err = FileTransferManager::PopulateMetadata(bin_file_path);

    if (err != EC::OK)
    {
        SPDLOG_ERROR("Failed to populate metadata for file transfer.");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FILE_NOT_AVAILABLE));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Send the ACK message
    std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::REQUEST_IMAGE);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_IMAGE);
}

void request_next_file_packet(std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Requesting next file packet..");

    uint16_t requested_packet_nb = (data[0] << 8) | data[1];
    SPDLOG_INFO("Requested packet number: {}", requested_packet_nb);

    // NEED TO START BY 1 so we can filter invalid commands from random commands filled with zeros
    if (requested_packet_nb == 0)
    {
        SPDLOG_ERROR("Invalid data size (0) for REQUEST_NEXT_FILE_PACKET command");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_NEXT_FILE_PACKET, to_uint8(EC::INVALID_COMMAND_ARGUMENTS));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // check if a file has been readied -> NO_FILE_READY
    if (!FileTransferManager::active_transfer())
    {
        SPDLOG_ERROR("No file available for transfer.");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_NEXT_FILE_PACKET, to_uint8(EC::NO_FILE_READY));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // check if requested seq number is valid -> NO_MORE_PACKET_FOR_FILE
    if (requested_packet_nb > FileTransferManager::total_seq_count())
    {
        SPDLOG_ERROR("Requested packet number {} is out of range.", requested_packet_nb);
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_NEXT_FILE_PACKET, to_uint8(EC::NO_MORE_PACKET_FOR_FILE)); // Important!
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Take the corresponding chunk of data, load it to ram, and send it
    std::vector<uint8_t> transmit_data;
    transmit_data.reserve(Packet::MAX_DATA_LENGTH);

    // GrabFileChunk handles both DH and non-DH files:
    // - DH files: extracts payload (≤240 bytes) from 242-byte records
    // - Non-DH files: reads raw data in 240-byte chunks
    EC err = FileTransferManager::GrabFileChunk(requested_packet_nb, transmit_data);
    if (err != EC::OK)
    {
        SPDLOG_ERROR("Failed to grab file chunk.");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_NEXT_FILE_PACKET, to_uint8(err));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // transmit_data now contains the payload only (≤240 bytes)
    // CreateMessage will pad to 240 bytes and add CRC to create 247-byte UART packet
    std::shared_ptr<Message> msg = CreateMessage(CommandID::REQUEST_NEXT_FILE_PACKET, transmit_data, requested_packet_nb);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_NEXT_FILE_PACKET);
}

void clear_storage([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Clearing storage..");

    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::CLEAR_STORAGE);
}

void ping_od_status([[maybe_unused]] std::vector<uint8_t> &data)
{

    SPDLOG_INFO("Pinging the status of the orbit determination process...");

    // TODO
    OD_STATE od_state = sys::od().GetState();
    std::vector<uint8_t> transmit_data;

    // Add the current ststae to pcket
    transmit_data.push_back(static_cast<uint8_t>(od_state));

    // Based on the state, return more information
    switch (od_state)
    {
    case OD_STATE::IDLE:
    {
        break;
    }
    case OD_STATE::INIT:
    {
        auto ds = DatasetManager::GetActiveDatasetManager(DATASET_KEY_OD);
        if (ds) // if it exists
        {
            DatasetProgress ds_progress = ds->QueryProgress();
            uint16_t nb_frames = ds_progress.current_frames;
            uint8_t completion = static_cast<uint8_t>(ds_progress.completion);
            transmit_data.push_back(completion);
            SerializeToBytes(nb_frames, transmit_data);
        }
        else
        {
            // Return (error) ACK telling that no dataset is running
            SPDLOG_ERROR("No dataset collection has been started on the command side");
            // TODO
        }
        break;
    }
    case OD_STATE::BATCH_OPT:
    {
        break;
    }
    }

    std::shared_ptr<Message> msg = CreateMessage(CommandID::PING_OD_STATUS, transmit_data);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::PING_OD_STATUS);
}

void run_od([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Running orbit determination..");

    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::RUN_OD);
}

void request_od_result([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Requesting OD result..");

    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_OD_RESULT);
}

void synchronize_time([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Synchronizing time..");
    // TODO
    // SHOULD SHUTDOWN and REBOOT
    sys::payload().SetLastExecutedCmdID(CommandID::SYNCHRONIZE_TIME);
}

void full_reset([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Performing full reset..");

    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::FULL_RESET);
}

void debug_display_camera([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Activating the display of the camera");

    if (sys::cameraManager().GetDisplayFlag() == true)
    {
        SPDLOG_WARN("Display already active");
        std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::DEBUG_DISPLAY_CAMERA);
        sys::payload().TransmitMessage(msg);
        return;
    }

    sys::cameraManager().SetDisplayFlag(true);
    // the command is already by a thread of the ThreadPool so no need to spawn a new thread here
    // This will block the thread until the display flag is set to false or all cameras are turned off
    sys::cameraManager().RunDisplayLoop();

    std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::DEBUG_DISPLAY_CAMERA);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::DEBUG_DISPLAY_CAMERA);
}

void debug_stop_display([[maybe_unused]] std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Stopping the display of the camera");
    sys::cameraManager().SetDisplayFlag(false);

    // return ACK
    std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::DEBUG_STOP_DISPLAY);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::DEBUG_STOP_DISPLAY);
}
void request_next_file_packets(std::vector<uint8_t> &data)
{
    SPDLOG_INFO("Requesting next file packets (batch)..");

    if (data.size() < 3)
    {
        SPDLOG_ERROR("Invalid data size for REQUEST_NEXT_FILE_PACKETS command");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(
            CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(EC::INVALID_COMMAND_ARGUMENTS));
        sys::payload().TransmitMessage(msg);
        return;
    }

    uint16_t start_packet_nb = (data[0] << 8) | data[1];
    uint8_t count = data[2];

    SPDLOG_INFO("Requested batch: start packet={}, count={}", start_packet_nb, count);

    // NEED TO START BY 1 so we can filter invalid commands from random commands filled with zeros
    if (start_packet_nb == 0)
    {
        SPDLOG_ERROR("Invalid start packet number (0) for REQUEST_NEXT_FILE_PACKETS command");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(
            CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(EC::INVALID_COMMAND_ARGUMENTS));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // check if a file has been readied -> NO_FILE_READY
    if (!FileTransferManager::active_transfer())
    {
        SPDLOG_ERROR("No file available for transfer.");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(
            CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(EC::NO_FILE_READY));
        sys::payload().TransmitMessage(msg);
        return;
    }

    uint16_t total_packets = FileTransferManager::total_seq_count();

    for (uint8_t i = 0; i < count; i++)
    {
        uint16_t current_packet = start_packet_nb + i;

        if (current_packet > total_packets)
        {
            SPDLOG_INFO("Reached end of file at packet {}", current_packet);
            std::shared_ptr<Message> msg = CreateErrorAckMessage(
                CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(EC::NO_MORE_PACKET_FOR_FILE));
            sys::payload().TransmitMessage(msg);
            return;
        }

        std::vector<uint8_t> transmit_data;
        transmit_data.reserve(Packet::MAX_DATA_LENGTH);

        // GrabFileChunk handles both DH and non-DH files:
        // - DH files: extracts payload (≤240 bytes) from 242-byte records
        // - Non-DH files: reads raw data in 240-byte chunks
        EC err = FileTransferManager::GrabFileChunk(current_packet, transmit_data);
        if (err != EC::OK)
        {
            SPDLOG_ERROR("Failed to grab file chunk for packet {}", current_packet);
            std::shared_ptr<Message> msg = CreateErrorAckMessage(
                CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(err));
            sys::payload().TransmitMessage(msg);
            return;
        }

        // transmit_data now contains the payload only (≤240 bytes)
        // CreateMessage will pad to 240 bytes and add CRC to create 247-byte UART packet
        std::shared_ptr<Message> msg = CreateMessage(
            CommandID::REQUEST_NEXT_FILE_PACKETS, transmit_data, current_packet);
        sys::payload().TransmitMessage(msg);

        // Add delay between packets to prevent UART buffer overflow on mainboard
        // CircuitPython polls at 1ms intervals, need sufficient time for processing
        // if (i < count - 1)  // Don't delay after last packet
        // {
        //     std::this_thread::sleep_for(std::chrono::milliseconds(15));
        // }
    }

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_NEXT_FILE_PACKETS);
}