#include "commands.hpp"
#include "messages.hpp"
#include "payload.hpp"
#include "core/data_handling.hpp"
#include "telemetry/telemetry.hpp"
#include "vision/dataset.hpp"
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
    ping_ack, // PING_ACK
    shutdown, // SHUTDOWN
    request_telemetry, // REQUEST_TELEMETRY
    enable_cameras, // ENABLE_CAMERAS
    disable_cameras, // DISABLE_CAMERAS
    capture_images, // CAPTURE_IMAGES
    start_capture_images_periodically, // START_CAPTURE_IMAGES_PERIODICALLY
    stop_capture_images, // STOP_CAPTURE_IMAGES
    request_storage_info, // REQUEST_STORAGE_INFO
    request_image, // REQUEST_IMAGE
    request_next_file_packet, // REQUEST_NEXT_FILE_PACKET
    clear_storage, // CLEAR_STORAGE
    ping_od_status, // PING_OD_STATUS
    run_od, // RUN_OD
    request_od_result, // REQUEST_OD_RESULT
    synchronize_time, // SYNCHRONIZE_TIME
    full_reset, // FULL_RESET (no implementation provided)
    debug_display_camera, // DEBUG_DISPLAY_CAMERA
    debug_stop_display, // DEBUG_STOP_DISPLAY
    request_next_file_packets // REQUEST_NEXT_FILE_PACKETS
};

// Define the array of strings mapping CommandID to command names
std::array<std::string_view, COMMAND_NUMBER> COMMAND_NAMES = {
    "PING_ACK",
    "SHUTDOWN",
    "REQUEST_TELEMETRY",
    "ENABLE_CAMERAS",
    "DISABLE_CAMERAS",
    "CAPTURE_IMAGES",
    "START_CAPTURE_IMAGES_PERIODICALLY",
    "STOP_CAPTURE_IMAGES",
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
    "REQUEST_NEXT_FILE_PACKETS"
};

void ping_ack([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Received PING_ACK");

    std::vector<uint8_t> transmit_data = {PING_RESP_VALUE};
    std::shared_ptr<Message> msg = CreateMessage(CommandID::PING_ACK, transmit_data);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::PING_ACK);
}

void shutdown([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Initiating Payload shutdown..");
    sys::payload().Stop();

    std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::SHUTDOWN);
    sys::payload().TransmitMessage(msg);
    
    sys::payload().SetLastExecutedCmdID(CommandID::SHUTDOWN);
}


void request_telemetry([[maybe_unused]] std::vector<uint8_t>& data)
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


void enable_cameras([[maybe_unused]] std::vector<uint8_t>& data)
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

void disable_cameras([[maybe_unused]] std::vector<uint8_t>& data)
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

void capture_images([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Capturing image now..");

    sys::cameraManager().SendCaptureRequest();

    // Need to return true or false based on the success of the operations
    // Basically should wait until the camera has captured the image or wait later to send the ocnfirmation and just exit the task? 
    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::CAPTURE_IMAGES);
}

void start_capture_images_periodically([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Starting capture images every X seconds..");

    /*if (!data.empty()) {
        uint8_t period = data[0]; // Period in seconds
        sys::cameraManager().SetPeriodicCaptureRate(period);

        if (data.size() > 1) {
            uint8_t frames = data[1]; // Number of frames to capture
            sys::cameraManager().SetPeriodicFramesToCapture(frames);
        }
    }

    sys::cameraManager().SetCaptureMode(CAPTURE_MODE::PERIODIC);*/

    // 1 byte for type
    // 2 bytes period (for now)
    // 2 bytes nb_frames
    DatasetType dataset_type = static_cast<DatasetType>(data[0]);
    double period = static_cast<double>((data[1] << 8) | data[2]);
    uint16_t nb_frames = (data[3] << 8) | data[4];


    if (period == 0.0 || nb_frames == 0)
    {
        // TODO: Send Error ACK
        uint8_t ERR_PERIODIC = 0x20; // TODO define elsewhere
        SPDLOG_ERROR("Invalid data for START_CAPTURE_IMAGES_PERIODICALLY command");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_IMAGES_PERIODICALLY, 0x20);
        sys::payload().TransmitMessage(msg);
        return;
    }

    auto ds = DatasetManager::GetActiveDataset(DATASET_KEY_CMD);

    if (ds) // if already exists
    {
        // need to ensure it's actually running
        if (ds->Running())
        {
            // if running: TODO: return ERROR ACK saying that a dataset is already running
            // If completed, stop it then too
        }
        else
        {
            ds->StopDataset(DATASET_KEY_CMD); // remove it (will create a new one)
        }
    }

    // Create a new Dataset
    SPDLOG_INFO("Starting dataset collection (type {}) for {} frames at a period of {} seconds.", static_cast<uint8_t>(dataset_type), nb_frames, period);

    try
    {
        ds = DatasetManager::Create(period, nb_frames, dataset_type, DATASET_KEY_CMD);
        ds->StartCollection();
    }
    catch(const std::exception& e)
    {
        
        SPDLOG_ERROR("Failed to start dataset collection: {}", e.what());
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::START_CAPTURE_IMAGES_PERIODICALLY, 0x21); // TODO example error code
        sys::payload().TransmitMessage(msg);
    }
    
    // All good
    std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::START_CAPTURE_IMAGES_PERIODICALLY);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::START_CAPTURE_IMAGES_PERIODICALLY);
}

void stop_capture_images([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Stopping capture images..");

    // sys::cameraManager().SetCaptureMode(CAPTURE_MODE::IDLE);
    // TODO return true or false based on the success of the operations

    auto ds = DatasetManager::GetActiveDataset(DATASET_KEY_CMD);
    if (ds) // if it exists
    {
        ds->StopCollection();
        DatasetProgress ds_progress = ds->QueryProgress();
        // include in message statistics about the collected data (how many frame)
        uint16_t nb_frames = ds_progress.current_frames;
        uint8_t completion = static_cast<uint8_t>(ds_progress.completion);

        // Stop dataset process and remove from registry
        ds->StopDataset(DATASET_KEY_CMD);

        // Create success ACK message with stats
        std::vector<uint8_t> transmit_data;
        transmit_data.push_back(ACK_SUCCESS);
        transmit_data.push_back(completion);
        SerializeToBytes(nb_frames, transmit_data);
        std::shared_ptr<Message> msg = CreateMessage(CommandID::STOP_CAPTURE_IMAGES, transmit_data);
        sys::payload().TransmitMessage(msg);
    }
    else
    {
        // Return (error) ACK telling that no dataset is running
        SPDLOG_ERROR("No dataset collection has been started on the command side");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::STOP_CAPTURE_IMAGES, 0x22); // TODO
        sys::payload().TransmitMessage(msg);

    }

    sys::payload().SetLastExecutedCmdID(CommandID::STOP_CAPTURE_IMAGES);
}

void request_storage_info([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Requesting storage information...");

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_STORAGE_INFO);
}

void request_image([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Requesting highest value image..");

    // Read the latest stored raw image
    Frame frame;
    bool res = DH::ReadLatestStoredRawImg(frame);
    
    if (!res)
    {
        SPDLOG_ERROR("Failed to read latest stored raw image");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FILE_NOT_AVAILABLE));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Get the image from the frame
    const cv::Mat& img = frame.GetImg();
    if (img.empty())
    {
        SPDLOG_ERROR("Frame contains empty image");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FILE_NOT_AVAILABLE));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Save the image temporarily so tilepack can load it
    std::string temp_img_path = std::string(COMMS_FOLDER) + "temp_img_" + std::to_string(frame.GetTimestamp()) + "_" + std::to_string(frame.GetCamID()) + ".png";
    if (!cv::imwrite(temp_img_path, img))
    {
        SPDLOG_ERROR("Failed to save temporary image: {}", temp_img_path);
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FAIL_TO_READ_FILE));
        sys::payload().TransmitMessage(msg);
        return;
    }
    SPDLOG_INFO("Saved temporary image: {}", temp_img_path);

    // Create tilepack encoder and process the image
    tilepack::TilepackEncoder encoder(
        1,  // page_id
        640,  // target_width
        480,  // target_height
        64,   // tile_w
        32,   // tile_h
        30    // jpeg_quality
    );

    if (!encoder.load_image(temp_img_path))
    {
        SPDLOG_ERROR("Failed to load image into tilepack encoder");
        std::filesystem::remove(temp_img_path);  // Clean up temp file
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
        std::filesystem::remove(temp_img_path);  // Clean up temp file
        std::shared_ptr<Message> msg = CreateErrorAckMessage(CommandID::REQUEST_IMAGE, to_uint8(EC::FAIL_TO_READ_FILE));
        sys::payload().TransmitMessage(msg);
        return;
    }

    // Clean up temporary image file
    std::filesystem::remove(temp_img_path);

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

void request_next_file_packet(std::vector<uint8_t>& data)
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

void clear_storage([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Clearing storage..");

    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::CLEAR_STORAGE);
}

void ping_od_status([[maybe_unused]] std::vector<uint8_t>& data)
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
            auto ds = DatasetManager::GetActiveDataset(DATASET_KEY_OD);
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

void run_od([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Running orbit determination..");

    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::RUN_OD);
}

void request_od_result([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Requesting OD result..");

    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_OD_RESULT);
}


void synchronize_time([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Synchronizing time..");
    // TODO
    // SHOULD SHUTDOWN and REBOOT
    sys::payload().SetLastExecutedCmdID(CommandID::SYNCHRONIZE_TIME);
}

void full_reset([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Performing full reset..");

    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::FULL_RESET);
}


void debug_display_camera([[maybe_unused]] std::vector<uint8_t>& data)
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

void debug_stop_display([[maybe_unused]] std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Stopping the display of the camera");
    sys::cameraManager().SetDisplayFlag(false);

    // return ACK
    std::shared_ptr<Message> msg = CreateSuccessAckMessage(CommandID::DEBUG_STOP_DISPLAY);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::DEBUG_STOP_DISPLAY);
}
void request_next_file_packets(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Requesting next file packets (batch)..");
    
    if (data.size() < 3)
    {
        SPDLOG_ERROR("Invalid data size for REQUEST_NEXT_FILE_PACKETS command");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(
            CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(EC::INVALID_COMMAND_ARGUMENTS)
        );
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
            CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(EC::INVALID_COMMAND_ARGUMENTS)
        );
        sys::payload().TransmitMessage(msg);
        return;
    }

    // check if a file has been readied -> NO_FILE_READY
    if (!FileTransferManager::active_transfer())
    {
        SPDLOG_ERROR("No file available for transfer.");
        std::shared_ptr<Message> msg = CreateErrorAckMessage(
            CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(EC::NO_FILE_READY)
        );
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
                CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(EC::NO_MORE_PACKET_FOR_FILE)
            );
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
                CommandID::REQUEST_NEXT_FILE_PACKETS, to_uint8(err)
            );
            sys::payload().TransmitMessage(msg);
            return;
        }

        // transmit_data now contains the payload only (≤240 bytes)
        // CreateMessage will pad to 240 bytes and add CRC to create 247-byte UART packet
        std::shared_ptr<Message> msg = CreateMessage(
            CommandID::REQUEST_NEXT_FILE_PACKETS, transmit_data, current_packet
        );
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
