#include "commands.hpp"
#include "messages.hpp"
#include "payload.hpp"
#include "core/data_handling.hpp"
#include "telemetry/telemetry.hpp"

// (void)data does nothing special, it is just used to avoid compiler warnings about a variable that is not used 

// Command functions array definition 
std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS = 
{
    ping_ack, // PING_ACK
    shutdown, // SHUTDOWN
    synchronize_time, // SYNCHRONIZE_TIME
    request_telemetry, // REQUEST_TELEMETRY
    enable_cameras, // ENABLE_CAMERAS
    disable_cameras, // DISABLE_CAMERAS
    capture_images, // CAPTURE_IMAGES
    start_capture_images_periodically, // START_CAPTURE_IMAGES_PERIODICALLY
    stop_capture_images, // STOP_CAPTURE_IMAGES
    stored_images, // STORED_IMAGES
    request_image, // REQUEST_IMAGE
    delete_images, // DELETE_IMAGES
    run_od, // RUN_OD
    ping_od_status, // PING_OD_STATUS
    debug_display_camera, // DEBUG_DISPLAY_CAMERA
    debug_stop_display // DEBUG_STOP_DISPLAY
};

// Define the array of strings mapping CommandID to command names
std::array<std::string_view, COMMAND_NUMBER> COMMAND_NAMES = {
    "PING_ACK",
    "SHUTDOWN",
    "SYNCHRONIZE_TIME",
    "REQUEST_TELEMETRY",
    "ENABLE_CAMERAS",
    "DISABLE_CAMERAS",
    "CAPTURE_IMAGES",
    "START_CAPTURE_IMAGES_PERIODICALLY",
    "STOP_CAPTURE_IMAGES",
    "STORED_IMAGES",
    "REQUEST_IMAGE",
    "DELETE_IMAGES",
    "RUN_OD",
    "PING_OD_STATUS",
    "DEBUG_DISPLAY_CAMERA",
    "DEBUG_STOP_DISPLAY"
};

void ping_ack(std::vector<uint8_t>& data)
{
    (void)data;
    SPDLOG_INFO("Received PING_ACK");

    std::vector<uint8_t> transmit_data = {PING_VALUE};
    auto msg = CreateMessage(CommandID::PING_ACK, transmit_data);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::PING_ACK);
}

void shutdown(std::vector<uint8_t>& data)
{
    (void)data;
    SPDLOG_INFO("Initiating Payload shutdown..");
    sys::payload().Stop();

    auto msg = CreateSuccessAckMessage(CommandID::SHUTDOWN);
    sys::payload().TransmitMessage(msg);
    
    sys::payload().SetLastExecutedCmdID(CommandID::SHUTDOWN);
}

void synchronize_time(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Synchronizing time..");
    (void)data;
    // TODO
    // SHOULD SHUTDOWN and REBOOT
    sys::payload().SetLastExecutedCmdID(CommandID::SYNCHRONIZE_TIME);
}

void request_telemetry(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Requesting last telemetry..");
    (void)data;


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

    auto msg = CreateMessage(CommandID::REQUEST_TELEMETRY, transmit_data);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_TELEMETRY);
}


// If a series of IDs is not given, it tries to activate all cameras by default 
void enable_cameras(std::vector<uint8_t>& data)
{
    if (data.empty())
    {
        SPDLOG_INFO("Trying to enable all cameras...");
        std::vector<int> ids;
        ids.reserve(4);
        sys::cameraManager().EnableCameras(ids);
        for (auto cam_id : ids)
        {
            SPDLOG_INFO("Enabled camera {}.", cam_id);
        }
    }
    else
    {
        for (uint8_t cam_id : data)
        {
            bool res = sys::cameraManager().EnableCamera(cam_id);
            if (res)
            {
                SPDLOG_INFO("Enabled camera {}.", cam_id);
                // Return a success msg 
            }
            else 
            {
                // transmit a failure msg 
            }
        }

    }

    sys::payload().SetLastExecutedCmdID(CommandID::ENABLE_CAMERAS);

}

void disable_cameras(std::vector<uint8_t>& data)
{
 
    if (data.empty())
    {
        SPDLOG_INFO("Trying to disable all cameras...");
        std::vector<int> ids;
        ids.reserve(4);
        sys::cameraManager().DisableCameras(ids);
        for (auto cam_id : ids)
        {
            SPDLOG_INFO("Disabled camera {}.", cam_id);
        }
    }
    else
    {
        for (uint8_t cam_id : data)
        {
            bool res = sys::cameraManager().DisableCamera(cam_id);
            if (res)
            {
                SPDLOG_INFO("Disabled camera {}.", cam_id);
                // Return a success msg 
            }
            else 
            {
                // transmit a failure msg 
            }
        }

    } 


    sys::payload().SetLastExecutedCmdID(CommandID::DISABLE_CAMERAS);
 
}

void capture_images(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Capturing image now..");

    sys::cameraManager().SendCaptureRequest();

    // Need to return true or false based on the success of the operations
    // Basically should wait until the camera has captured the image or wait later to send the ocnfirmation and just exit the task? 
    (void)data;
    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::CAPTURE_IMAGES);
}

void start_capture_images_periodically(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Starting capture images every X seconds..");

    if (!data.empty()) {
        uint8_t period = data[0]; // Period in seconds
        sys::cameraManager().SetPeriodicCaptureRate(period);

        if (data.size() > 1) {
            uint8_t frames = data[1]; // Number of frames to capture
            sys::cameraManager().SetPeriodicFramesToCapture(frames);
        }
    }

    sys::cameraManager().SetCaptureMode(CAPTURE_MODE::PERIODIC);

    // TODO: Implement a return type to indicate success or failure, if needed.

    sys::payload().SetLastExecutedCmdID(CommandID::START_CAPTURE_IMAGES_PERIODICALLY);
}

void stop_capture_images(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Stopping capture images..");

    (void)data;

    sys::cameraManager().SetCaptureMode(CAPTURE_MODE::IDLE);
    // TODO return true or false based on the success of the operations

    sys::payload().SetLastExecutedCmdID(CommandID::STOP_CAPTURE_IMAGES);
}



void stored_images(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Getting stored images..");
    
    (void)data;
    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::STORED_IMAGES);
}

void request_image(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Requesting last image..");
    
    (void)data;
    
    // Read the latest image from disk
    Frame frame;
    bool res = DH::ReadLatestStoredRawImg(frame);

    // TODO: Transmit the image if successful, else error ACK
    // Need to be flagged for deletion

    sys::payload().SetLastExecutedCmdID(CommandID::REQUEST_IMAGE);
}


void delete_images(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Deleting all stored images..");
    
    (void)data;
    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::DELETE_IMAGES);
}


void run_od(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Running orbit determination..");
    
    (void)data;
    // TODO

    sys::payload().SetLastExecutedCmdID(CommandID::RUN_OD);
}

void ping_od_status(std::vector<uint8_t>& data)
{

    SPDLOG_INFO("Pinging the status of the orbit determination process...");
    
    (void)data;
    // TODO


    sys::payload().SetLastExecutedCmdID(CommandID::PING_OD_STATUS);
}


void debug_display_camera(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Activating the display of the camera");

    if (sys::cameraManager().GetDisplayFlag() == true) 
    {
        SPDLOG_WARN("Display already active");
        auto msg = CreateSuccessAckMessage(CommandID::DEBUG_DISPLAY_CAMERA);
        sys::payload().TransmitMessage(msg);
        return;
    }

    sys::cameraManager().SetDisplayFlag(true);
    // the command is already by a thread of the ThreadPool so no need to spawn a new thread here
    // This will block the thread until the display flag is set to false or all cameras are turned off
    sys::cameraManager().RunDisplayLoop(); 

    auto msg = CreateSuccessAckMessage(CommandID::DEBUG_DISPLAY_CAMERA);
    sys::payload().TransmitMessage(msg);

    (void)data;

    sys::payload().SetLastExecutedCmdID(CommandID::DEBUG_DISPLAY_CAMERA);
}

void debug_stop_display(std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Stopping the display of the camera");
    sys::cameraManager().SetDisplayFlag(false);
    (void)data;
    // return ACK
    auto msg = CreateSuccessAckMessage(CommandID::DEBUG_STOP_DISPLAY);
    sys::payload().TransmitMessage(msg);

    sys::payload().SetLastExecutedCmdID(CommandID::DEBUG_STOP_DISPLAY);
}