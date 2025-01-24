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
    "DEBUG_DISPLAY_CAMERA",
    "DEBUG_STOP_DISPLAY"
};

void ping_ack(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Received PING_ACK");
    auto msg = std::make_shared<MSG_PING_ACK>();
    msg->serialize();

    payload.TransmitMessage(msg);
    (void)data;
}

void shutdown(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Initiating Payload shutdown..");
    payload.Stop();
    (void)data; // TODO
}

void synchronize_time(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Synchronizing time..");
    (void)payload;
    (void)data;
    // TODO
}

void request_telemetry(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Requesting last telemetry..");
    (void)payload;
    (void)data;
    PrintTelemetryFrame(payload.GetTelemetry().GetTmFrame());
    // TODO
}


// If a series of IDs is not given, it tries to activate all cameras by default 
void enable_cameras(Payload& payload, std::vector<uint8_t>& data)
{
    if (data.empty())
    {
        SPDLOG_INFO("Trying to enable all cameras...");
        std::vector<int> ids;
        ids.reserve(4);
        payload.GetCameraManager().EnableCameras(ids);
        for (auto cam_id : ids)
        {
            SPDLOG_INFO("Enabled camera {}.", cam_id);
        }
    }
    else
    {
        for (uint8_t cam_id : data)
        {
            bool res = payload.GetCameraManager().EnableCamera(cam_id);
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

}

void disable_cameras(Payload& payload, std::vector<uint8_t>& data)
{
 
    if (data.empty())
    {
        SPDLOG_INFO("Trying to disable all cameras...");
        std::vector<int> ids;
        ids.reserve(4);
        payload.GetCameraManager().DisableCameras(ids);
        for (auto cam_id : ids)
        {
            SPDLOG_INFO("Disabled camera {}.", cam_id);
        }
    }
    else
    {
        for (uint8_t cam_id : data)
        {
            bool res = payload.GetCameraManager().DisableCamera(cam_id);
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
 
}

void capture_images(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Capturing image now..");

    payload.GetCameraManager().SendCaptureRequest();

    // Need to return true or false based on the success of the operations
    // Basically should wait until the camera has captured the image or wait later to send the ocnfirmation and just exit the task? 
    (void)data;
    // TODO
}

void start_capture_images_periodically(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Starting capture images every X seconds..");

    if (!data.empty()) {
        uint8_t period = data[0]; // Period in seconds
        payload.GetCameraManager().SetPeriodicCaptureRate(period);

        if (data.size() > 1) {
            uint8_t frames = data[1]; // Number of frames to capture
            payload.GetCameraManager().SetPeriodicFramesToCapture(frames);
        }
    }

    payload.GetCameraManager().SetCaptureMode(CAPTURE_MODE::PERIODIC);

    // TODO: Implement a return type to indicate success or failure, if needed.
}

void stop_capture_images(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Stopping capture images..");

    (void)data;

    payload.GetCameraManager().SetCaptureMode(CAPTURE_MODE::IDLE);
    // TODO return true or false based on the success of the operations
}



void stored_images(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Getting stored images..");
    (void)payload;
    (void)data;
    // TODO
}

void request_image(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Requesting last image..");
    (void)payload;
    (void)data;
    
    // Read the latest image from disk
    Frame frame;
    DH::ReadLatestStoredRawImg(frame._img, frame._timestamp, frame._cam_id);

    // TODO: Transmit the image
    // Need to be flagged for deletion
}


void delete_images(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Deleting all stored images..");
    (void)payload;
    (void)data;
    // TODO
}


void run_od(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Running orbit determination..");
    (void)payload;
    (void)data;
    // TODO
}


void debug_display_camera(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Activating the display of the camera");

    if (payload.GetCameraManager().GetDisplayFlag() == true) 
    {
        SPDLOG_WARN("Display already active");
        // TODO: return success
        return;
    }

    payload.GetCameraManager().SetDisplayFlag(true);
    // the command is already by a thread of the ThreadPool so no need to spawn a new thread here
    // This will block the thread until the display flag is set to false or all cameras are turned off
    payload.GetCameraManager().RunDisplayLoop(); 

    (void)data; // TODO
}

void debug_stop_display(Payload& payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Stopping the display of the camera");
    payload.GetCameraManager().SetDisplayFlag(false);
    (void)data; // TODO
    // return ACK
}