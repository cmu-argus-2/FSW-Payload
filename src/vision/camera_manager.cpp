#include "vision/camera_manager.hpp"
#include "core/data_handling.hpp"
#include "inference/inference_manager.hpp"

CameraManager::CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs,
                             const CameraISPConfig& isp_config,
                             InferenceManager& inference_manager)
:
inferenceManager(inference_manager),
isp_config(isp_config),
capture_mode(CAPTURE_MODE::IDLE),
storage_folder(IMAGES_FOLDER),
camera_configs(camera_configs),
cameras{{
    Camera(camera_configs[0].id, camera_configs[0].path,
           static_cast<int>(camera_configs[0].width), static_cast<int>(camera_configs[0].height), isp_config),
    Camera(camera_configs[1].id, camera_configs[1].path,
           static_cast<int>(camera_configs[1].width), static_cast<int>(camera_configs[1].height), isp_config),
    Camera(camera_configs[2].id, camera_configs[2].path,
           static_cast<int>(camera_configs[2].width), static_cast<int>(camera_configs[2].height), isp_config),
    Camera(camera_configs[3].id, camera_configs[3].path,
           static_cast<int>(camera_configs[3].width), static_cast<int>(camera_configs[3].height), isp_config)
}}
{
    _UpdateCamStatus();
    SPDLOG_INFO("Camera Manager initialized");
}

CameraManager::~CameraManager()
{
    StopLoops();
}

void CameraManager::_UpdateCamStatus()
{
    // Initialize cam_status based on the status of each camera
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {
        cam_status[i] = cameras[i].GetStatus();
    }
}


void CameraManager::CaptureFrames()
{
    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE)
        {   
            cameras[i].CaptureFrame();
        }
    }
}


uint8_t CameraManager::SaveLatestFrames(CAPTURE_MODE mode)
{
    uint8_t saved_count = 0;
    std::vector<std::tuple<uint8_t, uint64_t>> new_ids;

    const bool needs_prefilter = (mode == CAPTURE_MODE::PERIODIC_EARTH ||
                                  mode == CAPTURE_MODE::PERIODIC_ROI   ||
                                  mode == CAPTURE_MODE::PERIODIC_LDMK);
    const bool needs_inference = (mode == CAPTURE_MODE::PERIODIC_ROI ||
                                  mode == CAPTURE_MODE::PERIODIC_LDMK);

    // grab all active camera buffers in a tight loop
    std::vector<std::shared_ptr<Frame>> grabbed;
    grabbed.reserve(CountActiveCameras());
    for (std::size_t i = 0; i < NUM_CAMERAS; ++i)
    {
        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE && cameras[i].IsNewFrameAvailable())
        {
            grabbed.push_back(std::make_shared<Frame>(cameras[i].GetBufferFrame()));
            cameras[i].SetOffNewFrameFlag();
        }
    }

    // prefilter, run inference, and persist
    std::vector<std::shared_ptr<Frame>> earth_frames;

    for (auto& frame_ptr : grabbed)
    {
        if (needs_prefilter && frame_ptr->GetProcessingStage() == ProcessingStage::NotPrefiltered)
            frame_ptr->RunPrefiltering();

        if (needs_prefilter && frame_ptr->GetImageState() < ImageState::Earth)
        {
            SPDLOG_INFO("CAM{}: Frame skipped (not Earth)", frame_ptr->GetCamID());
            continue;
        }

        if (!needs_inference)
        {
            // CAPTURE_SINGLE, PERIODIC, PERIODIC_EARTH
            if (!DH::StoreFrameToDisk(*frame_ptr, GetStorageFolder()).empty())
            {
                new_ids.emplace_back(frame_ptr->GetCamID(), frame_ptr->GetTimestamp());
                saved_count++;
            }
        }
        else
        {
            earth_frames.push_back(std::move(frame_ptr));
        }
    }

    // Inference pipeline (PERIODIC_ROI / PERIODIC_LDMK)
    if (needs_inference && !earth_frames.empty())
    {
        DisableCameras();
        SPDLOG_INFO("Cameras disabled before inference.");

        for (auto& frame_ptr : earth_frames)
        {
            EC rc_status = inferenceManager.ProcessFrame(frame_ptr, ProcessingStage::RCNeted);
            if (rc_status != EC::OK)
            {
                SPDLOG_ERROR("CAM{}: RCNet inference failed", frame_ptr->GetCamID());
                continue;
            }

            if (!frame_ptr->HasRegion())
            {
                SPDLOG_INFO("CAM{}: Frame skipped (no region)", frame_ptr->GetCamID());
                continue;
            }

            if (mode == CAPTURE_MODE::PERIODIC_ROI)
            {
                if (!DH::StoreFrameToDisk(*frame_ptr, GetStorageFolder()).empty())
                {
                    new_ids.emplace_back(frame_ptr->GetCamID(), frame_ptr->GetTimestamp());
                    saved_count++;
                }
                continue;
            }

            // PERIODIC_LDMK
            EC ld_status = inferenceManager.ProcessFrame(frame_ptr, ProcessingStage::LDNeted);
            if (ld_status != EC::OK)
            {
                SPDLOG_ERROR("CAM{}: LDNet inference failed", frame_ptr->GetCamID());
                continue;
            }

            if (frame_ptr->HasLandmark())
            {
                if (!DH::StoreFrameToDisk(*frame_ptr, GetStorageFolder()).empty())
                {
                    new_ids.emplace_back(frame_ptr->GetCamID(), frame_ptr->GetTimestamp());
                    saved_count++;
                }
            }
            else
            {
                SPDLOG_INFO("CAM{}: Frame skipped (no landmark)", frame_ptr->GetCamID());
            }
        }

        EnableCameras();
        SPDLOG_INFO("Cameras re-enabled after inference.");
    }

    {
        std::lock_guard<std::mutex> lock(buffer_frame_ids_m);
        buffer_frame_ids.insert(buffer_frame_ids.end(),
                                std::make_move_iterator(new_ids.begin()),
                                std::make_move_iterator(new_ids.end()));
    }
    return saved_count;
}


uint8_t CameraManager::CopyFrames(std::vector<Frame>& vec_frames, bool only_earth)
{

    uint8_t frame_count = 0;
    vec_frames.reserve(vec_frames.size() + NUM_CAMERAS); // assumes void vector

    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE)
        {   
            Frame new_frame;
            cameras[i].CopyBufferFrame(new_frame);
            vec_frames.emplace_back(std::move(new_frame)); // Move to avoid copying
            frame_count++;
        }
    }

    return frame_count;
}



void CameraManager::RunLoop()
{
    loop_flag.store(true);

    auto last_health_check = std::chrono::high_resolution_clock::now();
    auto last_capture      = std::chrono::high_resolution_clock::now();
    CAPTURE_MODE prev_mode = CAPTURE_MODE::IDLE;

    while (loop_flag.load())
    {
        const CAPTURE_MODE mode = capture_mode.load();

        switch (mode)
        {
            case CAPTURE_MODE::IDLE:
            {
                std::unique_lock<std::mutex> lock(capture_mode_mutex);
                capture_mode_cv.wait(lock, [this] {
                    return !loop_flag.load() || capture_mode.load() != CAPTURE_MODE::IDLE;
                });
                break;
            }

            case CAPTURE_MODE::CAPTURE_SINGLE:
            {
                int n = SaveLatestFrames(mode);
                SPDLOG_INFO("Single capture completed: {} frame(s)", n);
                _AutoDisableIfNeeded();
                // TODO: ACK command
                SetCaptureMode(CAPTURE_MODE::IDLE);
                break;
            }

            default: // all PERIODIC* modes share the same rate-limited dispatch
            {
                if (prev_mode != mode)
                    last_capture -= std::chrono::seconds(periodic_capture_rate.load());

                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_capture).count() >= periodic_capture_rate)
                {
                    {
                        const uint8_t  saved     = SaveLatestFrames(mode);
                        const uint16_t new_total = static_cast<uint16_t>(periodic_frames_captured.load()) + saved;
                        periodic_frames_captured = static_cast<uint8_t>(
                            std::min<uint16_t>(new_total, periodic_frames_to_capture.load()));
                    }
                    SPDLOG_INFO("Periodic capture: {}/{} frames saved",
                                periodic_frames_captured.load(), periodic_frames_to_capture.load());

                    if (periodic_frames_captured >= periodic_frames_to_capture)
                    {
                        SPDLOG_INFO("Periodic capture completed");
                        SetCaptureMode(CAPTURE_MODE::IDLE);
                        periodic_frames_captured = 0;
                        periodic_frames_to_capture = DEFAULT_PERIODIC_FRAMES_TO_CAPTURE;
                        break;
                    }
                    last_capture = now;
                }
                break;
            }
        }

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_health_check).count() >= CAMERA_HEALTH_CHECK_INTERVAL)
        {
            _PerformCameraHealthCheck();
            last_health_check = now;
        }

        prev_mode = mode;
        _UpdateCamStatus();
    }

    SPDLOG_INFO("Exiting Camera Manager Run Loop");
    SetDisplayFlag(false);
}



void CameraManager::RunDisplayLoop()
{

    int active_cams = 0;
    
    while (display_flag.load() && loop_flag.load()) 
    {
        active_cams = 0;
        
        for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
        {
            auto& cam = cameras[i]; // Alias for readability

            if (cam.GetStatus() == CAM_STATUS::ACTIVE) 
            {
                ++active_cams;

                if (cam.IsNewFrameAvailable())
                {
                    cam.DisplayLastFrame();
                }
            }
        }

        if (active_cams == 0) 
        {
            SPDLOG_WARN("No cameras are turned on. Exiting display loop.");
            break;
        }

        if (!display_flag.load() || !loop_flag.load()) 
        {
            SPDLOG_WARN("Display loop terminated by flags.");
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // Just displaying
    }

    display_flag.store(false);
    cv::destroyAllWindows();

    SPDLOG_INFO("Exiting Camera Manager Display Loop");
}




CameraConfig* CameraManager::GetCameraConfig(int cam_id) 
{
    for (auto& config : camera_configs)
    {
        if (config.id == cam_id)
        {
            return &config; // Return a pointer to the found config
        }
    }
    return nullptr; // Return nullptr if the ID is not found
}

int CameraManager::GetCapturedFramesCount() const
{
    return periodic_frames_captured.load();
}

std::vector<std::tuple<uint8_t, uint64_t>> CameraManager::GetBufferFrameIDs() const
{
    std::lock_guard<std::mutex> lock(buffer_frame_ids_m);
    return buffer_frame_ids; // non-destructive read
}

std::vector<std::tuple<uint8_t, uint64_t>> CameraManager::DrainBufferFrameIDs()
{
    std::lock_guard<std::mutex> lock(buffer_frame_ids_m);
    std::vector<std::tuple<uint8_t, uint64_t>> out;
    out.swap(buffer_frame_ids);
    return out;
}

void CameraManager::ResetCaptureState()
{
    std::lock_guard<std::mutex> lock(buffer_frame_ids_m);
    periodic_frames_captured = 0;
    periodic_frames_to_capture = DEFAULT_PERIODIC_FRAMES_TO_CAPTURE;
    periodic_capture_rate = 60;
    buffer_frame_ids.clear();
}

void CameraManager::StopLoops()
{
    display_flag.store(false);
    loop_flag.store(false);
    capture_mode_cv.notify_all();
    auto_disable_after_capture.store(false);

    for (auto& camera : cameras) 
    {
        camera.Disable();
    }

    SPDLOG_INFO("Stopped camera loops...");
}

void CameraManager::SetDisplayFlag(bool display_flag)
{
    this->display_flag.store(display_flag);
}

bool CameraManager::GetDisplayFlag() const
{
    return display_flag.load();
}

void CameraManager::SetCaptureMode(CAPTURE_MODE mode)
{
    capture_mode.store(mode);
    capture_mode_cv.notify_all();
}

/**/
bool CameraManager::SendCaptureRequest()
{
    const CAPTURE_MODE current_mode = GetCaptureMode();
    if (current_mode != CAPTURE_MODE::IDLE)
    {
        SPDLOG_WARN("Rejecting single capture request while capture mode is {}",
                    static_cast<uint8_t>(current_mode));
        return false;
    }

    if (!PrepareForCapture())
    {
        return false;
    }
    SetCaptureMode(CAPTURE_MODE::CAPTURE_SINGLE);
    return true;
}


void CameraManager::SetPeriodicCaptureRate(uint8_t period)
{
    periodic_capture_rate = period;
    SPDLOG_INFO("Periodic capture rate set to: {} seconds", period);
}


void CameraManager::SetPeriodicFramesToCapture(uint8_t frames)
{
    periodic_frames_to_capture = frames;
    SPDLOG_INFO("Periodic frames to capture set to: {}", frames);
}


void CameraManager::SetStorageFolder(const std::string& s)
{
    if (!DH::fs::exists(s))
    {
        if (!DH::MakeNewDirectory(s))
        {
            SPDLOG_ERROR("Failed to create storage folder: {}", s);
            return;
        }
    }
    std::lock_guard<std::mutex> lock(storage_folder_m);
    storage_folder = s;
    if (!storage_folder.empty() && storage_folder.back() != '/') {
        storage_folder += '/';
    }
    SPDLOG_INFO("Camera storage folder set to: {}", storage_folder);
}

void CameraManager::SetTargetProcessingStage(ProcessingStage stage)
{
    target_processing_stage.store(stage);
}

std::string CameraManager::GetStorageFolder()
{
    std::lock_guard<std::mutex> lock(storage_folder_m);
    return storage_folder;
}

ProcessingStage CameraManager::GetTargetProcessingStage() const
{
    return target_processing_stage.load();
}

bool CameraManager::PrepareForCapture()
{
    const int configured_count = CountConfiguredCameras();
    if (configured_count == 0)
    {
        SPDLOG_ERROR("No cameras are enabled in configuration; cannot prepare for capture.");
        auto_disable_after_capture.store(false);
        return false;
    }

    const int active_before = CountActiveCameras();
    if (active_before == configured_count)
    {
        auto_disable_after_capture.store(false);
        return true;
    }

    EnableCameras();
    const int active_after = CountActiveCameras();
    if (active_after == 0)
    {
        SPDLOG_ERROR("Failed to enable any cameras for capture.");
        auto_disable_after_capture.store(false);
        return false;
    }

    if (active_after != configured_count)
    {
        SPDLOG_ERROR("Only {}/{} configured camera(s) are active; aborting capture.",
                     active_after, configured_count);
        if (active_before == 0)
            DisableCameras();
        auto_disable_after_capture.store(false);
        return false;
    }

    auto_disable_after_capture.store(active_before == 0);
    const int newly_enabled = active_after > active_before ? (active_after - active_before) : 0;
    SPDLOG_INFO("Prepared {} additional camera(s) for capture.", newly_enabled);
    return true;
}

void CameraManager::_AutoDisableIfNeeded()
{
    if (!auto_disable_after_capture.exchange(false))
    {
        return;
    }

    int nb_disabled = DisableCameras();
    SPDLOG_INFO("Auto-disabled {} camera(s) after capture completion.", nb_disabled);
}


bool CameraManager::EnableCamera(int cam_id)
{
    for (auto& camera : cameras) 
    {
        if (camera.GetID() == cam_id)
        {
            const bool enabled = camera.Enable();
            _UpdateCamStatus();
            return enabled;
        }
    }
    return false;
}


bool CameraManager::DisableCamera(int cam_id)
{
    for (auto& camera : cameras) 
    {
        if (camera.GetID() == cam_id)
        {
            const bool disabled = camera.Disable();
            _UpdateCamStatus();
            return disabled;
        }
    }
    return false;
}


int CameraManager::EnableCameras()
{
    int count = 0;

    for (size_t i = 0; i < NUM_CAMERAS; ++i)
    {
        if (!camera_configs[i].enabled) {
            SPDLOG_INFO("CAM{}: disabled in config, skipping", camera_configs[i].id);
            continue;
        }

        const bool was_active = (cameras[i].GetStatus() == CAM_STATUS::ACTIVE);
        cameras[i].Enable();

        if (!was_active && cameras[i].GetStatus() == CAM_STATUS::ACTIVE)
            count++;
    }
    _UpdateCamStatus();
    return count;
}


int CameraManager::DisableCameras()
{
    int count = 0;

    for (size_t i = 0; i < NUM_CAMERAS; ++i)
    {
        if (!camera_configs[i].enabled)
            continue;

        const bool was_active = (cameras[i].GetStatus() == CAM_STATUS::ACTIVE);
        cameras[i].Disable();

        if (was_active && cameras[i].GetStatus() == CAM_STATUS::INACTIVE)
            count++;
    }

    _UpdateCamStatus();
    return count;
}


int CameraManager::CountConfiguredCameras() const
{
    int count = 0;
    for (const auto& cfg : camera_configs)
        if (cfg.enabled) count++;
    return count;
}

int CameraManager::CountActiveCameras() const
{
    int active_count = 0;
    for (auto& camera : cameras) 
    {
        // access atomic variable
        if (camera.GetStatus() == CAM_STATUS::ACTIVE)
        {
            active_count++;
        }
    }
    return active_count;
}

CAPTURE_MODE CameraManager::GetCaptureMode() const
{
    return capture_mode.load(); // atomic
}   

void CameraManager::FillCameraStatus(uint8_t* status) 
{
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {

        status[i] = static_cast<uint8_t>(this->cam_status[i]);

    }
}


void CameraManager::_PerformCameraHealthCheck()
{
    for (auto& camera : cameras) 
    {
        switch (camera.GetStatus())
        {
            case CAM_STATUS::ACTIVE:
            {
                if (camera.GetLastError() != CAM_ERROR::NO_ERROR) 
                {
                    // Restart the camera
                    camera.Disable();
                    camera.Enable();
                }
                break;
            }

            case CAM_STATUS::INACTIVE:
            {
                break;
            }

            default:
                // Restart to get out of an undefined state
                camera.Disable();
                camera.Enable();
                break;
        }
        
    }
}
