#ifndef CAMERA_MANAGER_HPP
#define CAMERA_MANAGER_HPP

#include <array>
#include "spdlog/spdlog.h"
#include "camera.hpp"

#define NUM_CAMERAS 4

// Main interface to manage the cameras 

class CameraManager
{


public:

    CameraManager(const std::array<int, 4>& camera_ids);

private:


    std::array<Camera, NUM_CAMERAS> cameras;

};







#endif // CAMERA_MANAGER_HPP