#include "camera_manager.hpp"

CameraManager::CameraManager(const std::array<int, 4>& camera_ids) :
cameras{Camera(camera_ids[0]), Camera(camera_ids[1]), Camera(camera_ids[2]), Camera(camera_ids[3])}
{

}