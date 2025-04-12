#ifndef POSE_INTEGRATOR_HPP
#define POSE_INTEGRATOR_HPP

#include <core/timing.hpp>

struct InertialData
{
    std::uint64_t timestamp;
    float gyro[3]; // Angular velocity in rad/s
    float mag[3]; // Magnetic field in uT
    bool valid; // Flag to indicate if the data is valid
};

class PoseIntegrator
{




};


# endif // POSE_INTEGRATOR_HPP