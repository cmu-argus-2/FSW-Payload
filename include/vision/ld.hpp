#ifndef LD_HPP
#define LD_HPP

#include <cstdint>
#include "vision/regions.hpp"

struct Landmark 
{
    float x, y;         // Centroid position
    uint16_t class_id;  // Class ID
    RegionID region_id; // Encoded MGRS region

    Landmark(float x_, float y_, uint16_t class_id_, RegionID region_id_)
        : x(x_), y(y_), class_id(class_id_), region_id(region_id_) {}
};


#endif // LD_HPP