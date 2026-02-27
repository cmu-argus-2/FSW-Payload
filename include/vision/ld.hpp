#ifndef LD_HPP
#define LD_HPP

#include <cstdint>
#include "vision/regions.hpp"

struct Landmark 
{
    float x, y;         // Centroid position
    float height, width;  // Bounding box dimensions
    float confidence;    // Confidence score
    uint16_t class_id;  // Class ID
    RegionID region_id; // Encoded MGRS region

    Landmark(float x_, float y_, uint16_t class_id_, RegionID region_id_, 
            float height_ = 0.0f, float width_ = 0.0f, float confidence_ = 0.0f)
        : x(x_), y(y_), height(height_), width(width_), confidence(confidence_), 
            class_id(class_id_), region_id(region_id_) {}
    
    bool operator<(const Landmark& other) const {
        return confidence < other.confidence;
    }
};


#endif // LD_HPP