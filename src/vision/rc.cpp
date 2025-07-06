#include "vision/rc.hpp"
#include <filesystem> 
#include "spdlog/spdlog.h"

namespace RegionMapping 
{
    
    // Static mappings
    const std::unordered_map<std::string, RegionID> region_to_id = 
    {
        {"10S", 1}, {"10T", 2}, {"11R", 3}, {"12R", 4},
        {"16T", 5}, {"17R", 6}, {"17T", 7}, {"18S", 8},
        {"32S", 9}, {"32T", 10}, {"33S", 11}, {"33T", 12},
        {"52S", 13}, {"53S", 14}, {"54S", 15}, {"54T", 16}
    };

    const std::unordered_map<RegionID, std::string> id_to_region = 
    {
        {1, "10S"}, {2, "10T"}, {3, "11R"}, {4, "12R"},
        {5, "16T"}, {6, "17R"}, {7, "17T"}, {8, "18S"},
        {9, "32S"}, {10, "32T"}, {11, "33S"}, {12, "33T"},
        {13, "52S"}, {14, "53S"}, {15, "54S"}, {16, "54T"}
    };

    RegionID GetRegionID(const std::string& region) 
    {
        auto it = region_to_id.find(region);
        return (it != region_to_id.end()) ? it->second : 0; // 0 = Invalid
    }

    std::string GetRegionString(RegionID id)
    {
        auto it = id_to_region.find(id);
        return (it != id_to_region.end()) ? it->second : "UNKNOWN";
    }

    std::string GetRegionLocation(RegionID id) 
    {
        switch (id) 
        {
            case 1: return "California";
            case 2: return "Washington / Oregon";
            case 3: return "Baja California, Mexico";
            case 4: return "Sonora, Mexico";
            case 5: return "Minnesota / Wisconsin / Iowa / Illinois";
            case 6: return "Florida";
            case 7: return "Toronto, Canada / Michigan / OH / PA";
            case 8: return "New Jersey / Washington DC";
            case 9: return "Tunisia (North Africa near Tyrrhenian Sea)";
            case 10: return "Switzerland / Italy / Tyrrhenian Sea";
            case 11: return "Sicilia, Italy";
            case 12: return "Italy / Adriatic Sea";
            case 13: return "Korea / Kumamoto, Japan";
            case 14: return "Hiroshima to Nagoya, Japan";
            case 15: return "Tokyo to Hachinohe, Japan";
            case 16: return "Sapporo, Japan";
            default: return "UNKNOWN";
        }
    }
}

