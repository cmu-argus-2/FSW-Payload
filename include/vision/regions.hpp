#ifndef REGIONS_HPP
#define REGIONS_HPP

#include <string>
#include <string_view>
#include <array>

/*
Currently supported regions (V1):
    '10S': 'California',
    '10T': 'Washington / Oregon',
    '11R': 'Baja California, Mexico',
    '12R': 'Sonora, Mexico',
    '16T': 'Minnesota / Wisconsin / Iowa / Illinois',
    '17R': 'Florida',
    '17T': 'Toronto, Canada / Michigan / OH / PA',
    '18S': 'New Jersey / Washington DC',
    '32S': 'Tunisia (North Africa near Tyrrhenian Sea)',
    '32T': 'Switzerland / Italy / Tyrrhenian Sea',
    '33S': 'Sicilia, Italy',
    '33T': 'Italy / Adriatic Sea',
    '52S': 'Korea / Kumamoto, Japan',
    '53S': 'Hiroshima to Nagoya, Japan',
    '54S': 'Tokyo to Hachinohe, Japan',
    '54T': 'Sapporo, Japan'
*/


enum class RegionID : uint8_t 
{
    R_10S,
    R_10T,
    R_11R,
    R_12R,
    R_16T,
    R_17R,
    R_17T,
    R_18S,
    R_32S,
    R_32T,
    R_33S,
    R_33T,
    R_52S,
    R_53S,
    R_54S,
    R_54T,
    UNKNOWN, // Placing at the end as a sentinel value instead of beginning to avoid mistakes when reading from GPU
};

// String → ID
constexpr RegionID GetRegionID(std::string_view region) {
    return
        region == "10S" ? RegionID::R_10S :
        region == "10T" ? RegionID::R_10T :
        region == "11R" ? RegionID::R_11R :
        region == "12R" ? RegionID::R_12R :
        region == "16T" ? RegionID::R_16T :
        region == "17R" ? RegionID::R_17R :
        region == "17T" ? RegionID::R_17T :
        region == "18S" ? RegionID::R_18S :
        region == "32S" ? RegionID::R_32S :
        region == "32T" ? RegionID::R_32T :
        region == "33S" ? RegionID::R_33S :
        region == "33T" ? RegionID::R_33T :
        region == "52S" ? RegionID::R_52S :
        region == "53S" ? RegionID::R_53S :
        region == "54S" ? RegionID::R_54S :
        region == "54T" ? RegionID::R_54T :
        RegionID::UNKNOWN;
}

// ID → Code string
constexpr std::string_view GetRegionString(RegionID id) {
    switch (id) {
        case RegionID::R_10S: return "10S";
        case RegionID::R_10T: return "10T";
        case RegionID::R_11R: return "11R";
        case RegionID::R_12R: return "12R";
        case RegionID::R_16T: return "16T";
        case RegionID::R_17R: return "17R";
        case RegionID::R_17T: return "17T";
        case RegionID::R_18S: return "18S";
        case RegionID::R_32S: return "32S";
        case RegionID::R_32T: return "32T";
        case RegionID::R_33S: return "33S";
        case RegionID::R_33T: return "33T";
        case RegionID::R_52S: return "52S";
        case RegionID::R_53S: return "53S";
        case RegionID::R_54S: return "54S";
        case RegionID::R_54T: return "54T";
        default: return "UNKNOWN";
    }
}

// ID → Location name
constexpr std::string_view GetRegionLocation(RegionID id) {
    switch (id) {
        case RegionID::R_10S: return "California";
        case RegionID::R_10T: return "Washington / Oregon";
        case RegionID::R_11R: return "Baja California, Mexico";
        case RegionID::R_12R: return "Sonora, Mexico";
        case RegionID::R_16T: return "Minnesota / Wisconsin / Iowa / Illinois";
        case RegionID::R_17R: return "Florida";
        case RegionID::R_17T: return "Toronto, Canada / Michigan / OH / PA";
        case RegionID::R_18S: return "New Jersey / Washington DC";
        case RegionID::R_32S: return "Tunisia (North Africa near Tyrrhenian Sea)";
        case RegionID::R_32T: return "Switzerland / Italy / Tyrrhenian Sea";
        case RegionID::R_33S: return "Sicilia, Italy";
        case RegionID::R_33T: return "Italy / Adriatic Sea";
        case RegionID::R_52S: return "Korea / Kumamoto, Japan";
        case RegionID::R_53S: return "Hiroshima to Nagoya, Japan";
        case RegionID::R_54S: return "Tokyo to Hachinohe, Japan";
        case RegionID::R_54T: return "Sapporo, Japan";
        default: return "UNKNOWN";
    }
}

// Get all valid region IDs (excluding UNKNOWN)
constexpr std::array<RegionID, 16> GetAllRegionIDs() {
    return {{
        RegionID::R_10S, RegionID::R_10T, RegionID::R_11R, RegionID::R_12R,
        RegionID::R_16T, RegionID::R_17R, RegionID::R_17T, RegionID::R_18S,
        RegionID::R_32S, RegionID::R_32T, RegionID::R_33S, RegionID::R_33T,
        RegionID::R_52S, RegionID::R_53S, RegionID::R_54S, RegionID::R_54T
    }};
}

// Get all region code strings (excluding UNKNOWN)
constexpr std::array<std::string_view, 16> GetAllRegionStrings() {
    return {{
        "10S", "10T", "11R", "12R",
        "16T", "17R", "17T", "18S",
        "32S", "32T", "33S", "33T",
        "52S", "53S", "54S", "54T"
    }};
}

struct Region
{
    RegionID id;
    float confidence;

    Region(RegionID id_, float confidence_) : id(id_), confidence(confidence_) {}
};

#endif // REGIONS_HPP