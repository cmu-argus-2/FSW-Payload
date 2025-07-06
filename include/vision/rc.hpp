#ifndef RC_HPP
#define RC_HPP

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

#include <string>

#define NUM_CLASSES 16

typedef uint8_t RegionID;


namespace RegionMapping 
{
    // Static mappings
    RegionID GetRegionID(const std::string& region);
    std::string GetRegionString(RegionID id);
    std::string GetRegionLocation(RegionID id);
}

#endif // RC_HPP