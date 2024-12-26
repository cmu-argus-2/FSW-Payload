#include <cstdlib>
#include "core/utils.hpp"


bool DetectJetsonPlatform()
{
    int res = std::system("uname -m | grep -q 'aarch64'");
    // -q suppresses output
    // exit code (0 if found, 1 otherwise).
    return (res == 0);
}
