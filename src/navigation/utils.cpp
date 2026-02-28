#include "navigation/utils.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <filesystem>

#include <cmath>
using namespace Eigen;
// seconds elapsed between the Unix and J2000 epoch
static constexpr int64_t J2000epochInUnixTime = 946727936;

// Astronomical Unit [m]
static constexpr double ASTRONOMICAL_UNIT = 149597870700;

int64_t unixToJ2000(int64_t unixSeconds) {
    return unixSeconds - J2000epochInUnixTime;
}

// Basic Utility functions
void loadAllKernels() {
    std::filesystem::path path(__FILE__);
    std::string root = path.parent_path().parent_path().parent_path().string(); // utils_and_transforms.cpp --> math --> world --> dynamics sim
    std::string data_folder = root + "/data/";


    std::string sol_system_spk = data_folder + "de440.bsp";
    std::string earth_rotation_pck = data_folder + "earth_latest_high_prec.bpc";
    std::string earth_dimensions_pck = data_folder + "pck00011.tpc";
    std::string leap_seconds_lsk = data_folder + "pck00011.tpc";
    std::string leap_seconds_lsk2 = data_folder + "naif0012.tls";
    
    SpiceInt count;
    ktotal_c("ALL", &count);

    if (count == 0) {
        furnsh_c(sol_system_spk.c_str());
        furnsh_c(earth_rotation_pck.c_str());
        furnsh_c(earth_dimensions_pck.c_str());
        furnsh_c(leap_seconds_lsk.c_str());
        furnsh_c(leap_seconds_lsk2.c_str());
    }; // only load kernel if not already loaded
    
    
}

// Convert CSPICE Double array to 3x3 Eigen Matrix
Matrix3d Cspice2Eigen(SpiceDouble M[3][3]) {
    Matrix3d R;
    R << M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2];
    return R;
}

// TRANSFORMS

Matrix3d ECI2ECEF(double t_J2000) {
    SpiceDouble Rot[3][3];

    loadAllKernels();
    pxform_c("J2000", "ITRF93", t_J2000, Rot);
    
    return Cspice2Eigen(Rot);
}

Matrix3d ECEF2ECI(double t_J2000) {
    SpiceDouble Rot[3][3];

    loadAllKernels();
    pxform_c("ITRF93", "J2000", t_J2000, Rot);
    
    return Cspice2Eigen(Rot);
}

Vector3d ECEF2LAT(Vector3d v_ecef) {

    SpiceDouble v[3]; //ECEF vector as a spice double

    SpiceDouble r, lon, lat;

    loadAllKernels();
    
    vpack_c(v_ecef(0), v_ecef(1), v_ecef(2), v); // cast Vector 3 to SpiceDouble[3]
    reclat_c(v, &r, &lon, &lat);
    // r [input units], longitude [rad], latitude [rad]
    Vector3d latcoord (r, lon, lat);
    
    return latcoord;
}

Vector3d LAT2ECEF(Vector3d v_lat) {

    SpiceDouble ecef[3]; //ECEF vector output

    loadAllKernels();
    // r, lon, lat
    latrec_c(v_lat(0), v_lat(1), v_lat(2), ecef); // r [input units], longitude [rad], latitude [rad]
    // r [input units], longitude [rad], latitude [rad]
    Vector3d ecefcoord (ecef[0], ecef[1], ecef[2]);
    
    return ecefcoord;
}