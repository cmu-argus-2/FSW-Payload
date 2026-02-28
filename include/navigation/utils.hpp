#ifndef UTILS_HPP
#define UTILS_HPP

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "SpiceUsr.h"

using namespace Eigen;

/**
 * @brief Converts seconds since the unix epoch to seconds since J2000 epoch
 *
 * @param unixSeconds # of seconds past the unix epoch (Jan 1, 1970 0:0:0)
 * @return int64_t # of seconds past the J2000 epoch (Jan 1, 2000 11:58:55.816
 * AM)
 */
int64_t unixToJ2000(int64_t unixSeconds);

/**
 * @brief Compute the skew-symmetric, 3x3 matrix corresponding to
 * cross product with v
 *
 * @param v vector on the left side of the hypothetical cross product
 * @return Matrix_3x3 3x3 skew symmetric matrix
 */
Matrix3d toSkew(const Vector3d &v);


// CSPICE COORDINATE TRANSFORMS

/**
 * @brief Load all kernels from datapaths necessary for SPICE
 *
 * TOREAD : types of kernel according to SPICE [read https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html#Kernel%20Types]
 */
void loadAllKernels();

/**
 * @brief Cast a double[3][3] into an Eigen <double, 3, 3> matrix
 *
 * @param M SpiceDouble 3x3 matrix
 * @return R Eigen 3x3 matrix
 */
Matrix3d Cspice2Eigen(SpiceDouble M[3][3]);

/**
 * @brief Computes the rotation matrix from ECI to ECEF at a given time
 *
 * @param t_J2000 - seconds past J2000 i.e., seconds past Jan 1st 2000, 12:00:00 PM
 * @return R Eigen 3x3 matrix representing roation from ECI to ECEF
 */
Matrix3d ECI2ECEF(double t_J2000);

/**
 * @brief Computes the rotation matrix from ECEF to ECI at a given time
 *
 * @param t_J2000 - seconds past J2000 i.e., seconds past Jan 1st 2000, 12:00:00 PM
 * @return R Eigen 3x3 matrix representing roation from ECEF to ECI
 */
Matrix3d ECEF2ECI(double t_J2000);

/**
 * @brief Transforms a vector in ECEF frame to latitudinal coordinates
 *
 * @param v_ecef - vector in ECEF frame [UNITS : m]
 * @return vector in latitudinal coordinates (r, lon, lat) [m, rad, rad]
 */
Vector3d ECEF2LAT(Vector3d v_ecef);

/**
 * @brief Transforms a vector in latitudinal coordinates to ECEF frame
 *
 * @param v_lat - vector in latitudinal coordinates (r, lon, lat) [m, rad, rad]
 * @return vector in ECEF frame [UNITS : m]
 */
Vector3d LAT2ECEF(Vector3d v_lat);

#endif // UTILS_HPP