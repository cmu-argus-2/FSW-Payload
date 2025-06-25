#include "navigation/batch_optimization.hpp"

#include <ceres/internal/eigen.h>
// #include <xtensor/xtensor.hpp>
#include <Eigen/Eigen>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>

// using namespace HighFive;
 

int main() {
      // TODO: WIP Fix path to correct HDF5 file
      std::string filename = "/home/frederik/cmu/GNC-Payload/batch_opt_gen/orbit_measurements.h5";

      HighFive::File file(filename, HighFive::File::ReadOnly);

      // Directly load into an Eigen MatrixXd (column-major):
      Eigen::MatrixXd landmarks = H5Easy::load<Eigen::MatrixXd>(file, "landmark_measurements");
      Eigen::MatrixXd gyro_measurements = H5Easy::load<Eigen::MatrixXd>(file, "gyro_measurements");
      Eigen::MatrixXd landmark_group_starts = H5Easy::load<Eigen::MatrixXd>(file, "group_starts");

      LandmarkMeasurements       lm   = landmarks;
      GyroMeasurements           gm   = gyro_measurements;
      LandmarkGroupStarts        gs   = landmark_group_starts.cast<bool>();

      // correct dims?
      assert(lm.cols()   == LandmarkMeasurementIdx::LANDMARK_COUNT);
      assert(gm.cols()   == GyroMeasurementIdx::GYRO_MEAS_COUNT);
      assert(gs.rows()   == lm.rows());

      std::cout << "Loaded measurements from HDF5 file." << std::endl;
      // Run Ceres batch optimization
      StateEstimates state_estimates = solve_ceres_batch_opt(lm, gs, gm, 60.0);
    return 0;
}
