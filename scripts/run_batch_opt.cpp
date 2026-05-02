#include "navigation/batch_optimization.hpp"
#include "navigation/od.hpp"

#include <ceres/internal/eigen.h>
// #include <xtensor/xtensor.hpp>
#include <Eigen/Eigen>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include <string>
#include <iostream>
#include <cassert>
#include <exception>

int main(int argc, char** argv) {
      // TODO: WIP Fix path to correct HDF5 file
      std::string foldername = "data/datasets/batch_opt_gen/";
      // std::string filename = "data/datasets/batch_opt_gen_no_bias/orbit_measurements.h5";
      std::string filename = foldername + "orbit_measurements.h5";

      if (argc > 1) {
        filename = std::string(argv[1]);
      }

      // Load OD configuration file
      std::string config_filename = "config/od.toml";
      OD_Config od_config = ReadODConfig(config_filename);

      std::cout << "Loaded OD configuration from: " << config_filename << std::endl;

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

      //Print first few rows of each measurement
      std::cout << "Landmark Measurements (first 5 rows):\n" <<
          lm.topRows(5) << std::endl;
      std::cout << "Gyro Measurements (first 5 rows):\n" <<
          gm.topRows(5) << std::endl;

      // Run Ceres batch optimization
      auto [state_estimates, covariance, residuals] = solve_ceres_batch_opt(lm, gs, gm, od_config.batch_opt);
      // Compute residuals of state estimates
      
    try {
        const std::string out_filename = foldername + "state_estimates.h5";
        H5Easy::File outfile(out_filename, HighFive::File::Overwrite);
        // write state_estimates to an HDF5 file (overwrites if exists)
        H5Easy::dump(outfile, "state_estimates", state_estimates);
        H5Easy::dump(outfile, "state_estimate_covariance_diagonal", covariance);
        H5Easy::dump(outfile, "residuals", residuals);
        std::cout << "Saved state estimates to " << out_filename << std::endl;
    } catch (const HighFive::Exception& e) {
        std::cerr << "Failed to write state estimates: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error writing state estimates: " << e.what() << std::endl;
    }
    return 0;
}
