#include "navigation/batch_optimization.hpp"

#include <ceres/internal/eigen.h>
// #include <xtensor/xtensor.hpp>
#include <Eigen/Eigen>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>

// using namespace HighFive;
 

int main(int argc, char** argv) {
      // TODO: WIP Fix path to correct HDF5 file
    std::string filename = "data/datasets/batch_opt_gen/orbit_measurements.h5";
    
      if (argc > 1) {
        filename = std::string(argv[1]);
      }


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
      StateEstimates state_estimates = solve_ceres_batch_opt(lm, gs, gm, 60.0);

    try {
        const std::string out_filename = "data/datasets/batch_opt_gen/state_estimates.h5";
        H5Easy::File outfile(out_filename, H5Easy::File::Overwrite);
        // write state_estimates to an HDF5 file (overwrites if exists)
        H5Easy::dump(outfile, "state_estimates", state_estimates);
        std::cout << "Saved state estimates to " << out_filename << std::endl;
    } catch (const HighFive::Exception& e) {
        std::cerr << "Failed to write state estimates: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error writing state estimates: " << e.what() << std::endl;
    }
    return 0;
}
