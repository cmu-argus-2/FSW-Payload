#ifndef TRAJECTORY_INITIALIZER_HPP
#define TRAJECTORY_INITIALIZER_HPP

#include "navigation/batch_optimization.hpp"
#include <string>
#include <vector>

// Computes an initial trajectory guess from landmark measurements and IMU data.
//
// Position is initialized from the mean ECI direction of landmarks per group,
// scaled to Earth radius + 600 km. Attitude is estimated per group via Wahba's
// problem (SVD) on bearing-to-ECI vector pairs. Velocity is derived from finite
// differences of position. Gyro bias is estimated by differentiating the attitude
// and comparing against the measured angular rate.
//
// Intermediate results are accessible via state_estimates(). write_csv() writes
// the full trajectory in the same column format as state_estimates.csv for easy
// overlay comparison in plot_batch_opt.py.
class TrajectoryInitializer {
public:
    TrajectoryInitializer(
        const std::vector<double>&  state_timestamps,
        const LandmarkMeasurements& landmark_measurements,
        const LandmarkGroupStarts&  landmark_group_starts,
        BIAS_MODE                   bias_mode,
        const GyroMeasurements&     gyro_measurements);

    const StateEstimates& state_estimates() const { return state_estimates_; }

    // Writes initial_trajectory.csv into results_dir with the same header and
    // column layout as state_estimates.csv.
    void write_csv(const std::string& results_dir) const;

private:
    StateEstimates state_estimates_;
};

#endif // TRAJECTORY_INITIALIZER_HPP
