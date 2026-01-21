#ifndef TEST_JACOBIANS_HPP
#define TEST_JACOBIANS_HPP

#include "navigation/batch_optimization.hpp"
#include "navigation/pose_dynamics.hpp"
#include <ceres/ceres.h>
#include <ceres/internal/eigen.h>
// #include <xtensor/xtensor.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <math.h>


// TODO: Once ready, move to tests folder 
int main(int argc, char** argv);

void test_linear_dynamics_cost_functor();

void test_angular_dynamics_cost_functor();

#endif // TEST_JACOBIANS_HPP