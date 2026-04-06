#include "navigation/batch_optimization.hpp"
#include "navigation/pose_dynamics.hpp"
#include "navigation/measurement_residuals.hpp"
#include <ceres/ceres.h>
#include <ceres/internal/eigen.h>
// #include <xtensor/xtensor.hpp>
#include <Eigen/Eigen>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <memory>

// tests/navigation/test_jacobians.cpp relies on M_PI, std::abs, and std::max but does not include <cmath> / <algorithm>. 
// It also allocates multiple arrays with new (jacobians_*, AutoDiffCostFunction) without freeing them, 
// which makes this test executable leak memory and depend on transitive includes. 
// Add the missing standard headers and use RAII containers (std::array/std::vector/std::unique_ptr) for allocated buffers.

void test_linear_dynamics_cost_functor() {
    std::cout << "Testing LinearDynamicsCostFunctor Jacobians...\n";
    double dt = 1.0;
    LinearDynamicsCostFunctor linear_dyn(dt, 1.0, 1.0);
    LinearDynamicsAnalytic linear_dyn_analytic(dt, 1.0, 1.0);

    auto* ad_linear_dyn = new ceres::AutoDiffCostFunction<LinearDynamicsCostFunctor, 6, 3, 3, 3, 3>(
        new LinearDynamicsCostFunctor{dt, 1.0, 1.0}
    );

    
    // D: state/residual dimension
    const int D = 6;
    const int TD = 12;
    const int V = 3;
    const double tol = 1e-9;

    double residuals_ad[D];
    double residuals_an[D];

    double* jacobians_an[4];
    for (int i = 0; i < 4; ++i) {
        jacobians_an[i] = new double[D * V];
    }
    double* jacobians_ad[4];
    for (int i = 0; i < 4; ++i) {
        jacobians_ad[i] = new double[D * V];
    }

    // Predefined list of state vectors (r0(3), v0(3), r1(3), v1(3))
    std::vector<std::array<double, TD>> tests{
        std::array<double, TD>{0.0,0.0,0.0, 0.0,0.0,0.0,  0.0,0.0,0.0, 0.0,0.0,0.0},
        std::array<double, TD>{1.0,2.0,3.0, 0.1,0.2,0.3,  1.5,2.5,3.5, 0.15,0.25,0.35},
        std::array<double, TD>{-1.0,0.5,2.0, 0.0,-0.1,0.2,  0.0,1.0,-1.0, 0.1,0.0,-0.1},
        std::array<double, TD>{10.0, -5.0, 3.3, 1.1, 2.2, -0.5, 9.9, -4.8, 3.0, 1.0, 2.0, -0.4}
    };

    for (size_t t = 0; t < tests.size(); ++t) {
        const auto &data = tests[t];

        // parameter block pointers for autodiff (point to x0, x1)
        const double* param_blocks[4] = {
            data.data() + 0,   // r0
            data.data() + 3,   // v0
            data.data() + 6,   // r1
            data.data() + 9    // v1
        };

        // prepare analytic input (double** with single block as before)
        const double* rv_01[2] = {
            data.data() + 0,   // r0
            data.data() + 6    // r1
        };


        // time autodiff Evaluate
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok_ad = ad_linear_dyn->Evaluate(param_blocks, residuals_ad, jacobians_ad);
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_ad_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // time analytic Evaluate
        auto t2 = std::chrono::high_resolution_clock::now();
        bool ok_an = linear_dyn_analytic.Evaluate(param_blocks, residuals_an, jacobians_an);
        auto t3 = std::chrono::high_resolution_clock::now();
        double time_an_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "time autodiff Evaluate: " << time_ad_ms << " ms\n";
        std::cout << "time analytic Evaluate: " << time_an_ms << " ms\n";

        if (!ok_ad) std::cout << "autodiff Evaluate failed\n";
        if (!ok_an) std::cout << "analytic Evaluate failed\n";

        if (ok_ad && ok_an) {
            // print residuals and their difference
            std::cout << "residuals (autodiff): ";
            for (int i = 0; i < D; ++i) std::cout << residuals_ad[i] << " ";
            std::cout << "\nresiduals (analytic): ";
            for (int i = 0; i < D; ++i) std::cout << residuals_an[i] << " ";
            std::cout << "\nresiduals diff: ";
            double max_res_diff = 0.0;
            for (int i = 0; i < D; ++i) {
                double d = std::abs(residuals_ad[i] - residuals_an[i]);
                max_res_diff = std::max(max_res_diff, d);
                std::cout << (residuals_ad[i] - residuals_an[i]) << " ";
            }
            std::cout << "\nmax residual abs diff = " << max_res_diff << "\n";

            // assemble full autodiff jacobian into row-major D x TD
            std::vector<double> J_ad(D * TD, 0.0);
            for (int b = 0; b < 4; ++b) {
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < V; ++j) {
                        int col = b * V + j;
                        J_ad[i * TD + col] = jacobians_ad[b][i * V + j];
                    }
                }
            }

            // double* J_an = jacobian2[0];
            std::vector<double> J_an(D * TD, 0.0);
            for (int b = 0; b < 4; ++b) {
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < V; ++j) {
                        int col = b * V + j;
                        J_an[i * TD + col] = jacobians_an[b][i * V + j];
                    }
                }
            }

            // compare jacobians
            double max_jdiff = 0.0;
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < TD; ++j) {
                    double a = J_ad[i * TD + j];
                    double bval = J_an[i * TD + j];
                    double d = std::abs(a - bval);
                    max_jdiff = std::max(max_jdiff, d);
                }
            }

            std::cout << "Jacobian (autodiff):\n";
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < TD; ++j) {
                    std::cout << J_ad[i * TD + j] << (j + 1 == TD ? "" : " ");
                }
                std::cout << "\n";
            }

            std::cout << "Jacobian (analytic):\n";
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < TD; ++j) {
                    std::cout << J_an[i * TD + j] << (j + 1 == TD ? "" : " ");
                }
                std::cout << "\n";
            }

            std::cout << "Jacobian diff (autodiff - analytic):\n";
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < TD; ++j) {
                    double d = J_ad[i * TD + j] - J_an[i * TD + j];
                    std::cout << d << (j + 1 == TD ? "" : " ");
                }
                std::cout << "\n";
            }

            std::cout << "max jacobian abs diff = " << max_jdiff;
            if (max_jdiff > tol) std::cout << "  (exceeds tol " << tol << ")";
            std::cout << "\n";
        } else {
            std::cout << "Skipping comparison due to Evaluate failure\n";
        }
    }

    // cleanup
    for (int i = 0; i < 4; ++i) {
        delete[] jacobians_an[i];
        delete[] jacobians_ad[i];
    }

    delete ad_linear_dyn;

    return;
}

void test_angular_dynamics_cost_functor() {
    std::cout << "Testing AngularDynamicsCostFunctor Jacobians...\n";
    double dt = 1.0;

    GyroMeasurements gyro_measurements = Eigen::MatrixXd::Zero(4, GyroMeasurementIdx::GYRO_MEAS_COUNT);
    gyro_measurements(1,GyroMeasurementIdx::ANG_VEL_X) = M_PI / 2.0;
    gyro_measurements(2,GyroMeasurementIdx::ANG_VEL_X) = M_PI / 2.0+0.1;
    gyro_measurements(3,GyroMeasurementIdx::ANG_VEL_X) = M_PI / 2.0;
    AngularDynamicsCostFunctor angular_dyn(gyro_measurements.data(), dt, 1.0, 1.0);
    // AngularDynamicsAnalytic angular_dyn_analytic(dt);

    auto* ad_angular_dyn = new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctor, 6, 4, 3, 4, 3>(
        new AngularDynamicsCostFunctor{gyro_measurements.data(), dt, 1.0, 1.0}
    );
    
    // D: state/residual dimension
    const int D = 6;
    const int TD = 14;
    const int V_array[4] = {4, 3, 4, 3};
    const double tol = 1e-9;

    double residuals_ad[D];
    double residuals_an[D];

    double* jacobians_an[4];
    for (int i = 0; i < 4; ++i) {
        jacobians_an[i] = new double[D * V_array[i]];
    }
    double* jacobians_ad[4];
    for (int i = 0; i < 4; ++i) {
        jacobians_ad[i] = new double[D * V_array[i]];
    }

    // Predefined list of state vectors (q0(4), omega0(3), q1(4), omega1(3))
    // q = [x,y,z,w]
    std::vector<std::array<double, TD>> tests{
        std::array<double, TD>{0.0,0.0,0.0,1.0, 0.0,0.0,0.0, 0.0,0.0,0.0,1.0, 0.0,0.0,0.0},
        std::array<double, TD>{0.0,0.0,0.0,1.0, 0.0,0.0,0.0, 0.7071,0.0,0.0,0.7071, 0.0,0.0,0.0},
        std::array<double, TD>{0.0,0.0,0.0,1.0, 0.1,0.0,0.0, 0.7071,0.0,0.0,0.7071, 0.1,0.0,0.0},
        std::array<double, TD>{0.0,0.0,0.0,1.0, 0.1,0.0,0.0, 0.7071,0.0,0.0,0.7071, 0.0,0.0,0.0},
    };

    for (size_t t = 0; t < tests.size(); ++t) {
        const auto &data = tests[t];

        // parameter block pointers for autodiff (point to x0, x1)
        const double* param_blocks[4] = {
            data.data() + 0,   // q0
            data.data() + 4,   // bw0
            data.data() + 7,   // q1
            data.data() + 11   // bw1
        };

        auto* gyro_row = gyro_measurements.data() + GyroMeasurementIdx::GYRO_MEAS_COUNT * t;
        
        auto* ad_angular_dyn = new ceres::AutoDiffCostFunction<AngularDynamicsCostFunctor, 6, 4, 3, 4, 3>(
            new AngularDynamicsCostFunctor{gyro_row, dt, 1.0, 1.0}
        );

        // time autodiff Evaluate
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok_ad = ad_angular_dyn->Evaluate(param_blocks, residuals_ad, jacobians_ad);
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_ad_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "time autodiff Evaluate: " << time_ad_ms << " ms\n";

        if (!ok_ad) std::cout << "autodiff Evaluate failed\n";

        if (ok_ad) {
            // print residuals and their difference
            std::cout << "residuals (autodiff): ";
            for (int i = 0; i < D; ++i) std::cout << residuals_ad[i] << " ";

            // assemble full autodiff jacobian into row-major D x TD
            std::vector<double> J_ad(D * TD, 0.0);
            int col_offset = 0;
            for (int b = 0; b < 4; ++b) {
                col_offset = (b == 0) ? 0 : col_offset + V_array[b-1];
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < V_array[b]; ++j) {
                        int col = col_offset + j;
                        J_ad[i * TD + col] = jacobians_ad[b][i * V_array[b] + j];
                    }
                }
            }

            std::cout << "Jacobian (autodiff):\n";
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < TD; ++j) {
                    std::cout << J_ad[i * TD + j] << (j + 1 == TD ? "" : " ");
                }
                std::cout << "\n";
            }

        } else {
            std::cout << "Skipping comparison due to Evaluate failure\n";
        }

    }

    // cleanup
    for (int i = 0; i < 4; ++i) {
        delete[] jacobians_an[i];
        delete[] jacobians_ad[i];
    }

    delete ad_angular_dyn;

    return;
}

// TODO: Landmark measurement test function
void test_landmark_residuals_cost_functor() {
    std::cout << "Testing LandmarkCostFunctor Jacobians...\n";
    double dt = 1.0;
    
    // Create landmark measurements: timestamp, earing vector, landmark position
    LandmarkMeasurements landmark_measurements = Eigen::MatrixXd::Zero(7, LandmarkMeasurementIdx::LANDMARK_COUNT);
    landmark_measurements(0, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP) = 0.0;
    landmark_measurements(0, LandmarkMeasurementIdx::BEARING_VEC_X) = 0.7071;
    landmark_measurements(0, LandmarkMeasurementIdx::BEARING_VEC_Y) = 0.0;
    landmark_measurements(0, LandmarkMeasurementIdx::BEARING_VEC_Z) = 0.7071;
    landmark_measurements(0, LandmarkMeasurementIdx::LANDMARK_POS_X) = 10.0;
    landmark_measurements(0, LandmarkMeasurementIdx::LANDMARK_POS_Y) = 5.0;
    landmark_measurements(0, LandmarkMeasurementIdx::LANDMARK_POS_Z) = 2.0;
    
    landmark_measurements(1, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP) = 1.0;
    landmark_measurements(1, LandmarkMeasurementIdx::BEARING_VEC_X) = 0.7071;
    landmark_measurements(1, LandmarkMeasurementIdx::BEARING_VEC_Y) = 0.0;
    landmark_measurements(1, LandmarkMeasurementIdx::BEARING_VEC_Z) = 0.7071;
    landmark_measurements(1, LandmarkMeasurementIdx::LANDMARK_POS_X) = 10.5;
    landmark_measurements(1, LandmarkMeasurementIdx::LANDMARK_POS_Y) = 5.2;
    landmark_measurements(1, LandmarkMeasurementIdx::LANDMARK_POS_Z) = 2.1;
    double landmark_std_dev = 1.0;
    
    // auto* ad_landmark = new ceres::AutoDiffCostFunction<LandmarkCostFunctor, 3, 3, 4>(
    //     new LandmarkCostFunctor{landmark_measurements.data(), landmark_std_dev}
    // );
    
    // D: residual dimension (3 for position error)
    const int D = 3;
    const int TD = 7;  // r0(3) + q0(4)
    const int V_array[2] = {3, 4};
    const double tol = 1e-8;
    
    double residuals_ad[D];
    
    double* jacobians_ad[2];
    for (int i = 0; i < 2; ++i) {
        jacobians_ad[i] = new double[D * V_array[i]];
    }
    
    // Test cases: r0(3), q0(4)
    std::vector<std::array<double, TD>> tests{
        std::array<double, TD>{9.9, 5.1, 1.9,  0.0, 0.0, 0.0, 1.0},
        std::array<double, TD>{10.0, 5.0, 2.0,  0.0, 0.0, 0.0, 1.0},
    };
    
    for (size_t t = 0; t < tests.size(); ++t) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Test case " << t << ":\n";
        const auto &data = tests[t];
        
        // parameter block pointers: r0, q0
        const double* param_blocks[2] = {
            data.data() + 0,   // r0
            data.data() + 3,   // q0
        };
        
        auto* landmark_row = landmark_measurements.data() + LandmarkMeasurementIdx::LANDMARK_COUNT * t;
        
        auto* ad_landmark = new ceres::AutoDiffCostFunction<LandmarkCostFunctor, 3, 3, 4>(
            new LandmarkCostFunctor{landmark_row, landmark_std_dev}
        );
        
        // time autodiff Evaluate
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok_ad = ad_landmark->Evaluate(param_blocks, residuals_ad, jacobians_ad);
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_ad_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        
        std::cout << "time autodiff Evaluate: " << time_ad_ms << " ms\n";
        
        if (!ok_ad) std::cout << "autodiff Evaluate failed\n";
        
        if (ok_ad) {
            std::cout << "residuals (autodiff): ";
            for (int i = 0; i < D; ++i) std::cout << residuals_ad[i] << " ";
            std::cout << "\n";
            
            // assemble full autodiff jacobian
            std::vector<double> J_ad(D * TD, 0.0);
            int col_offset = 0;
            for (int b = 0; b < 2; ++b) {
                col_offset = (b == 0) ? 0 : col_offset + V_array[b-1];
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < V_array[b]; ++j) {
                        int col = col_offset + j;
                        J_ad[i * TD + col] = jacobians_ad[b][i * V_array[b] + j];
                    }
                }
            }
            
            std::cout << "Jacobian (autodiff):\n";
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < TD; ++j) {
                    std::cout << J_ad[i * TD + j] << (j + 1 == TD ? "" : " ");
                }
                std::cout << "\n";
            }
        } else {
            std::cout << "Skipping jacobian output due to Evaluate failure\n";
        }
        std::cout << "\n";
        delete ad_landmark;
    }

    // cleanup
    for (int i = 0; i < 2; ++i) {
        delete[] jacobians_ad[i];
    }
    
    return;
}

// TODO: Once ready, adapt these to the Google Test framework
int main(int argc, char** argv) {
    test_linear_dynamics_cost_functor();
    test_angular_dynamics_cost_functor();
    test_landmark_residuals_cost_functor();
    std::cout << "All Jacobian tests completed.\n";
    return 0;
}