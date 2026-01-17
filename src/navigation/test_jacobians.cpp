#include "navigation/batch_optimization.hpp"
#include "navigation/pose_dynamics.hpp"
#include <ceres/ceres.h>
#include <ceres/internal/eigen.h>
// #include <xtensor/xtensor.hpp>
#include <Eigen/Eigen>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    double dt = 1.0;
    LinearDynamicsCostFunctor linear_dyn(dt);
    LinearDynamicsAnalytic linear_dyn_analytic(dt);

    auto* ad_linear_dyn = new ceres::AutoDiffCostFunction<LinearDynamicsCostFunctor, 6, 3, 3, 3, 3>(
        new LinearDynamicsCostFunctor{dt}
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

    // fix the angular dynamics

    // cleanup
    // free the single contiguous jacobian block and the row pointer array
    // for (int i = 0; i < 4; ++i) delete [] jacobians_ad[i];
    // for (int i = 0; i < 2; ++i) delete [] jacobian2[i];


    /*
    // Gradient checking
    GradientChecker gradient_checker(my_cost_function,
                                 manifolds,
                                 numeric_diff_options);
    GradientCheckResults results;
    if (!gradient_checker.Probe(parameter_blocks.data(), 1e-9, &results) {
    LOG(ERROR) << "An error has occurred:\n" << results.error_log;
    }
    */
}

