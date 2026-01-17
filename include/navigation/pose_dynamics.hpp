#ifndef POSE_DYNAMICS_HPP
#define POSE_DYNAMICS_HPP

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <core/timing.hpp>

struct InertialData
{
    std::uint64_t timestamp;
    float gyro[3]; // Angular velocity in rad/s
    float mag[3]; // Magnetic field in uT
    bool valid; // Flag to indicate if the data is valid
};

// Gyro measurements are treated as inputs to the attitude dynamics
enum GyroMeasurementIdx {
    GYRO_MEAS_TIMESTAMP = 0,
    ANG_VEL_X = 1,
    ANG_VEL_Y = 2,
    ANG_VEL_Z = 3,
    GYRO_MEAS_COUNT = 4
};


static constexpr double GM_EARTH = 3.9860044188e5;  // km^3/s^2

enum class IntegratorType {
    ForwardEuler,
    RK4
};

class DynamicsResidual {
protected:
    const int nx;
    const int num_control;
    double dt;
    const IntegratorType integrator_type = IntegratorType::ForwardEuler;
    DynamicsResidual(int nx_, int num_control_, double dt_, 
        IntegratorType integrator_type_ = IntegratorType::ForwardEuler)
        : nx(nx_), num_control(num_control_), dt(dt_), integrator_type(integrator_type_) {}
        
public:

    virtual ~DynamicsResidual() = default;

    // ceres::CostFunction override
    bool residuals(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const
    {
        const Eigen::Map<const Eigen::VectorXd> x0(parameters[0], nx);
        const Eigen::Map<const Eigen::VectorXd> x1(parameters[1], nx);

        // empty control (no controls for this dynamics)
        Eigen::VectorXd u; // size 0

        // Predict next state using the integrator (request discrete-time Phi when jacobians wanted)
        Eigen::VectorXd x_pred;
        if (jacobians) {
            Eigen::MatrixXd Phi(nx,nx);
            x_pred = this->integrate(0.0, x0, u, dt, integrator_type, &Phi);

            // Fill jacobian blocks using full state x (NX) parameterization
            // Param 0: x0 -> -Phi (NX x NX)
            if (jacobians[0]) {
                for (int row = 0; row < nx; ++row)
                    for (int col = 0; col < nx; ++col)
                        jacobians[0][row*nx + col] = -Phi(row, col);
            }
            // Param 1: x1 -> Identity (NX x NX)
            if (jacobians[1]) {
                for (int row = 0; row < nx; ++row)
                    for (int col = 0; col < nx; ++col)
                        jacobians[1][row*nx + col] = (row == col) ? 1.0 : 0.0;
            }

            // Prevent the later manual-jacobian block from overwriting these analytic blocks.
            // for (int i = 0; i < 2; ++i) jacobians[i] = nullptr;
        } else {
            x_pred = this->integrate(0.0, x0, u, dt, integrator_type);
        }

        // residuals: full state nx difference: x1 - x_pred
        Eigen::VectorXd x_res = x1 - x_pred;
        for (int i = 0; i < nx; ++i) residuals[i] = x_res(i);

        return true;
    }

    virtual int state_dim()   const = 0;
    virtual int control_dim() const = 0;

    // Continuous-time dynamics: xdot = f(t, x, u)
    virtual Eigen::VectorXd f(double t, const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;

    // Analytic Jacobians of f wrt x: xdot = f(t,x,u)
    // fx = df/dx (nx x nx)
    virtual void f_analytic_jacobians(double t,
                                      const Eigen::VectorXd& x,
                                      const Eigen::VectorXd& u,
                                      Eigen::MatrixXd* fx) const = 0;

    // Integrator (discrete-time step): x_{k+1} = fd(x_k, u_k; t_k, dt)
    Eigen::VectorXd integrate(double t,
                  const Eigen::VectorXd& x,
                  const Eigen::VectorXd& u,
                  double dt,
                  IntegratorType type = IntegratorType::ForwardEuler,
                  Eigen::MatrixXd* phi = nullptr) const
    {
        const int n = x.size();

        switch (type) {
        case IntegratorType::ForwardEuler: {
            // x_next = x + dt * f(t,x,u)
            Eigen::VectorXd x_next = x + dt * f(t, x, u);

            if (phi) {
                // Phi = I + dt * fx(t,x,u)
                phi->setZero(n, n);
                this->f_analytic_jacobians(t, x, u, phi);
                *phi = Eigen::MatrixXd::Identity(n, n) + dt * (*phi);
            }

            return x_next;
        }

        case IntegratorType::RK4: {
            // Standard RK4 state update
            Eigen::VectorXd k1 = f(t,          x,                 u);
            Eigen::VectorXd k2 = f(t + 0.5*dt, x + 0.5*dt*k1,     u);
            Eigen::VectorXd k3 = f(t + 0.5*dt, x + 0.5*dt*k2,     u);
            Eigen::VectorXd k4 = f(t + dt,     x + dt * k3,       u);
            Eigen::VectorXd x_next = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

            if (phi) {
                // Compute analytic jacobians at RK4 stages and form discrete-time Phi
                Eigen::MatrixXd fx1(n,n), fx2(n,n), fx3(n,n), fx4(n,n);

                // fx1 at x
                this->f_analytic_jacobians(t, x, u, &fx1);

                // fx2 at x + 0.5*dt*k1
                Eigen::VectorXd x_k2 = x + 0.5 * dt * k1;
                this->f_analytic_jacobians(t + 0.5*dt, x_k2, u, &fx2);

                // fx3 at x + 0.5*dt*k2
                Eigen::VectorXd x_k3 = x + 0.5 * dt * k2;
                this->f_analytic_jacobians(t + 0.5*dt, x_k3, u, &fx3);

                // fx4 at x + dt*k3
                Eigen::VectorXd x_k4 = x + dt * k3;
                this->f_analytic_jacobians(t + dt, x_k4, u, &fx4);

                // RK4 variational approximation:
                // Phi ≈ I + dt/6*(fx1 + 2*fx2 + 2*fx3 + fx4)
                *phi = Eigen::MatrixXd::Identity(n,n)
                       + (dt/6.0) * (fx1 + 2.0*fx2 + 2.0*fx3 + fx4);
            }

            return x_next;
        }

        throw std::runtime_error("DynamicsBase::integrate: unsupported IntegratorType");
        }
        return Eigen::VectorXd::Zero(n);
    }
};

class LinearDynamicsAnalytic
    : public ceres::SizedCostFunction<6, 3, 3, 3, 3>
    , public DynamicsResidual {
public:
    static constexpr int NX = 6;
    LinearDynamicsAnalytic(const double dt) : DynamicsResidual(NX, 0, dt) {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override
    {
        const Eigen::Map<const Eigen::Matrix<double, 3, 1>> r0(parameters[0]);
        const Eigen::Map<const Eigen::Matrix<double, 3, 1>> v0(parameters[1]);
        const Eigen::Map<const Eigen::Matrix<double, 3, 1>> r1(parameters[2]);
        const Eigen::Map<const Eigen::Matrix<double, 3, 1>> v1(parameters[3]);
        
        Eigen::VectorXd x0(NX);
        x0 << r0, v0;
        Eigen::VectorXd x1(NX);
        x1 << r1, v1;

        const double*params[2] = { x0.data(), x1.data() };

        if (jacobians) {
            double* jacobians_x[2];
            for (int i = 0; i < 2; ++i) {
                jacobians_x[i] = new double[NX * NX];
            }

            bool result = this->residuals(params, residuals, jacobians_x);

            // Copy back jacobians to ceres format
            // Param 0: x0 -> r0,v0
            if (jacobians[0]) {
                for (int row = 0; row < NX; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        jacobians[0][row*3 + col] = jacobians_x[0][row*NX + col];
                        jacobians[1][row*3 + col] = jacobians_x[0][row*NX + col + 3];
                    }
                }
            }
            // Param 1: x1 -> r1,v1
            if (jacobians[1]) {
                for (int row = 0; row < NX; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        jacobians[2][row*3 + col] = jacobians_x[1][row*NX + col];
                        jacobians[3][row*3 + col] = jacobians_x[1][row*NX + col + 3];
                    }
                }
            }

            for (int i = 0; i < 2; ++i) {
                delete[] jacobians_x[i];
            }

            return result;
        }
        bool result = this->residuals(params, residuals, jacobians);

        return result;
    }

    // DynamicsResidual overrides
    virtual int state_dim() const override { return nx; }
    virtual int control_dim() const override { return num_control; }

    virtual Eigen::VectorXd f(double /*t*/, const Eigen::VectorXd& x, const Eigen::VectorXd& /*u*/) const override {
        Eigen::VectorXd xdot(NX);
        Eigen::Vector3d r = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);
        double r_norm = std::sqrt(r.squaredNorm());
        if (r_norm < 1e-1) {
            // Avoid division by zero; set acceleration to zero
            xdot.segment<3>(0) = v;
            xdot.segment<3>(3) = Eigen::Vector3d::Zero();
            return xdot;
        }
        double denom = r_norm * r_norm * r_norm;
        xdot.segment<3>(0) = v;
        xdot.segment<3>(3) = -GM_EARTH * r / denom;
        return xdot;
    }

    virtual void f_analytic_jacobians(double /*t*/,
                                      const Eigen::VectorXd& x,
                                      const Eigen::VectorXd& /*u*/,
                                      Eigen::MatrixXd* fx) const override
    {
        fx->setZero(NX,NX);
        Eigen::Vector3d r = x.segment<3>(0);
        double r_norm = std::sqrt(r.squaredNorm());
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        if (r_norm < 1e-1) {
            // Avoid division by zero; set dv/dr to zero matrix
            fx->block<3,3>(0,3) = Eigen::Matrix3d::Identity(); // dr/dv
            return;
        }
        Eigen::Matrix3d uuT = (r / r_norm) * (r / r_norm).transpose();
        Eigen::Matrix3d J = (I - 3.0 * uuT) / (r_norm * r_norm * r_norm);

        // dr/dv
        fx->block<3,3>(0,3) = Eigen::Matrix3d::Identity();

        // dv/dr = -GM_EARTH * J
        fx->block<3,3>(3,0) = -GM_EARTH * J;
    }

};

// TODO: Needs to be normalized by the linear dynamics process noise covariance
// Old Autodiff function
class LinearDynamicsCostFunctor {
public:
    LinearDynamicsCostFunctor(const double dt) : dt(dt) {}

    template<typename T>
    bool operator()(const T* const pos_curr,
                    const T* const vel_curr,
                    const T* const pos_next,
                    const T* const vel_next,
                    T* const residuals) const {
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r0(pos_curr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> v0(vel_curr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> r1(pos_next);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> v1(vel_next);

    Eigen::Map <Eigen::Matrix<T, 3, 1>> r_res(residuals);
    Eigen::Map <Eigen::Matrix<T, 3, 1>> v_res(residuals + 3);

    const T eps = T(1e-8);
    const T r_norm_safe = ceres::sqrt(r0.squaredNorm() + eps*eps);
    const T denom = r_norm_safe * r_norm_safe * r_norm_safe;

    r_res = r1 - (r0 + v0 * dt);
    v_res = v1 - (v0 - (GM_EARTH * r0 / denom) * dt);
    return true;
    }

private:
    const double dt;
};

// TODO: Needs to be normalized by the angular dynamics process noise covariance

struct AngularDynamicsCostFunctor {
public:
    AngularDynamicsCostFunctor(const double* const gyro_row, const double& dt) :
            gyro_ang_vel(gyro_row + GyroMeasurementIdx::ANG_VEL_X), dt(dt) {}

    template<typename T>
    bool operator()(const T* const quat_curr,
                    const T* const gyro_bias_curr,
                    const T* const quat_next,
                    const T* const gyro_bias_next,
                    T* const residuals) const {
                        // const Eigen::Map<const Eigen::Matrix<T, 3, 1>> w0(ang_vel_curr);
    const Eigen::Matrix<T, 3, 1> gyro_w_0 = gyro_ang_vel.template cast<T>();
    
    const Eigen::Map<const Eigen::Quaternion <T>> q0(quat_curr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> b_w_0(gyro_bias_curr);
    const Eigen::Map<const Eigen::Quaternion <T>> q1(quat_next);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> b_w_1(gyro_bias_next);

    Eigen::Map <Eigen::Matrix<T, 3, 1>> q_res(residuals);  // in axis-angle form
    Eigen::Map <Eigen::Matrix<T, 3, 1>> b_res(residuals + 3);

    
    Eigen::Matrix<T, 3, 1> unbiased_gyro_ang_vel = gyro_w_0 - b_w_0;
    
    // Halving not needed because Eigen Quaternion base handles half angle
    // https://github.com/libigl/eigen/blob/1f05f51517ec4fd91eed711e0f89e97a7c028c0e/Eigen/src/Geometry/Quaternion.h#L505

    // const T half_dt = T(0.5) * T(dt);

    //Safe normalization of angular velocity vector
    // const T w_norm_sq = unbiased_gyro_ang_vel.squaredNorm();
    // const T eps = T(1e-8);
    // const T inv_w_norm = T(1.0) / ceres::sqrt(w_norm_sq + eps);
    // ceres::AngleAxisRotatePoint(unbiased_gyro_ang_vel, const T pt[3], T result[3])
    // const Eigen::Quaternion <T> dq = Eigen::Quaternion<T>(
    //         Eigen::AngleAxis<T>(unbiased_gyro_ang_vel.norm() * T(dt), unbiased_gyro_ang_vel * inv_w_norm));
    // const Eigen::AngleAxis <T> q_error = Eigen::AngleAxis<T>(q_pred.conjugate() * q1);
    // q_res = q_error.angle() * q_error.axis();

    
    Eigen::Quaternion <T> dq;
    {
        // Build angle-axis array (angle = |w|*dt, axis = w/|w|) in T type
        T angle_axis[3] = {
            unbiased_gyro_ang_vel[0] * T(dt),
            unbiased_gyro_ang_vel[1] * T(dt),
            unbiased_gyro_ang_vel[2] * T(dt)
        };
        // Ceres expects quaternion in order [x, y, z, w]
        T dq_arr[4];
        ceres::AngleAxisToQuaternion(angle_axis, dq_arr);
        // Eigen::Quaternion constructor takes (w, x, y, z)
        dq = Eigen::Quaternion<T>(dq_arr[3], dq_arr[0], dq_arr[1], dq_arr[2]);
    }
    const Eigen::Quaternion <T> q_pred = q0 * dq;
    {
        // Compute error quaternion q_err = q_pred.conjugate() * q1 and convert to array [w,x,y,z]
        const Eigen::Quaternion<T> q_err = q_pred.conjugate() * q1;
        T q_err_arr[4] = { q_err.w(), q_err.x(), q_err.y(), q_err.z() };
        // Write angle-axis into residuals
        ceres::QuaternionToAngleAxis(q_err_arr, residuals);
    }
    
    b_res = b_w_1 - b_w_0; // assuming constant bias for now
    return true;
    }

private:
    const Eigen::Map<const Eigen::Vector3d> gyro_ang_vel;
    const double& dt;
};
/*
struct AngularDynamicsCostFunctor {
public:
    AngularDynamicsCostFunctor(const double* const gyro_row, const double& dt) :
            gyro_ang_vel(gyro_row + GyroMeasurementIdx::ANG_VEL_X), dt(dt) {}

    template<typename T>
    bool operator()(const T* const quat_curr,
                    const T* const gyro_bias_curr,
                    const T* const quat_next,
                    const T* const gyro_bias_next,
                    T* const residuals) const {
                        // const Eigen::Map<const Eigen::Matrix<T, 3, 1>> w0(ang_vel_curr);
    const Eigen::Matrix<T, 3, 1> gyro_w_0 = gyro_ang_vel.template cast<T>();
    
    const Eigen::Map<const Eigen::Quaternion <T>> q0(quat_curr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> b_w_0(gyro_bias_curr);
    const Eigen::Map<const Eigen::Quaternion <T>> q1(quat_next);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> b_w_1(gyro_bias_next);

    Eigen::Map <Eigen::Matrix<T, 3, 1>> q_res(residuals);  // in axis-angle form
    Eigen::Map <Eigen::Matrix<T, 3, 1>> b_res(residuals + 3);

    
    Eigen::Matrix<T, 3, 1> unbiased_gyro_ang_vel = gyro_w_0 - b_w_0;
    
    // Halving not needed because Eigen Quaternion base handles half angle
    // https://github.com/libigl/eigen/blob/1f05f51517ec4fd91eed711e0f89e97a7c028c0e/Eigen/src/Geometry/Quaternion.h#L505

    // const T half_dt = T(0.5) * T(dt);

    //Safe normalization of angular velocity vector
    const T w_norm_sq = unbiased_gyro_ang_vel.squaredNorm();
    const T eps = T(1e-8);
    const T inv_w_norm = T(1.0) / ceres::sqrt(w_norm_sq + eps);

    const Eigen::Quaternion <T> dq = Eigen::Quaternion<T>(
            Eigen::AngleAxis<T>(unbiased_gyro_ang_vel.norm() * T(dt), unbiased_gyro_ang_vel * inv_w_norm));
    const Eigen::Quaternion <T> q_pred = q0 * dq;
    const Eigen::AngleAxis <T> q_error = Eigen::AngleAxis<T>(q_pred.conjugate() * q1);
    q_res = q_error.angle() * q_error.axis();
    b_res = b_w_1 - b_w_0; // assuming constant bias for now
    return true;
    }

private:
    const Eigen::Map<const Eigen::Vector3d> gyro_ang_vel;
    const double& dt;
};

*/

# endif // POSE_DYNAMICS_HPP