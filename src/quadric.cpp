#include "quadric.h"
#include "utils.h"
#include "common.h"

#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <utility>

#define MIN_ANGLE 0.04

typedef Eigen::Matrix<double, 5, 1> Vector5; // elliptic paraboloid variables/coefficients
typedef Eigen::Matrix<double, 6, 1> Vector6; // arbitrary quadric variables/coefficients

class QuadraticCostFunction : public ceres::SizedCostFunction<1, 5> {
public:
    QuadraticCostFunction(Vector5 coefs, double value) : coefs_(std::move(coefs)), value_(value) {}

    virtual ~QuadraticCostFunction() {}

    virtual bool Evaluate(double const *const *x,
                          double *residuals,
                          double **jacobians) const {
        residuals[0] = 0.0;
        for (int j = 0; j < 5; ++j) {
            residuals[0] += x[0][j] * coefs_[j];
        }
        residuals[0] -= value_;
        // Compute the Jacobian if asked for.
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            for (int j = 0; j < 5; ++j) {
                jacobians[0][j] = coefs_[j];
            }
        }
        return true;
    }

protected:
    Vector5 coefs_;
    double value_;
};

void saveSaliencies(const Eigen::MatrixXd &xs, const Eigen::MatrixXd &ys, const Eigen::VectorXd &values,
                    const Vector5 &coefs, int index) {
    std::cerr << coefs(0) << " " << coefs(1) << " " << coefs(2) << " " << coefs(3) << " " << coefs(4) << "\n";
    PointNCloud::Ptr pcd(new PointNCloud), pcd_pd(new PointNCloud);
    double m = *std::max_element(values.data(), values.data() + values.size());
    double min_x = std::numeric_limits<double>::max(), max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max(), max_y = std::numeric_limits<double>::lowest();
    for (int i = 0; i < values.size(); ++i) {
        pcd->push_back(PointN{(float)xs(i), (float)ys(i), (float) (values(i) / m / 200.0)});
        min_x = std::min(min_x, xs(i));
        min_y = std::min(min_y, ys(i));
        max_x = std::max(max_x, xs(i));
        max_y = std::max(max_y, ys(i));
    }
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            float x = min_x + i * (max_x - min_x) / 100;
            float y = min_y + j * (max_y - min_y) / 100;
            float z = (coefs(0) * x * x + coefs(1) * y * y + coefs(2) * x + coefs(3) * y + coefs(4)) / m / 200.0;
            pcd_pd->push_back(PointN{x, y, z});
        }
    }
    saveColorizedPointCloud(pcd, Eigen::Matrix4f::Identity(), COLOR_PURPLE, constructPath("saliencies", std::to_string(index), "ply", false));
    saveColorizedPointCloud(pcd_pd, Eigen::Matrix4f::Identity(), COLOR_PURPLE, constructPath("saliencies_all", std::to_string(index), "ply", false));
}

Eigen::Vector2d estimateMaximumPointEP(const Eigen::VectorXd &xs, const Eigen::VectorXd &ys,
                                       const Eigen::VectorXd &values, int index) {
    Vector5 estimated_coefs;
    estimated_coefs.setZero();

    ceres::Problem problem;
    for (int i = 0; i < values.size(); ++i) {
        Vector5 a;
        a << xs(i) * xs(i), ys(i) * ys(i), xs(i), ys(i), 1;
        problem.AddResidualBlock(new QuadraticCostFunction(a, values(i)),
                                 nullptr, estimated_coefs.data());
    }
//    problem.SetParameterUpperBound(estimated_coefs.data(), 0, 0.0);
//    problem.SetParameterUpperBound(estimated_coefs.data(), 1, 0.0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    saveSaliencies(xs, ys, values, estimated_coefs, index);
    double x = xs(0), y= ys(0);
    if (std::fabs(estimated_coefs[0]) >= 1e-9) {
        x = -estimated_coefs[2] / 2.0 / estimated_coefs[0];
    }
    if (std::fabs(estimated_coefs[1]) >= 1e-9) {
        y = -estimated_coefs[3] / 2.0 / estimated_coefs[1];
    }
    if (std::fabs(estimated_coefs[0]) <= 1e-9 && std::fabs(estimated_coefs[1]) <= 1e-9) {
        std::cerr << xs.transpose() << "\n";
        std::cerr << ys.transpose() << "\n";
        std::cerr << values.transpose() << "\n";
        std::cerr << "coefs: ";
        for (int i = 0; i < 5; ++i) {
            std::cerr << estimated_coefs[i] << " ";
        }
        std::cerr << "\n";
        x = 1;
        y = 1;
    }
    rassert(std::isfinite(x) && std::isfinite(y), 13049387429308932);
    return {x, y};
}

double estimatePointOnQuadric(const Eigen::VectorXd &xs, const Eigen::VectorXd &ys,
                              const Eigen::VectorXd &values, double x, double y) {
    Eigen::MatrixXd A;
    A.resize(xs.size(), 6);
    for (int i = 0; i < xs.size(); ++i) {
        A(i, 0) = xs(i) * xs(i);
        A(i, 1) = xs(i) * ys(i);
        A(i, 2) = ys(i) * ys(i);
        A(i, 3) = xs(i);
        A(i, 4) = ys(i);
        A(i, 5) = 1;
    }
    Vector6 coefs = (A.transpose() * A).inverse() * A.transpose() * values;
    double z = coefs(0) * x * x + coefs(1) * x * y + coefs(2) * y * y + coefs(3) * x + coefs(4) * y + coefs(5);
    rassert(std::isfinite(z), 435842394921);
    return z;
}

Eigen::Matrix3d calculateRotationToAlignZAxis(const Eigen::Vector3d &vector) {
    Eigen::Vector3d z_axis = {0.0, 0.0, 1.0};
    double angle = std::acos(vector.normalized().dot(z_axis));
    if (std::fabs(angle) < MIN_ANGLE) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d axis = z_axis.cross(vector.normalized());
    return Eigen::AngleAxis(angle, axis).toRotationMatrix();
}

Eigen::Vector3d estimateMaximumPoint(const Eigen::MatrixXd &points, const Eigen::Vector3d &normal,
                                     const Eigen::VectorXd &values, int index) {
    Eigen::Matrix3d rot_matrix = calculateRotationToAlignZAxis(normal);
    Eigen::MatrixXd rotated_points = (rot_matrix * (points.transpose())).transpose();
    Eigen::Vector2d maximum_point = estimateMaximumPointEP(rotated_points.col(0), rotated_points.col(1), values, index);
    double z = estimatePointOnQuadric(rotated_points.col(0), rotated_points.col(1), rotated_points.col(2),
                                      maximum_point.x(), maximum_point.y());
    return rot_matrix.inverse() * Eigen::Vector3d{maximum_point.x(), maximum_point.y(), z};
}