#include "quadric.h"
#include "utils.h"
#include "common.h"

#include <Eigen/Geometry>
#include <utility>

#define MIN_ANGLE 0.04

typedef Eigen::Matrix<double, 5, 1> Vector5; // elliptic paraboloid variables/coefficients
typedef Eigen::Matrix<double, 6, 1> Vector6; // arbitrary quadric variables/coefficients

void placeCenterAtBeginning(Eigen::MatrixXd &points, Eigen::VectorXd &values) {
    if (points.rows() == 0) return;
    double max_value = std::numeric_limits<double>::lowest();
    int index = -1;
    for (int i = 0; i < points.rows(); ++i) {
        if (values(i) > max_value) {
            max_value = values(i);
            index = i;
        }
    }
    std::swap(points(index, 0), points(0, 0));
    std::swap(points(index, 1), points(0, 1));
    std::swap(points(index, 2), points(0, 2));
    std::swap(values(index), points(0));
}

double estimateRadius(const Eigen::MatrixXd &xs, const Eigen::MatrixXd &ys) {
    double radius2 = 0.0;
    for (int i = 0; i < xs.rows(); ++i) {
        radius2 = std::max(radius2, (xs(i) - xs(0)) * (xs(i) - xs(0)) + (ys(i) - ys(0)) * (ys(i) - ys(0)));
    }
    return std::sqrt(radius2);
}
std::tuple<double, double, double> estimateMaximumPointOnInterval(const Vector6 &coefs,
                                                                  double x1, double y1, double x2, double y2) {
    // p0 * x^2 + p1 * x * y + p2 * y^2 + p3 * x + p4 * y + p5
    // (x - x1) / (x1 - x2) = (y - y1) / (y1 - y2)
    // y = (x - x1) / (x1 - x2) * (y1 - y2) + y1
    return {};
}
void saveSaliencies(const Eigen::MatrixXd &xs, const Eigen::MatrixXd &ys, const Eigen::VectorXd &values,
                    const Vector6 &coefs, int index) {
    PointNCloud::Ptr pcd(new PointNCloud), pcd_pd(new PointNCloud);
    double m = *std::max_element(values.data(), values.data() + values.size());
    double min_x = std::numeric_limits<double>::max(), max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max(), max_y = std::numeric_limits<double>::lowest();
    for (int i = 0; i < values.size(); ++i) {
        pcd->push_back(PointN{(float)xs(i), (float)ys(i), (float) (values(i) / m / 200.0)});
        pcd->push_back(PointN{(float)xs(i), (float)ys(i), 0.f});
        min_x = std::min(min_x, xs(i));
        min_y = std::min(min_y, ys(i));
        max_x = std::max(max_x, xs(i));
        max_y = std::max(max_y, ys(i));
    }
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            float x = min_x + i * (max_x - min_x) / 100;
            float y = min_y + j * (max_y - min_y) / 100;
            float z = (coefs(0) * x * x + coefs(1) * x * y + coefs(2) * y * y + coefs(3) * x + coefs(4) * y + coefs(5)) / m / 200.0;
            pcd_pd->push_back(PointN{x, y, z});
            pcd_pd->push_back(PointN{x, y, 0.f});
        }
    }
    saveColorizedPointCloud(pcd, Eigen::Matrix4f::Identity(), COLOR_PURPLE, constructPath("saliencies", std::to_string(index), "ply", false));
    saveColorizedPointCloud(pcd_pd, Eigen::Matrix4f::Identity(), COLOR_PURPLE, constructPath("saliencies_all", std::to_string(index), "ply", false));
}

Eigen::Vector2d estimateMaximumPointEP(const Eigen::VectorXd &xs, const Eigen::VectorXd &ys,
                                       const Eigen::VectorXd &values, int index) {
    Vector6 estimated_coefs_cf;
    Eigen::MatrixXd A;
    A.resize(xs.rows(), 6);
    for (int i = 0; i < xs.size(); ++i) {
        A(i, 0) = xs(i) * xs(i);
        A(i, 1) = xs(i) * ys(i);
        A(i, 2) = ys(i) * ys(i);
        A(i, 3) = xs(i);
        A(i, 4) = ys(i);
        A(i, 5) = 1.f;
    }
    estimated_coefs_cf = (A.transpose() * A).inverse() * A.transpose() * values;
    std::cerr << "closed form coefs: " << estimated_coefs_cf.transpose() << "\n";
    saveSaliencies(xs, ys, values, estimated_coefs_cf, index);
    Eigen::Matrix2d A_;
    A_ << 2 * estimated_coefs_cf(0), estimated_coefs_cf(1), estimated_coefs_cf(1), 2 * estimated_coefs_cf(2);
    Eigen::Vector2d b_;
    b_ << -estimated_coefs_cf(3), - estimated_coefs_cf(4);
    Eigen::Vector2d point = A_.inverse() * b_;
    double radius = estimateRadius(xs, ys);
    if ((point.x() - xs(0)) * (point.x() - xs(0)) + (point.y() - ys(0)) * (point.y() - ys(0)) < radius * radius) {
        return point;
    }
    std::cerr << "unsuccessful attempt\n";
    for (int i = 0; i < xs.rows(); ++i) {
        for (int j = i + 1; j < xs.rows(); ++j) {
            std::tuple<double, double, double> xyv = estimateMaximumPointOnInterval(estimated_coefs_cf, xs(i), ys(i), xs(j), ys(j));

        }
    }
    return {1, 1};
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

Eigen::Vector3d estimateMaximumPoint(Eigen::MatrixXd points, const Eigen::Vector3d &normal,
                                     Eigen::VectorXd values, int index) {
    placeCenterAtBeginning(points, values);
    Eigen::Matrix3d rot_matrix = calculateRotationToAlignZAxis(normal);
    Eigen::MatrixXd rotated_points = (rot_matrix * (points.transpose())).transpose();
    Eigen::Vector2d maximum_point = estimateMaximumPointEP(rotated_points.col(0), rotated_points.col(1), values, index);
    double z = estimatePointOnQuadric(rotated_points.col(0), rotated_points.col(1), rotated_points.col(2),
                                      maximum_point.x(), maximum_point.y());
    return rot_matrix.inverse() * Eigen::Vector3d{maximum_point.x(), maximum_point.y(), z};
}