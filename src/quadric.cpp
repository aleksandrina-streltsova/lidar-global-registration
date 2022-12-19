#include "quadric.h"
#include "utils.h"
#include "common.h"

#include <utility>

#define MIN_ANGLE 0.04
#define EPS 1e-9

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
    std::swap(values(index), values(0));
}

double estimateRadius(const Eigen::MatrixXd &xs, const Eigen::MatrixXd &ys) {
    double radius2 = 0.0;
    for (int i = 0; i < xs.rows(); ++i) {
        radius2 = std::max(radius2, (xs(i) - xs(0)) * (xs(i) - xs(0)) + (ys(i) - ys(0)) * (ys(i) - ys(0)));
    }
    return std::sqrt(radius2);
}

inline double calculateValueAtPoint(const Vector6 &coefs, double x, double y) {
    return coefs(0) * x * x +
           coefs(1) * x * y +
           coefs(2) * y * y +
           coefs(3) * x +
           coefs(4) * y +
           coefs(5);
}

std::tuple<double, double, double> estimateMaximumPointOnInterval(const Vector6 &coefs,
                                                                  double x1, double y1, double x2, double y2) {
    // f(x, y) = p0 * x^2 + p1 * x * y + p2 * y^2 + p3 * x + p4 * y + p5
    // g(x, y) = (x - x1) * (y1 - y2) - (y - y1) * (x1 - x2)
    // L(x, y, l) = f(x, y) - l * g(x, y) -- Lagrangian
    // linear equations:
    // dL(x, y, l) / dx = 2 * p0 * x + p1 * y + p3 - l * (y1 - y2)
    // dL(x, y, l) / dy = 2 * p2 * y + p1 * x + p4 + l * (x1 - x2)
    // g(x, y) = (x1 - x) * (y2 - y1) - (y1 - y) * (x2 - x1)
    Eigen::Matrix3d A;
    Eigen::Vector3d b, estimated_solution;
    A << 2 * coefs[0], coefs[1], -(y2 - y1),
            coefs[1], 2 * coefs[2], (x2 - x1),
            (y2 - y1), -(x2 - x1), 0;
    b << -coefs[3], -coefs[4], x1 * (y2 - y1) - y1 * (x2 - x1);
    estimated_solution = A.inverse() * b;
    double x = estimated_solution(0), y = estimated_solution(1);
    if (x1 > x2) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }

    double v = calculateValueAtPoint(coefs, x, y);
    double v1 = calculateValueAtPoint(coefs, x1, y1);
    double v2 = calculateValueAtPoint(coefs, x2, y2);

    bool outside_interval = x < x1 || x > x2 || std::abs((x1 - x) * (y2 - y1) - (y1 - y) * (x2 - x1)) > EPS;
    if (outside_interval || !std::isfinite(v) || v < v1 || v < v2) {
        if (v1 > v2) return {x1, y1, v1};
        return {x2, y2, v2};
    }
    return {x, y, calculateValueAtPoint(coefs, x, y)};
}

void saveSaliencies(const Eigen::MatrixXd &xs, const Eigen::MatrixXd &ys, const Eigen::VectorXd &values,
                    const Vector6 &coefs, int index) {
    PointNCloud::Ptr pcd(new PointNCloud), pcd_pd(new PointNCloud);
    double m = *std::max_element(values.data(), values.data() + values.size());
    double min_x = std::numeric_limits<double>::max(), max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max(), max_y = std::numeric_limits<double>::lowest();
    for (int i = 0; i < values.size(); ++i) {
        pcd->push_back(PointN{(float) xs(i), (float) ys(i), (float) (values(i) / m / 200.0)});
        pcd->push_back(PointN{(float) xs(i), (float) ys(i), 0.f});
        min_x = std::min(min_x, xs(i));
        min_y = std::min(min_y, ys(i));
        max_x = std::max(max_x, xs(i));
        max_y = std::max(max_y, ys(i));
    }
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            double x = min_x + i * (max_x - min_x) / 100;
            double y = min_y + j * (max_y - min_y) / 100;
            double z = calculateValueAtPoint(coefs, x, y) / m / 200.0;
            pcd_pd->push_back(PointN{(float) x, (float) y, (float) z});
            pcd_pd->push_back(PointN{(float) x, (float) y, 0.f});
        }
    }
    saveColorizedPointCloud(pcd, Eigen::Matrix4f::Identity(), COLOR_PURPLE,
                            constructPath("saliencies", std::to_string(index), "ply", false));
    saveColorizedPointCloud(pcd_pd, Eigen::Matrix4f::Identity(), COLOR_PURPLE,
                            constructPath("saliencies_all", std::to_string(index), "ply", false));
}

Eigen::Vector2d estimateMaximumPointEP(const Eigen::VectorXd &xs, const Eigen::VectorXd &ys,
                                       const Eigen::VectorXd &values, int index) {
    Vector6 estimated_coefs_cf;
    Eigen::MatrixXd A;
    // estimate coefficients of quadric (closed form)
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
//    saveSaliencies(xs, ys, values, estimated_coefs_cf, index);

    // estimate maximum point of quadric
    A.resize(2, 2);
    Eigen::Vector2d b;
    A << 2 * estimated_coefs_cf(0), estimated_coefs_cf(1), estimated_coefs_cf(1), 2 * estimated_coefs_cf(2);
    b << -estimated_coefs_cf(3), -estimated_coefs_cf(4);
    Eigen::Vector2d point = A.inverse() * b;
    double radius = estimateRadius(xs, ys);
    double dist2 = (point.x() - xs(0)) * (point.x() - xs(0)) + (point.y() - ys(0)) * (point.y() - ys(0));
    if (dist2 < radius * radius && calculateValueAtPoint(estimated_coefs_cf, point.x(), point.y()) > values(0)) {
        return point;
    }

    // TODO: rewrite calculation of maximum point for boundary intervals
    Eigen::Vector2d max_point;
    max_point << xs(0), ys(0);
    double max_v = calculateValueAtPoint(estimated_coefs_cf, max_point.x(), max_point.y());
    for (int i = 0; i < xs.rows(); ++i) {
        for (int j = i + 1; j < xs.rows(); ++j) {
            auto [x, y, v] = estimateMaximumPointOnInterval(estimated_coefs_cf, xs(i), ys(i), xs(j), ys(j));
            if (max_v < v) {
                max_v = v;
                max_point << x, y;
            }
        }
    }
    dist2 = (max_point.x() - xs(0)) * (max_point.x() - xs(0)) +
            (max_point.y() - ys(0)) * (max_point.y() - ys(0));
    if (dist2 > radius * radius) {
        max_point << xs(0), ys(0);
    }
    return max_point;
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