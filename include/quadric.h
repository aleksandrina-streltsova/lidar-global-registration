#ifndef REGISTRATION_QUADRIC_H
#define REGISTRATION_QUADRIC_H

#include <Eigen/Core>

Eigen::Vector3d estimateMaximumPoint(const Eigen::MatrixXd &points, const Eigen::Vector3d &normal,
                                     const Eigen::VectorXd &values, int index);
#endif
