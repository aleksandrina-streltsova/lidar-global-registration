#ifndef REGISTRATION_TRANSFORMATION_H
#define REGISTRATION_TRANSFORMATION_H

#include "common.h"

void estimateOptimalRigidTransformation(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                        const std::vector<InlierPair> &inlier_pairs, Eigen::Matrix4f &transformation);

#endif
