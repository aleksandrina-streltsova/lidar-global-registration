#ifndef REGISTRATION_HYPOTHESES_H
#define REGISTRATION_HYPOTHESES_H

#include <vector>

#include <Eigen/Core>

#include "common.h"

void updateHypotheses(std::vector<Eigen::Matrix4f> &transformations, std::vector<float> &metrics,
                      const Eigen::Matrix4f &new_transformation, float new_metric,
                      const AlignmentParameters &parameters);

Eigen::Matrix4f chooseBestHypothesis(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                     const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                                     const CorrespondencesConstPtr &correspondences,
                                     const AlignmentParameters &params, std::vector<Eigen::Matrix4f> &tns);

#endif
