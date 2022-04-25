#ifndef REGISTRATION_WEIGHTS_H
#define REGISTRATION_WEIGHTS_H

#include <vector>

#include "common.h"

typedef std::vector<float> (*WeightFunction)(float curvature_radius, const PointNCloud::ConstPtr &pcd);

WeightFunction getWeightFunction(const std::string &identifier);

#endif
