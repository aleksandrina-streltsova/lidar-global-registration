#ifndef REGISTRATION_WEIGHTS_H
#define REGISTRATION_WEIGHTS_H

#include <vector>

#include "common.h"

typedef std::vector<float> (*WeightFunction)(int nr_points, const PointNCloud::ConstPtr &pcd);

WeightFunction getWeightFunction(const std::string &identifier);

#endif
