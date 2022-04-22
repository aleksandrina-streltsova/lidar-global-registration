#ifndef REGISTRATION_WEIGHTS_H
#define REGISTRATION_WEIGHTS_H

#include <vector>

#include "common.h"

typedef std::vector<float> (*WeightFunction) (float radius_search, const PointNCloud::ConstPtr &pcd, const NormalCloud::ConstPtr &normals);

WeightFunction getWeightFunction(const std::string &identifier);

#endif
