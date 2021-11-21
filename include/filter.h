#ifndef REGISTRATION_FILTER_H
#define REGISTRATION_FILTER_H

#include "common.h"

#define UNIQUENESS_THRESHOLD 10.f
#define N_RANDOM_FEATURES 50.f

typedef std::vector<float> (*UniquenessFunction)(const FeatureCloudT::Ptr &);

UniquenessFunction getUniquenessFunction(const std::string& identifier);

void filterPointCloud(UniquenessFunction func, const std::string &func_identifier,
                      const PointCloudT::Ptr &pcd, const FeatureCloudT::Ptr &features,
                      PointCloudT::Ptr &dst_pcd, FeatureCloudT::Ptr &dst_features,
                      const Eigen::Matrix4f &transformation_gt,
                      const std::string &testname, bool is_source);

void filter_duplicate_points(PointCloudT::Ptr &pcd);

#endif