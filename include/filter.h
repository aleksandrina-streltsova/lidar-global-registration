#ifndef REGISTRATION_FILTER_H
#define REGISTRATION_FILTER_H

#include "common.h"

#define UNIQUENESS_THRESHOLD 10.f
#define N_RANDOM_FEATURES 50.f

typedef std::vector<float> (*UniquenessFunction)(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &);

UniquenessFunction getUniquenessFunction(const std::string& identifier);

void filterPointCloud(UniquenessFunction func, const std::string &func_identifier,
                      const PointCloud::Ptr &pcd, const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features,
                      PointCloud::Ptr &dst_pcd, pcl::PointCloud<pcl::FPFHSignature33>::Ptr &dst_features,
                      const Eigen::Matrix4f &transformation_gt,
                      const AlignmentParameters &parameters, bool is_source);

void filterDuplicatePoints(PointNCloud::Ptr &pcd);

#endif