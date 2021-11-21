#ifndef REGISTRATION_FEATURE_ANALYSIS_H
#define REGISTRATION_FEATURE_ANALYSIS_H

#include <Eigen/Core>
#include <pcl/point_cloud.h>

void saveHistograms(const FeatureCloudT::Ptr &features, const std::string &testname, bool is_source);

void saveFeatures(float radius_search, const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                  const std::string &testname, bool is_source);

void saveNormals(const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                 const Eigen::Matrix4f &transformation_gt, bool is_source, const std::string &testname);

#endif
