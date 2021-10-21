#ifndef REGISTRATION_ALIGN_H
#define REGISTRATION_ALIGN_H

#include <Eigen/Core>

#include "pch.h"
#include "config.h"
#include "common.h"

float getAABBDiagonal(const PointCloudT::Ptr &pcd);

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename);

void downsamplePointCloud(const PointCloudT::Ptr &pcd_fullsize, PointCloudT::Ptr &pcd_down, float voxel_size);

void estimateNormals(float radius_search, const PointCloudT::Ptr &pcd, PointCloudN::Ptr &normals);

void estimateFeatures(float radius_search, const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                      FeatureCloudT::Ptr &features);

Eigen::Matrix4f align(const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                      const FeatureCloudT::Ptr &features_src, const FeatureCloudT::Ptr &features_tgt,
                      const Eigen::Matrix4f &transformation_gt, const YamlConfig &config, const std::string &testname);

#endif
