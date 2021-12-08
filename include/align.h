#ifndef REGISTRATION_ALIGN_H
#define REGISTRATION_ALIGN_H

#include <Eigen/Core>

#include "config.h"
#include "common.h"
#include "sac_prerejective_omp.h"

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename);

void estimateNormals(float radius_search, const PointCloudT::Ptr &pcd, PointCloudN::Ptr &normals);

void estimateFeatures(float radius_search, const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                      FeatureCloudT::Ptr &features);

SampleConsensusPrerejectiveOMP<PointT, PointT, FeatureT> align_point_clouds(const PointCloudT::Ptr &src,
                                                                            const PointCloudT::Ptr &tgt,
                                                                            const FeatureCloudT::Ptr &features_src,
                                                                            const FeatureCloudT::Ptr &features_tgt,
                                                                            const YamlConfig &config);

#endif
