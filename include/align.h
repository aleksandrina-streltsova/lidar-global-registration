#ifndef REGISTRATION_ALIGN_H
#define REGISTRATION_ALIGN_H

#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/fpfh_omp.h>

#include "config.h"

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::Normal> PointCloudN;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointT, pcl::Normal, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

void printTransformation(const Eigen::Matrix4f &transformation);

int countCorrectCorrespondences(const Eigen::Matrix4f &transformation_gt,
                                const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                                std::vector<int> inliers, float error_threshold);

float getAABBDiagonal(const PointCloudT::Ptr &pcd);

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename);

void filterReciprocalCorrespondences(PointCloudT::Ptr &src, FeatureCloudT::Ptr &features_src,
                                     PointCloudT::Ptr &tgt, FeatureCloudT::Ptr &features_tgt);

void downsamplePointCloud(const PointCloudT::Ptr &pcd_fullsize, PointCloudT::Ptr &pcd_down, float voxel_size);

void estimateNormals(float radius_search, const PointCloudT::Ptr &pcd, PointCloudN::Ptr &normals);

void estimateFeatures(float radius_search, const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                      FeatureCloudT::Ptr &features);

Eigen::Matrix4f align(PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                      const FeatureCloudT::Ptr &features_src, const FeatureCloudT::Ptr &features_tgt,
                      const Eigen::Matrix4f &transformation_gt, const YamlConfig &config);
#endif
