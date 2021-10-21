#ifndef REGISTRATION_COMMON_H
#define REGISTRATION_COMMON_H

#include <pcl/types.h>

#define COLOR_BEIGE 0xf8c471
#define COLOR_BROWN 0xd68910
#define COLOR_PURPLE 0xaf7ac5

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointColoredT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointColoredT> PointCloudColoredT;
typedef pcl::PointCloud<pcl::Normal> PointCloudN;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointT, pcl::Normal, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

struct MultivaluedCorrespondence {
    int query_idx;
    pcl::Indices match_indices;
};

extern const std::string DATA_DEBUG_PATH;
extern const std::string VERSION;

void printTransformation(const Eigen::Matrix4f &transformation);

void saveColorizedPointCloud(const PointCloudT::Ptr &src,
                             const std::vector<MultivaluedCorrespondence> &correspondences,
                             const pcl::Indices &inliers,
                             const std::string &testname);

void setPointColor(PointColoredT &point, int color);

std::string constructPath(const std::string &test, const std::string &name, const std::string &extension = "ply");

#endif
