#ifndef REGISTRATION_COMMON_H
#define REGISTRATION_COMMON_H

#include <pcl/types.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>

#include "utils.h"

#define COLOR_BEIGE 0xf8c471
#define COLOR_BROWN 0xd68910
#define COLOR_PURPLE 0xaf7ac5
#define COLOR_RED 0xff0000
#define COLOR_BLUE 0x0000ff
#define COLOR_WHITE 0xffffff
#define DEBUG_N_EDGES 100ul
// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointColoredT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointColoredT> PointCloudColoredT;
typedef pcl::PointCloud<pcl::Normal> PointCloudN;

struct MultivaluedCorrespondence {
    int query_idx;
    pcl::Indices match_indices;
};

struct PointHash {
    inline size_t operator()(const PointT &point) const
    {
        size_t seed = 0;
        combineHash(seed, point.x);
        combineHash(seed, point.y);
        combineHash(seed, point.z);
        return seed;
    }
};

template <typename T>
struct HashEigen {
    std::size_t operator()(T const& matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct PointEqual {
public:
    bool operator()(const PointT &point1, const PointT &point2) const {
        if (point1.x == point2.x && point1.y == point2.y && point1.z == point2.z)
            return true;
        else
            return false;
    }
};

extern const std::string DATA_DEBUG_PATH;
extern const std::string VERSION;
extern const std::string DEFAULT_DESCRIPTOR;

void printTransformation(const Eigen::Matrix4f &transformation);

std::pair<PointT, PointT> calculateBoundingBox(const PointCloudT::Ptr &pcd);

float getAABBDiagonal(const PointCloudT::Ptr &pcd);

void saveColorizedPointCloud(const PointCloudT::ConstPtr &src,
                             const std::vector<MultivaluedCorrespondence> &correspondences,
                             const std::vector<MultivaluedCorrespondence> &correct_correspondences,
                             const pcl::Indices &inliers,
                             const std::string &testname);

void saveCorrespondences(const PointCloudT::ConstPtr &src, const PointCloudT::ConstPtr &tgt,
                         const std::vector<MultivaluedCorrespondence> &correspondences,
                         const Eigen::Matrix4f &transformation_gt,
                         const std::string &testname, bool sparse = false);

void saveCorrespondenceDistances(const PointCloudT::ConstPtr &src, const PointCloudT::ConstPtr &tgt,
                                 const std::vector<MultivaluedCorrespondence> &correspondences,
                                 const Eigen::Matrix4f &transformation_gt, float voxel_size,
                                 const std::string &testname);

void setPointColor(PointColoredT &point, int color);

void mixPointColor(PointColoredT &point, int color);

void setPointColor(PointColoredT &point, std::uint8_t red, std::uint8_t green, std::uint8_t blue);

std::string constructPath(const std::string &test, const std::string &name,
                          const std::string &extension = "ply", bool with_version = true);

#endif
