#ifndef REGISTRATION_COMMON_H
#define REGISTRATION_COMMON_H

#include <algorithm>

#include <pcl/types.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Core>

#include "config.h"
#include "utils.h"

#define COLOR_BEIGE 0xf8c471
#define COLOR_BROWN 0xd68910
#define COLOR_PURPLE 0xaf7ac5
#define COLOR_RED 0xff0000
#define COLOR_BLUE 0x0000ff
#define COLOR_WHITE 0xffffff
#define DEBUG_N_EDGES 100ul
#define ROPS_DIM 135

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointTN;
typedef pcl::PointXYZRGBNormal PointColoredTN;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointColoredTN> PointCloudColoredTN;
typedef pcl::PointCloud<pcl::Normal> PointCloudN;
typedef pcl::PointCloud<PointTN> PointCloudTN;

// Feature types
typedef pcl::Histogram<ROPS_DIM> RoPS135;
typedef pcl::FPFHSignature33 FPFH;
typedef pcl::UniqueShapeContext1960 USC;
typedef pcl::SHOT352 SHOT;

POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::Histogram<ROPS_DIM>, (float[ROPS_DIM], histogram, histogram))

template<>
class pcl::DefaultPointRepresentation<RoPS135> : public pcl::DefaultFeatureRepresentation<RoPS135> {
};

struct AlignmentParameters {
    bool downsample, use_normals, normals_available;
    float voxel_size;
    float edge_thr_coef, distance_thr_coef;
    float normal_radius_coef, feature_radius_coef;
    float confidence, inlier_fraction;
    bool reciprocal, use_bfmatcher;
    int bf_block_size;
    int randomness, n_samples;
    std::string func_id, descriptor_id;
    std::optional<int> max_iterations;

    bool save_features;
    std::string testname;
};

std::vector<AlignmentParameters> getParametersFromConfig(const YamlConfig &config,
                                                         const std::vector<::pcl::PCLPointField> &fields_src,
                                                         const std::vector<::pcl::PCLPointField> &fields_tgt);

struct MultivaluedCorrespondence {
    int query_idx = -1;
    pcl::Indices match_indices;
    std::vector<float> distances;
};

struct PointHash {
    inline size_t operator()(const PointTN &point) const {
        size_t seed = 0;
        combineHash(seed, point.x);
        combineHash(seed, point.y);
        combineHash(seed, point.z);
        return seed;
    }
};

template<typename T>
struct HashEigen {
    std::size_t operator()(T const &matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int) matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

template<typename PointT>
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

template<typename PointT>
std::pair<PointT, PointT> calculateBoundingBox(const typename pcl::PointCloud<PointT>::Ptr &pcd) {
    float min = std::numeric_limits<float>::min(), max = std::numeric_limits<float>::max();
    PointT min_point_AABB(max, max, max);
    PointT max_point_AABB(min, min, min);
    for (auto p: pcd->points) {
        min_point_AABB.x = std::min(min_point_AABB.x, p.x);
        min_point_AABB.y = std::min(min_point_AABB.y, p.y);
        min_point_AABB.z = std::min(min_point_AABB.z, p.z);
        max_point_AABB.x = std::max(max_point_AABB.x, p.x);
        max_point_AABB.y = std::max(max_point_AABB.y, p.y);
        max_point_AABB.z = std::max(max_point_AABB.z, p.z);
    }
    return {min_point_AABB, max_point_AABB};
}

template<typename PointT>
float calculatePointCloudDensity(const typename pcl::PointCloud<PointT>::Ptr &pcd) {
    pcl::KdTreeFLANN<PointT> tree(new pcl::KdTreeFLANN<PointT>);
    tree.setInputCloud(pcd);

    int k_neighbours = 8, n_points = pcd->size();
    pcl::Indices match_indices(k_neighbours);
    std::vector<float> match_distances(k_neighbours), distances(n_points);
    for (int i = 0; i < n_points; ++i) {
        tree.nearestKSearch(*pcd,
                            i,
                            k_neighbours,
                            match_indices,
                            match_distances);
        std::nth_element(match_distances.begin(), match_distances.begin() + k_neighbours / 2, match_distances.end());
        distances[i] = std::sqrt(match_distances[k_neighbours / 2]);
    }
    std::nth_element(distances.begin(), distances.begin() + n_points / 2, distances.end());
    return distances[n_points / 2];
}

float getAABBDiagonal(const PointCloudTN::Ptr &pcd);

void saveColorizedPointCloud(const PointCloudTN::ConstPtr &pcd,
                             const std::vector<MultivaluedCorrespondence> &correspondences,
                             const std::vector<MultivaluedCorrespondence> &correct_correspondences,
                             const pcl::Indices &inliers, const AlignmentParameters &parameters,
                             const Eigen::Matrix4f &transformation_gt, bool is_source);

void saveCorrespondences(const PointCloudTN::ConstPtr &src, const PointCloudTN::ConstPtr &tgt,
                         const std::vector<MultivaluedCorrespondence> &correspondences,
                         const Eigen::Matrix4f &transformation_gt,
                         const AlignmentParameters &parameters, bool sparse = false);

void saveCorrespondenceDistances(const PointCloudTN::ConstPtr &src, const PointCloudTN::ConstPtr &tgt,
                                 const std::vector<MultivaluedCorrespondence> &correspondences,
                                 const Eigen::Matrix4f &transformation_gt, float voxel_size,
                                 const AlignmentParameters &parameters);

void saveInlierIds(const std::vector<MultivaluedCorrespondence> &correspondences,
                   const std::vector<MultivaluedCorrespondence> &correct_correspondences,
                   const pcl::Indices &inliers, const AlignmentParameters &parameters);

void setPointColor(PointColoredTN &point, int color);

void mixPointColor(PointColoredTN &point, int color);

void setPointColor(PointColoredTN &point, std::uint8_t red, std::uint8_t green, std::uint8_t blue);

std::string constructPath(const std::string &test, const std::string &name,
                          const std::string &extension = "ply", bool with_version = true);

std::string constructPath(const AlignmentParameters &parameters, const std::string &name,
                          const std::string &extension = "ply", bool with_version = true);

template<typename PointT>
bool pointCloudHasNormals(const std::vector<pcl::PCLPointField> &fields) {
    bool normal_x = false, normal_y = false, normal_z = false;
    for (const auto &field: fields) {
        if (pcl::FieldMatches<PointT, pcl::fields::normal_x>()(field)) {
            normal_x = true;
        }
        if (pcl::FieldMatches<PointT, pcl::fields::normal_y>()(field)) {
            normal_y = true;
        }
        if (pcl::FieldMatches<PointT, pcl::fields::normal_x>()(field)) {
            normal_z = true;
        }
    }
    return normal_x && normal_y && normal_z;
}

#endif
