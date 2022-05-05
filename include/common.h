#ifndef REGISTRATION_COMMON_H
#define REGISTRATION_COMMON_H

#include <algorithm>

#include <pcl/types.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>

#include <Eigen/Core>

#include "config.h"
#include "utils.h"

#define SEED 566
#define COLOR_BEIGE 0xf8c471
#define COLOR_BROWN 0xd68910
#define COLOR_PURPLE 0xaf7ac5
#define COLOR_RED 0xff0000
#define COLOR_BLUE 0x0000ff
#define COLOR_WHITE 0xffffff
#define DEBUG_N_EDGES 100ul
#define ROPS_DIM 135

// Types
typedef pcl::PointXYZ Point;
typedef pcl::PointNormal PointN;
typedef pcl::ReferenceFrame PointRF;
typedef pcl::PointXYZRGBNormal PointColoredN;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointCloud<PointColoredN> PointColoredNCloud;
typedef pcl::PointCloud<pcl::Normal> NormalCloud;
typedef pcl::PointCloud<PointN> PointNCloud;
typedef pcl::PointCloud<PointRF> PointRFCloud;

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
    bool use_normals, normals_available;
    float voxel_size;
    float edge_thr_coef, distance_thr_coef;
    float normal_radius_coef, feature_radius_coef;
    float confidence, inlier_fraction;
    bool use_bfmatcher;
    int bf_block_size;
    int randomness, n_samples;
    std::string func_id, descriptor_id, lrf_id, metric_id, matching_id, weight_id;
    std::optional<int> max_iterations;

    bool save_features;
    std::string testname;
    std::shared_ptr<Eigen::Matrix4f> ground_truth{nullptr};

    // these parameters cannot be set in config, they are set before alignment steps
    float match_search_radius = 0;
    std::shared_ptr<Eigen::Matrix4f> guess{nullptr};
};

std::vector<AlignmentParameters> getParametersFromConfig(const YamlConfig &config,
                                                         const std::vector<::pcl::PCLPointField> &fields_src,
                                                         const std::vector<::pcl::PCLPointField> &fields_tgt,
                                                         float min_voxel_size);

struct MultivaluedCorrespondence {
    int query_idx = -1;
    pcl::Indices match_indices;
    std::vector<float> distances;
};

void updateMultivaluedCorrespondence(MultivaluedCorrespondence &corr, int query_idx,
                                     int k_matches, int match_idx, float distance);

struct InlierPair {
    int idx_src, idx_tgt;
};

struct PointHash {
    inline size_t operator()(const PointN &point) const {
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
extern const std::string TRANSFORMATIONS_CSV;
extern const std::string ITERATIONS_CSV;
extern const std::string VERSION;
extern const std::string DEFAULT_DESCRIPTOR;
extern const std::string DEFAULT_LRF;
extern const std::string METRIC_CORRESPONDENCES;
extern const std::string METRIC_CLOSEST_POINT;
extern const std::string METRIC_WEIGHTED_CLOSEST_POINT;
extern const std::string METRIC_COMBINATION;
extern const std::string MATCHING_LEFT_TO_RIGHT;
extern const std::string MATCHING_RATIO;
extern const std::string MATCHING_CLUSTER;
extern const std::string METRIC_WEIGHT_CONSTANT;
extern const std::string METRIC_WEIGHT_EXP_CURVATURE;
extern const std::string METRIC_WEIGHT_CURVEDNESS;
extern const std::string METRIC_WEIGHT_HARRIS;
extern const std::string METRIC_WEIGHT_TOMASI;
extern const std::string METRIC_WEIGHT_CURVATURE;

void printTransformation(const Eigen::Matrix4f &transformation);

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename);

Eigen::Matrix4f getTransformation(const std::string &csv_path, const std::string &transformation_name);

void saveTransformation(const std::string &csv_path, const std::string &transformation_name,
                        const Eigen::Matrix4f &transformation);

void getIterationsInfo(const std::string &csv_path, const std::string &name, std::vector<float> voxel_sizes);

void saveIterationsInfo(const std::string &csv_path, const std::string &name, const std::vector<float> &voxel_sizes);

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
bool pointInBoundingBox(PointT point, PointT min_point, PointT max_point) {
    return min_point.x < point.x && point.x < max_point.x && min_point.y < point.y && point.y < max_point.y &&
           min_point.z < point.z && point.z < max_point.z;
}

template<typename PointT>
float calculatePointCloudDensity(const typename pcl::PointCloud<PointT>::Ptr &pcd) {
    pcl::KdTreeFLANN<PointT> tree;
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

float getAABBDiagonal(const PointNCloud::Ptr &pcd);

void saveColorizedPointCloud(const PointNCloud::ConstPtr &pcd,
                             const pcl::Correspondences &correspondences,
                             const pcl::Correspondences &correct_correspondences,
                             const std::vector<InlierPair> &inlier_pairs, const AlignmentParameters &parameters,
                             const Eigen::Matrix4f &transformation_gt, bool is_source);

void saveColorizedWeights(const PointNCloud::ConstPtr &pcd, std::vector<float> &weights, const std::string &name,
                          const AlignmentParameters &parameters, const Eigen::Matrix4f &transformation_gt);

void saveCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                         const pcl::Correspondences &correspondences,
                         const Eigen::Matrix4f &transformation_gt,
                         const AlignmentParameters &parameters, bool sparse = false);

void saveCorrespondenceDistances(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                 const pcl::Correspondences &correspondences,
                                 const Eigen::Matrix4f &transformation_gt, float voxel_size,
                                 const AlignmentParameters &parameters);

void saveCorrespondencesDebug(const pcl::Correspondences &correspondences,
                              const pcl::Correspondences &correct_correspondences,
                              const AlignmentParameters &parameters);

void setPointColor(PointColoredN &point, int color);

void mixPointColor(PointColoredN &point, int color);

void setPointColor(PointColoredN &point, std::uint8_t red, std::uint8_t green, std::uint8_t blue);

std::string constructName(const AlignmentParameters &parameters, const std::string &name,
                          bool with_version = true, bool with_metric = true, bool with_weights = true);

std::string constructPath(const std::string &test, const std::string &name,
                          const std::string &extension = "ply", bool with_version = true);

std::string constructPath(const AlignmentParameters &parameters, const std::string &name,
                          const std::string &extension = "ply",
                          bool with_version = true, bool with_metric = true, bool with_weights = true);

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

void readCorrespondencesFromCSV(const std::string &filepath, pcl::Correspondences &correspondences, bool &success);

void saveCorrespondencesToCSV(const std::string &filepath, const pcl::Correspondences &correspondences);

#endif
