#ifndef REGISTRATION_COMMON_H
#define REGISTRATION_COMMON_H

#include <algorithm>

#include <pcl/types.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/usc.h>
#include <pcl/surface/gp3.h>

#include <Eigen/Core>

#include "config.h"
#include "utils.h"
#include "rops_custom_lrf.h"

#define SEED 566
#define COLOR_BEIGE 0xf8c471
#define COLOR_BROWN 0xd68910
#define COLOR_PURPLE 0xaf7ac5
#define COLOR_RED 0xff0000
#define COLOR_GREEN 0x00ff00
#define COLOR_PARAKEET 0x03c04a
#define COLOR_ROSE 0xe3242b
#define COLOR_BLUE 0x0000ff
#define COLOR_WHITE 0xffffff
#define DEBUG_N_EDGES 100ul
#define ROPS_DIM 135

#define ALIGNMENT_COARSE_TO_FINE false
#define ALIGNMENT_EDGE_THR 0.95
#define ALIGNMENT_CONFIDENCE 0.999
#define ALIGNMENT_INLIER_FRACTION 0.1
#define ALIGNMENT_USE_BFMATCHER true
#define ALIGNMENT_USE_NORMALS false
#define ALIGNMENT_RANDOMNESS 1
#define ALIGNMENT_N_SAMPLES 3
#define ALIGNMENT_SAVE_FEATURES false
#define ALIGNMENT_BLOCK_SIZE 10000
#define ALIGNMENT_NORMAL_RADIUS_COEF 3
#define ALIGNMENT_FEATURE_RADIUS_COEF 15

#define GROR_ISS_COEF 4.0

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

extern const std::string DATA_DEBUG_PATH;
extern const std::string TRANSFORMATIONS_CSV;
extern const std::string ITERATIONS_CSV;
extern const std::string VERSION;
extern const std::string SUBVERSION;
extern const std::string ALIGNMENT_DEFAULT;
extern const std::string ALIGNMENT_GROR;
extern const std::string KEYPOINT_ANY;
extern const std::string KEYPOINT_ISS;
extern const std::string DESCRIPTOR_FPFH;
extern const std::string DESCRIPTOR_SHOT;
extern const std::string DESCRIPTOR_ROPS;
extern const std::string DESCRIPTOR_USC;
extern const std::string DEFAULT_LRF;
extern const std::string METRIC_CORRESPONDENCES;
extern const std::string METRIC_CLOSEST_PLANE;
extern const std::string METRIC_WEIGHTED_CLOSEST_PLANE;
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

struct InlierPair {
    int idx_src, idx_tgt;
};

struct AlignmentParameters {
    bool coarse_to_fine{ALIGNMENT_COARSE_TO_FINE};
    bool use_normals{ALIGNMENT_USE_NORMALS}, normals_available;
    float voxel_size;
    float edge_thr_coef{ALIGNMENT_EDGE_THR}, distance_thr_coef, gror_iss_coef{GROR_ISS_COEF};
    float normal_radius_coef{ALIGNMENT_NORMAL_RADIUS_COEF}, feature_radius_coef{ALIGNMENT_FEATURE_RADIUS_COEF};
    float confidence{ALIGNMENT_CONFIDENCE}, inlier_fraction{ALIGNMENT_INLIER_FRACTION};
    bool use_bfmatcher{ALIGNMENT_USE_BFMATCHER};
    int bf_block_size{ALIGNMENT_BLOCK_SIZE};
    int randomness{ALIGNMENT_RANDOMNESS}, n_samples{ALIGNMENT_N_SAMPLES};
    std::string alignment_id{ALIGNMENT_DEFAULT}, descriptor_id{DESCRIPTOR_SHOT}, keypoint_id{KEYPOINT_ISS};
    std::string metric_id{METRIC_COMBINATION}, matching_id{MATCHING_CLUSTER}, lrf_id{DEFAULT_LRF};
    std::string weight_id{METRIC_WEIGHT_CONSTANT}, func_id;
    int max_iterations;

    bool save_features;
    std::string testname;
    std::shared_ptr<Eigen::Matrix4f> ground_truth{nullptr};

    // these parameters cannot be set in config, they are set before alignment steps
    bool fix_seed = true;
    float match_search_radius = 0;
    std::optional<Eigen::Matrix4f> guess{std::nullopt};
    std::string dir_path{DATA_DEBUG_PATH};
};

struct AlignmentResult {
    PointNCloud::ConstPtr src, tgt;
    Eigen::Matrix4f transformation;
    pcl::CorrespondencesConstPtr correspondences;
    int iterations;
    bool converged{false};

    int time;
};

std::vector<AlignmentParameters> getParametersFromConfig(const YamlConfig &config,
                                                         const std::vector<::pcl::PCLPointField> &fields_src,
                                                         const std::vector<::pcl::PCLPointField> &fields_tgt,
                                                         float min_voxel_size);

void loadPointClouds(const std::string &src_path, const std::string &tgt_path,
                     std::string &testname, PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                     std::vector<::pcl::PCLPointField> &fields_src, std::vector<::pcl::PCLPointField> &fields_tgt,
                     const std::optional<float> &density, float &min_voxel_size);

void loadTransformationGt(const std::string &src_path, const std::string &tgt_path,
                          const std::string &csv_path, Eigen::Matrix4f &transformation_gt);

struct MultivaluedCorrespondence {
    int query_idx = -1;
    pcl::Indices match_indices;
    std::vector<float> distances;
};

void updateMultivaluedCorrespondence(MultivaluedCorrespondence &corr, int query_idx,
                                     int k_matches, int match_idx, float distance);

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

void printTransformation(const Eigen::Matrix4f &transformation);

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename);

Eigen::Matrix4f getTransformation(const std::string &csv_path, const std::string &transformation_name);

void saveTransformation(const std::string &csv_path, const std::string &transformation_name,
                        const Eigen::Matrix4f &transformation);

void getIterationsInfo(const std::string &csv_path, const std::string &name,
                       std::vector<float> &voxel_sizes,
                       std::vector<std::string> &matching_ids);

void saveIterationsInfo(const std::string &csv_path, const std::string &name,
                        const std::vector<float> &voxel_sizes,
                        const std::vector<std::string> &matching_ids);

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

void estimateNormalsRadius(float radius_search, const PointNCloud::Ptr &pcd, NormalCloud::Ptr &normals,
                           bool normals_available);

void estimateNormalsPoints(int k_points, const PointNCloud::Ptr &pcd, NormalCloud::Ptr &normals,
                           bool normals_available);

void smoothNormals(float radius_search, float voxel_size, const PointNCloud::Ptr &pcd);

pcl::IndicesPtr detectKeyPoints(const PointNCloud::ConstPtr &pcd, const AlignmentParameters &parameters);

void estimateReferenceFrames(const PointNCloud::ConstPtr &pcd, const pcl::IndicesConstPtr &indices,
                             PointRFCloud::Ptr &frames_kps, const AlignmentParameters &parameters);

template<typename FeatureT>
void estimateFeatures(const PointNCloud::ConstPtr &pcd, const pcl::IndicesConstPtr &indices,
                      typename pcl::PointCloud<FeatureT>::Ptr &features, const AlignmentParameters &parameters) {
    throw std::runtime_error("Feature with proposed reference frame isn't supported!");
}


template<>
inline void estimateFeatures<FPFH>(const PointNCloud::ConstPtr &pcd, const pcl::IndicesConstPtr &indices,
                                   pcl::PointCloud<FPFH>::Ptr &features, const AlignmentParameters &parameters) {
    int nr_kps = indices ? indices->size() : pcd->size();
    pcl::FPFHEstimationOMP<PointN, PointN, FPFH> fpfh_estimation;
    fpfh_estimation.setRadiusSearch(parameters.feature_radius_coef * parameters.voxel_size);
    fpfh_estimation.setInputCloud(pcd);
    fpfh_estimation.setInputNormals(pcd);
    if (indices) fpfh_estimation.setIndices(indices);
    fpfh_estimation.compute(*features);
    rassert(features->size() == nr_kps, 104935923)
}

template<>
inline void estimateFeatures<USC>(const PointNCloud::ConstPtr &pcd, const pcl::IndicesConstPtr &indices,
                                  pcl::PointCloud<USC>::Ptr &features, const AlignmentParameters &parameters) {
    int nr_kps = indices ? indices->size() : pcd->size();
    float radius_search = parameters.feature_radius_coef * parameters.voxel_size;
    pcl::UniqueShapeContext<PointN, USC, PointRF> shape_context;
    shape_context.setInputCloud(pcd);
    if (indices) shape_context.setIndices(indices);
    shape_context.setMinimalRadius(radius_search / 10.f);
    shape_context.setRadiusSearch(radius_search);
    shape_context.setPointDensityRadius(radius_search / 5.f);
    shape_context.setLocalRadius(radius_search);
    shape_context.compute(*features);
    rassert(features->size() == nr_kps, 3489281234)
    std::cout << "output points.size (): " << features->points.size() << std::endl;

}

template<>
inline void estimateFeatures<RoPS135>(const PointNCloud::ConstPtr &pcd, const pcl::IndicesConstPtr &indices,
                                      pcl::PointCloud<RoPS135>::Ptr &features, const AlignmentParameters &parameters) {
    int nr_kps = indices ? indices->size() : pcd->size();
    float radius_search = parameters.feature_radius_coef * parameters.voxel_size;
    // RoPs estimation object.
    ROPSEstimationWithLocalReferenceFrames<PointN, RoPS135> rops;
    rops.setInputCloud(pcd);
    if (indices) rops.setIndices(indices);
    rops.setRadiusSearch(radius_search);
    // Number of partition bins that is used for distribution matrix calculation.
    rops.setNumberOfPartitionBins(5);
    // The greater the number of rotations is, the bigger the resulting descriptor.
    // Make sure to change the histogram size accordingly.
    rops.setNumberOfRotations(3);
    // Support radius that is used to crop the local surface of the point.
    rops.setSupportRadius(radius_search);

    PointRFCloud::Ptr frames{nullptr};
    estimateReferenceFrames(pcd, indices, frames, parameters);
    if (!frames) {
        // Perform triangulation.
        pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
        pcl::search::KdTree<PointN>::Ptr tree_n(new pcl::search::KdTree<PointN>);
        tree_n->setInputCloud(pcd);

        pcl::GreedyProjectionTriangulation<PointN> triangulation;
        pcl::PolygonMesh triangles;
        triangulation.setSearchRadius(radius_search);
        triangulation.setMu(2.5);
        triangulation.setMaximumNearestNeighbors(100);
        triangulation.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees.
        triangulation.setNormalConsistency(false);
        triangulation.setMinimumAngle(M_PI / 18); // 10 degrees.
        triangulation.setMaximumAngle(2 * M_PI / 3); // 120 degrees.
        triangulation.setInputCloud(pcd);
        triangulation.setSearchMethod(tree_n);
        triangulation.reconstruct(triangles);

        rops.setTriangles(triangles.polygons);
    } else {
        rops.setInputReferenceFrames(frames);
    }

    rops.compute(*features);
    rassert(features->size() == nr_kps, 2434751037284)
}

template<>
inline void estimateFeatures<SHOT>(const PointNCloud::ConstPtr &pcd, const pcl::IndicesConstPtr &indices,
                                   pcl::PointCloud<SHOT>::Ptr &features, const AlignmentParameters &parameters) {
    int nr_kps = indices ? indices->size() : pcd->size();
    // SHOT estimation object.
    pcl::SHOTEstimationOMP<PointN, PointN, SHOT> shot;
    shot.setInputCloud(pcd);
    if (indices) shot.setIndices(indices);
//	shot.setSearchSurface(surface);
    shot.setInputNormals(pcd);
    // The radius that defines which of the keypoint's neighbors are described.
    // If too large, there may be clutter, and if too small, not enough points may be found.
    shot.setRadiusSearch(parameters.feature_radius_coef * parameters.voxel_size);
    PointRFCloud::Ptr frames{nullptr};
    estimateReferenceFrames(pcd, indices, frames, parameters);
    if (frames) shot.setInputReferenceFrames(frames);
    PCL_WARN("[estimateFeatures<SHOT>] Points probably have NaN normals in their neighbourhood\n");
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    shot.compute(*features);
    rassert(features->size() == nr_kps, 845637470190)
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
}

void saveColorizedPointCloud(const PointNCloud::ConstPtr &pcd,
                             const pcl::IndicesConstPtr &key_point_indices,
                             const pcl::Correspondences &correspondences,
                             const pcl::Correspondences &correct_correspondences,
                             const std::vector<InlierPair> &inlier_pairs, const AlignmentParameters &parameters,
                             const Eigen::Matrix4f &transformation_gt, bool is_source);

void saveColorizedWeights(const PointNCloud::ConstPtr &pcd, std::vector<float> &weights, const std::string &name,
                          const AlignmentParameters &parameters, const Eigen::Matrix4f &transformation_gt);

void saveTemperatureMaps(PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                         const std::string &name, const AlignmentParameters &parameters,
                         const Eigen::Matrix4f &transformation, bool normals_available = true);

void saveCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                         const pcl::Correspondences &correspondences,
                         const Eigen::Matrix4f &transformation_gt,
                         const AlignmentParameters &parameters, bool sparse = false);

void saveCorrectCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                const pcl::Correspondences &correspondences,
                                const pcl::Correspondences &correct_correspondences,
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
                          bool with_version = true, bool with_metric = true,
                          bool with_weights = true, bool with_subversion = false);

std::string constructPath(const std::string &test, const std::string &name,
                          const std::string &extension = "ply", bool with_version = true, bool with_subversion = false);

std::string constructPath(const AlignmentParameters &parameters, const std::string &name,
                          const std::string &extension = "ply",
                          bool with_version = true, bool with_metric = true,
                          bool with_weights = true, bool with_subversion = false);

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

pcl::CorrespondencesPtr readCorrespondencesFromCSV(const std::string &filepath, bool &success);

void saveCorrespondencesToCSV(const std::string &filepath,
                              const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                              const pcl::CorrespondencesConstPtr &correspondences);

#endif
