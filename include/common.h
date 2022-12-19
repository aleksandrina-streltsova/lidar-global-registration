#ifndef REGISTRATION_COMMON_H
#define REGISTRATION_COMMON_H

#include <algorithm>
#include <unordered_set>

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
#include "pcl/rops_custom_lrf.h"
#include "pcl/shot_debug.h"

#define SEED 566ul
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

#define ALIGNMENT_EDGE_THR 0.95
#define ALIGNMENT_CONFIDENCE 0.999
#define ALIGNMENT_INLIER_FRACTION 0.1
#define ALIGNMENT_USE_BFMATCHER true
#define ALIGNMENT_RANDOMNESS 1
#define ALIGNMENT_N_SAMPLES 3
#define ALIGNMENT_SAVE_FEATURES false
#define ALIGNMENT_BLOCK_SIZE 10000

#define FEATURES_SCALE_FACTOR 2.0
#define FEATURES_REESTIMATE_FRAMES true

#define MATCHING_RATIO_THRESHOLD 1.1f
#define MATCHING_RATIO_K 2
#define MATCHING_CLUSTER_THRESHOLD 0.95f
#define MATCHING_CLUSTER_K 40

#define SPARSE_POINTS_FRACTION 0.01
#define FEATURE_NR_POINTS 352
#define NORMAL_NR_POINTS 30
#define SPARSE_POINTS_FRACTION 0.01
#define FINE_VOXEL_SIZE_COEFFICIENT 2
#define DIST_TO_PLANE_COEFFICIENT 2

// Point types
typedef pcl::PointXYZI Point;
typedef pcl::PointXYZINormal PointN;
typedef pcl::ReferenceFrame PointRF;
typedef pcl::PointXYZRGBNormal PointColoredN;

// Feature types
typedef pcl::Histogram<ROPS_DIM> RoPS135;
typedef pcl::FPFHSignature33 FPFH;
typedef pcl::UniqueShapeContext1960 USC;
typedef pcl::SHOT352 SHOT;

POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::Histogram<ROPS_DIM>, (float[ROPS_DIM], histogram, histogram))

template<>
class pcl::DefaultPointRepresentation<RoPS135> : public pcl::DefaultFeatureRepresentation<RoPS135> {
};

// Point cloud types
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointCloud<PointColoredN> PointColoredNCloud;
typedef pcl::PointCloud<pcl::Normal> NormalCloud;
typedef pcl::PointCloud<PointN> PointNCloud;
typedef pcl::PointCloud<PointRF> PointRFCloud;

extern const std::string DATA_DEBUG_PATH;
extern const std::string TRANSFORMATIONS_CSV;
extern const std::string ITERATIONS_CSV;
extern const std::string VERSION;
extern const std::string SUBVERSION;
extern const std::string ALIGNMENT_RANSAC;
extern const std::string ALIGNMENT_GROR;
extern const std::string ALIGNMENT_TEASER;
extern const std::string KEYPOINT_ANY;
extern const std::string KEYPOINT_ISS;
extern const std::string DESCRIPTOR_FPFH;
extern const std::string DESCRIPTOR_SHOT;
extern const std::string DESCRIPTOR_ROPS;
extern const std::string DESCRIPTOR_USC;
extern const std::string DEFAULT_LRF;
extern const std::string METRIC_CORRESPONDENCES;
extern const std::string METRIC_UNIFORMITY;
extern const std::string METRIC_CLOSEST_PLANE;
extern const std::string METRIC_WEIGHTED_CLOSEST_PLANE;
extern const std::string METRIC_COMBINATION;
extern const std::string MATCHING_LEFT_TO_RIGHT;
extern const std::string MATCHING_RATIO;
extern const std::string MATCHING_CLUSTER;
extern const std::string MATCHING_ONE_SIDED;
extern const std::string METRIC_WEIGHT_CONSTANT;
extern const std::string METRIC_WEIGHT_EXP_CURVATURE;
extern const std::string METRIC_WEIGHT_CURVEDNESS;
extern const std::string METRIC_WEIGHT_HARRIS;
extern const std::string METRIC_WEIGHT_TOMASI;
extern const std::string METRIC_WEIGHT_CURVATURE;
extern const std::string METRIC_WEIGHT_NSS;
extern const std::string METRIC_SCORE_CONSTANT;
extern const std::string METRIC_SCORE_MAE;
extern const std::string METRIC_SCORE_MSE;
extern const std::string METRIC_SCORE_EXP;

struct Correspondence : pcl::Correspondence {
    Correspondence() : pcl::Correspondence(), threshold(0.f) {}

    Correspondence(int index_query, int index_match, float distance, float thr) :
            pcl::Correspondence(index_query, index_match, distance), threshold(thr) {}

    float threshold;
};

typedef std::vector<Correspondence> Correspondences;
typedef std::shared_ptr<std::vector<Correspondence>> CorrespondencesPtr;
typedef std::shared_ptr<const std::vector<Correspondence>> CorrespondencesConstPtr;

pcl::Correspondences correspondencesToPCL(const Correspondences &correspondences);

struct AlignmentParameters {
    bool reestimate_frames{FEATURES_REESTIMATE_FRAMES};
    int feature_nr_points{FEATURE_NR_POINTS}, normal_nr_points{NORMAL_NR_POINTS};
    float edge_thr_coef{ALIGNMENT_EDGE_THR};
    float distance_thr, iss_radius_src, iss_radius_tgt;
    // optional parameter, if radius isn't set multi-scale matching is used
    std::optional<float> feature_radius;
    float scale_factor{FEATURES_SCALE_FACTOR};
    float confidence{ALIGNMENT_CONFIDENCE};
    bool use_bfmatcher{ALIGNMENT_USE_BFMATCHER};
    int bf_block_size{ALIGNMENT_BLOCK_SIZE};
    int ratio_k{MATCHING_RATIO_K}, cluster_k{MATCHING_CLUSTER_K};
    int randomness{ALIGNMENT_RANDOMNESS}, n_samples{ALIGNMENT_N_SAMPLES};
    std::string alignment_id{ALIGNMENT_RANSAC}, descriptor_id{DESCRIPTOR_SHOT}, keypoint_id{KEYPOINT_ISS};
    std::string metric_id{METRIC_COMBINATION}, matching_id{MATCHING_CLUSTER}, lrf_id{DEFAULT_LRF};
    std::string weight_id{METRIC_WEIGHT_CONSTANT}, score_id{METRIC_SCORE_MSE};
    int max_iterations;

    bool save_features;
    std::string testname;
    std::optional<Eigen::Matrix4f> ground_truth{std::nullopt};

    // these parameters cannot be set in config, they are set before alignment steps
    bool fix_seed = true, normals_available;
    float match_search_radius = 0;
    std::optional<Eigen::Matrix4f> guess{std::nullopt};
    std::string dir_path{DATA_DEBUG_PATH};
    std::optional<Eigen::Vector3f> vp_src, vp_tgt;
};

struct AlignmentResult {
    PointNCloud::ConstPtr src, tgt;
    PointNCloud::ConstPtr kps_src, kps_tgt;
    Eigen::Matrix4f transformation;
    CorrespondencesConstPtr correspondences;
    int iterations;
    bool converged;

    // transformation estimation time, correspondence search time
    double time_te, time_cs{0.0};
};

std::vector<AlignmentParameters> getParametersFromConfig(const YamlConfig &config,
                                                         const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                                                         const std::vector<::pcl::PCLPointField> &fields_src,
                                                         const std::vector<::pcl::PCLPointField> &fields_tgt);

void filterDuplicatePoints(PointNCloud::Ptr &pcd);

void loadPointClouds(const YamlConfig &config, std::string &testname, PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                     std::vector<::pcl::PCLPointField> &fields_src, std::vector<::pcl::PCLPointField> &fields_tgt);

void loadTransformationGt(const YamlConfig &config, const std::optional<std::string> &csv_path,
                          std::optional<Eigen::Matrix4f> &transformation_gt);

void loadViewpoint(const std::optional<std::string> &viewpoints_path,
                   const std::string &pcd_path, std::optional<Eigen::Vector3f> &viewpoint);

struct MultivaluedCorrespondence {
    pcl::Indices match_indices;
    std::vector<float> distances;
};

std::ostream &operator<<(std::ostream &out, const MultivaluedCorrespondence &corr);

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

std::string constructName(const AlignmentParameters &parameters, const std::string &name,
                          bool with_version = true, bool with_metric = true,
                          bool with_weights = true, bool with_subversion = false);

std::string constructPath(const std::string &test, const std::string &name,
                          const std::string &extension = "ply", bool with_version = true, bool with_subversion = false);

std::string constructPath(const AlignmentParameters &parameters, const std::string &name,
                          const std::string &extension = "ply",
                          bool with_version = true, bool with_metric = true,
                          bool with_weights = true, bool with_subversion = false);

void printTransformation(const Eigen::Matrix4f &transformation);

std::optional<Eigen::Matrix4f> getTransformation(const std::string &csv_path,
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
std::pair<PointT, PointT> calculateBoundingBox(const typename pcl::PointCloud<PointT>::ConstPtr &pcd) {
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

float calculatePointCloudDensity(const typename pcl::PointCloud<PointN>::ConstPtr &pcd, float quantile = 0.8);

std::vector<float> calculateSmoothedDensities(const PointNCloud::ConstPtr &pcd, int k = 2);

float getAABBDiagonal(const PointNCloud::Ptr &pcd);

void mergeOverlaps(const PointNCloud::ConstPtr &pcd1, const PointNCloud::ConstPtr &pcd2, PointNCloud::Ptr &dst,
                   float distance_thr);

void estimateNormalsRadius(float radius_search, PointNCloud::Ptr &pcd, const PointNCloud::ConstPtr &surface,
                           const std::optional<Eigen::Vector3f> &vp, bool normals_available);

void estimateNormalsPoints(int k_points, PointNCloud::Ptr &pcd, const PointNCloud::ConstPtr &surface,
                           const std::optional<Eigen::Vector3f> &vp, bool normals_available);

PointNCloud::ConstPtr detectKeyPoints(const PointNCloud::ConstPtr &pcd, const AlignmentParameters &parameters,
                                      float iss_radius, bool debug = false, bool subvoxel = false);

void estimateReferenceFrames(const PointNCloud::ConstPtr &pcd, const PointNCloud::ConstPtr &surface,
                             PointRFCloud::Ptr &frames, float radius_search, const AlignmentParameters &parameters);

template<typename FeatureT>
void estimateFeatures(const PointNCloud::ConstPtr &pcd, const PointNCloud::ConstPtr &surface,
                      typename pcl::PointCloud<FeatureT>::Ptr &features,
                      float radius_search, const AlignmentParameters &parameters) {
    throw std::runtime_error("Feature with proposed reference frame isn't supported!");
}

template<>
inline void estimateFeatures<FPFH>(const PointNCloud::ConstPtr &pcd, const PointNCloud::ConstPtr &surface,
                                   pcl::PointCloud<FPFH>::Ptr &features,
                                   float radius_search, const AlignmentParameters &parameters) {
    pcl::FPFHEstimationOMP<PointN, PointN, FPFH> fpfh_estimation;
    fpfh_estimation.setRadiusSearch(radius_search);
    fpfh_estimation.setInputCloud(pcd);
    fpfh_estimation.setSearchSurface(surface);
    fpfh_estimation.setInputNormals(surface);
    fpfh_estimation.compute(*features);
}

template<>
inline void estimateFeatures<USC>(const PointNCloud::ConstPtr &pcd, const PointNCloud::ConstPtr &surface,
                                  pcl::PointCloud<USC>::Ptr &features,
                                  float radius_search, const AlignmentParameters &parameters) {
    pcl::UniqueShapeContext<PointN, USC, PointRF> shape_context;
    shape_context.setInputCloud(pcd);
    shape_context.setSearchSurface(surface);
    shape_context.setMinimalRadius(radius_search / 10.f);
    shape_context.setRadiusSearch(radius_search);
    shape_context.setPointDensityRadius(radius_search / 5.f);
    shape_context.setLocalRadius(radius_search);
    shape_context.compute(*features);
}

template<>
inline void estimateFeatures<RoPS135>(const PointNCloud::ConstPtr &pcd, const PointNCloud::ConstPtr &surface,
                                      pcl::PointCloud<RoPS135>::Ptr &features,
                                      float radius_search, const AlignmentParameters &parameters) {
    // RoPs estimation object.
    ROPSEstimationWithLocalReferenceFrames<PointN, RoPS135> rops;
    rops.setInputCloud(pcd);
    rops.setSearchSurface(surface);
    rops.setRadiusSearch(radius_search);
    // Number of partition bins that is used for distribution matrix calculation.
    rops.setNumberOfPartitionBins(5);
    // The greater the number of rotations is, the bigger the resulting descriptor.
    // Make sure to change the histogram size accordingly.
    rops.setNumberOfRotations(3);
    // Support radius that is used to crop the local surface of the point.
    rops.setSupportRadius(radius_search);

    PointRFCloud::Ptr frames{nullptr};
    estimateReferenceFrames(pcd, surface, frames, radius_search, parameters);
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
}

template<>
inline void estimateFeatures<SHOT>(const PointNCloud::ConstPtr &pcd, const PointNCloud::ConstPtr &surface,
                                   pcl::PointCloud<SHOT>::Ptr &features,
                                   float radius_search, const AlignmentParameters &parameters) {
    // SHOT estimation object.
    SHOTEstimationDebug shot;
    shot.setInputCloud(pcd);
    shot.setSearchSurface(surface);
    shot.setInputNormals(surface);
    // The radius that defines which of the keypoint's neighbors are described.
    // If too large, there may be clutter, and if too small, not enough points may be found.
    shot.setRadiusSearch(radius_search);
    PointRFCloud::Ptr frames{nullptr};
    estimateReferenceFrames(pcd, surface, frames, radius_search, parameters);
    if (frames) shot.setInputReferenceFrames(frames);
    PCL_WARN("[estimateFeatures<SHOT>] Points probably have NaN normals in their neighbourhood\n");
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    shot.compute(*features);
    std::string volumes_path = constructPath(parameters, "volumes", "csv", true, false, false);
//    saveVectorOfArrays<int, 32>(shot.getVolumesDebugInfo(), volumes_path);
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
}

void saveColorizedPointCloud(const PointNCloud::ConstPtr &pcd, const Eigen::Matrix4f &transformation_gt,
                             int color, const std::string &filepath);

void savePointCloudWithCorrespondences(const PointNCloud::ConstPtr &pcd,
                                       const PointNCloud::ConstPtr &key_points,
                                       const Correspondences &correspondences,
                                       const Correspondences &correct_correspondences,
                                       const Correspondences &inliers,
                                       const AlignmentParameters &parameters,
                                       const Eigen::Matrix4f &transformation_gt, bool is_source);

void saveColorizedWeights(const PointNCloud::ConstPtr &pcd, std::vector<float> &weights, const std::string &name,
                          const AlignmentParameters &parameters, const Eigen::Matrix4f &transformation_gt);

enum TemperatureType {
    Distance, NormalDifference
};

void saveTemperatureMaps(PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                         const std::string &name, const AlignmentParameters &params, float distance_thr,
                         const Eigen::Matrix4f &transformation, bool normals_available = true);

void saveCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                         const Correspondences &correspondences,
                         const Eigen::Matrix4f &transformation_gt,
                         const AlignmentParameters &parameters, bool sparse = false);

void saveCorrectCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                const Correspondences &correspondences,
                                const Correspondences &correct_correspondences,
                                const Eigen::Matrix4f &transformation_gt,
                                const AlignmentParameters &parameters, bool sparse = false);

void saveCorrespondenceDistances(const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                                 const Correspondences &correspondences,
                                 const Eigen::Matrix4f &transformation_gt,
                                 const AlignmentParameters &parameters);

void saveCorrespondencesDebug(const Correspondences &correspondences,
                              const Correspondences &correct_correspondences,
                              const AlignmentParameters &parameters);

void setPointColor(PointColoredN &point, int color);

void mixPointColor(PointColoredN &point, int color);

void setPointColor(PointColoredN &point, std::uint8_t red, std::uint8_t green, std::uint8_t blue);

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

void readKeyPointsAndCorrespondences(CorrespondencesConstPtr &correspondences,
                                     PointNCloud::ConstPtr &kps_src, PointNCloud::ConstPtr &kps_tgt,
                                     const AlignmentParameters &params, bool &success);

void saveKeyPointsAndCorrespondences(const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                                     const CorrespondencesConstPtr &correspondences,
                                     const AlignmentParameters &params);

#endif
