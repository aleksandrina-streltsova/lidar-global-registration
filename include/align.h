#ifndef REGISTRATION_ALIGN_H
#define REGISTRATION_ALIGN_H

#include <Eigen/Core>

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/usc.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/common/time.h>
#include <pcl/surface/gp3.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/features/shot_lrf.h>

#include "common.h"
#include "downsample.h"
#include "rops_custom_lrf.h"
#include "sac_prerejective_omp.h"
#include "feature_analysis.h"
#include "weights.h"

#define INITIAL_STEP_VOXEL_SIZE_COEF 8
#define FINAL_STEP_EDG_THR_COEF 0.99
#define MATCH_SEARCH_RADIUS_COEF 4

namespace fs = std::filesystem;

void estimateNormals(float radius_search, const PointNCloud::Ptr &pcd, NormalCloud::Ptr &normals,
                     bool normals_available);

void smoothNormals(float radius_search, float voxel_size, const PointNCloud::Ptr &pcd);

void detectKeyPoints(const PointNCloud::ConstPtr &pcd, const NormalCloud::ConstPtr &normals, pcl::IndicesPtr &indices,
                     const AlignmentParameters &parameters);

void estimateReferenceFrames(const PointNCloud::Ptr &pcd, const NormalCloud::Ptr &normals,
                             const pcl::IndicesPtr &indices, PointRFCloud::Ptr &frames_kps,
                             const AlignmentParameters &parameters, bool is_source);

template<typename FeatureT, typename PointRFT = PointRF>
void estimateFeatures(float radius_search, const PointNCloud::Ptr &pcd,
                      const NormalCloud::Ptr &normals,
                      const pcl::IndicesConstPtr &indices,
                      const typename pcl::PointCloud<PointRFT>::Ptr &frames,
                      typename pcl::PointCloud<FeatureT>::Ptr &features) {
    throw std::runtime_error("Feature with proposed reference frame isn't supported!");
}

template<>
inline void estimateFeatures<FPFH>(float radius_search, const PointNCloud::Ptr &pcd,
                                   const NormalCloud::Ptr &normals,
                                   const pcl::IndicesConstPtr &indices,
                                   const PointRFCloud::Ptr &frames,
                                   pcl::PointCloud<FPFH>::Ptr &features) {
    int nr_kps = indices ? indices->size() : pcd->size();
    pcl::FPFHEstimationOMP<PointN, pcl::Normal, FPFH> fpfh_estimation;
    fpfh_estimation.setRadiusSearch(radius_search);
    fpfh_estimation.setInputCloud(pcd);
    fpfh_estimation.setInputNormals(normals);
    if (indices) fpfh_estimation.setIndices(indices);
    fpfh_estimation.compute(*features);
    rassert(features->size() == nr_kps, 104935923)
}

template<>
inline void estimateFeatures<USC>(float radius_search, const PointNCloud::Ptr &pcd,
                                  const NormalCloud::Ptr &normals,
                                  const pcl::IndicesConstPtr &indices,
                                  const PointRFCloud::Ptr &frames,
                                  pcl::PointCloud<USC>::Ptr &features) {
    int nr_kps = indices ? indices->size() : pcd->size();
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
inline void estimateFeatures<pcl::ShapeContext1980>(float radius_search, const PointNCloud::Ptr &pcd,
                                                    const NormalCloud::Ptr &normals,
                                                    const pcl::IndicesConstPtr &indices,
                                                    const PointRFCloud::Ptr &frames,
                                                    pcl::PointCloud<pcl::ShapeContext1980>::Ptr &features) {
    int nr_kps = indices ? indices->size() : pcd->size();
    pcl::ShapeContext3DEstimation<PointN, pcl::Normal, pcl::ShapeContext1980> shape_context;
    shape_context.setInputCloud(pcd);
    shape_context.setInputNormals(normals);
    if (indices) shape_context.setIndices(indices);
    shape_context.setMinimalRadius(radius_search / 10.f);
    shape_context.setRadiusSearch(radius_search);
    shape_context.compute(*features);
    rassert(features->size() == nr_kps, 1398423940)
    std::cout << "output points.size (): " << features->points.size() << std::endl;
}

template<>
inline void estimateFeatures<RoPS135>(float radius_search, const PointNCloud::Ptr &pcd,
                                      const NormalCloud::Ptr &normals,
                                      const pcl::IndicesConstPtr &indices,
                                      const PointRFCloud::Ptr &frames,
                                      pcl::PointCloud<RoPS135>::Ptr &features) {
    int nr_kps = indices ? indices->size() : pcd->size();
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
inline void estimateFeatures<SHOT>(float radius_search, const PointNCloud::Ptr &pcd,
                                   const NormalCloud::Ptr &normals,
                                   const pcl::IndicesConstPtr &indices,
                                   const PointRFCloud::Ptr &frames,
                                   pcl::PointCloud<SHOT>::Ptr &features) {
    int nr_kps = indices ? indices->size() : pcd->size();
    // SHOT estimation object.
    pcl::SHOTEstimationOMP<PointN, pcl::Normal, SHOT> shot;
    shot.setInputCloud(pcd);
    if (indices) shot.setIndices(indices);
//	shot.setSearchSurface(surface);
    shot.setInputNormals(normals);
    // The radius that defines which of the keypoint's neighbors are described.
    // If too large, there may be clutter, and if too small, not enough points may be found.
    shot.setRadiusSearch(radius_search);
    if (frames) shot.setInputReferenceFrames(frames);
    PCL_WARN("[estimateFeatures<SHOT>] Points probably have NaN normals in their neighbourhood\n");
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    shot.compute(*features);
    rassert(features->size() == nr_kps, 845637470190)
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
}

template<typename FeatureT>
SampleConsensusPrerejectiveOMP<FeatureT> executeAlignmentStep(const PointNCloud::Ptr &src_final,
                                                              const PointNCloud::Ptr &tgt_final,
                                                              const AlignmentParameters &parameters) {
    bool is_initial = parameters.guess == nullptr;
    pcl::console::print_highlight("Starting%s alignment step...\n", std::string(is_initial ? " initial" : "").c_str());

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    PointNCloud::Ptr src_aligned(new PointNCloud);
    SampleConsensusPrerejectiveOMP<FeatureT> align;

    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    pcl::IndicesPtr indices_src(nullptr), indices_tgt(nullptr);
    PointRFCloud::Ptr frames_kps_src(nullptr), frames_kps_tgt(nullptr);
    typename pcl::PointCloud<FeatureT>::Ptr features_kps_src(new pcl::PointCloud<FeatureT>);
    typename pcl::PointCloud<FeatureT>::Ptr features_kps_tgt(new pcl::PointCloud<FeatureT>);

    float voxel_size = parameters.voxel_size;
    float normal_radius = parameters.normal_radius_coef * voxel_size;
    float feature_radius = parameters.feature_radius_coef * voxel_size;

    // Downsample
    pcl::console::print_highlight("Downsampling [voxel_size = %.5f]...\n", parameters.voxel_size);
    downsamplePointCloud(src_final, src, parameters);
    downsamplePointCloud(tgt_final, tgt, parameters);

    // Estimate normals
    if (parameters.use_normals) {
        pcl::console::print_highlight("Normals are from point clouds. Smoothing normals...\n");
        smoothNormals(normal_radius, voxel_size, src);
        smoothNormals(normal_radius, voxel_size, tgt);
        pcl::copyPointCloud(*src, *normals_src);
        pcl::copyPointCloud(*tgt, *normals_tgt);
    } else {
        pcl::console::print_highlight("Estimating normals...\n");
        estimateNormals(normal_radius, src, normals_src, parameters.normals_available);
        estimateNormals(normal_radius, tgt, normals_tgt, parameters.normals_available);
        pcl::concatenateFields(*src, *normals_src, *src);
        pcl::concatenateFields(*tgt, *normals_tgt, *tgt);
    }

    // align.readCorrespondences(parameters);
    if (!align.correspondencesFromFile()) {
        // Detect key points
        pcl::console::print_highlight("Detecting key points...\n");
        detectKeyPoints(src, normals_src, indices_src, parameters);
        detectKeyPoints(tgt, normals_tgt, indices_tgt, parameters);

        // Estimate reference frames
        pcl::console::print_highlight("Estimating local reference frames...\n");
        estimateReferenceFrames(src, normals_src, indices_src, frames_kps_src, parameters, true);
        estimateReferenceFrames(tgt, normals_tgt, indices_tgt, frames_kps_tgt, parameters, false);

        // Estimate features
        pcl::console::print_highlight("Estimating features...\n");
        estimateFeatures<FeatureT>(feature_radius, src, normals_src, indices_src, frames_kps_src, features_kps_src);
        estimateFeatures<FeatureT>(feature_radius, tgt, normals_tgt, indices_tgt, frames_kps_tgt, features_kps_tgt);

        if (parameters.save_features) {
            saveFeatures<FeatureT>(features_kps_src, indices_src, parameters, true);
            saveFeatures<FeatureT>(features_kps_tgt, indices_tgt, parameters, false);
        }

        align.setSourceFeatures(features_kps_src);
        align.setTargetFeatures(features_kps_tgt);

        align.setSourceIndices(indices_src);
        align.setTargetIndices(indices_tgt);
    }
    align.setInputSource(src);
    align.setInputTarget(tgt);

    align.setFeatureMatcher(getFeatureMatcher<FeatureT>(parameters));
    align.setMetricEstimator(getMetricEstimator(parameters, true));

    int n_samples = parameters.n_samples;
    int iteration_brute_force = calculate_combination_or_max<int>((int) std::min(src->size(), tgt->size()), n_samples);
    align.setMaximumIterations(parameters.max_iterations.value_or(iteration_brute_force)); // Number of RANSAC iterations
    align.setNumberOfSamples(n_samples); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(parameters.randomness); // Number of nearest features to use
    align.setSimilarityThreshold(parameters.edge_thr_coef); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(parameters.distance_thr_coef * voxel_size); // Inlier threshold
    align.setConfidence(parameters.confidence); // Confidence in adaptive RANSAC
    align.setInlierFraction(parameters.inlier_fraction); // Required inlier fraction for accepting a pose hypothesis
    if (!is_initial) {
        align.setTransformationGuess(*parameters.guess);
    }
    align.setRandomness(parameters.fix_seed);
    {
        pcl::ScopeTime t(is_initial ? "Initial alignment step" : "Alignment step");
        align.align(*src_aligned);
    }
    align.saveCorrespondences(parameters);
    saveTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                       constructName(parameters, "transformation"), align.getFinalTransformation());
    return align;
}

template<typename FeatureT>
std::pair<SampleConsensusPrerejectiveOMP<FeatureT>, double> align_point_clouds(
        const PointNCloud::Ptr &src_fullsize,
        const PointNCloud::Ptr &tgt_fullsize,
        const AlignmentParameters &parameters
) {
    SampleConsensusPrerejectiveOMP<FeatureT> final_alignment;
    double time;
    {
        pcl::ScopeTime t("Alignment");
        if (parameters.coarse_to_fine) {
            PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
            downsamplePointCloud(src_fullsize, src, parameters);
            downsamplePointCloud(tgt_fullsize, tgt, parameters);
            float final_voxel_size = parameters.voxel_size;
            float initial_voxel_size = INITIAL_STEP_VOXEL_SIZE_COEF * final_voxel_size;

            // initial step
            AlignmentParameters initial_parameters(parameters);
            initial_parameters.voxel_size = initial_voxel_size;
            auto initial_alignment = executeAlignmentStep<FeatureT>(src, tgt, initial_parameters);

            // final step
            AlignmentParameters final_parameters(parameters);
            final_parameters.voxel_size = final_voxel_size;
            final_parameters.match_search_radius = MATCH_SEARCH_RADIUS_COEF * initial_voxel_size;
            final_parameters.guess = std::make_shared<Eigen::Matrix4f>(initial_alignment.getFinalTransformation());
            final_parameters.matching_id = MATCHING_LEFT_TO_RIGHT;
            final_parameters.metric_id = METRIC_CORRESPONDENCES;
            final_parameters.edge_thr_coef = FINAL_STEP_EDG_THR_COEF;
            final_alignment = executeAlignmentStep<FeatureT>(src, tgt, final_parameters);

            saveIterationsInfo(fs::path(DATA_DEBUG_PATH) / fs::path(ITERATIONS_CSV),
                               constructName(parameters, "iterations"),
                               {initial_voxel_size, final_voxel_size},
                               {initial_parameters.matching_id, final_parameters.matching_id});
        } else {
            final_alignment = executeAlignmentStep<FeatureT>(src_fullsize, tgt_fullsize, parameters);
            saveIterationsInfo(fs::path(DATA_DEBUG_PATH) / fs::path(ITERATIONS_CSV),
                               constructName(parameters, "iterations"),
                               {parameters.voxel_size}, {parameters.matching_id});
        }
        time = t.getTimeSeconds();
    }
    return {final_alignment, time};
}

#endif
