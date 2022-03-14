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
#include "sac_prerejective_omp.h"
#include "feature_analysis.h"

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename);

void estimateNormals(float radius_search, const PointCloudTN::Ptr &pcd, PointCloudN::Ptr &normals,
                     bool normals_available);

void smoothNormals(float radius_search, float voxel_size, const PointCloudTN::Ptr &pcd);

void estimateReferenceFrames(const PointCloudTN::Ptr &pcd, const PointCloudN::Ptr &normals,
                             PointCloudRF::Ptr &frames, const AlignmentParameters &parameters, bool is_source);

template<typename FeatureT, typename PointRFT = PointRF>
void estimateFeatures(float radius_search, const PointCloudTN::Ptr &pcd, const PointCloudTN::Ptr &surface,
                      const PointCloudN::Ptr &normals, const typename pcl::PointCloud<PointRFT>::Ptr &frames,
                      typename pcl::PointCloud<FeatureT>::Ptr &features) {
    throw std::runtime_error("Feature with proposed reference frame isn't supported!");
}

template<>
inline void estimateFeatures<FPFH>(float radius_search, const PointCloudTN::Ptr &pcd,
                                   const PointCloudTN::Ptr &surface,
                                   const PointCloudN::Ptr &normals,
                                   const PointCloudRF::Ptr &frames,
                                   pcl::PointCloud<FPFH>::Ptr &features) {
    pcl::FPFHEstimationOMP<PointTN, pcl::Normal, FPFH> fpfh_estimation;
    fpfh_estimation.setRadiusSearch(radius_search);
    fpfh_estimation.setInputCloud(pcd);
    fpfh_estimation.setInputNormals(normals);
    fpfh_estimation.compute(*features);
}

template<>
inline void estimateFeatures<USC>(float radius_search, const PointCloudTN::Ptr &pcd,
                                  const PointCloudTN::Ptr &surface,
                                  const PointCloudN::Ptr &normals,
                                  const PointCloudRF::Ptr &frames,
                                  pcl::PointCloud<USC>::Ptr &features) {
    pcl::UniqueShapeContext<PointTN, USC, PointRF> shape_context;
    shape_context.setInputCloud(pcd);
    shape_context.setMinimalRadius(radius_search / 10.f);
    shape_context.setRadiusSearch(radius_search);
    shape_context.setPointDensityRadius(radius_search / 5.f);
    shape_context.setLocalRadius(radius_search);
    shape_context.compute(*features);
    std::cout << "output points.size (): " << features->points.size() << std::endl;
}

template<>
inline void estimateFeatures<pcl::ShapeContext1980>(float radius_search, const PointCloudTN::Ptr &pcd,
                                                    const PointCloudTN::Ptr &surface,
                                                    const PointCloudN::Ptr &normals,
                                                    const PointCloudRF::Ptr &frames,
                                                    pcl::PointCloud<pcl::ShapeContext1980>::Ptr &features) {
    pcl::ShapeContext3DEstimation<PointTN, pcl::Normal, pcl::ShapeContext1980> shape_context;
    shape_context.setInputCloud(pcd);
    shape_context.setInputNormals(normals);
    shape_context.setMinimalRadius(radius_search / 10.f);
    shape_context.setRadiusSearch(radius_search);
    shape_context.compute(*features);
    std::cout << "output points.size (): " << features->points.size() << std::endl;
}

template<>
inline void estimateFeatures<RoPS135>(float radius_search, const PointCloudTN::Ptr &pcd,
                                      const PointCloudTN::Ptr &surface,
                                      const PointCloudN::Ptr &normals,
                                      const PointCloudRF::Ptr &frames,
                                      pcl::PointCloud<RoPS135>::Ptr &features) {
    // Perform triangulation.
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::search::KdTree<PointTN>::Ptr tree_n(new pcl::search::KdTree<PointTN>);
    tree_n->setInputCloud(pcd);

    pcl::GreedyProjectionTriangulation<PointTN> triangulation;
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

    // Note: you should only compute descriptors for chosen keypoints. It has
    // been omitted here for simplicity.

    // RoPs estimation object.
    pcl::ROPSEstimation<PointTN, RoPS135> rops;
    rops.setInputCloud(pcd);
//	rops.setSearchMethod(tree); TODO: is it necessary?
    rops.setRadiusSearch(radius_search);
    rops.setTriangles(triangles.polygons);
    // Number of partition bins that is used for distribution matrix calculation.
    rops.setNumberOfPartitionBins(5);
    // The greater the number of rotations is, the bigger the resulting descriptor.
    // Make sure to change the histogram size accordingly.
    rops.setNumberOfRotations(3);
    // Support radius that is used to crop the local surface of the point.
    rops.setSupportRadius(radius_search);
    rops.compute(*features);
}

template<>
inline void estimateFeatures<SHOT>(float radius_search, const PointCloudTN::Ptr &pcd,
                                   const PointCloudTN::Ptr &surface,
                                   const PointCloudN::Ptr &normals,
                                   const PointCloudRF::Ptr &frames,
                                   pcl::PointCloud<SHOT>::Ptr &features) {

    // SHOT estimation object.
    pcl::SHOTEstimationOMP<PointTN, pcl::Normal, SHOT> shot;
    shot.setInputCloud(pcd);
//	shot.setSearchSurface(surface);
    shot.setInputNormals(normals);
    // The radius that defines which of the keypoint's neighbors are described.
    // If too large, there may be clutter, and if too small, not enough points may be found.
    shot.setRadiusSearch(radius_search);
    if (frames) {
        shot.setInputReferenceFrames(frames);
    }
    PCL_WARN("[estimateFeatures<SHOT>] Points probably have NaN normals in their neighbourhood\n");
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    shot.compute(*features);
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
}

template<typename FeatureT>
SampleConsensusPrerejectiveOMP<PointTN, PointTN, FeatureT> align_point_clouds(
        const PointCloudTN::Ptr &src_fullsize,
        const PointCloudTN::Ptr &tgt_fullsize,
        const AlignmentParameters &parameters
) {
    PointCloudTN::Ptr src(new PointCloudTN), tgt(new PointCloudTN);
    // Downsample
    if (parameters.downsample) {
        pcl::console::print_highlight("Downsampling...\n");
        downsamplePointCloud(src_fullsize, src, parameters);
        downsamplePointCloud(tgt_fullsize, tgt, parameters);
    } else {
        pcl::console::print_highlight("Filtering duplicate points...\n");
        pcl::copyPointCloud(*src_fullsize, *src);
        pcl::copyPointCloud(*tgt_fullsize, *tgt);
    }

    PointCloudTN::Ptr src_aligned(new PointCloudTN);
    SampleConsensusPrerejectiveOMP<PointTN, PointTN, FeatureT> align;

    PointCloudN::Ptr normals_src(new PointCloudN), normals_tgt(new PointCloudN);
    PointCloudRF::Ptr frames_src(new PointCloudRF), frames_tgt(new PointCloudRF);
    typename pcl::PointCloud<FeatureT>::Ptr features_src(new pcl::PointCloud<FeatureT>);
    typename pcl::PointCloud<FeatureT>::Ptr features_tgt(new pcl::PointCloud<FeatureT>);

    float voxel_size = parameters.voxel_size;
    float normal_radius = parameters.normal_radius_coef * voxel_size;
    float feature_radius = parameters.feature_radius_coef * voxel_size;

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

    // Estimate reference frames
    estimateReferenceFrames(src, normals_src, frames_src, parameters, true);
    estimateReferenceFrames(tgt, normals_tgt, frames_tgt, parameters, false);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    estimateFeatures<FeatureT>(feature_radius, src, src_fullsize, normals_src, frames_src, features_src);
    estimateFeatures<FeatureT>(feature_radius, tgt, tgt_fullsize, normals_tgt, frames_tgt, features_tgt);

    if (parameters.save_features) {
        saveFeatures<FeatureT>(features_src, parameters, true);
        saveFeatures<FeatureT>(features_tgt, parameters, false);
    }
    // Filter point clouds
    // TODO: fix filtering (separate debug from actual filtering)
//    auto func_id = parameters.func_id;
//    auto func = getUniquenessFunction(func_id);
//    if (func != nullptr) {
//        std::cout << "Point cloud downsampled after filtration (" << func_id << ") from " << src->size();
//        filterPointCloud(func, func_id, src, features_src, src, features_src, transformation_gt, testname, true);
//        std::cout << " to " << src->size() << "\n";
//        std::cout << "Point cloud downsampled after filtration (" << func_id << ") from " << tgt->size();
//        filterPointCloud(func, func_id, tgt, features_tgt, tgt, features_tgt, transformation_gt, testname, false);
//        std::cout << " to " << tgt->size() << "\n";
//    }

    if (parameters.reciprocal) {
        align.enableMutualFiltering();
    }
    if (parameters.use_bfmatcher) {
        align.useBFMatcher();
        align.setBFBlockSize(parameters.bf_block_size);
    }

    align.setInputSource(src);
    align.setSourceFeatures(features_src);

    align.setInputTarget(tgt);
    align.setTargetFeatures(features_tgt);

    int n_samples = parameters.n_samples;
    int iteration_brute_force = calculate_combination_or_max<int>((int) std::min(src->size(), tgt->size()), n_samples);
    align.setMaximumIterations(parameters.max_iterations.value_or(iteration_brute_force)); // Number of RANSAC iterations
    align.setNumberOfSamples(n_samples); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(parameters.randomness); // Number of nearest features to use
    align.setSimilarityThreshold(parameters.edge_thr_coef); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(parameters.distance_thr_coef * voxel_size); // Inlier threshold
    align.setConfidence(parameters.confidence); // Confidence in adaptive RANSAC
    align.setInlierFraction(parameters.inlier_fraction); // Required inlier fraction for accepting a pose hypothesis
    std::cout << "    iteration: " << align.getMaximumIterations() << std::endl;
    std::cout << "    voxel size: " << voxel_size << std::endl;
    {
        pcl::ScopeTime t("Alignment");
        align.align(*src_aligned);
    }
    return align;
}

#endif
