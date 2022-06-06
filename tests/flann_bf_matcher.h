#ifndef REGISTRATION_FLANN_BF_MATCHER_H
#define REGISTRATION_FLANN_BF_MATCHER_H

#include <vector>

#include "common.h"
#include "downsample.h"
#include "align.h"

inline bool isclose(float a, float b, float rtol=1e-5, float atol=1e-8) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

void assertCorrespondencesEqual(int i,
                                const MultivaluedCorrespondence &corr1,
                                const MultivaluedCorrespondence &corr2) {
    if (corr1.query_idx != corr2.query_idx) {
        std::cerr << "{" << i << "} query indices differ: [" << corr1.query_idx << "] [" << corr2.query_idx << "]" << std::endl;
        abort();
    }
    if (corr1.match_indices.size() != corr2.match_indices.size()) {
        std::cerr << "{" << i << "} size of match indices differ: [" << corr1.match_indices.size() << "] [" << corr2.match_indices.size() << "]\n";
    }
    for (int j = 0; j < corr1.match_indices.size(); ++j) {
        if (!corr1.match_indices.empty() && corr1.match_indices[j] != corr2.match_indices[j]) {
            std::cerr << "{" << i << "} match indices at position " << j << " differ: [" << corr1.match_indices[j] << "] [" << corr2.match_indices[j] << "]\n";
            std::cerr << "{" << i << "} with distances: [" << corr1.distances[j] << "] [" << corr2.distances[j] << "]" << std::endl;
            abort();
        }
    }
}

template<typename FeatureT>
void runTest(const PointNCloud::Ptr &src_fullsize,
             const PointNCloud::Ptr &tgt_fullsize,
             const AlignmentParameters &parameters) {
    PointNCloud::Ptr src_downsize(new PointNCloud), tgt_downsize(new PointNCloud);
    // Downsample
    pcl::console::print_highlight("Downsampling...\n");
    downsamplePointCloud(src_fullsize, src_downsize, parameters);
    downsamplePointCloud(tgt_fullsize, tgt_downsize, parameters);

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud), src_aligned(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    PointRFCloud::Ptr frames_src(nullptr), frames_tgt(nullptr);
    typename pcl::PointCloud<FeatureT>::Ptr features_src(new pcl::PointCloud<FeatureT>);
    typename pcl::PointCloud<FeatureT>::Ptr features_tgt(new pcl::PointCloud<FeatureT>);
    pcl::search::KdTree<PointN>::Ptr tree_src(new pcl::search::KdTree<PointN>), tree_tgt(new pcl::search::KdTree<PointN>);

    float voxel_size = parameters.voxel_size;
    float normal_radius = parameters.normal_radius_coef * voxel_size;
    float feature_radius = parameters.feature_radius_coef * voxel_size;

    if (!parameters.use_normals) {
        estimateNormals(normal_radius, src_downsize, normals_src, parameters.normals_available);
        estimateNormals(normal_radius, tgt_downsize, normals_tgt, parameters.normals_available);
    }

    pcl::concatenateFields(*src_downsize, *normals_src, *src);
    pcl::concatenateFields(*tgt_downsize, *normals_tgt, *tgt);

    tree_src->setInputCloud(src);
    tree_tgt->setInputCloud(tgt);

    // Estimate reference frames
    estimateReferenceFrames(src, normals_src, nullptr, frames_src, parameters, true);
    estimateReferenceFrames(tgt, normals_tgt, nullptr, frames_tgt, parameters, false);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    estimateFeatures<FeatureT>(feature_radius, src, normals_src, nullptr, frames_src, features_src);
    estimateFeatures<FeatureT>(feature_radius, tgt, normals_tgt, nullptr, frames_tgt, features_tgt);

    std::vector<MultivaluedCorrespondence> mv_correspondences_bf;
    std::vector<MultivaluedCorrespondence> mv_correspondences_flann;
    std::vector<MultivaluedCorrespondence> mv_correspondences_local;

    auto point_representation = std::shared_ptr<pcl::PointRepresentation<FeatureT>>(new pcl::DefaultPointRepresentation<FeatureT>);
    int nr_dims = point_representation->getNumberOfDimensions();
    int threads = 1;
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
    threads = omp_get_num_procs();
#endif
    int k_matches = parameters.randomness;
    matchBF<FeatureT>(features_src, features_tgt, mv_correspondences_bf, point_representation, k_matches, nr_dims, parameters.bf_block_size);
    matchFLANN<FeatureT>(features_src, features_tgt, mv_correspondences_flann, point_representation, k_matches, threads);
    matchLocal<FeatureT>(src, tree_tgt, features_src, features_tgt, mv_correspondences_local, point_representation,
                         Eigen::Matrix4f::Identity(), std::numeric_limits<float>::max(), k_matches, threads);
    for (int i = 0; i < features_src->size(); ++i) {
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_flann[i]);
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_local[i]);
    }

    mv_correspondences_bf.clear();
    mv_correspondences_flann.clear();
    mv_correspondences_local.clear();

    matchBF<FeatureT>(features_tgt, features_src, mv_correspondences_bf, point_representation, k_matches, nr_dims, parameters.bf_block_size);
    matchFLANN<FeatureT>(features_tgt, features_src, mv_correspondences_flann, point_representation, k_matches, threads);
    matchLocal<FeatureT>(tgt, tree_src, features_tgt, features_src, mv_correspondences_local, point_representation,
                         Eigen::Matrix4f::Identity(), std::numeric_limits<float>::max(), k_matches, threads);
    for (int i = 0; i < features_tgt->size(); ++i) {
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_flann[i]);
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_local[i]);
    }
}

#endif
