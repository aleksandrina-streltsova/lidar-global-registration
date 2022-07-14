#ifndef REGISTRATION_FLANN_BF_MATCHER_H
#define REGISTRATION_FLANN_BF_MATCHER_H

#include <vector>
#include <pcl/common/io.h>

#include "common.h"
#include "downsample.h"
#include "alignment.h"
#include "matching.h"

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

    if (!parameters.use_normals) {
        estimateNormalsRadius(normal_radius, src_downsize, normals_src, parameters.normals_available);
        estimateNormalsRadius(normal_radius, tgt_downsize, normals_tgt, parameters.normals_available);
    }

    pcl::concatenateFields(*src_downsize, *normals_src, *src);
    pcl::concatenateFields(*tgt_downsize, *normals_tgt, *tgt);

    tree_src->setInputCloud(src);
    tree_tgt->setInputCloud(tgt);

    // Detect key points
    auto indices_src = detectKeyPoints(src, parameters);
    auto indices_tgt = detectKeyPoints(tgt, parameters);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    estimateFeatures<FeatureT>(src, indices_src, features_src, parameters);
    estimateFeatures<FeatureT>(tgt, indices_tgt, features_tgt, parameters);

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
    AlignmentParameters parameters_local(parameters);
    parameters_local.guess = std::optional<Eigen::Matrix4f>(Eigen::Matrix4f::Identity());
    parameters_local.match_search_radius = std::numeric_limits<float>::max();

    mv_correspondences_bf = matchBF<FeatureT>(features_src, features_tgt, parameters);
    mv_correspondences_flann = matchFLANN<FeatureT>(features_src, features_tgt, parameters);
    mv_correspondences_local = matchLocal<FeatureT>(src, tree_tgt, features_src, features_tgt, parameters_local);
    for (int i = 0; i < features_src->size(); ++i) {
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_flann[i]);
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_local[i]);
    }

    mv_correspondences_bf.clear();
    mv_correspondences_flann.clear();
    mv_correspondences_local.clear();

    mv_correspondences_bf = matchBF<FeatureT>(features_tgt, features_src, parameters);
    mv_correspondences_flann = matchFLANN<FeatureT>(features_tgt, features_src, parameters);
    mv_correspondences_local = matchLocal<FeatureT>(tgt, tree_src, features_tgt, features_src, parameters_local);
    for (int i = 0; i < features_tgt->size(); ++i) {
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_flann[i]);
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_local[i]);
    }
}

#endif
