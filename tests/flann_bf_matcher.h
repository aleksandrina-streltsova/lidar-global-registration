#ifndef REGISTRATION_FLANN_BF_MATCHER_H
#define REGISTRATION_FLANN_BF_MATCHER_H

#include <vector>
#include <pcl/common/io.h>

#include "common.h"
#include "downsample.h"
#include "alignment.h"
#include "matching.h"

inline bool isclose(float a, float b, float rtol = 1e-5, float atol = 1e-8) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

void assertCorrespondencesEqual(int i,
                                const MultivaluedCorrespondence &corr1,
                                const MultivaluedCorrespondence &corr2) {
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
void runTest(const PointNCloud::Ptr &src,
             const PointNCloud::Ptr &tgt,
             const AlignmentParameters &params) {
    PointNCloud::Ptr src_aligned(new PointNCloud);
    PointRFCloud::Ptr frames_src(nullptr), frames_tgt(nullptr);
    typename pcl::PointCloud<FeatureT>::Ptr features_src(new pcl::PointCloud<FeatureT>);
    typename pcl::PointCloud<FeatureT>::Ptr features_tgt(new pcl::PointCloud<FeatureT>);
    pcl::search::KdTree<PointN>::Ptr tree_src(new pcl::search::KdTree<PointN>), tree_tgt(new pcl::search::KdTree<PointN>);

    tree_src->setInputCloud(src);
    tree_tgt->setInputCloud(tgt);

    // Detect key points
    auto kps_src = detectKeyPoints(src, params, params.iss_radius_src);
    auto kps_tgt = detectKeyPoints(tgt, params, params.iss_radius_tgt);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    rassert(params.feature_radius.has_value(), 237859320932);
    estimateFeatures<FeatureT>(kps_src, src, features_src, params.feature_radius.value(), params);
    estimateFeatures<FeatureT>(kps_tgt, tgt, features_tgt, params.feature_radius.value(), params);

    std::vector<MultivaluedCorrespondence> mv_correspondences_bf;
    std::vector<MultivaluedCorrespondence> mv_correspondences_flann;
    std::vector<MultivaluedCorrespondence> mv_correspondences_local;

    auto point_representation = std::shared_ptr<pcl::PointRepresentation<FeatureT>>(new pcl::DefaultPointRepresentation<FeatureT>);
    AlignmentParameters params_local(params);
    params_local.guess = std::optional<Eigen::Matrix4f>(Eigen::Matrix4f::Identity());
    params_local.match_search_radius = std::numeric_limits<float>::max();

    mv_correspondences_bf = matchBF<FeatureT>(features_src, features_tgt, params);
    mv_correspondences_flann = matchFLANN<FeatureT>(features_src, features_tgt, params);
    mv_correspondences_local = matchLocal<FeatureT>(src, tree_tgt, features_src, features_tgt, params_local, params_local.guess.value());
    for (int i = 0; i < features_src->size(); ++i) {
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_flann[i]);
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_local[i]);
    }

    mv_correspondences_bf.clear();
    mv_correspondences_flann.clear();
    mv_correspondences_local.clear();

    mv_correspondences_bf = matchBF<FeatureT>(features_tgt, features_src, params);
    mv_correspondences_flann = matchFLANN<FeatureT>(features_tgt, features_src, params);
    mv_correspondences_local = matchLocal<FeatureT>(tgt, tree_src, features_tgt, features_src, params_local, params_local.guess.value().inverse());
    for (int i = 0; i < features_tgt->size(); ++i) {
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_flann[i]);
        assertCorrespondencesEqual(i, mv_correspondences_bf[i], mv_correspondences_local[i]);
    }
    pcl::console::print_highlight("%d correspondences received after matching target features.\n", mv_correspondences_bf.size());
}

#endif
