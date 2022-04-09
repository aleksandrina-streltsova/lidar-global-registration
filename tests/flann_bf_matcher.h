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
    if (corr1.match_indices[0] != corr2.match_indices[0]) {
        std::cerr << "{" << i << "} match indices differ: [" << corr1.match_indices[0] << "] [" << corr2.match_indices[0] << "]\n";
        std::cerr << "{" << i << "} with distances: [" << corr1.distances[0] << "] [" << corr2.distances[0] << "]" << std::endl;
        abort();
    }
}

template<typename FeatureT>
void run_test(const PointCloudTN::Ptr &src_fullsize,
              const PointCloudTN::Ptr &tgt_fullsize,
              const AlignmentParameters &parameters) {
    PointCloudTN::Ptr src_downsize(new PointCloudTN), tgt_downsize(new PointCloudTN);
    // Downsample
    if (parameters.downsample) {
        pcl::console::print_highlight("Downsampling...\n");
        downsamplePointCloud(src_fullsize, src_downsize, parameters);
        downsamplePointCloud(tgt_fullsize, tgt_downsize, parameters);
    }

    PointCloudTN::Ptr src(new PointCloudTN), tgt(new PointCloudTN), src_aligned(new PointCloudTN);
    PointCloudN::Ptr normals_src(new PointCloudN), normals_tgt(new PointCloudN);
    PointCloudRF::Ptr frames_src(nullptr), frames_tgt(nullptr);
    typename pcl::PointCloud<FeatureT>::Ptr features_src(new pcl::PointCloud<FeatureT>);
    typename pcl::PointCloud<FeatureT>::Ptr features_tgt(new pcl::PointCloud<FeatureT>);

    float voxel_size = parameters.voxel_size;
    float normal_radius = parameters.normal_radius_coef * voxel_size;
    float feature_radius = parameters.feature_radius_coef * voxel_size;

    if (!parameters.use_normals) {
        estimateNormals(normal_radius, src_downsize, normals_src, parameters.normals_available);
        estimateNormals(normal_radius, tgt_downsize, normals_tgt, parameters.normals_available);
    }

    pcl::concatenateFields(*src_downsize, *normals_src, *src);
    pcl::concatenateFields(*tgt_downsize, *normals_tgt, *tgt);

    // Estimate reference frames
    estimateReferenceFrames(src, normals_src, frames_src, parameters, true);
    estimateReferenceFrames(tgt, normals_tgt, frames_tgt, parameters, false);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    estimateFeatures<FeatureT>(feature_radius, src, src_fullsize, normals_src, frames_src, features_src);
    estimateFeatures<FeatureT>(feature_radius, tgt, tgt_fullsize, normals_tgt, frames_tgt, features_tgt);

    std::vector<MultivaluedCorrespondence> correspondences_bf;
    std::vector<MultivaluedCorrespondence> correspondences_flann;

    auto point_representation = std::shared_ptr<pcl::PointRepresentation<FeatureT>>(new pcl::DefaultPointRepresentation<FeatureT>);
    int nr_dims = point_representation->getNumberOfDimensions();
    int threads = 1;
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
    threads = omp_get_num_procs();
#endif
    int k_matches = parameters.randomness;
    matchBF<FeatureT>(features_src, features_tgt, correspondences_bf, point_representation, k_matches, nr_dims, parameters.bf_block_size);
    matchFLANN<FeatureT>(features_src, features_tgt, correspondences_flann, point_representation, k_matches, threads);
    for (int i = 0; i < features_src->size(); ++i) {
        if (correspondences_flann[i].match_indices.empty()) {
            std::cout << "invalid point [" << i << "]" << std::endl;
        } else {
            assertCorrespondencesEqual(i, correspondences_bf[i], correspondences_flann[i]);
        }
    }

    correspondences_bf.clear();
    correspondences_flann.clear();

    matchBF<FeatureT>(features_tgt, features_src, correspondences_bf, point_representation, k_matches, nr_dims, parameters.bf_block_size);
    matchFLANN<FeatureT>(features_tgt, features_src, correspondences_flann, point_representation, k_matches, threads);
    for (int i = 0; i < features_tgt->size(); ++i) {
        if (correspondences_flann[i].match_indices.empty()) {
            std::cout << "invalid point [" << i << "]" << std::endl;
        } else {
            assertCorrespondencesEqual(i, correspondences_bf[i], correspondences_flann[i]);
        }
    }
}

#endif
