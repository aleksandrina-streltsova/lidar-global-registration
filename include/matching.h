#ifndef REGISTRATION_MATCHING_H
#define REGISTRATION_MATCHING_H

#include <pcl/point_cloud.h>

#include <opencv2/features2d.hpp>

template<typename FeatureT>
void pcl2cv(int nr_dims, const typename pcl::PointCloud<FeatureT>::ConstPtr &src, cv::OutputArray &dst, int size = 0,
            int offset = 0) {
    if (src->empty()) return;
    int rows = size == 0 ? (src->size() - offset) : std::min((int) (src->size() - offset), size);
    cv::Mat _src(rows, nr_dims, CV_32FC1, (void *) &src->points[offset], sizeof(src->points[0]));
    _src.copyTo(dst);
}

template<typename FeatureT>
void matchFLANN(const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                std::vector<MultivaluedCorrespondence> &correspondences,
                const typename pcl::PointRepresentation<FeatureT>::Ptr point_representation,
                int k_matches, int threads) {
    pcl::KdTreeFLANN<FeatureT> feature_tree(new pcl::KdTreeFLANN<FeatureT>);
    feature_tree.setInputCloud(train_features);
    int n = query_features->size();
    correspondences.resize(n);
#pragma omp parallel for num_threads(threads) default(none) shared(correspondences, query_features, point_representation, feature_tree) firstprivate(n, k_matches)
    for (int i = 0; i < n; i++) {
        if (point_representation->isValid(query_features->points[i])) {
            correspondences[i].query_idx = i;
            pcl::Indices &match_indices = correspondences[i].match_indices;
            std::vector<float> &match_distances = correspondences[i].distances;
            match_indices.resize(k_matches);
            match_distances.resize(k_matches);
            feature_tree.nearestKSearch(*query_features,
                                        i,
                                        k_matches,
                                        match_indices,
                                        match_distances);
            for (int j = 0; j < k_matches; ++j) {
                match_distances[j] = std::sqrt(match_distances[j]);
            }
        }
    }
}

template<typename FeatureT>
void matchBF(const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
             const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
             std::vector<MultivaluedCorrespondence> &correspondences,
             const typename pcl::PointRepresentation<FeatureT>::Ptr point_representation,
             int k_matches, int nr_dims, int block_size) {
    // TODO: support k > 1
    assert(k_matches == 1);

    auto matcher = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<cv::DMatch> matches;

    correspondences.resize(query_features->size());
    int n_query_blocks = (query_features->size() + block_size - 1) / block_size;
    for (int i = 0; i < n_query_blocks; ++i) {
        for (int j = 0; j < (train_features->size() + block_size - 1) / block_size; ++j) {
            cv::UMat query_features_batch, train_features_batch;
            pcl2cv<FeatureT>(nr_dims, query_features, query_features_batch, block_size, i * block_size);
            pcl2cv<FeatureT>(nr_dims, train_features, train_features_batch, block_size, j * block_size);
            matcher->match(query_features_batch, train_features_batch, matches);
            for (int k = 0; k < matches.size(); ++k) {
                if (j == 0 || correspondences[i * block_size + k].distances[0] > matches[k].distance) {
                    correspondences[i * block_size + k] = MultivaluedCorrespondence{
                            matches[k].queryIdx + i * block_size, {matches[k].trainIdx + j * block_size},
                            {matches[k].distance}};
                }
            }
            matches.clear();
        }
        PCL_DEBUG("[matchBF] %d / % d blocks processed.\n", i + 1, n_query_blocks);
    }
    for (int i = 0; i < query_features->size(); i++) {
        if (!point_representation->isValid(query_features->points[i])) {
            correspondences[i].query_idx = -1;
        }
    }
}

#endif
