#ifndef REGISTRATION_MATCHING_H
#define REGISTRATION_MATCHING_H

#include <pcl/point_cloud.h>

#include <opencv2/features2d.hpp>

template<typename FeatureT>
class FeatureMatcher {
public:
    using Ptr = std::shared_ptr<FeatureMatcher<FeatureT>>;
    using ConstPtr = std::shared_ptr<const FeatureMatcher<FeatureT>>;

    virtual std::vector<MultivaluedCorrespondence> match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                                                         const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                                                         const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                                                         int k_corrs, int threads,
                                                         bool use_bfmatcher, int bf_block_size) = 0;

    virtual std::string getClassName() = 0;
};

template<typename FeatureT>
class LeftToRightMatcher : public FeatureMatcher<FeatureT> {
public:
    std::vector<MultivaluedCorrespondence> match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                                                 const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                                                 const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                                                 int k_corrs, int threads,
                                                 bool use_bfmatcher, int bf_block_size) override {
        int nr_dims = point_representation->getNumberOfDimensions();
        std::vector<MultivaluedCorrespondence> correspondences_ij;
        if (use_bfmatcher) {
            matchBF<FeatureT>(src, tgt, correspondences_ij, point_representation, k_corrs, nr_dims, bf_block_size);
        } else {
            matchFLANN<FeatureT>(src, tgt, correspondences_ij, point_representation, k_corrs, threads);
        }

        {
            float dists_sum = 0.f;
            int n_dists = 0;
            for (int i = 0; i < src->size(); i++) {
                if (correspondences_ij[i].query_idx >= 0) {
                    dists_sum += correspondences_ij[i].distances[0];
                    n_dists++;
                }
            }
            if (n_dists == 0) {
                PCL_ERROR("[%s::match] no distances were calculated.\n", getClassName().c_str());
            } else {
                PCL_DEBUG("[%s::match] average distance to nearest neighbour: %0.7f.\n",
                          getClassName().c_str(),
                          dists_sum / (float) n_dists);
            }
        }

        std::vector<MultivaluedCorrespondence> correspondences_ji;
        if (use_bfmatcher) {
            matchBF<FeatureT>(tgt, src, correspondences_ji, point_representation, k_corrs, nr_dims, bf_block_size);
        } else {
            matchFLANN<FeatureT>(tgt, src, correspondences_ji, point_representation, k_corrs, threads);
        }

        std::vector<MultivaluedCorrespondence> correspondences_mutual;
        for (int i = 0; i < src->size(); ++i) {
            MultivaluedCorrespondence corr_i{i};
            for (const int &j: correspondences_ij[i].match_indices) {
                auto &corr_j = correspondences_ji[j];
                for (int k = 0; k < corr_j.match_indices.size(); ++k) {
                    if (corr_j.match_indices[k] == i) {
                        corr_i.match_indices.push_back(j);
                        corr_i.distances.push_back(corr_j.distances[k]);
                        break;
                    }
                }
            }
            if (!corr_i.match_indices.empty()) {
                correspondences_mutual.emplace_back(corr_i);
            }
        }
        correspondences_ij = correspondences_mutual;
        PCL_DEBUG("[%s::match] %i correspondences remain after mutual filter.\n",
                  getClassName().c_str(),
                  correspondences_mutual.size());

        std::vector<MultivaluedCorrespondence> correspondences;
        for (auto corr: correspondences_ij) {
            if (corr.query_idx >= 0) {
                correspondences.emplace_back(corr);
            }
        }
        return correspondences;
    }

    inline std::string getClassName() override {
        return "LeftToRightFeatureMatcher";
    }
};

template<typename FeatureT>
typename FeatureMatcher<FeatureT>::Ptr getFeatureMatcher(const std::string &matching_id) {
    return std::make_shared<LeftToRightMatcher<FeatureT>>();
}

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
    correspondences.resize(n, MultivaluedCorrespondence{});
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
    auto matcher = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> matches;

    correspondences.resize(query_features->size(), MultivaluedCorrespondence{});
    int n_query_blocks = (query_features->size() + block_size - 1) / block_size;
    for (int i = 0; i < n_query_blocks; ++i) {
        for (int j = 0; j < (train_features->size() + block_size - 1) / block_size; ++j) {
            cv::UMat query_features_batch, train_features_batch;
            pcl2cv<FeatureT>(nr_dims, query_features, query_features_batch, block_size, i * block_size);
            pcl2cv<FeatureT>(nr_dims, train_features, train_features_batch, block_size, j * block_size);
            matcher->knnMatch(query_features_batch, train_features_batch, matches, k_matches);
            for (int l = 0; l < matches.size(); ++l) {
                if (matches[l].empty() || matches[l][0].queryIdx == -1) {
                    continue;
                }
                int query_idx_local = matches[l][0].queryIdx;
                int query_idx = i * block_size + query_idx_local;
                for (int m = 0; m < matches[l].size(); ++m) {
                    if (matches[l][m].queryIdx != query_idx_local) {
                        PCL_ERROR("[matchBF] unexpected query index in brute-force matches!");
                        exit(1);
                    }
                    updateMultivaluedCorrespondence(correspondences[query_idx], query_idx, k_matches,
                                                    j * block_size + matches[l][m].trainIdx, matches[l][m].distance);
                }
            }
            matches.clear();
        }
        PCL_DEBUG("[matchBF] %d / % d blocks processed.\n", i + 1, n_query_blocks);
    }
    for (int i = 0; i < query_features->size(); i++) {
        if (!point_representation->isValid(query_features->points[i])) {
            correspondences[i] = MultivaluedCorrespondence{};
        }
    }
}

#endif
