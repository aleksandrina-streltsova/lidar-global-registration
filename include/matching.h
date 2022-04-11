#ifndef REGISTRATION_MATCHING_H
#define REGISTRATION_MATCHING_H

#include <unordered_set>

#include <pcl/point_cloud.h>
#include <opencv2/features2d.hpp>

#define MATCHING_RATIO_THRESHOLD 0.95f
#define MATCHING_CLUSTER_THRESHOLD 0.6f
#define MATCHING_CLUSTER_K_NEIGHBORS 100

template<typename FeatureT>
class FeatureMatcher {
public:
    using Ptr = std::shared_ptr<FeatureMatcher<FeatureT>>;
    using ConstPtr = std::shared_ptr<const FeatureMatcher<FeatureT>>;

    virtual pcl::Correspondences match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                                       const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                                       const PointCloudTN::ConstPtr &pcd_src, const PointCloudTN::ConstPtr &pcd_tgt,
                                       const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                                       int k_corrs, int threads, bool use_bfmatcher, int bf_block_size) = 0;

    virtual std::string getClassName() = 0;

protected:
    void printDebugInfo(const std::vector<MultivaluedCorrespondence> &mv_correspondences) {
        float dists_sum = 0.f;
        int n_dists = 0;
        for (int i = 0; i < mv_correspondences.size(); i++) {
            if (mv_correspondences[i].query_idx >= 0) {
                dists_sum += mv_correspondences[i].distances[0];
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
};

template<typename FeatureT>
class LeftToRightFeatureMatcher : public FeatureMatcher<FeatureT> {
public:
    LeftToRightFeatureMatcher() = default;

    pcl::Correspondences match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                               const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                               const PointCloudTN::ConstPtr &pcd_src, const PointCloudTN::ConstPtr &pcd_tgt,
                               const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                               int k_corrs, int threads, bool use_bfmatcher, int bf_block_size) override {
        int nr_dims = point_representation->getNumberOfDimensions();
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (use_bfmatcher) {
            matchBF<FeatureT>(src, tgt, mv_correspondences_ij, point_representation, k_corrs, nr_dims, bf_block_size);
        } else {
            matchFLANN<FeatureT>(src, tgt, mv_correspondences_ij, point_representation, k_corrs, threads);
        }

        this->printDebugInfo(mv_correspondences_ij);

        std::vector<MultivaluedCorrespondence> mv_correspondences_ji;
        if (use_bfmatcher) {
            matchBF<FeatureT>(tgt, src, mv_correspondences_ji, point_representation, k_corrs, nr_dims, bf_block_size);
        } else {
            matchFLANN<FeatureT>(tgt, src, mv_correspondences_ji, point_representation, k_corrs, threads);
        }

        pcl::Correspondences correspondences_mutual;
        for (int i = 0; i < src->size(); ++i) {
            for (const int &j: mv_correspondences_ij[i].match_indices) {
                auto &corr_j = mv_correspondences_ji[j];
                for (int k = 0; k < corr_j.match_indices.size(); ++k) {
                    if (corr_j.match_indices[k] == i) {
                        correspondences_mutual.push_back({i, j, corr_j.distances[k]});
                        break;
                    }
                }
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after mutual filtering.\n",
                  getClassName().c_str(),
                  correspondences_mutual.size());
        return correspondences_mutual;
    }

    inline std::string getClassName() override {
        return "LeftToRightFeatureMatcher";
    }
};

template<typename FeatureT>
class RatioFeatureMatcher : public FeatureMatcher<FeatureT> {
public:
    RatioFeatureMatcher() = default;

    pcl::Correspondences match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                               const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                               const PointCloudTN::ConstPtr &pcd_src, const PointCloudTN::ConstPtr &pcd_tgt,
                               const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                               int k_corrs, int threads, bool use_bfmatcher, int bf_block_size) override {
        if (k_corrs != 1) {
            PCL_WARN("[%s::match] k_corrs different from 1 cannot be used with ratio filtering, using k_corrs = 1.\n",
                     getClassName().c_str());
        }
        int nr_dims = point_representation->getNumberOfDimensions();
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (use_bfmatcher) {
            matchBF<FeatureT>(src, tgt, mv_correspondences_ij, point_representation, 2, nr_dims, bf_block_size);
        } else {
            matchFLANN<FeatureT>(src, tgt, mv_correspondences_ij, point_representation, 2, threads);
        }

        this->printDebugInfo(mv_correspondences_ij);

        float dist1, dist2, ratio;
        pcl::Correspondences correspondences_ratio;
        for (auto &mv_corr: mv_correspondences_ij) {
            if (mv_corr.match_indices.size() != 2) {
                continue;
            }
            dist1 = std::min(mv_corr.distances[0], mv_corr.distances[1]);
            dist2 = std::max(mv_corr.distances[0], mv_corr.distances[1]);
            ratio = (dist2 == 0.f) ? 1.f : (dist1 / dist2);
            if (ratio < MATCHING_RATIO_THRESHOLD) {
                int i = (dist1 < dist2) ? 0 : 1;
                correspondences_ratio.push_back({mv_corr.query_idx, mv_corr.match_indices[i], ratio});
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after ratio filtering.\n",
                  getClassName().c_str(),
                  correspondences_ratio.size());
        return correspondences_ratio;
    }

    inline std::string getClassName() override {
        return "RatioFeatureMatcher";
    }
};

template<typename FeatureT>
class ClusterFeatureMatcher : public FeatureMatcher<FeatureT> {
public:
    ClusterFeatureMatcher() = default;
    typedef std::vector<std::unordered_set<int>> Neighbors;
    pcl::Correspondences match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                               const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                               const PointCloudTN::ConstPtr &pcd_src, const PointCloudTN::ConstPtr &pcd_tgt,
                               const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                               int k_corrs, int threads, bool use_bfmatcher, int bf_block_size) override {
        int nr_dims = point_representation->getNumberOfDimensions();
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (use_bfmatcher) {
            matchBF<FeatureT>(src, tgt, mv_correspondences_ij, point_representation, k_corrs, nr_dims, bf_block_size);
        } else {
            matchFLANN<FeatureT>(src, tgt, mv_correspondences_ij, point_representation, k_corrs, threads);
        }

        this->printDebugInfo(mv_correspondences_ij);

        pcl::KdTreeFLANN<PointTN> pcd_tree_src, pcd_tree_tgt;
        pcd_tree_src.setInputCloud(pcd_src);
        pcd_tree_tgt.setInputCloud(pcd_tgt);
        Neighbors src_neighbors(src->size()), tgt_neighbors(tgt->size());

        pcl::Indices match_indices;
        std::vector<float> distances;

        for (int i = 0; i < src->size(); ++i) {
            pcd_tree_src.nearestKSearch(i, MATCHING_CLUSTER_K_NEIGHBORS, match_indices, distances);
            std::copy(match_indices.begin(), match_indices.end(),
                      std::inserter(src_neighbors[i], src_neighbors[i].begin()));
        }
        for (int i = 0; i < tgt->size(); ++i) {
            pcd_tree_tgt.nearestKSearch(i, MATCHING_CLUSTER_K_NEIGHBORS, match_indices, distances);
            std::copy(match_indices.begin(), match_indices.end(),
                      std::inserter(tgt_neighbors[i], tgt_neighbors[i].begin()));
        }

        pcl::Correspondences correspondences_cluster;
        for (int i = 0; i < src->size(); ++i) {
            for (int j: mv_correspondences_ij[i].match_indices) {
                float weight = calculateCorrespondenceWeight(i, j, mv_correspondences_ij, src_neighbors, tgt_neighbors);
                if (weight < MATCHING_CLUSTER_THRESHOLD) {
                    correspondences_cluster.push_back({i, j, weight});
                }
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after cluster filtering.\n",
                  getClassName().c_str(),
                  correspondences_cluster.size());
        return correspondences_cluster;
    }

    inline std::string getClassName() override {
        return "ClusterFeatureMatcher";
    }

protected:
    float calculateCorrespondenceWeight(int i, int j,
                                        const std::vector<MultivaluedCorrespondence> &mv_correspondences_ij,
                                        const Neighbors &src_neighbors, const Neighbors &tgt_neighbors) {
        int count_consistent_pairs = 0, count_pairs = 0;
        for (int i_neighbor: src_neighbors[i]) {
            for (int i_neighbor_match: mv_correspondences_ij[i_neighbor].match_indices) {
                if (tgt_neighbors[j].contains(i_neighbor_match)) {
                    count_consistent_pairs++;
                }
                count_pairs++;
            }
        }
        if (count_pairs == 0) {
            return 0;
        }
        return 1.f - (float) count_consistent_pairs / (float) count_pairs;
    }
};

template<typename FeatureT>
typename FeatureMatcher<FeatureT>::Ptr getFeatureMatcher(const std::string &matching_id) {
    if (matching_id == MATCHING_RATIO) {
        return std::make_shared<RatioFeatureMatcher<FeatureT>>();
    } else if (matching_id == MATCHING_CLUSTER) {
        return std::make_shared<ClusterFeatureMatcher<FeatureT>>();
    } else if (matching_id != MATCHING_LEFT_TO_RIGHT) {
        PCL_WARN("[getFeatureMatcher] feature matcher %s isn't supported, left-to-right matcher will be used.",
                 matching_id.c_str());
    }
    return std::make_shared<LeftToRightFeatureMatcher<FeatureT>>();
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
                std::vector<MultivaluedCorrespondence> &mv_correspondences,
                const typename pcl::PointRepresentation<FeatureT>::Ptr point_representation,
                int k_matches, int threads) {
    pcl::KdTreeFLANN<FeatureT> feature_tree(new pcl::KdTreeFLANN<FeatureT>);
    feature_tree.setInputCloud(train_features);
    int n = query_features->size();
    mv_correspondences.resize(n, MultivaluedCorrespondence{});
#pragma omp parallel for num_threads(threads) default(none) shared(mv_correspondences, query_features, point_representation, feature_tree) firstprivate(n, k_matches)
    for (int i = 0; i < n; i++) {
        if (point_representation->isValid(query_features->points[i])) {
            mv_correspondences[i].query_idx = i;
            pcl::Indices &match_indices = mv_correspondences[i].match_indices;
            std::vector<float> &match_distances = mv_correspondences[i].distances;
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
             std::vector<MultivaluedCorrespondence> &mv_correspondences,
             const typename pcl::PointRepresentation<FeatureT>::Ptr point_representation,
             int k_matches, int nr_dims, int block_size) {
    auto matcher = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> matches;

    mv_correspondences.resize(query_features->size(), MultivaluedCorrespondence{});
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
                    updateMultivaluedCorrespondence(mv_correspondences[query_idx], query_idx, k_matches,
                                                    j * block_size + matches[l][m].trainIdx, matches[l][m].distance);
                }
            }
            matches.clear();
        }
        PCL_DEBUG("[matchBF] %d / % d blocks processed.\n", i + 1, n_query_blocks);
    }
    for (int i = 0; i < query_features->size(); i++) {
        if (!point_representation->isValid(query_features->points[i])) {
            mv_correspondences[i] = MultivaluedCorrespondence{};
        }
    }
}

#endif
