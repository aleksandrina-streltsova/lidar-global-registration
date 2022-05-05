#ifndef REGISTRATION_MATCHING_H
#define REGISTRATION_MATCHING_H

#include <unordered_set>
#include <utility>
#include <opencv2/features2d.hpp>

#include <pcl/common/norms.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>

#include "common.h"

#define MATCHING_RATIO_THRESHOLD 0.95f
#define MATCHING_CLUSTER_THRESHOLD 0.8f
#define MATCHING_CLUSTER_RADIUS_COEF 7.f

template<typename FeatureT>
class FeatureMatcher {
public:
    using Ptr = std::shared_ptr<FeatureMatcher<FeatureT>>;
    using ConstPtr = std::shared_ptr<const FeatureMatcher<FeatureT>>;
    using KdTreeConstPtr = typename pcl::search::KdTree<PointN>::ConstPtr;

    virtual pcl::Correspondences match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                                       const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                                       const KdTreeConstPtr &pcd_tree_src, const KdTreeConstPtr &pcd_tree_tgt,
                                       const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                                       int threads) = 0;

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

    AlignmentParameters parameters_;
};

template<typename FeatureT>
class LeftToRightFeatureMatcher : public FeatureMatcher<FeatureT> {
public:
    using KdTreeConstPtr = typename pcl::search::KdTree<PointN>::ConstPtr;

    LeftToRightFeatureMatcher() = delete;

    LeftToRightFeatureMatcher(AlignmentParameters parameters) : parameters_(std::move(parameters)) {};

    pcl::Correspondences match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                               const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                               const KdTreeConstPtr &pcd_tree_src, const KdTreeConstPtr &pcd_tree_tgt,
                               const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                               int threads) override {
        int nr_dims = point_representation->getNumberOfDimensions();
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess != nullptr) {
            matchLocal<FeatureT>(pcd_tree_src->getInputCloud(), pcd_tree_tgt, src, tgt, mv_correspondences_ij,
                                 point_representation, *parameters_.guess,
                                 parameters_.match_search_radius, parameters_.randomness, threads);
        } else if (this->parameters_.use_bfmatcher) {
            matchBF<FeatureT>(src, tgt, mv_correspondences_ij, point_representation,
                              this->parameters_.randomness, nr_dims, this->parameters_.bf_block_size);
        } else {
            matchFLANN<FeatureT>(src, tgt, mv_correspondences_ij, point_representation,
                                 this->parameters_.randomness, threads);
        }

        this->printDebugInfo(mv_correspondences_ij);

        std::vector<MultivaluedCorrespondence> mv_correspondences_ji;
        if (this->parameters_.use_bfmatcher) {
            matchBF<FeatureT>(tgt, src, mv_correspondences_ji, point_representation,
                              this->parameters_.randomness, nr_dims, this->parameters_.bf_block_size);
        } else {
            matchFLANN<FeatureT>(tgt, src, mv_correspondences_ji, point_representation,
                                 this->parameters_.randomness, threads);
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

protected:
    AlignmentParameters parameters_;
};

template<typename FeatureT>
class RatioFeatureMatcher : public FeatureMatcher<FeatureT> {
public:
    using KdTreeConstPtr = typename pcl::search::KdTree<PointN>::ConstPtr;

    RatioFeatureMatcher() = delete;

    RatioFeatureMatcher(AlignmentParameters parameters) : parameters_(std::move(parameters)) {};

    pcl::Correspondences match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                               const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                               const KdTreeConstPtr &pcd_tree_src, const KdTreeConstPtr &pcd_tree_tgt,
                               const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                               int threads) override {
        if (this->parameters_.randomness != 1) {
            PCL_WARN("[%s::match] k_corrs different from 1 cannot be used with ratio filtering, using k_corrs = 1.\n",
                     getClassName().c_str());
        }
        int nr_dims = point_representation->getNumberOfDimensions();
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess != nullptr) {
            matchLocal<FeatureT>(pcd_tree_src->getInputCloud(), pcd_tree_tgt, src, tgt, mv_correspondences_ij,
                                 point_representation, *parameters_.guess,
                                 parameters_.match_search_radius, parameters_.randomness, threads);
        } else if (this->parameters_.use_bfmatcher) {
            matchBF<FeatureT>(src, tgt, mv_correspondences_ij, point_representation,
                              2, nr_dims, this->parameters_.bf_block_size);
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

protected:
    AlignmentParameters parameters_;
};

template<typename FeatureT>
class ClusterFeatureMatcher : public FeatureMatcher<FeatureT> {
public:
    using KdTreeConstPtr = typename pcl::search::KdTree<PointN>::ConstPtr;

    ClusterFeatureMatcher() = delete;

    ClusterFeatureMatcher(AlignmentParameters parameters) : parameters_(std::move(parameters)) {};

    pcl::Correspondences match(const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
                               const typename pcl::PointCloud<FeatureT>::ConstPtr &tgt,
                               const KdTreeConstPtr &pcd_tree_src, const KdTreeConstPtr &pcd_tree_tgt,
                               const typename pcl::PointRepresentation<FeatureT>::Ptr &point_representation,
                               int threads) override {
        int nr_dims = point_representation->getNumberOfDimensions();
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess != nullptr) {
            matchLocal<FeatureT>(pcd_tree_src->getInputCloud(), pcd_tree_tgt, src, tgt, mv_correspondences_ij,
                                 point_representation, *parameters_.guess,
                                 parameters_.match_search_radius, parameters_.randomness, threads);
        } else if (this->parameters_.use_bfmatcher) {
            matchBF<FeatureT>(src, tgt, mv_correspondences_ij, point_representation,
                              this->parameters_.randomness, nr_dims, this->parameters_.bf_block_size);
        } else {
            matchFLANN<FeatureT>(src, tgt, mv_correspondences_ij, point_representation,
                                 this->parameters_.randomness, threads);
        }

        this->printDebugInfo(mv_correspondences_ij);

        float matching_cluster_radius = MATCHING_CLUSTER_RADIUS_COEF * this->parameters_.voxel_size;
        pcl::Correspondences correspondences_cluster;
        for (int i = 0; i < src->size(); ++i) {
            for (int j: mv_correspondences_ij[i].match_indices) {
                float distance = calculateCorrespondenceDistance(i, j, matching_cluster_radius, mv_correspondences_ij,
                                                                 pcd_tree_src, pcd_tree_tgt);
                if (distance < MATCHING_CLUSTER_THRESHOLD) {
                    correspondences_cluster.push_back({i, j, distance});
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
    float calculateCorrespondenceDistance(int i, int j, float radius,
                                          const std::vector<MultivaluedCorrespondence> &mv_correspondences_ij,
                                          const KdTreeConstPtr &pcd_tree_src, const KdTreeConstPtr &pcd_tree_tgt) {
        std::unordered_set<int> i_neighbors, j_neighbors;
        pcl::Indices match_indices;
        std::vector<float> distances;

        pcd_tree_src->radiusSearch(i, radius, match_indices, distances);
        std::copy(match_indices.begin(), match_indices.end(), std::inserter(i_neighbors, i_neighbors.begin()));

        pcd_tree_tgt->radiusSearch(j, radius, match_indices, distances);
        std::copy(match_indices.begin(), match_indices.end(), std::inserter(j_neighbors, j_neighbors.begin()));

        int count_consistent_pairs = 0, count_pairs = 0;
        for (int i_neighbor: i_neighbors) {
            for (int i_neighbor_match: mv_correspondences_ij[i_neighbor].match_indices) {
                if (j_neighbors.contains(i_neighbor_match)) {
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

    AlignmentParameters parameters_;
};

template<typename FeatureT>
typename FeatureMatcher<FeatureT>::Ptr getFeatureMatcher(const AlignmentParameters &parameters) {
    if (parameters.matching_id == MATCHING_RATIO) {
        return std::make_shared<RatioFeatureMatcher<FeatureT>>(parameters);
    } else if (parameters.matching_id == MATCHING_CLUSTER) {
        return std::make_shared<ClusterFeatureMatcher<FeatureT>>(parameters);
    } else if (parameters.matching_id != MATCHING_LEFT_TO_RIGHT) {
        PCL_WARN("[getFeatureMatcher] feature matcher %s isn't supported, left-to-right matcher will be used.",
                 parameters.matching_id.c_str());
    }
    return std::make_shared<LeftToRightFeatureMatcher<FeatureT>>(parameters);
}

template<typename T>
class KNNResult {
private:
    int capacity_;
    int count_;
    std::vector<int> indices_;
    std::vector<T> dists_;
public:
    inline KNNResult(int capacity) : capacity_(capacity), count_(0) {
        indices_.reserve(capacity);
        dists_.reserve(capacity);
    }

    inline int size() const {
        return count_;
    }

    inline std::vector<int> getIndices() const {
        return indices_;
    }

    inline std::vector<T> getDistances() const {
        return dists_;
    }

    inline bool addPoint(T dist, int index) {
        if (count_ < capacity_) {
            indices_.resize(count_ + 1);
            dists_.resize(count_ + 1);
        }
        int i;
        for (i = count_; i > 0; --i) {
            if (dists_[i - 1] > dist) {
                if (i < capacity_) {
                    dists_[i] = dists_[i - 1];
                    indices_[i] = indices_[i - 1];
                }
            } else {
                break;
            }
        }
        if (i < capacity_) {
            dists_[i] = dist;
            indices_[i] = index;
        }
        if (count_ < capacity_) {
            count_++;
        }
        return true;
    }
};

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

template<typename FeatureT>
void matchLocal(const PointNCloud::ConstPtr &query_pcd,
                const typename pcl::search::KdTree<PointN>::ConstPtr &train_tree,
                const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                std::vector<MultivaluedCorrespondence> &mv_correspondences,
                const typename pcl::PointRepresentation<FeatureT>::Ptr point_representation,
                const Eigen::Matrix4f &guess, float match_search_radius, int k_matches, int threads) {
    PointNCloud transformed_query_pcd;
    pcl::transformPointCloudWithNormals(*query_pcd, transformed_query_pcd, guess);
    int n = transformed_query_pcd.size();
    mv_correspondences.resize(query_features->size(), MultivaluedCorrespondence{});

#pragma omp parallel num_threads(threads) default(none) \
    shared(transformed_query_pcd, train_tree, query_features, train_features, mv_correspondences, point_representation) \
    firstprivate(n, k_matches, match_search_radius)
    {
        std::vector<float> distances;
        pcl::Indices indices;

        int nr_dims = point_representation->getNumberOfDimensions();
        auto *query_feature = new float[nr_dims];
        auto *train_feature = new float[nr_dims];
#pragma omp for
        for (int query_idx = 0; query_idx < n; ++query_idx) {
            if (point_representation->isValid(query_features->points[query_idx])) {
                point_representation->copyToFloatArray(query_features->points[query_idx], query_feature);
                KNNResult<float> knnResult(k_matches);
                train_tree->radiusSearch(transformed_query_pcd.points[query_idx],
                                         match_search_radius, indices, distances);
                for (int train_idx: indices) {
                    if (point_representation->isValid(train_features->points[train_idx])) {
                        point_representation->copyToFloatArray(train_features->points[train_idx], train_feature);
                        knnResult.addPoint(pcl::L2_Norm(query_feature, train_feature, nr_dims), train_idx);
                    }
                }
                if (knnResult.size() > 0) {
                    mv_correspondences[query_idx].query_idx = query_idx;
                    mv_correspondences[query_idx].match_indices = knnResult.getIndices();
                    mv_correspondences[query_idx].distances = knnResult.getDistances();
                }
            }
        }
        delete[] query_feature;
        delete[] train_feature;
    }
}

#endif
