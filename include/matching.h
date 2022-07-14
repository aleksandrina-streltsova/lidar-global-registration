#ifndef REGISTRATION_MATCHING_H
#define REGISTRATION_MATCHING_H

#include <unordered_set>
#include <utility>
#include <opencv2/features2d.hpp>

#include <pcl/common/norms.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/io.h>
#include <pcl/common/time.h>

#include "common.h"

#define MATCHING_RATIO_THRESHOLD 0.95f
#define MATCHING_CLUSTER_THRESHOLD 0.8f
#define MATCHING_CLUSTER_RADIUS_COEF 7.f

class FeatureBasedMatcher {
public:
    using Ptr = std::shared_ptr<FeatureBasedMatcher>;
    using ConstPtr = std::shared_ptr<const FeatureBasedMatcher>;

    virtual pcl::CorrespondencesPtr match() = 0;

    inline float getAverageDistance() const {
        return average_distance_;
    }

    void printDebugInfo(const std::vector<MultivaluedCorrespondence> &mv_correspondences);

    virtual std::string getClassName() = 0;

protected:
    float average_distance_ = std::numeric_limits<float>::max();
};

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
class FeatureBasedMatcherImpl : public FeatureBasedMatcher {
public:
    FeatureBasedMatcherImpl(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                            pcl::IndicesConstPtr indices_src, pcl::IndicesConstPtr indices_tgt,
                            AlignmentParameters parameters) : src_(std::move(src)), tgt_(std::move(tgt)),
                                                              kps_indices_src_(std::move(indices_src)),
                                                              kps_indices_tgt_(std::move(indices_tgt)),
                                                              parameters_(std::move(parameters)) {}

protected:
    using FeatureCloud = pcl::PointCloud<FeatureT>;
    using KdTree = pcl::search::KdTree<PointN>;

    // estimate features and initialize k-d trees
    void initialize();

    virtual pcl::CorrespondencesPtr match_impl() = 0;

    // convert local indices into global indices
    void finalize(const pcl::CorrespondencesPtr &correspondences);

    pcl::CorrespondencesPtr match() override {
        initialize();
        {
            pcl::ScopeTime t("Matching");
            auto correspondences = match_impl();
            finalize(correspondences);
            return correspondences;
        }
    }

    PointNCloud::ConstPtr src_, tgt_;
    typename pcl::IndicesConstPtr kps_indices_src_, kps_indices_tgt_;
    PointNCloud::Ptr kps_src_{new PointNCloud}, kps_tgt_{new PointNCloud};
    typename FeatureCloud::Ptr kps_features_src_{new FeatureCloud}, kps_features_tgt_{new FeatureCloud};
    typename KdTree::Ptr kps_tree_src_{new KdTree}, kps_tree_tgt_{new KdTree};
    AlignmentParameters parameters_;
};

template<typename FeatureT>
void FeatureBasedMatcherImpl<FeatureT>::initialize() {
    float radius_search = parameters_.feature_radius_coef * parameters_.voxel_size;
    {
        pcl::ScopeTime t("Feature estimation");
        AlignmentParameters parameters(parameters_);
        estimateFeatures<FeatureT>(src_, kps_indices_src_, kps_features_src_, parameters);
        // need to set gt to id in case lrf == 'gt' in estimateReferenceFrames
        parameters.ground_truth = std::make_shared<Eigen::Matrix4f>(Eigen::Matrix4f::Identity());
        estimateFeatures<FeatureT>(tgt_, kps_indices_tgt_, kps_features_tgt_, parameters);
    }
    pcl::copyPointCloud(*src_, *kps_indices_src_, *kps_src_);
    pcl::copyPointCloud(*tgt_, *kps_indices_tgt_, *kps_tgt_);
    kps_tree_src_->setInputCloud(kps_src_);
    kps_tree_tgt_->setInputCloud(kps_tgt_);
}

template<typename FeatureT>
void FeatureBasedMatcherImpl<FeatureT>::finalize(const pcl::CorrespondencesPtr &correspondences) {
    for (auto &corr: *correspondences) {
        corr.index_query = this->kps_indices_src_->operator[](corr.index_query);
        corr.index_match = this->kps_indices_tgt_->operator[](corr.index_match);
    }
}

template<typename FeatureT>
void pcl2cv(int nr_dims, const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
            cv::OutputArray &dst, int size, int offset);

template<typename FeatureT>
std::vector<MultivaluedCorrespondence> matchFLANN(const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                                                  const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                                                  const AlignmentParameters &parameters);

template<typename FeatureT>
std::vector<MultivaluedCorrespondence> matchBF(const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                                               const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                                               const AlignmentParameters &parameters);

template<typename FeatureT>
std::vector<MultivaluedCorrespondence> matchLocal(const PointNCloud::ConstPtr &query_pcd,
                                                  const typename pcl::search::KdTree<PointN>::ConstPtr &train_tree,
                                                  const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                                                  const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                                                  const AlignmentParameters &parameters);

template<typename FeatureT>
class LeftToRightMatcher : public FeatureBasedMatcherImpl<FeatureT> {
public:
    LeftToRightMatcher() = delete;

    LeftToRightMatcher(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                       pcl::IndicesConstPtr indices_src, pcl::IndicesConstPtr indices_tgt,
                       AlignmentParameters parameters) : FeatureBasedMatcherImpl<FeatureT>(src, tgt, indices_src,
                                                                                           indices_tgt, parameters) {}

    pcl::CorrespondencesPtr match_impl() override {
        this->initialize();
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ij = matchLocal<FeatureT>(this->kps_src_, this->kps_tree_tgt_,
                                                         this->kps_features_src_, this->kps_features_tgt_,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ij = matchBF<FeatureT>(this->kps_features_src_, this->kps_features_tgt_,
                                                      this->parameters_);
        } else {
            mv_correspondences_ij = matchFLANN<FeatureT>(this->kps_features_src_, this->kps_features_tgt_,
                                                         this->parameters_);
        }
        this->printDebugInfo(mv_correspondences_ij);
        std::vector<MultivaluedCorrespondence> mv_correspondences_ji;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ji = matchLocal<FeatureT>(this->kps_tgt_, this->kps_tree_src_,
                                                         this->kps_features_tgt_, this->kps_features_src_,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ji = matchBF<FeatureT>(this->kps_features_tgt_, this->kps_features_src_,
                                                      this->parameters_);
        } else {
            mv_correspondences_ji = matchFLANN<FeatureT>(this->kps_features_tgt_, this->kps_features_src_,
                                                         this->parameters_);
        }
        pcl::CorrespondencesPtr correspondences_mutual(new pcl::Correspondences);
        for (int i = 0; i < this->kps_src_->size(); ++i) {
            for (const int &j: mv_correspondences_ij[i].match_indices) {
                auto &corr_j = mv_correspondences_ji[j];
                for (int k = 0; k < corr_j.match_indices.size(); ++k) {
                    if (corr_j.match_indices[k] == i) {
                        correspondences_mutual->push_back({i, j, corr_j.distances[k]});
                        break;
                    }
                }
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after mutual filtering.\n",
                  getClassName().c_str(),
                  correspondences_mutual->size());
        return correspondences_mutual;
    }

    inline std::string getClassName() override {
        return "LeftToRightMatcher";
    }
};

template<typename FeatureT>
class RatioMatcher : public FeatureBasedMatcherImpl<FeatureT> {
public:
    RatioMatcher() = delete;

    RatioMatcher(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                 pcl::IndicesConstPtr indices_src, pcl::IndicesConstPtr indices_tgt,
                 AlignmentParameters parameters) : FeatureBasedMatcherImpl<FeatureT>(src, tgt, indices_src,
                                                                                     indices_tgt, parameters) {}

    pcl::CorrespondencesPtr match_impl() override {
        if (this->parameters_.randomness != 1) {
            PCL_WARN("[%s::match] k_corrs different from 1 cannot be used with ratio filtering, using k_corrs = 1.\n",
                     getClassName().c_str());
        }
        this->initialize();
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ij = matchLocal<FeatureT>(this->kps_src_, this->kps_tree_tgt_,
                                                         this->kps_features_src_, this->kps_features_tgt_,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ij = matchBF<FeatureT>(this->kps_features_src_, this->kps_features_tgt_,
                                                      this->parameters_);
        } else {
            mv_correspondences_ij = matchFLANN<FeatureT>(this->kps_features_src_, this->kps_features_tgt_,
                                                         this->parameters_);
        }
        this->printDebugInfo(mv_correspondences_ij);
        float dist1, dist2, ratio;
        pcl::CorrespondencesPtr correspondences_ratio(new pcl::Correspondences);
        for (auto &mv_corr: mv_correspondences_ij) {
            if (mv_corr.match_indices.size() != 2) {
                continue;
            }
            dist1 = std::min(mv_corr.distances[0], mv_corr.distances[1]);
            dist2 = std::max(mv_corr.distances[0], mv_corr.distances[1]);
            ratio = (dist2 == 0.f) ? 1.f : (dist1 / dist2);
            if (ratio < MATCHING_RATIO_THRESHOLD) {
                int i = (dist1 < dist2) ? 0 : 1;
                correspondences_ratio->push_back({mv_corr.query_idx, mv_corr.match_indices[i], ratio});
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after ratio filtering.\n",
                  getClassName().c_str(),
                  correspondences_ratio->size());
        return correspondences_ratio;
    }

    inline std::string getClassName() override {
        return "RatioMatcher";
    }
};

template<typename FeatureT>
class ClusterMatcher : public FeatureBasedMatcherImpl<FeatureT> {
public:
    using KdTreeConstPtr = typename pcl::search::KdTree<PointN>::ConstPtr;

    ClusterMatcher() = delete;

    ClusterMatcher(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                   pcl::IndicesConstPtr indices_src, pcl::IndicesConstPtr indices_tgt,
                   AlignmentParameters parameters) : FeatureBasedMatcherImpl<FeatureT>(src, tgt, indices_src,
                                                                                       indices_tgt, parameters) {}

    pcl::CorrespondencesPtr match_impl() override {
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ij = matchLocal<FeatureT>(this->kps_src_, this->kps_tree_tgt_,
                                                         this->kps_features_src_, this->kps_features_tgt_,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ij = matchBF<FeatureT>(this->kps_features_src_, this->kps_features_tgt_,
                                                      this->parameters_);
        } else {
            mv_correspondences_ij = matchFLANN<FeatureT>(this->kps_features_src_, this->kps_features_tgt_,
                                                         this->parameters_);
        }
        this->printDebugInfo(mv_correspondences_ij);
        std::vector<MultivaluedCorrespondence> mv_correspondences_ji;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ji = matchLocal<FeatureT>(this->kps_tgt_, this->kps_tree_src_,
                                                         this->kps_features_tgt_, this->kps_features_src_,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ji = matchBF<FeatureT>(this->kps_features_tgt_, this->kps_features_src_,
                                                      this->parameters_);
        } else {
            mv_correspondences_ji = matchFLANN<FeatureT>(this->kps_features_tgt_, this->kps_features_src_,
                                                         this->parameters_);
        }
        float matching_cluster_radius = MATCHING_CLUSTER_RADIUS_COEF * this->parameters_.voxel_size;
        pcl::CorrespondencesPtr correspondences_cluster(new pcl::Correspondences);
        for (int i = 0; i < this->kps_src_->size(); ++i) {
            for (int j: mv_correspondences_ij[i].match_indices) {
                float distance_i = calculateCorrespondenceDistance(i, j, matching_cluster_radius, mv_correspondences_ij,
                                                                   this->kps_tree_src_, this->kps_tree_tgt_);
                float distance_j = calculateCorrespondenceDistance(j, i, matching_cluster_radius, mv_correspondences_ji,
                                                                   this->kps_tree_tgt_, this->kps_tree_src_);
                if (distance_i < MATCHING_CLUSTER_THRESHOLD && distance_j < MATCHING_CLUSTER_THRESHOLD) {
                    correspondences_cluster->push_back({i, j, std::max(distance_i, distance_j)});
                }
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after cluster filtering.\n",
                  getClassName().c_str(),
                  correspondences_cluster->size());
        return correspondences_cluster;
    }

    inline std::string getClassName() override {
        return "ClusterMatcher";
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
};

template<typename FeatureT>
void pcl2cv(int nr_dims, const typename pcl::PointCloud<FeatureT>::ConstPtr &src,
            cv::OutputArray &dst, int size, int offset) {
    if (src->empty()) return;
    int rows = size == 0 ? (src->size() - offset) : std::min((int) (src->size() - offset), size);
    cv::Mat _src(rows, nr_dims, CV_32FC1, (void *) &src->points[offset], sizeof(src->points[0]));
    _src.copyTo(dst);
}

template<typename FeatureT>
std::vector<MultivaluedCorrespondence> matchFLANN(const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                                                  const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                                                  const AlignmentParameters &parameters) {
    auto point_representation = pcl::DefaultPointRepresentation<FeatureT>();
    pcl::KdTreeFLANN<FeatureT> feature_tree(new pcl::KdTreeFLANN<FeatureT>);
    feature_tree.setInputCloud(train_features);
    auto n = query_features->size();
    std::vector<MultivaluedCorrespondence> mv_correspondences(n, MultivaluedCorrespondence{});
#pragma omp parallel for \
    default(none) \
    shared(mv_correspondences, query_features, feature_tree) \
    firstprivate(n, point_representation, parameters)
    for (int i = 0; i < n; i++) {
        if (point_representation.isValid(query_features->points[i])) {
            mv_correspondences[i].query_idx = i;
            pcl::Indices &match_indices = mv_correspondences[i].match_indices;
            std::vector<float> &match_distances = mv_correspondences[i].distances;
            match_indices.resize(parameters.randomness);
            match_distances.resize(parameters.randomness);
            feature_tree.nearestKSearch(*query_features,
                                        i,
                                        parameters.randomness,
                                        match_indices,
                                        match_distances);
            for (int j = 0; j < parameters.randomness; ++j) {
                match_distances[j] = std::sqrt(match_distances[j]);
            }
        }
    }
    return mv_correspondences;
}

template<typename FeatureT>
std::vector<MultivaluedCorrespondence> matchBF(const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                                               const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                                               const AlignmentParameters &parameters) {
    auto point_representation = pcl::DefaultPointRepresentation<FeatureT>();
    int nr_dims = point_representation.getNumberOfDimensions();
    auto matcher = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> matches;

    std::vector<MultivaluedCorrespondence> mv_correspondences(query_features->size(), MultivaluedCorrespondence{});
    int block_size = parameters.bf_block_size;
    int n_query_blocks = (query_features->size() + block_size - 1) / block_size;
    int n_train_blocks = (train_features->size() + block_size - 1) / block_size;
    for (int i = 0; i < n_query_blocks; ++i) {
        for (int j = 0; j < n_train_blocks; ++j) {
            cv::UMat query_features_batch, train_features_batch;
            pcl2cv<FeatureT>(nr_dims, query_features, query_features_batch, block_size, i * block_size);
            pcl2cv<FeatureT>(nr_dims, train_features, train_features_batch, block_size, j * block_size);
            matcher->knnMatch(query_features_batch, train_features_batch, matches, parameters.randomness);
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
                    updateMultivaluedCorrespondence(mv_correspondences[query_idx], query_idx, parameters.randomness,
                                                    j * block_size + matches[l][m].trainIdx, matches[l][m].distance);
                }
            }
            matches.clear();
            PCL_DEBUG("\t[matchBF] %d / % d blocks processed.\n", j, n_train_blocks);
        }
        PCL_DEBUG("[matchBF] %d / % d blocks processed.\n", i + 1, n_query_blocks);
    }
    for (int i = 0; i < query_features->size(); i++) {
        if (!point_representation.isValid(query_features->points[i])) {
            mv_correspondences[i] = MultivaluedCorrespondence{};
        }
    }
    return mv_correspondences;
}


template<typename FeatureT>
std::vector<MultivaluedCorrespondence> matchLocal(const PointNCloud::ConstPtr &query_pcd,
                                                  const typename pcl::search::KdTree<PointN>::ConstPtr &train_tree,
                                                  const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                                                  const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                                                  const AlignmentParameters &parameters) {
    PointNCloud transformed_query_pcd;
    pcl::transformPointCloudWithNormals(*query_pcd, transformed_query_pcd, parameters.guess.value());
    auto n = transformed_query_pcd.size();
    std::vector<MultivaluedCorrespondence> mv_correspondences(query_features->size(), MultivaluedCorrespondence{});

#pragma omp parallel default(none) \
    shared(transformed_query_pcd, train_tree, query_features, train_features, mv_correspondences) \
    firstprivate(n, parameters)
    {
        std::vector<float> distances;
        pcl::Indices indices;
        auto point_representation = pcl::DefaultPointRepresentation<FeatureT>();
        int nr_dims = point_representation.getNumberOfDimensions();

#pragma omp for
        for (int query_idx = 0; query_idx < n; ++query_idx) {
            if (point_representation.isValid(query_features->points[query_idx])) {
                KNNResult<float> knnResult(parameters.randomness);
                train_tree->radiusSearch(transformed_query_pcd.points[query_idx],
                                         parameters.match_search_radius, indices, distances);
                for (int train_idx: indices) {
                    if (point_representation.isValid(train_features->points[train_idx])) {
                        float dist = pcl::L2_Norm((float *) &query_features->points[query_idx],
                                                  (float *) &train_features->points[train_idx], nr_dims);
                        knnResult.addPoint(dist, train_idx);
                    }
                }
                if (knnResult.size() > 0) {
                    mv_correspondences[query_idx].query_idx = query_idx;
                    mv_correspondences[query_idx].match_indices = knnResult.getIndices();
                    mv_correspondences[query_idx].distances = knnResult.getDistances();
                }
            }
        }
    }
    return mv_correspondences;
}

#endif
