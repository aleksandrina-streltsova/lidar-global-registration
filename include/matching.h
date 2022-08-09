#ifndef REGISTRATION_MATCHING_H
#define REGISTRATION_MATCHING_H

#include <memory>
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
#include "downsample.h"
#include "analysis.h"

#define MATCHING_RATIO_THRESHOLD 1.1f
#define MATCHING_CLUSTER_THRESHOLD 0.8f
#define MATCHING_CLUSTER_K 150

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
    std::vector<PointNCloud::Ptr> initialize(const PointNCloud::ConstPtr &pcd,
                                             const pcl::IndicesConstPtr &kps_indices,
                                             std::vector<pcl::Indices> &kps_indices_multiscale,
                                             std::vector<PointNCloud::Ptr> &kps_multiscale,
                                             std::vector<typename KdTree::Ptr> &kps_tree_multiscale,
                                             std::vector<typename FeatureCloud::Ptr> &kps_features_multiscale,
                                             int &min_log2_radius, const AlignmentParameters &parameters);

    virtual CorrespondencesWithFlagsPtr match_impl(int idx_src, int idx_tgt) = 0;

    pcl::CorrespondencesPtr match() override {
        pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
        std::vector<MultivaluedCorrespondence> mv_correspondences(kps_indices_src_->size());
        std::vector<PointNCloud::Ptr> srcs_ds, tgts_ds;
        {
            // temporary parameters
            AlignmentParameters parameters(parameters_);
            srcs_ds = initialize(src_, kps_indices_src_, kps_indices_src_multiscale_,
                                 kps_src_multiscale_, kps_tree_src_multiscale_, kps_features_src_multiscale_,
                                 min_log2_radius_src_, parameters);
            // need to set gt to id in case lrf == 'gt' in estimateReferenceFrames
            parameters.ground_truth = std::optional<Eigen::Matrix4f>(Eigen::Matrix4f::Identity());
            tgts_ds = initialize(tgt_, kps_indices_tgt_, kps_indices_tgt_multiscale_,
                                 kps_tgt_multiscale_, kps_tree_tgt_multiscale_, kps_features_tgt_multiscale_,
                                 min_log2_radius_tgt_, parameters);
        }
        {
            pcl::ScopeTime t("Matching");
            int nr_scales_src = kps_features_src_multiscale_.size(), nr_scales_tgt = kps_features_tgt_multiscale_.size();
            int max_log2_radius_src = min_log2_radius_src_ - 1 + nr_scales_src;
            int max_log2_radius_tgt = min_log2_radius_tgt_ - 1 + nr_scales_tgt;
            std::vector<float> norm_diffs_pcd, norm_diffs_kps;
            for (int log2_radius = std::max(min_log2_radius_src_, min_log2_radius_tgt_);
                 log2_radius <= std::min(max_log2_radius_src, max_log2_radius_tgt); ++log2_radius) {
                int idx_src = log2_radius - min_log2_radius_src_, idx_tgt = log2_radius - min_log2_radius_tgt_;
                float search_radius = powf(2.0, (float) log2_radius);
                float voxel_size = sqrtf(M_PI * search_radius * search_radius / (float) parameters_.feature_nr_points);
                if (parameters_.ground_truth.has_value()) {
                    norm_diffs_pcd.push_back(calculateNormalDifference(srcs_ds[idx_src], tgts_ds[idx_tgt], voxel_size, parameters_.ground_truth.value()));
                    norm_diffs_kps.push_back(calculateNormalDifference(kps_src_multiscale_[idx_src], kps_tgt_multiscale_[idx_tgt], parameters_.distance_thr, parameters_.ground_truth.value()));
//                    saveTemperatureMaps(srcs_ds[idx_src], tgts_ds[idx_tgt], "t" + std::to_string(idx_src), parameters_,voxel_size,parameters_.ground_truth.value());
                }
                auto correspondences_with_flags = match_impl(idx_src, idx_tgt);
                rassert(correspondences_with_flags->size() == kps_features_src_multiscale_[idx_src]->size(),
                        9082472354896501);
                int keypoint_idx = 0;
                for (int i = 0; i < correspondences_with_flags->size(); ++i) {
                    pcl::Correspondence corr = correspondences_with_flags->operator[](i).first;
                    int index_query = kps_indices_src_multiscale_[idx_src][corr.index_query];
                    int index_match = corr.index_match > 0 ? kps_indices_tgt_multiscale_[idx_tgt][corr.index_match] : -1;
                    while (kps_indices_src_->operator[](keypoint_idx) != index_query) keypoint_idx++;
                    mv_correspondences[keypoint_idx].query_idx = index_query;
                    mv_correspondences[keypoint_idx].match_indices.push_back(index_match);
                    mv_correspondences[keypoint_idx].distances.push_back(corr.distance);
                    if (index_match > 0) {
                        float dist = (parameters_.ground_truth.value() * src_->points[index_query].getVector4fMap() -
                                      tgt_->points[index_match].getVector4fMap()).norm();
                        mv_correspondences[keypoint_idx].distances.push_back(dist);
                    } else {
                        mv_correspondences[keypoint_idx].distances.push_back(-1.f);
                    }
                }
            }
            saveVector(norm_diffs_pcd, constructPath(parameters_, "median_norm_diff", "csv", true, false, false));
            saveVector(norm_diffs_kps, constructPath(parameters_, "median_norm_diff_kps", "csv", true, false, false));
        }
        saveVector(mv_correspondences, constructPath(parameters_, "multiscale", "csv", true, false, false));
        for (const auto &mv_corr: mv_correspondences) {
            std::unordered_map<int, std::pair<int, float>> counter;
            for (int i = 0; i < mv_corr.match_indices.size(); ++i) {
                if (mv_corr.match_indices[i] < 0) continue;
                if (counter.find(mv_corr.match_indices[i]) != counter.end()) {
                    counter[mv_corr.match_indices[i]].first++;
                    counter[mv_corr.match_indices[i]].second += mv_corr.distances[2 * i];
                } else {
                    counter[mv_corr.match_indices[i]] = {1, mv_corr.distances[2 * i]};
                }
            }
            std::pair<int, float> best_match{0, 0.f};
            pcl::Correspondence corr;
            corr.index_query = mv_corr.query_idx;
            for (auto[index_match, count_dist]: counter) {
                if (count_dist.first > best_match.first ||
                    (count_dist.first == best_match.first && count_dist.second < best_match.second)) {
                    best_match = count_dist;
                    corr.index_match = index_match;
                    corr.distance = count_dist.second;
                }
            }
            if (best_match.first > 0) correspondences->push_back(corr);
        }
        return correspondences;
    }

    // downsampling and normal estimation time, feature estimation time
    double time_ds_ne_{0.0}, time_fe_{0.0};
    int min_log2_radius_src_{std::numeric_limits<int>::max()}, min_log2_radius_tgt_{std::numeric_limits<int>::max()};
    PointNCloud::ConstPtr src_, tgt_;
    typename pcl::IndicesConstPtr kps_indices_src_, kps_indices_tgt_;
    std::vector<pcl::Indices> kps_indices_src_multiscale_, kps_indices_tgt_multiscale_;
    std::vector<PointNCloud::Ptr> kps_src_multiscale_, kps_tgt_multiscale_;
    std::vector<typename FeatureCloud::Ptr> kps_features_src_multiscale_, kps_features_tgt_multiscale_;
    std::vector<typename KdTree::Ptr> kps_tree_src_multiscale_, kps_tree_tgt_multiscale_;
    pcl::DefaultPointRepresentation<FeatureT> point_representation_{};
    AlignmentParameters parameters_;
};

template<typename FeatureT>
std::vector<PointNCloud::Ptr>
FeatureBasedMatcherImpl<FeatureT>::initialize(const PointNCloud::ConstPtr &pcd,
                                              const pcl::IndicesConstPtr &kps_indices,
                                              std::vector<pcl::Indices> &kps_indices_multiscale,
                                              std::vector<PointNCloud::Ptr> &kps_multiscale,
                                              std::vector<typename KdTree::Ptr> &kps_tree_multiscale,
                                              std::vector<typename FeatureCloud::Ptr> &kps_features_multiscale,
                                              int &min_log2_radius, const AlignmentParameters &parameters) {
    KdTree tree(true);
    tree.setInputCloud(pcd);
    std::vector<int> log2_radii(kps_indices->size());
    int max_log2_radius = std::numeric_limits<int>::lowest();
    pcl::Indices nn_indices;
    std::vector<float> nn_sqr_dists;
    int k = 5, nr_extra_scales = 3;
    for (int i = 0; i < kps_indices->size(); ++i) {
        tree.nearestKSearch(*pcd, kps_indices->operator[](i), k, nn_indices, nn_sqr_dists);
        float density = sqrtf(nn_sqr_dists[k - 1]);
        float feature_radius = sqrtf((float) parameters.feature_nr_points * density * density / M_PI);
        log2_radii[i] = (int) std::floor(std::log2(feature_radius));
        min_log2_radius = std::min(log2_radii[i], min_log2_radius);
        max_log2_radius = std::max(log2_radii[i], max_log2_radius);
    }
    max_log2_radius += nr_extra_scales;
    int nr_scales = max_log2_radius - (min_log2_radius - 1);
    kps_multiscale.resize(nr_scales);
    kps_indices_multiscale.resize(nr_scales);
    kps_tree_multiscale.resize(nr_scales);
    kps_features_multiscale.resize(nr_scales);
    for (int i = 0; i < nr_scales; ++i) {
        kps_multiscale[i] = std::make_shared<PointNCloud>();
        kps_indices_multiscale[i].reserve(kps_indices->size());
        kps_tree_multiscale[i] = std::make_shared<KdTree>();
        kps_features_multiscale[i] = std::make_shared<FeatureCloud>();
    }
    // for each scale build indices
    for (int i = 0; i < kps_indices->size(); ++i) {
        for (int j = log2_radii[i]; j <= max_log2_radius; ++j) {
            kps_indices_multiscale[j - min_log2_radius].push_back(kps_indices->operator[](i));
        }
    }
    // for each scale estimate features
    std::vector<PointNCloud::Ptr> pcds_ds;
    for (int i = 0; i < nr_scales; i++) {
        PointNCloud::Ptr pcd_ds(new PointNCloud);
        float search_radius = powf(2.0, (float) (min_log2_radius + i));
        float voxel_size = sqrtf(M_PI * search_radius * search_radius / (float) parameters.feature_nr_points);
        {
            pcl::ScopeTime t("Downsampling and normal estimation");
            downsamplePointCloud(pcd, pcd_ds, voxel_size);
            estimateNormalsPoints(parameters_.normal_nr_points, pcd_ds, {nullptr}, parameters_.normals_available);
            pcds_ds.push_back(pcd_ds);
            time_ds_ne_ += t.getTimeSeconds();
        }
        {
            pcl::ScopeTime t("Feature estimation");
            pcl::copyPointCloud(*pcd, kps_indices_multiscale[i], *kps_multiscale[i]);
            if (parameters.reestimate_frames) {
                estimateNormalsPoints(parameters_.normal_nr_points, kps_multiscale[i], pcd_ds, true);
            }
            pcl::console::print_highlight("Estimating features  [search_radius = %.5f]...\n", search_radius);
            estimateFeatures<FeatureT>(kps_multiscale[i], pcd_ds, kps_features_multiscale[i],
                                       search_radius, parameters);
            int count_invalid = 0;
            for (int j = 0; j < kps_features_multiscale[i]->size(); ++j) {
                if (!point_representation_.isValid(kps_features_multiscale[i]->points[j]))
                    count_invalid++;
            }
            PCL_DEBUG("[%s::initialize] %i/%i invalid features\n",
                      getClassName().c_str(), count_invalid, kps_features_multiscale[i]->size());
            time_fe_ += t.getTimeSeconds();
        }
        kps_tree_multiscale[i]->setInputCloud(kps_multiscale[i]);
    }
    return pcds_ds;
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

    CorrespondencesWithFlagsPtr match_impl(int idx_src, int idx_tgt) override {
        const auto &kps_src = this->kps_src_multiscale_[idx_src];
        const auto &kps_tgt = this->kps_tgt_multiscale_[idx_tgt];
        const auto &kps_tree_src = this->kps_tree_src_multiscale_[idx_src];
        const auto &kps_tree_tgt = this->kps_tree_tgt_multiscale_[idx_tgt];
        const auto &kps_features_src = this->kps_features_src_multiscale_[idx_src];
        const auto &kps_features_tgt = this->kps_features_tgt_multiscale_[idx_tgt];
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ij = matchLocal<FeatureT>(kps_src, kps_tree_tgt, kps_features_src, kps_features_tgt,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ij = matchBF<FeatureT>(kps_features_src, kps_features_tgt, this->parameters_);
        } else {
            mv_correspondences_ij = matchFLANN<FeatureT>(kps_features_src, kps_features_tgt, this->parameters_);
        }
        this->printDebugInfo(mv_correspondences_ij);
        std::vector<MultivaluedCorrespondence> mv_correspondences_ji;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ji = matchLocal<FeatureT>(kps_tgt, kps_tree_src, kps_features_tgt, kps_features_src,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ji = matchBF<FeatureT>(kps_features_tgt, kps_features_src, this->parameters_);
        } else {
            mv_correspondences_ji = matchFLANN<FeatureT>(kps_features_tgt, kps_features_src, this->parameters_);
        }
        int n = kps_features_src->size(), count_successful = 0;
        CorrespondencesWithFlagsPtr correspondences_mutual(new CorrespondencesWithFlags);
        correspondences_mutual->reserve(n);
        for (int i = 0; i < n; ++i) {
            bool success = false;
            for (const int &j: mv_correspondences_ij[i].match_indices) {
                auto &corr_j = mv_correspondences_ji[j];
                for (int k = 0; k < corr_j.match_indices.size(); ++k) {
                    if (corr_j.match_indices[k] == i) {
                        correspondences_mutual->emplace_back(pcl::Correspondence{i, j, corr_j.distances[k]}, true);
                        success = true;
                        break;
                    }
                }
            }
            if (!success) {
                correspondences_mutual->emplace_back(pcl::Correspondence{i, -1, 1.f}, false);
            } else {
                count_successful++;
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after mutual filtering.\n",
                  getClassName().c_str(), count_successful);
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

    CorrespondencesWithFlagsPtr match_impl(int idx_src, int idx_tgt) override {
        if (this->parameters_.randomness != 1) {
            PCL_WARN("[%s::match] k_corrs different from 1 cannot be used with ratio filtering, using k_corrs = 1.\n",
                     getClassName().c_str());
        }
        const auto &kps_src = this->kps_src_multiscale_[idx_src];
        const auto &kps_tgt = this->kps_tgt_multiscale_[idx_tgt];
        const auto &kps_tree_src = this->kps_tree_src_multiscale_[idx_src];
        const auto &kps_tree_tgt = this->kps_tree_tgt_multiscale_[idx_tgt];
        const auto &kps_features_src = this->kps_features_src_multiscale_[idx_src];
        const auto &kps_features_tgt = this->kps_features_tgt_multiscale_[idx_tgt];
        this->parameters_.randomness = this->parameters_.ratio_parameter;
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ij = matchLocal<FeatureT>(kps_src, kps_tree_tgt, kps_features_src, kps_features_tgt,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ij = matchBF<FeatureT>(kps_features_src, kps_features_tgt, this->parameters_);
        } else {
            mv_correspondences_ij = matchFLANN<FeatureT>(kps_features_src, kps_features_tgt, this->parameters_);
        }
        this->printDebugInfo(mv_correspondences_ij);
        float dist1, dist2, ratio;
        int n = kps_features_src->size(), count_successful = 0, match_idx;
        CorrespondencesWithFlagsPtr correspondences_ratio(new CorrespondencesWithFlags);
        for (int i = 0; i < n; ++i) {
            const auto &mv_corr = mv_correspondences_ij[i];
            if (mv_corr.match_indices.size() != this->parameters_.ratio_parameter) {
                correspondences_ratio->emplace_back(pcl::Correspondence{i, -1, 1.f}, false);
                continue;
            }
            dist1 = mv_corr.distances[0];
            dist2 = mv_corr.distances[this->parameters_.ratio_parameter - 1];
            ratio = (dist2 == 0.f) ? 1.f : (dist1 / dist2);
            if (ratio < MATCHING_RATIO_THRESHOLD) {
                match_idx = mv_corr.match_indices[0];
                correspondences_ratio->emplace_back(pcl::Correspondence{i, match_idx, ratio}, true);
                count_successful++;
            } else {
                correspondences_ratio->emplace_back(pcl::Correspondence{i, -1, 1.f}, false);
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

    CorrespondencesWithFlagsPtr match_impl(int idx_src, int idx_tgt) override {
        const auto &kps_src = this->kps_src_multiscale_[idx_src];
        const auto &kps_tgt = this->kps_tgt_multiscale_[idx_tgt];
        const auto &kps_tree_src = this->kps_tree_src_multiscale_[idx_src];
        const auto &kps_tree_tgt = this->kps_tree_tgt_multiscale_[idx_tgt];
        const auto &kps_features_src = this->kps_features_src_multiscale_[idx_src];
        const auto &kps_features_tgt = this->kps_features_tgt_multiscale_[idx_tgt];
        std::vector<MultivaluedCorrespondence> mv_correspondences_ij;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ij = matchLocal<FeatureT>(kps_src, kps_tree_tgt, kps_features_src, kps_features_tgt,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ij = matchBF<FeatureT>(kps_features_src, kps_features_tgt, this->parameters_);
        } else {
            mv_correspondences_ij = matchFLANN<FeatureT>(kps_features_src, kps_features_tgt, this->parameters_);
        }
        this->printDebugInfo(mv_correspondences_ij);
        std::vector<MultivaluedCorrespondence> mv_correspondences_ji;
        if (this->parameters_.guess.has_value()) {
            mv_correspondences_ji = matchLocal<FeatureT>(kps_tgt, kps_tree_src, kps_features_tgt, kps_features_src,
                                                         this->parameters_);
        } else if (this->parameters_.use_bfmatcher) {
            mv_correspondences_ji = matchBF<FeatureT>(kps_features_tgt, kps_features_src, this->parameters_);
        } else {
            mv_correspondences_ji = matchFLANN<FeatureT>(kps_features_tgt, kps_features_src, this->parameters_);
        }
        int n = kps_features_src->size(), count_successful = 0;
        CorrespondencesWithFlagsPtr correspondences_cluster(new CorrespondencesWithFlags);
        correspondences_cluster->reserve(n);
        for (int i = 0; i < n; ++i) {
            for (int j: mv_correspondences_ij[i].match_indices) {
                float distance_i = calculateCorrespondenceDistance(i, j, MATCHING_CLUSTER_K, mv_correspondences_ij,
                                                                   kps_tree_src, kps_tree_tgt);
                float distance_j = calculateCorrespondenceDistance(j, i, MATCHING_CLUSTER_K, mv_correspondences_ji,
                                                                   kps_tree_tgt, kps_tree_src);
                if (distance_i < MATCHING_CLUSTER_THRESHOLD && distance_j < MATCHING_CLUSTER_THRESHOLD) {
                    correspondences_cluster->emplace_back(pcl::Correspondence{i, j, std::max(distance_i, distance_j)},
                                                          true);
                    count_successful++;
                } else {
                    correspondences_cluster->emplace_back(pcl::Correspondence{i, -1, 1.f}, false);
                }
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after cluster filtering.\n",
                  getClassName().c_str(), count_successful);
        return correspondences_cluster;
    }

    inline std::string getClassName() override {
        return "ClusterMatcher";
    }

protected:
    float calculateCorrespondenceDistance(int i, int j, int k,
                                          const std::vector<MultivaluedCorrespondence> &mv_correspondences_ij,
                                          const KdTreeConstPtr &pcd_tree_src, const KdTreeConstPtr &pcd_tree_tgt) {
        std::unordered_set<int> i_neighbors, j_neighbors;
        pcl::Indices match_indices;
        std::vector<float> distances;

        pcd_tree_src->nearestKSearch(i, k, match_indices, distances);
        std::copy(match_indices.begin(), match_indices.end(), std::inserter(i_neighbors, i_neighbors.begin()));

        pcd_tree_tgt->nearestKSearch(j, k, match_indices, distances);
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
            PCL_DEBUG("\t[matchBF] %d/%d blocks processed.\n", j, n_train_blocks);
        }
        PCL_DEBUG("[matchBF] %d/%d blocks processed.\n", i + 1, n_query_blocks);
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

FeatureBasedMatcher::Ptr getFeatureBasedMatcherFromParameters(const PointNCloud::ConstPtr &src,
                                                              const PointNCloud::ConstPtr &tgt,
                                                              const pcl::IndicesConstPtr &indices_src,
                                                              const pcl::IndicesConstPtr &indices_tgt,
                                                              const AlignmentParameters &parameters);

#endif
