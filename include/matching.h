#ifndef REGISTRATION_MATCHING_H
#define REGISTRATION_MATCHING_H

#include <memory>
#include <unordered_set>
#include <deque>
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
#include "feature_analysis.h"

#define DEBUG_NORMAL_DIFFERENCE false

class FeatureBasedMatcher {
public:
    using Ptr = std::shared_ptr<FeatureBasedMatcher>;
    using ConstPtr = std::shared_ptr<const FeatureBasedMatcher>;

    virtual CorrespondencesPtr match() = 0;

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
    FeatureBasedMatcherImpl(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                            const pcl::IndicesConstPtr &indices_src, const pcl::IndicesConstPtr &indices_tgt,
                            AlignmentParameters parameters) : parameters_(std::move(parameters)) {
        st_src_.pcd = src;
        st_tgt_.pcd = tgt;
        st_src_.kps_indices = indices_src;
        st_tgt_.kps_indices = indices_tgt;
    }

    CorrespondencesPtr match() override;

protected:
    using FeatureCloud = pcl::PointCloud<FeatureT>;
    using KdTree = pcl::search::KdTree<PointN>;

    struct Storage {
        int min_log2_radius{std::numeric_limits<int>::max()}, max_log2_radius{std::numeric_limits<int>::lowest()};
        PointNCloud::ConstPtr pcd;
        PointNCloud::Ptr kps{new PointNCloud};
        typename KdTree::Ptr kps_tree{new KdTree};
        typename pcl::IndicesConstPtr kps_indices; // indices in pcd
        std::vector<PointNCloud::Ptr> pcds_ds;
        std::vector<pcl::Indices> kps_indices_multiscale; // indices in kps_indices
        std::vector<PointNCloud::Ptr> kps_multiscale;
        std::vector<typename FeatureCloud::Ptr> kps_features_multiscale;
        std::vector<typename KdTree::Ptr> kps_tree_multiscale;
        float iss_radius;
    };

    // estimate features on different scales and initialize k-d trees
    void initialize(Storage &storage, const AlignmentParameters &parameters,
                    const std::optional<Eigen::Vector3f> &viewpoint, float iss_radius);

    // get correspondences from features calculated on different scales (this function should be used in match_impl)
    // query and match indices are local, use finalize to get global
    std::vector<MultivaluedCorrespondence> match_multiscale(const Storage &st_query, const Storage &st_train,
                                                            bool inverse_tn = false);

    virtual CorrespondencesPtr match_impl() = 0;

    void finalize(const CorrespondencesPtr &correspondences);

    // downsampling and normal estimation time, feature estimation time
    double time_ds_ne_{0.0}, time_fe_{0.0};
    Storage st_src_, st_tgt_;
    pcl::DefaultPointRepresentation<FeatureT> point_representation_{};
    AlignmentParameters parameters_;
};

template<typename FeatureT>
CorrespondencesPtr FeatureBasedMatcherImpl<FeatureT>::match() {
    // temporary parameters
    AlignmentParameters parameters(parameters_);
    initialize(st_src_, parameters, parameters.vp_src, parameters.iss_radius_src);
    // need to set gt to id in case lrf == 'gt' in estimateReferenceFrames
    parameters.ground_truth = std::optional<Eigen::Matrix4f>(Eigen::Matrix4f::Identity());
    initialize(st_tgt_, parameters, parameters.vp_tgt, parameters.iss_radius_tgt);
    std::cerr << "Downsampling and normal estimation took " << 1000.0 * time_ds_ne_ << "ms.\n";
    std::cerr << "Feature estimation took " << 1000.0 * time_fe_ << "ms.\n";
    auto correspondences = match_impl();
    finalize(correspondences);
    return correspondences;
}

template<typename FeatureT>
void FeatureBasedMatcherImpl<FeatureT>::initialize(Storage &storage, const AlignmentParameters &parameters,
                                                   const std::optional<Eigen::Vector3f> &viewpoint, float iss_radius) {
    storage.iss_radius = iss_radius;
    pcl::copyPointCloud(*storage.pcd, *storage.kps_indices, *storage.kps);
    storage.kps_tree->setInputCloud(storage.kps);
    std::vector<int> log2_radii(storage.kps->size());
    if (parameters.feature_radius.has_value()) {
        float feature_radius = parameters.feature_radius.value();
        int log2_radius = (int) std::floor(std::log2(feature_radius) / std::log2(parameters_.scale_factor));
        storage.min_log2_radius = log2_radius;
        storage.max_log2_radius = log2_radius;
        std::fill(log2_radii.begin(), log2_radii.end(), log2_radius);
    } else {
        KdTree tree(true);
        tree.setInputCloud(storage.pcd);
        pcl::Indices nn_indices;
        std::vector<float> nn_sqr_dists;
        int k = 5;
        for (int i = 0; i < storage.kps->size(); ++i) {
            tree.nearestKSearch(*storage.pcd, storage.kps_indices->operator[](i), k, nn_indices, nn_sqr_dists);
            float density = sqrtf(nn_sqr_dists[k - 1]);
            float feature_radius = sqrtf((float) parameters.feature_nr_points * density * density / M_PI);
            log2_radii[i] = (int) std::floor(std::log2(feature_radius) / std::log2(parameters_.scale_factor));
            storage.min_log2_radius = std::min(log2_radii[i], storage.min_log2_radius);
            storage.max_log2_radius = std::max(log2_radii[i], storage.max_log2_radius);
        }
        // how many points on each level?
        std::deque<int> count_nr_points(storage.max_log2_radius - (storage.min_log2_radius - 1), 0);
        for (int i = 0; i < log2_radii.size(); ++i) {
            count_nr_points[log2_radii[i] - storage.min_log2_radius]++;
        }
        int max_nr_points = *std::max_element(count_nr_points.begin(), count_nr_points.end());
        // remove levels with too little points
        while (10 * count_nr_points.front() < max_nr_points) {
            count_nr_points.pop_front();
            storage.min_log2_radius++;
        }
        while (1000 * count_nr_points.back() < max_nr_points) {
            count_nr_points.pop_back();
            storage.max_log2_radius--;
        }
        for (int &log2_radius: log2_radii) {
            log2_radius = std::min(std::max(log2_radius, storage.min_log2_radius), storage.max_log2_radius);
        }
    }
    int nr_scales = storage.max_log2_radius - (storage.min_log2_radius - 1);
    storage.pcds_ds.resize(nr_scales);
    storage.kps_multiscale.resize(nr_scales);
    storage.kps_indices_multiscale.resize(nr_scales);
    storage.kps_tree_multiscale.resize(nr_scales);
    storage.kps_features_multiscale.resize(nr_scales);
    for (int i = 0; i < nr_scales; ++i) {
        storage.pcds_ds[i] = std::make_shared<PointNCloud>();
        storage.kps_multiscale[i] = std::make_shared<PointNCloud>();
        storage.kps_indices_multiscale[i].reserve(storage.kps_indices->size());
        storage.kps_tree_multiscale[i] = std::make_shared<KdTree>();
        storage.kps_features_multiscale[i] = std::make_shared<FeatureCloud>();
    }
    // for each scale build indices
    for (int i = 0; i < storage.kps->size(); ++i) {
        for (int j = log2_radii[i]; j <= storage.max_log2_radius; ++j) {
            storage.kps_indices_multiscale[j - storage.min_log2_radius].push_back(i);
        }
    }
    // for each scale estimate features
    for (int i = 0; i < nr_scales; i++) {
        float search_radius = powf(parameters.scale_factor, (float) (storage.min_log2_radius + i));
        float voxel_size = sqrtf(M_PI * search_radius * search_radius / (float) parameters.feature_nr_points);
        {
            auto t1 = std::chrono::system_clock::now();
            downsamplePointCloud(i == 0 ? storage.pcd : storage.pcds_ds[i - 1], storage.pcds_ds[i], voxel_size);
            estimateNormalsPoints(parameters_.normal_nr_points, storage.pcds_ds[i], {nullptr}, viewpoint,
                                  parameters_.normals_available);
            auto t2 = std::chrono::system_clock::now();
            time_ds_ne_ += double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0;
        }
        {
            auto t1 = std::chrono::system_clock::now();
            pcl::copyPointCloud(*storage.kps, storage.kps_indices_multiscale[i], *storage.kps_multiscale[i]);
            if (parameters.reestimate_frames) {
                estimateNormalsPoints(parameters_.normal_nr_points, storage.kps_multiscale[i], storage.pcds_ds[i],
                                      viewpoint, true);
            }
            pcl::console::print_highlight("Estimating features  [search_radius = %.5f]...\n", search_radius);
            estimateFeatures<FeatureT>(storage.kps_multiscale[i], storage.pcds_ds[i],
                                       storage.kps_features_multiscale[i], search_radius, parameters);
            int count_invalid = 0;
            for (int j = 0; j < storage.kps_features_multiscale[i]->size(); ++j) {
                if (!point_representation_.isValid(storage.kps_features_multiscale[i]->points[j]))
                    count_invalid++;
            }
            PCL_DEBUG("[%s::initialize] %i/%i invalid features\n",
                      getClassName().c_str(), count_invalid, storage.kps_features_multiscale[i]->size());
            auto t2 = std::chrono::system_clock::now();
            time_fe_ += double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0;
        }
        storage.kps_tree_multiscale[i]->setInputCloud(storage.kps_multiscale[i]);
    }
}

template<typename FeatureT>
std::vector<MultivaluedCorrespondence> FeatureBasedMatcherImpl<FeatureT>::match_multiscale(const Storage &st_query,
                                                                                           const Storage &st_train,
                                                                                           bool inverse_tn) {
    std::vector<MultivaluedCorrespondence> mv_correspondences(st_query.kps->size());
    std::vector<float> norm_diffs_pcd, norm_diffs_kps;
    int min_log2_radius = std::max(st_query.min_log2_radius, st_train.min_log2_radius);
    int max_log2_radius = std::min(st_query.max_log2_radius, st_train.max_log2_radius);
    for (int log2_radius = min_log2_radius; log2_radius <= max_log2_radius; ++log2_radius) {
        if (parameters_.save_features && !inverse_tn) {
            std::string scale = parameters_.feature_radius.has_value() ? "" : std::to_string(log2_radius);
            saveFeatures<FeatureT>(st_query.kps_features_multiscale[log2_radius - st_query.min_log2_radius], {},
                                   parameters_, true, scale);
            saveFeatures<FeatureT>(st_train.kps_features_multiscale[log2_radius - st_train.min_log2_radius], {},
                                   parameters_, false, scale);
        }
        int idx_query = log2_radius - st_query.min_log2_radius, idx_train = log2_radius - st_train.min_log2_radius;
        int nr_kps_query = st_query.kps_multiscale[idx_query]->size();
        float search_radius = powf(2.0, (float) log2_radius);
        float voxel_size = sqrtf(M_PI * search_radius * search_radius / (float) parameters_.feature_nr_points);
#if DEBUG_NORMAL_DIFFERENCE
        if (parameters_.ground_truth.has_value() ) {
            Eigen::Matrix4f gt = parameters_.ground_truth.value();
            if (inverse_tn) gt = gt.inverse();
            norm_diffs_pcd.push_back(calculateNormalDifference(st_query.pcds_ds[idx_query], st_train.pcds_ds[idx_train],
                                                               voxel_size, gt));
            norm_diffs_kps.push_back(calculateNormalDifference(st_query.kps_multiscale[idx_query],
                                                               st_train.kps_multiscale[idx_train],
                                                               parameters_.distance_thr, gt));
        }
#endif
        // correspondences calculated on one level
        std::vector<MultivaluedCorrespondence> mv_corrs_fixed_level;
        if (parameters_.guess.has_value()) {
            Eigen::Matrix4f guess = parameters_.guess.value();
            if (inverse_tn) guess = guess.inverse();
            mv_corrs_fixed_level = matchLocal<FeatureT>(st_query.kps_multiscale[idx_query],
                                                        st_train.kps_tree_multiscale[idx_train],
                                                        st_query.kps_features_multiscale[idx_query],
                                                        st_train.kps_features_multiscale[idx_train],
                                                        parameters_, guess);
        } else if (this->parameters_.use_bfmatcher) {
            mv_corrs_fixed_level = matchBF<FeatureT>(st_query.kps_features_multiscale[idx_query],
                                                     st_train.kps_features_multiscale[idx_train], parameters_);
        } else {
            mv_corrs_fixed_level = matchFLANN<FeatureT>(st_query.kps_features_multiscale[idx_query],
                                                        st_train.kps_features_multiscale[idx_train], parameters_);
        }
        rassert(mv_corrs_fixed_level.size() == nr_kps_query, 9082472354896501)
        for (int i = 0; i < nr_kps_query; ++i) {
            MultivaluedCorrespondence mv_corr = mv_corrs_fixed_level[i];
            int index_query = st_query.kps_indices_multiscale[idx_query][i];
            for (int j = 0; j < mv_corr.match_indices.size(); ++j) {
                int index_match = st_train.kps_indices_multiscale[idx_train][mv_corr.match_indices[j]];
                mv_correspondences[index_query].match_indices.push_back(index_match);
                mv_correspondences[index_query].distances.push_back(mv_corr.distances[j]);
            }
        }
    }
    if (DEBUG_NORMAL_DIFFERENCE) {
        saveVector(norm_diffs_pcd, constructPath(parameters_, "median_norm_diff", "csv", true, false, false));
        saveVector(norm_diffs_kps, constructPath(parameters_, "median_norm_diff_kps", "csv", true, false, false));
    }
    for (int i = 0; i < mv_correspondences.size(); ++i) {
        const auto &mv_corr = mv_correspondences[i];
        std::vector<std::pair<float, float>> counter;
        for (int m1 = 0; m1 < mv_corr.match_indices.size(); ++m1) {
            counter.emplace_back(0.f, mv_corr.distances[m1]);
            for (int m2 = m1; m2 < mv_corr.match_indices.size(); ++m2) {
                if (mv_corr.match_indices[m1] < 0 || mv_corr.match_indices[m2] < 0) continue;
                float dist_l2 = (st_train.kps->points[mv_corr.match_indices[m1]].getVector3fMap() -
                                 st_train.kps->points[mv_corr.match_indices[m2]].getVector3fMap()).norm();
                if (dist_l2 < 32 * st_train.iss_radius) {
                    counter[m1].first += st_train.iss_radius / std::max(dist_l2, st_train.iss_radius);
                }
            }
        }
        std::pair<float, float> best_match{0.f, 0.f};
        MultivaluedCorrespondence final_mv_corr;
        for (int m = 0; m < mv_corr.match_indices.size(); ++m) {
            auto[count, dist] = counter[m];
            if (count > best_match.first || (count == best_match.first && dist < best_match.second)) {
                best_match = {count, dist};
                final_mv_corr = {{mv_corr.match_indices[m]},
                                 {mv_corr.distances[m]}};
            }
        }
        mv_correspondences[i] = final_mv_corr;
    }
    return mv_correspondences;
}

template<typename FeatureT>
void FeatureBasedMatcherImpl<FeatureT>::finalize(const CorrespondencesPtr &correspondences) {
    for (auto &corr: *correspondences) {
        corr.index_query = st_src_.kps_indices->operator[](corr.index_query);
        corr.index_match = st_tgt_.kps_indices->operator[](corr.index_match);
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
                                                  const AlignmentParameters &parameters, const Eigen::Matrix4f &guess);

template<typename FeatureT>
class OneSidedMatcher : public FeatureBasedMatcherImpl<FeatureT> {
public:
    OneSidedMatcher() = delete;

    OneSidedMatcher(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                    pcl::IndicesConstPtr indices_src, pcl::IndicesConstPtr indices_tgt,
                    AlignmentParameters parameters) : FeatureBasedMatcherImpl<FeatureT>(src, tgt, indices_src,
                                                                                        indices_tgt, parameters) {}

    CorrespondencesPtr match_impl() override {
        std::vector<float> thresholds_src = calculateSmoothedDensities(this->st_src_.kps);
        std::vector<float> thresholds_tgt = calculateSmoothedDensities(this->st_tgt_.kps);
        std::vector<MultivaluedCorrespondence> mv_corrs_ij = this->match_multiscale(this->st_src_, this->st_tgt_);
        this->printDebugInfo(mv_corrs_ij);
        CorrespondencesPtr correspondences(new Correspondences);
        int nr_kps = this->st_src_.kps_indices->size();
        correspondences->reserve(nr_kps);
        for (int i = 0; i < nr_kps; ++i) {
            if (mv_corrs_ij[i].match_indices.empty()) continue;
            float threshold = std::min(std::max(thresholds_src[i], thresholds_tgt[mv_corrs_ij[i].match_indices[0]]),
                                       this->parameters_.distance_thr);
            correspondences->emplace_back(i, mv_corrs_ij[i].match_indices[0], mv_corrs_ij[i].distances[0], threshold);
        }
        PCL_DEBUG("[%s::match] %i correspondences.\n", getClassName().c_str(), correspondences->size());
        return correspondences;
    }

    inline std::string getClassName() override {
        return "OneSidedMatcher";
    }
};

template<typename FeatureT>
class LeftToRightMatcher : public FeatureBasedMatcherImpl<FeatureT> {
public:
    LeftToRightMatcher() = delete;

    LeftToRightMatcher(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                       pcl::IndicesConstPtr indices_src, pcl::IndicesConstPtr indices_tgt,
                       AlignmentParameters parameters) : FeatureBasedMatcherImpl<FeatureT>(src, tgt, indices_src,
                                                                                           indices_tgt, parameters) {}

    CorrespondencesPtr match_impl() override {
        std::vector<float> thresholds_src = calculateSmoothedDensities(this->st_src_.kps);
        std::vector<float> thresholds_tgt = calculateSmoothedDensities(this->st_tgt_.kps);
        std::vector<MultivaluedCorrespondence> mv_corrs_ij = this->match_multiscale(this->st_src_, this->st_tgt_);
        std::vector<MultivaluedCorrespondence> mv_corrs_ji = this->match_multiscale(this->st_tgt_, this->st_src_, true);
        this->printDebugInfo(mv_corrs_ij);
        CorrespondencesPtr correspondences_mutual(new Correspondences);
        int nr_kps = this->st_src_.kps_indices->size();
        correspondences_mutual->reserve(nr_kps);
        for (int i = 0; i < nr_kps; ++i) {
            for (const int &j: mv_corrs_ij[i].match_indices) {
                auto &corr_j = mv_corrs_ji[j];
                for (int k = 0; k < corr_j.match_indices.size(); ++k) {
                    if (corr_j.match_indices[k] == i) {
                        float threshold = std::min(std::max(thresholds_src[i], thresholds_tgt[j]),
                                                   this->parameters_.distance_thr);
                        correspondences_mutual->emplace_back(i, j, corr_j.distances[k], threshold);
                        break;
                    }
                }
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after mutual filtering.\n",
                  getClassName().c_str(), correspondences_mutual->size());
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

    CorrespondencesPtr match_impl() override {
        // TODO: implement
        return {};
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

    CorrespondencesPtr match_impl() override {
        std::vector<float> thresholds_src = calculateSmoothedDensities(this->st_src_.kps);
        std::vector<float> thresholds_tgt = calculateSmoothedDensities(this->st_tgt_.kps);
        std::vector<MultivaluedCorrespondence> mv_corrs_ij = this->match_multiscale(this->st_src_, this->st_tgt_);
        std::vector<MultivaluedCorrespondence> mv_corrs_ji = this->match_multiscale(this->st_tgt_, this->st_src_, true);
        this->printDebugInfo(mv_corrs_ij);
        CorrespondencesPtr correspondences_cluster(new Correspondences);
        int nr_kps = this->st_src_.kps_indices->size();
        correspondences_cluster->reserve(nr_kps);
        for (int i = 0; i < nr_kps; ++i) {
            for (int j: mv_corrs_ij[i].match_indices) {
                float distance_i = calculateCorrespondenceDistance(i, j, this->parameters_.cluster_k, mv_corrs_ij,
                                                                   this->st_src_.kps_tree, this->st_tgt_.kps_tree);
                float distance_j = calculateCorrespondenceDistance(j, i, this->parameters_.cluster_k, mv_corrs_ji,
                                                                   this->st_tgt_.kps_tree, this->st_src_.kps_tree);
                if (distance_i < MATCHING_CLUSTER_THRESHOLD && distance_j < MATCHING_CLUSTER_THRESHOLD) {
                    float threshold = std::min(std::max(thresholds_src[i], thresholds_tgt[j]),
                                               this->parameters_.distance_thr);
                    correspondences_cluster->emplace_back(i, j, std::max(distance_i, distance_j), threshold);
                }
            }
        }
        PCL_DEBUG("[%s::match] %i correspondences remain after cluster filtering.\n",
                  getClassName().c_str(), correspondences_cluster->size());
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
    pcl::KdTreeFLANN<FeatureT> feature_tree;
    feature_tree.setInputCloud(train_features);
    auto n = query_features->size();
    std::vector<MultivaluedCorrespondence> mv_correspondences(n, MultivaluedCorrespondence{});
#pragma omp parallel for \
    default(none) \
    shared(mv_correspondences, query_features, feature_tree) \
    firstprivate(n, point_representation, parameters)
    for (int i = 0; i < n; i++) {
        if (point_representation.isValid(query_features->points[i])) {
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
    return mv_correspondences;
}


template<typename FeatureT>
std::vector<MultivaluedCorrespondence> matchLocal(const PointNCloud::ConstPtr &query_pcd,
                                                  const typename pcl::search::KdTree<PointN>::ConstPtr &train_tree,
                                                  const typename pcl::PointCloud<FeatureT>::ConstPtr &query_features,
                                                  const typename pcl::PointCloud<FeatureT>::ConstPtr &train_features,
                                                  const AlignmentParameters &parameters, const Eigen::Matrix4f &guess) {
    PointNCloud transformed_query_pcd;
    pcl::transformPointCloudWithNormals(*query_pcd, transformed_query_pcd, guess);
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
