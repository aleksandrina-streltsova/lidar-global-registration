#include "metric.h"

#include <unordered_set>

#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>

void MetricEstimator::buildCorrectInlierPairs(const std::vector<InlierPair> &inlier_pairs,
                                              std::vector<InlierPair> &correct_inlier_pairs,
                                              const Eigen::Matrix4f &transformation_gt,
                                              float error_thr) const {
    correct_inlier_pairs.clear();
    correct_inlier_pairs.reserve(inlier_pairs.size());

    PointCloudTN src_transformed;
    src_transformed.resize(src_->size());
    pcl::transformPointCloud(*src_, src_transformed, transformation_gt);

    for (const auto &ip: inlier_pairs) {
        PointTN source_point(src_transformed.points[ip.idx_src]);
        PointTN target_point(tgt_->points[ip.idx_tgt]);
        float e = pcl::L2_Norm(source_point.data, target_point.data, 3);
        if (e < error_thr) {
            correct_inlier_pairs.push_back(ip);
        }
    }
}

void CorrespondencesMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                      std::vector<InlierPair> &inlier_pairs,
                                                      float &rmse) const {
    inlier_pairs.clear();
    inlier_pairs.reserve(correspondences_.size());
    rmse = 0.0f;

    PointCloudTN src_transformed;
    src_transformed.resize(src_->size());
    pcl::transformPointCloud(*src_, src_transformed, transformation);

    // For each point from correspondences in the source dataset
    for (int i = 0; i < correspondences_.size(); ++i) {
        int query_idx = correspondences_[i].query_idx;
        int match_idx = correspondences_[i].match_indices[0];
        PointTN source_point(src_transformed.points[query_idx]);
        PointTN target_point(tgt_->points[match_idx]);

        // Calculate correspondence distance
        float dist = pcl::L2_Norm(source_point.data, target_point.data, 3);

        // Check if correspondence is an inlier
        if (dist < inlier_threshold_) {
            // Update inliers and rmse
            inlier_pairs.push_back({query_idx, match_idx});
            rmse += dist * dist;
        }
    }

    // Calculate RMSE
    if (!inlier_pairs.empty())
        rmse = std::sqrt(rmse / static_cast<float>(inlier_pairs.size()));
    else
        rmse = std::numeric_limits<float>::max();
}

void CorrespondencesMetricEstimator::estimateMetric(const std::vector<InlierPair> &inlier_pairs, float &metric) const {
    metric = (float) inlier_pairs.size() / (float) correspondences_.size();
}

int CorrespondencesMetricEstimator::estimateMaxIterations(std::vector<InlierPair> &inlier_pairs,
                                                          float confidence, int nr_samples) const {
    float inlier_fraction = (float) inlier_pairs.size() / (float) correspondences_.size();
    if (inlier_fraction <= 0.0) {
        return std::numeric_limits<int>::max();
    }
    double iterations = std::log(1.0 - confidence) / std::log(1.0 - std::pow(inlier_fraction, nr_samples));
    return static_cast<int>(std::min((double) std::numeric_limits<int>::max(), iterations));
}

void ClosestPointMetricEstimator::setTargetCloud(const PointCloudTN::ConstPtr &tgt) {
    tgt_ = tgt;
    tree_tgt_.setInputCloud(tgt);
}

void ClosestPointMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                   std::vector<InlierPair> &inlier_pairs,
                                                   float &rmse) const {
    inlier_pairs.clear();
    inlier_pairs.reserve(src_->size());
    rmse = 0.0f;

    PointCloudTN src_transformed;
    src_transformed.resize(src_->size());
    pcl::transformPointCloud(*src_, src_transformed, transformation);

    // For each point in the source dataset
    for (std::size_t i = 0; i < src_transformed.size(); ++i) {
        // Find its nearest neighbor in the target
        pcl::Indices nn_indices(1);
        std::vector<float> nn_dists(1);
        tree_tgt_.nearestKSearch(src_transformed[i], 1, nn_indices, nn_dists);

        // Check if point is an inlier
        if (nn_dists[0] < inlier_threshold_ * inlier_threshold_) {
            // Update inliers and rmse
            inlier_pairs.push_back({(int) i, nn_indices[0]});
            rmse += nn_dists[0];
        }
    }

    // Calculate RMSE
    if (!inlier_pairs.empty())
        rmse = std::sqrt(rmse / static_cast<float>(inlier_pairs.size()));
    else
        rmse = std::numeric_limits<float>::max();
}

void ClosestPointMetricEstimator::estimateMetric(const std::vector<InlierPair> &inlier_pairs, float &metric) const {
    metric = (float) inlier_pairs.size() / (float) src_->size();
}

int ClosestPointMetricEstimator::estimateMaxIterations(std::vector<InlierPair> &inlier_pairs,
                                                       float confidence, int nr_samples) const {
    return std::numeric_limits<int>::max();
}

MetricEstimator::Ptr getMetricEstimator(const std::string &metric_id) {
    if (metric_id == "closest_point") {
        return std::make_shared<ClosestPointMetricEstimator>();
    }
    return std::make_shared<CorrespondencesMetricEstimator>();
}