#include <unordered_set>

#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>

#include "metric.h"
#include "weights.h"

void buildClosestPlaneInliers(const PointNCloud &src,
                              const pcl::KdTreeFLANN<PointN> &tree_tgt,
                              const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                              float inlier_threshold, float &rmse, bool sparse, UniformRandIntGenerator &rand) {
    inlier_pairs.clear();
    inlier_pairs.reserve(src.size());
    rmse = 0.0f;
    const PointNCloud &tgt = *tree_tgt.getInputCloud();
    int n = (int) ((sparse ? SPARSE_POINTS_FRACTION : 1.f) * (float) src.size());
    float search_radius = 2.f * inlier_threshold;
    float dist_to_plane;
    Eigen::Vector3f point_transformed, nearest_point;

    // For point in the source dataset
    for (int i = 0; i < n; ++i) {
        int idx = sparse ? rand() % (int) src.size() : i;
        point_transformed = (transformation * src[idx].getVector4fMap()).block<3, 1>(0, 0);
        // Find its nearest neighbor in the target
        pcl::Indices nn_indices(1);
        std::vector<float> nn_dists(1);
        tree_tgt.radiusSearch(PointN(point_transformed.x(), point_transformed.y(), point_transformed.z()),
                              search_radius, nn_indices, nn_dists);
        if (!nn_dists.empty()) {
            nearest_point = tgt[nn_indices[0]].getNormalVector3fMap();
            dist_to_plane = std::fabs(nearest_point.transpose() * (nearest_point - point_transformed));
            // Check if point is an inlier
            if (dist_to_plane < inlier_threshold) {
                // Update inliers and rmse
                inlier_pairs.push_back({(int) idx, nn_indices[0]});
                rmse += dist_to_plane * dist_to_plane;
            }
        }
    }

    // Calculate RMSE
    if (!inlier_pairs.empty())
        rmse = std::sqrt(rmse / static_cast<float>(inlier_pairs.size()));
    else
        rmse = std::numeric_limits<float>::max();
}

void MetricEstimator::buildCorrectInlierPairs(const std::vector<InlierPair> &inlier_pairs,
                                              std::vector<InlierPair> &correct_inlier_pairs,
                                              const Eigen::Matrix4f &transformation_gt) const {
    correct_inlier_pairs.clear();
    correct_inlier_pairs.reserve(inlier_pairs.size());

    PointNCloud src_transformed;
    src_transformed.resize(src_->size());
    pcl::transformPointCloud(*src_, src_transformed, transformation_gt);

    for (const auto &ip: inlier_pairs) {
        PointN source_point(src_transformed.points[ip.idx_src]);
        PointN target_point(tgt_->points[ip.idx_tgt]);
        float e = pcl::L2_Norm(source_point.data, target_point.data, 3);
        if (e < inlier_threshold_) {
            correct_inlier_pairs.push_back(ip);
        }
    }
}

int MetricEstimator::estimateMaxIterations(const Eigen::Matrix4f &transformation,
                                           float confidence, int nr_samples) const {
    int count_supporting_corrs = 0;
    for (const auto &corr: correspondences_) {
        Eigen::Vector4f source_point(0, 0, 0, 1);
        source_point.block(0, 0, 3, 1) = src_->points[corr.index_query].getArray3fMap();
        Eigen::Vector4f target_point(0, 0, 0, 1);
        target_point.block(0, 0, 3, 1) = tgt_->points[corr.index_match].getArray3fMap();
        float e = (transformation * source_point - target_point).block(0, 0, 3, 1).norm();
        if (e < inlier_threshold_) {
            count_supporting_corrs++;
        }
    }
    float supporting_corr_fraction = (float) count_supporting_corrs / (float) correspondences_.size();
    if (supporting_corr_fraction <= 0.0 || std::log(1.0 - std::pow(supporting_corr_fraction, nr_samples)) >= 0.0) {
        return std::numeric_limits<int>::max();
    }
    double iterations = std::log(1.0 - confidence) / std::log(1.0 - std::pow(supporting_corr_fraction, nr_samples));
    return static_cast<int>(std::min((double) std::numeric_limits<int>::max(), iterations));
}

void CorrespondencesMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                      std::vector<InlierPair> &inlier_pairs,
                                                      float &rmse) {
    inlier_pairs.clear();
    inlier_pairs.reserve(correspondences_.size());
    rmse = 0.0f;

    // For each point from correspondences in the source dataset
    Eigen::Vector4f source_point, target_point;
    for (int i = 0; i < correspondences_.size(); ++i) {
        int query_idx = correspondences_[i].index_query;
        int match_idx = correspondences_[i].index_match;
        source_point = transformation * src_->points[query_idx].getVector4fMap();
        target_point = tgt_->points[match_idx].getVector4fMap();

        // Calculate correspondence distance
        float dist = (source_point - target_point).norm();

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

void CorrespondencesMetricEstimator::buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                       std::vector<InlierPair> &inlier_pairs,
                                                                       float &rmse, float &metric) {
    buildInlierPairs(transformation, inlier_pairs, rmse);
    metric = (float) inlier_pairs.size() / (float) correspondences_.size();
}

void ClosestPlaneMetricEstimator::setTargetCloud(const PointNCloud::ConstPtr &tgt) {
    tgt_ = tgt;
    tree_tgt_.setInputCloud(tgt);
}

void ClosestPlaneMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                   std::vector<InlierPair> &inlier_pairs,
                                                   float &rmse) {
    buildClosestPlaneInliers(*src_, tree_tgt_, transformation, inlier_pairs, inlier_threshold_, rmse, sparse_, rand_);
}

void ClosestPlaneMetricEstimator::buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                    std::vector<InlierPair> &inlier_pairs,
                                                                    float &rmse, float &metric) {
    buildInlierPairs(transformation, inlier_pairs, rmse);
    metric = (float) inlier_pairs.size() / ((sparse_ ? SPARSE_POINTS_FRACTION : 1.f) * (float) src_->size());
}

void WeightedClosestPlaneMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                           std::vector<InlierPair> &inlier_pairs,
                                                           float &rmse) {
    buildClosestPlaneInliers(*src_, tree_tgt_, transformation, inlier_pairs, inlier_threshold_, rmse, sparse_, rand_);
}

void WeightedClosestPlaneMetricEstimator::buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                            std::vector<InlierPair> &inlier_pairs,
                                                                            float &rmse, float &metric) {
    buildInlierPairs(transformation, inlier_pairs, rmse);
    float sum = 0.0;
    for (auto &inlier_pair: inlier_pairs) {
        sum += weights_[inlier_pair.idx_src];
    }
    metric = sum / ((sparse_ ? SPARSE_POINTS_FRACTION : 1.f) * weights_sum_);
}

void WeightedClosestPlaneMetricEstimator::setSourceCloud(const PointNCloud::ConstPtr &src) {
    src_ = src;
    auto weight_function = getWeightFunction(weight_id_);
    weights_ = weight_function(curvature_radius_, src_);
    weights_sum_ = 0.f;
    for (float weight: weights_) {
        weights_sum_ += weight;
    }
}

void WeightedClosestPlaneMetricEstimator::setTargetCloud(const PointNCloud::ConstPtr &tgt) {
    tgt_ = tgt;
    tree_tgt_.setInputCloud(tgt);
}

void CombinationMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                  std::vector<InlierPair> &inlier_pairs, float &rmse) {
    correspondences_estimator.buildInlierPairs(transformation, inlier_pairs, rmse);
}

void CombinationMetricEstimator::buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                   std::vector<InlierPair> &inlier_pairs,
                                                                   float &rmse, float &metric) {
    float metric_cs, metric_cp;
    float rmse_cp;
    std::vector<InlierPair> inlier_pairs_cp;
    correspondences_estimator.buildInlierPairsAndEstimateMetric(transformation, inlier_pairs, rmse, metric_cs);
    closest_plane_estimator.buildInlierPairsAndEstimateMetric(transformation, inlier_pairs_cp, rmse_cp, metric_cp);
    metric = metric_cs * metric_cp;
}

void CombinationMetricEstimator::setCorrespondences(const pcl::Correspondences &correspondences) {
    correspondences_ = correspondences;
    correspondences_estimator.setCorrespondences(correspondences);
    closest_plane_estimator.setCorrespondences(correspondences);
}

void CombinationMetricEstimator::setSourceCloud(const PointNCloud::ConstPtr &src) {
    src_ = src;
    correspondences_estimator.setSourceCloud(src);
    closest_plane_estimator.setSourceCloud(src);
}

void CombinationMetricEstimator::setTargetCloud(const PointNCloud::ConstPtr &tgt) {
    tgt_ = tgt;
    correspondences_estimator.setTargetCloud(tgt);
    closest_plane_estimator.setTargetCloud(tgt);
}

void CombinationMetricEstimator::setInlierThreshold(float inlier_threshold) {
    inlier_threshold_ = inlier_threshold;
    correspondences_estimator.setInlierThreshold(inlier_threshold);
    closest_plane_estimator.setInlierThreshold(inlier_threshold);
}

// if sparse is true then only fixed percentage of points from source point cloud will be used
// to estimate metrics based on closest plane
MetricEstimator::Ptr getMetricEstimatorFromParameters(const AlignmentParameters &parameters, bool sparse) {
    if (parameters.metric_id == METRIC_CLOSEST_PLANE) {
        return std::make_shared<ClosestPlaneMetricEstimator>(sparse);
    } else if (parameters.metric_id == METRIC_WEIGHTED_CLOSEST_PLANE) {
        float curvature_radius = 2.f * parameters.normal_radius_coef * parameters.voxel_size;
        return std::make_shared<WeightedClosestPlaneMetricEstimator>(parameters.weight_id, curvature_radius, sparse);
    } else if (parameters.metric_id == METRIC_COMBINATION) {
        return std::make_shared<CombinationMetricEstimator>(sparse);
    } else if (parameters.metric_id != METRIC_CORRESPONDENCES) {
        PCL_WARN("[getMetricEstimator] metric estimator %s isn't supported, correspondences will be used\n",
                 parameters.metric_id.c_str());
    }
    return std::make_shared<CorrespondencesMetricEstimator>();
}