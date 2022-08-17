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
    Eigen::Vector3f point_transformed, nearest_point, normal;
    std::vector<bool> visited(src.size(), false);

    // For point in the source dataset
    for (int i = 0; i < n; ++i) {
        int idx = sparse ? rand() % (int) src.size() : i;
        while (visited[idx]) idx = (idx + 1) % (int) src.size();
        visited[idx] = true;

        point_transformed = (transformation * src[idx].getVector4fMap()).block<3, 1>(0, 0);
        // Find its nearest neighbor in the target
        pcl::Indices nn_indices(1);
        std::vector<float> nn_sqr_dists(1);
        tree_tgt.nearestKSearch(PointN(point_transformed.x(), point_transformed.y(), point_transformed.z()),
                                1, nn_indices, nn_sqr_dists);
        nearest_point = tgt[nn_indices[0]].getVector3fMap();
        normal = tgt[nn_indices[0]].getNormalVector3fMap();
        dist_to_plane = std::fabs(normal.transpose() * (nearest_point - point_transformed));
        // Check if point is an inlier
        if (dist_to_plane < inlier_threshold) {
            // Update inliers and rmse
            inlier_pairs.push_back({(int) idx, nn_indices[0], dist_to_plane});
            rmse += dist_to_plane * dist_to_plane;
        }
    }

    // Calculate RMSE
    if (!inlier_pairs.empty())
        rmse = std::sqrt(rmse / static_cast<float>(inlier_pairs.size()));
    else
        rmse = std::numeric_limits<float>::max();
}

float calculateScore(const std::vector<InlierPair> &inlier_pairs, float threshold, ScoreFunction score_function,
                     const std::vector<float> &weights = {}) {
    bool with_weights = !weights.empty();
    float score = 0.f, value;
    for (const auto &ip: inlier_pairs) {
        switch (score_function) {
            case Constant:
                value = 1.f;
                break;
            case MAE:
                value = std::fabs(ip.dist - threshold) / threshold;
                break;
            case MSE:
                value = (ip.dist - threshold) * (ip.dist - threshold) / (threshold * threshold);
                break;
            case EXP:
                value = (std::exp(-ip.dist * ip.dist / (2 * threshold * threshold)));
                break;
        }
        if (with_weights) {
            value *= weights[ip.idx_src];
        }
        score += value;
    }
    return score;
}

void MetricEstimator::buildCorrectInlierPairs(const std::vector<InlierPair> &inlier_pairs,
                                              std::vector<InlierPair> &correct_inlier_pairs,
                                              const Eigen::Matrix4f &transformation_gt) const {
    correct_inlier_pairs.clear();
    correct_inlier_pairs.reserve(inlier_pairs.size());

    PointNCloud src_transformed;
    src_transformed.resize(src_->size());
    pcl::transformPointCloudWithNormals(*src_, src_transformed, transformation_gt);

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
    for (const auto &corr: *correspondences_) {
        Eigen::Vector4f source_point(0, 0, 0, 1);
        source_point.block(0, 0, 3, 1) = src_->points[corr.index_query].getArray3fMap();
        Eigen::Vector4f target_point(0, 0, 0, 1);
        target_point.block(0, 0, 3, 1) = tgt_->points[corr.index_match].getArray3fMap();
        float e = (transformation * source_point - target_point).block(0, 0, 3, 1).norm();
        if (e < inlier_threshold_) {
            count_supporting_corrs++;
        }
    }
    float supporting_corr_fraction = (float) count_supporting_corrs / (float) correspondences_->size();
    supporting_corr_fraction /= 4.f;    // estimate number of iterations pessimistically
    if (supporting_corr_fraction <= 0.0 || std::log(1.0 - std::pow(supporting_corr_fraction, nr_samples)) >= 0.0) {
        return std::numeric_limits<int>::max();
    }
    double iterations = std::log(1.0 - confidence) / std::log(1.0 - std::pow(supporting_corr_fraction, nr_samples));
    return static_cast<int>(std::min((double) std::numeric_limits<int>::max(), iterations));
}

void CorrespondencesMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                      std::vector<InlierPair> &inlier_pairs,
                                                      float &rmse, UniformRandIntGenerator &) const {
    inlier_pairs.clear();
    inlier_pairs.reserve(correspondences_->size());
    rmse = 0.0f;

    // For each point from correspondences in the source dataset
    Eigen::Vector4f source_point, target_point;
    for (int i = 0; i < correspondences_->size(); ++i) {
        int query_idx = correspondences_->operator[](i).index_query;
        int match_idx = correspondences_->operator[](i).index_match;
        source_point = transformation * src_->points[query_idx].getVector4fMap();
        target_point = tgt_->points[match_idx].getVector4fMap();

        // Calculate correspondence distance
        float dist = (source_point - target_point).norm();

        // Check if correspondence is an inlier
        if (dist < inlier_threshold_) {
            // Update inliers and rmse
            inlier_pairs.push_back({query_idx, match_idx, dist});
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
                                                                       float &rmse, float &metric,
                                                                       UniformRandIntGenerator &rand) const {
    buildInlierPairs(transformation, inlier_pairs, rmse, rand);
    float score = calculateScore(inlier_pairs, inlier_threshold_, this->score_function_);
    metric = score / (float) correspondences_->size();
}

void ClosestPlaneMetricEstimator::setTargetCloud(const PointNCloud::ConstPtr &tgt) {
    tgt_ = tgt;
    tree_tgt_.setInputCloud(tgt);
}

void ClosestPlaneMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                   std::vector<InlierPair> &inlier_pairs,
                                                   float &rmse, UniformRandIntGenerator &rand) const {
    buildClosestPlaneInliers(*src_, tree_tgt_, transformation, inlier_pairs, inlier_threshold_, rmse, sparse_, rand);
}

void ClosestPlaneMetricEstimator::buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                    std::vector<InlierPair> &inlier_pairs,
                                                                    float &rmse, float &metric,
                                                                    UniformRandIntGenerator &rand) const {
    buildInlierPairs(transformation, inlier_pairs, rmse, rand);
    float score = calculateScore(inlier_pairs, inlier_threshold_, this->score_function_);
    metric = score / ((sparse_ ? SPARSE_POINTS_FRACTION : 1.f) * (float) src_->size());
}

void WeightedClosestPlaneMetricEstimator::buildInlierPairs(const Eigen::Matrix4f &transformation,
                                                           std::vector<InlierPair> &inlier_pairs,
                                                           float &rmse, UniformRandIntGenerator &rand) const {
    buildClosestPlaneInliers(*src_, tree_tgt_, transformation, inlier_pairs, inlier_threshold_, rmse, sparse_, rand);
}

void WeightedClosestPlaneMetricEstimator::buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                            std::vector<InlierPair> &inlier_pairs,
                                                                            float &rmse, float &metric,
                                                                            UniformRandIntGenerator &rand) const {
    buildInlierPairs(transformation, inlier_pairs, rmse, rand);
    float score = calculateScore(inlier_pairs, inlier_threshold_, this->score_function_, weights_);
    metric = score / ((sparse_ ? SPARSE_POINTS_FRACTION : 1.f) * weights_sum_);
}

void WeightedClosestPlaneMetricEstimator::setSourceCloud(const PointNCloud::ConstPtr &src) {
    src_ = src;
    auto weight_function = getWeightFunction(weight_id_);
    weights_ = weight_function(nr_points_, src_);
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
                                                  std::vector<InlierPair> &inlier_pairs, float &rmse,
                                                  UniformRandIntGenerator &rand) const {
    correspondences_estimator.buildInlierPairs(transformation, inlier_pairs, rmse, rand);
}

void CombinationMetricEstimator::buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                   std::vector<InlierPair> &inlier_pairs,
                                                                   float &rmse, float &metric,
                                                                   UniformRandIntGenerator &rand) const {
    float metric_cs, metric_cp;
    float rmse_cp;
    std::vector<InlierPair> inlier_pairs_cp;
    correspondences_estimator.buildInlierPairsAndEstimateMetric(transformation, inlier_pairs, rmse, metric_cs, rand);
    closest_plane_estimator.buildInlierPairsAndEstimateMetric(transformation, inlier_pairs_cp, rmse_cp, metric_cp,
                                                              rand);
    metric = metric_cs * metric_cp;
}

void CombinationMetricEstimator::setCorrespondences(const pcl::CorrespondencesConstPtr &correspondences) {
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
    ScoreFunction score_function;
    if (parameters.score_id == METRIC_SCORE_MAE) {
        score_function = ScoreFunction::MAE;
    } else if (parameters.score_id == METRIC_SCORE_MSE) {
        score_function = ScoreFunction::MSE;
    } else if (parameters.score_id == METRIC_SCORE_EXP) {
        score_function = ScoreFunction::EXP;
    } else {
        if (parameters.score_id != METRIC_SCORE_CONSTANT) {
            PCL_WARN("[getMetricEstimator] score function %s isn't supported, constant will be used\n",
                     parameters.score_id.c_str());
        }
        score_function = ScoreFunction::Constant;
    }
    if (parameters.metric_id == METRIC_CLOSEST_PLANE) {
        return std::make_shared<ClosestPlaneMetricEstimator>(sparse, score_function);
    } else if (parameters.metric_id == METRIC_WEIGHTED_CLOSEST_PLANE) {
        return std::make_shared<WeightedClosestPlaneMetricEstimator>(parameters.weight_id, NORMAL_NR_POINTS, sparse,
                                                                     score_function);
    } else if (parameters.metric_id == METRIC_COMBINATION) {
        return std::make_shared<CombinationMetricEstimator>(sparse, score_function);
    } else if (parameters.metric_id != METRIC_CORRESPONDENCES) {
        PCL_WARN("[getMetricEstimator] metric estimator %s isn't supported, correspondences will be used\n",
                 parameters.metric_id.c_str());
    }
    return std::make_shared<CorrespondencesMetricEstimator>(score_function);
}