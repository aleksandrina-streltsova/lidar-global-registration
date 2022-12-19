#include <unordered_set>

#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>

#include "metric.h"
#include "weights.h"
#include "analysis.h"

void buildClosestPlaneInliers(const PointNCloud &src,
                              const pcl::KdTreeFLANN<PointN> &tree_tgt,
                              const Eigen::Matrix4f &transformation, Correspondences &inliers,
                              float inlier_threshold, float &rmse, bool sparse, UniformRandIntGenerator &rand) {
    inliers.clear();
    inliers.reserve(src.size());
    rmse = 0.0f;
    const PointNCloud &tgt = *tree_tgt.getInputCloud();
    int n = (int) ((sparse ? SPARSE_POINTS_FRACTION : 1.f) * (float) src.size());
    float search_radius = DIST_TO_PLANE_COEFFICIENT * inlier_threshold;
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
        tree_tgt.radiusSearch(PointN(point_transformed.x(), point_transformed.y(), point_transformed.z()),
                                search_radius, nn_indices, nn_sqr_dists, 1);
        if (nn_indices.empty()) continue;
        nearest_point = tgt[nn_indices[0]].getVector3fMap();
        normal = tgt[nn_indices[0]].getNormalVector3fMap();
        dist_to_plane = std::fabs(normal.transpose() * (nearest_point - point_transformed));
        // Check if point is an inlier
        if (dist_to_plane < inlier_threshold) {
            // Update inliers and rmse
            inliers.push_back({(int) idx, nn_indices[0], dist_to_plane, inlier_threshold});
            rmse += dist_to_plane * dist_to_plane;
        }
    }

    // Calculate RMSE
    if (!inliers.empty())
        rmse = std::sqrt(rmse / static_cast<float>(inliers.size()));
    else
        rmse = std::numeric_limits<float>::max();
}

float calculateScore(const Correspondences &inliers, ScoreFunction score_function) {
    float score = 0.f, value;
    for (const auto &inlier: inliers) {
        switch (score_function) {
            case Constant:
                value = 1.f;
                break;
            case MAE:
                value = std::fabs(inlier.distance - inlier.threshold) / inlier.threshold;
                break;
            case MSE:
                value = (inlier.distance - inlier.threshold) * (inlier.distance - inlier.threshold) /
                        (inlier.threshold * inlier.threshold);
                break;
            case EXP:
                value = (std::exp(-inlier.distance * inlier.distance / (2 * inlier.threshold * inlier.threshold)));
                break;
        }
        score += value;
    }
    return score;
}

void MetricEstimator::buildCorrectInliers(const Correspondences &inliers,
                                          Correspondences &correct_inliers,
                                          const Eigen::Matrix4f &transformation_gt) const {
    correct_inliers.clear();
    correct_inliers.reserve(inliers.size());

    PointNCloud kps_src_transformed;
    kps_src_transformed.resize(src_->size());
    pcl::transformPointCloudWithNormals(*kps_src_, kps_src_transformed, transformation_gt);

    for (const auto &inlier: inliers) {
        PointN point_src(kps_src_transformed.points[inlier.index_query]);
        PointN point_tgt(kps_tgt_->points[inlier.index_match]);
        float e = pcl::L2_Norm(point_src.data, point_tgt.data, 3);
        if (e < inlier.threshold) {
            correct_inliers.push_back(inlier);
        }
    }
}

int MetricEstimator::estimateMaxIterations(const Eigen::Matrix4f &transformation,
                                           float confidence, int nr_samples) const {
    int count_supporting_corrs = 0;
    for (const auto &corr: *correspondences_) {
        Eigen::Vector4f point_src(0, 0, 0, 1);
        point_src.block(0, 0, 3, 1) = kps_src_->points[corr.index_query].getArray3fMap();
        Eigen::Vector4f point_tgt(0, 0, 0, 1);
        point_tgt.block(0, 0, 3, 1) = kps_tgt_->points[corr.index_match].getArray3fMap();
        float e = (transformation * point_src - point_tgt).block(0, 0, 3, 1).norm();
        if (e < corr.threshold) {
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

void CorrespondencesMetricEstimator::buildInliers(const Eigen::Matrix4f &transformation,
                                                  Correspondences &inliers,
                                                  float &rmse, UniformRandIntGenerator &rand) const {
    inliers.clear();
    inliers.reserve(correspondences_->size());
    rmse = 0.0f;

    // For each point from correspondences in the source dataset
    Eigen::Vector4f source_point, target_point;
    for (const auto &corr: *correspondences_) {
        int query_idx = corr.index_query;
        int match_idx = corr.index_match;
        source_point = transformation * kps_src_->points[query_idx].getVector4fMap();
        target_point = kps_tgt_->points[match_idx].getVector4fMap();

        // Calculate correspondence distance
        float dist = (source_point - target_point).norm();

        // Check if correspondence is an inlier
        if (dist < corr.threshold) {
            // Update inliers and rmse
            inliers.push_back({query_idx, match_idx, dist, corr.threshold});
            rmse += dist * dist;
        }
    }

    // Calculate RMSE
    if (!inliers.empty())
        rmse = std::sqrt(rmse / static_cast<float>(inliers.size()));
    else
        rmse = std::numeric_limits<float>::max();
}

void CorrespondencesMetricEstimator::buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                   Correspondences &inliers,
                                                                   float &rmse, float &metric,
                                                                   UniformRandIntGenerator &rand) const {
    buildInliers(transformation, inliers, rmse, rand);
    float score = calculateScore(inliers, this->score_function_);
    metric = score / (float) correspondences_->size();
}

void UniformityMetricEstimator::setSourceCloud(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &kps_src) {
    MetricEstimator::setSourceCloud(src, kps_src);
    bbox_ = calculateBoundingBox<PointN>(src_);
}

void UniformityMetricEstimator::buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                              Correspondences &inliers,
                                                              float &rmse, float &metric,
                                                              UniformRandIntGenerator &rand) const {
    buildInliers(transformation, inliers, rmse, rand);
    if (inliers.empty()) metric = 0.0;
    else metric = calculateCorrespondenceUniformity(src_, bbox_, inliers);
}

void ClosestPlaneMetricEstimator::setTargetCloud(const PointNCloud::ConstPtr &tgt, const PointNCloud::ConstPtr &kps_tgt) {
    MetricEstimator::setTargetCloud(tgt, kps_tgt);
    tree_tgt_.setInputCloud(tgt);
    inlier_threshold_ = calculatePointCloudDensity(tgt_);
}

void ClosestPlaneMetricEstimator::buildInliers(const Eigen::Matrix4f &transformation,
                                               Correspondences &inliers,
                                               float &rmse, UniformRandIntGenerator &rand) const {
    buildClosestPlaneInliers(*src_, tree_tgt_, transformation, inliers, inlier_threshold_, rmse, sparse_, rand);
}

void ClosestPlaneMetricEstimator::buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                Correspondences &inliers,
                                                                float &rmse, float &metric,
                                                                UniformRandIntGenerator &rand) const {
    buildInliers(transformation, inliers, rmse, rand);
    float score = calculateScore(inliers, this->score_function_);
    metric = score / ((sparse_ ? SPARSE_POINTS_FRACTION : 1.f) * (float) src_->size());
}

void WeightedClosestPlaneMetricEstimator::buildInliers(const Eigen::Matrix4f &transformation,
                                                       Correspondences &inliers,
                                                       float &rmse, UniformRandIntGenerator &rand) const {
    buildClosestPlaneInliers(*src_, tree_tgt_, transformation, inliers, inlier_threshold_, rmse, sparse_, rand);
}

void WeightedClosestPlaneMetricEstimator::buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                                        Correspondences &inliers,
                                                                        float &rmse, float &metric,
                                                                        UniformRandIntGenerator &rand) const {
    buildInliers(transformation, inliers, rmse, rand);
    float score = calculateScore(inliers, this->score_function_);
    metric = score / ((sparse_ ? SPARSE_POINTS_FRACTION : 1.f) * weights_sum_);
}

void WeightedClosestPlaneMetricEstimator::setSourceCloud(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &kps_src) {
    MetricEstimator::setSourceCloud(src, kps_src);
    auto weight_function = getWeightFunction(weight_id_);
    weights_ = weight_function(nr_points_, src_);
    weights_sum_ = 0.f;
    for (float weight: weights_) {
        weights_sum_ += weight;
    }
}

void WeightedClosestPlaneMetricEstimator::setTargetCloud(const PointNCloud::ConstPtr &tgt, const PointNCloud::ConstPtr &kps_tgt) {
    MetricEstimator::setTargetCloud(tgt, kps_tgt);
    tree_tgt_.setInputCloud(tgt);
    inlier_threshold_ = calculatePointCloudDensity(tgt_);
}

void CombinationMetricEstimator::buildInliers(const Eigen::Matrix4f &transformation,
                                              Correspondences &inliers, float &rmse,
                                              UniformRandIntGenerator &rand) const {
    correspondences_estimator.buildInliers(transformation, inliers, rmse, rand);
}

void CombinationMetricEstimator::buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                               Correspondences &inliers,
                                                               float &rmse, float &metric,
                                                               UniformRandIntGenerator &rand) const {
    float metric_cs, metric_cp;
    float rmse_cp;
    Correspondences inliers_cp;
    correspondences_estimator.buildInliersAndEstimateMetric(transformation, inliers, rmse, metric_cs, rand);
    closest_plane_estimator.buildInliersAndEstimateMetric(transformation, inliers_cp, rmse_cp, metric_cp,
                                                          rand);
    metric = metric_cs * metric_cp;
}

void CombinationMetricEstimator::setCorrespondences(const CorrespondencesConstPtr &correspondences) {
    correspondences_ = correspondences;
    correspondences_estimator.setCorrespondences(correspondences);
    closest_plane_estimator.setCorrespondences(correspondences);
}

void CombinationMetricEstimator::setSourceCloud(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &kps_src) {
    MetricEstimator::setSourceCloud(src, kps_src);
    correspondences_estimator.setSourceCloud(src, kps_src);
    closest_plane_estimator.setSourceCloud(src, kps_src);
}

void CombinationMetricEstimator::setTargetCloud(const PointNCloud::ConstPtr &tgt, const PointNCloud::ConstPtr &kps_tgt) {
    MetricEstimator::setTargetCloud(tgt, kps_tgt);
    correspondences_estimator.setTargetCloud(tgt, kps_tgt);
    closest_plane_estimator.setTargetCloud(tgt, kps_tgt);
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
    if (parameters.metric_id == METRIC_UNIFORMITY) {
        return std::make_shared<UniformityMetricEstimator>();
    } else if (parameters.metric_id == METRIC_CLOSEST_PLANE) {
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