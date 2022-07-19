#ifndef REGISTRATION_METRIC_H
#define REGISTRATION_METRIC_H

#include <memory>
#include <numeric>
#include <vector>
#include <Eigen/Core>

#include <pcl/types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>

#include "common.h"

#define SPARSE_POINTS_FRACTION 0.01f

class MetricEstimator {
public:
    using Ptr = std::shared_ptr<MetricEstimator>;
    using ConstPtr = std::shared_ptr<const MetricEstimator>;

    virtual float getInitialMetric() const = 0;

    virtual bool isBetter(float new_value, float old_value) const = 0;

    virtual void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                                  float &rmse, UniformRandIntGenerator &rand) const = 0;

    virtual void buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                                   std::vector<InlierPair> &inlier_pairs,
                                                   float &rmse, float &metric, UniformRandIntGenerator &rand) const = 0;

    virtual int estimateMaxIterations(const Eigen::Matrix4f &transformation, float confidence, int nr_samples) const;

    virtual void buildCorrectInlierPairs(const std::vector<InlierPair> &inlier_pairs,
                                         std::vector<InlierPair> &correct_inlier_pairs,
                                         const Eigen::Matrix4f &transformation_gt) const;

    virtual inline void setCorrespondences(const pcl::CorrespondencesConstPtr &correspondences) {
        correspondences_ = correspondences;
    }

    virtual inline void setSourceCloud(const PointNCloud::ConstPtr &src) {
        src_ = src;
    }

    virtual inline void setTargetCloud(const PointNCloud::ConstPtr &tgt) {
        tgt_ = tgt;
    }

    virtual inline void setInlierThreshold(float inlier_threshold) {
        inlier_threshold_ = inlier_threshold;
    }

    virtual std::string getClassName() const = 0;

protected:
    pcl::CorrespondencesConstPtr correspondences_;
    PointNCloud::ConstPtr src_, tgt_;
    float inlier_threshold_;
};

class CorrespondencesMetricEstimator : public MetricEstimator {
public:
    CorrespondencesMetricEstimator() = default;

    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline bool isBetter(float new_value, float old_value) const override {
        return new_value > old_value;
    }

    void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                          float &rmse, UniformRandIntGenerator &rand) const override;

    void buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                           std::vector<InlierPair> &inlier_pairs,
                                           float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    inline std::string getClassName() const override {
        return "CorrespondencesMetricEstimator";
    }
};

class ClosestPlaneMetricEstimator : public MetricEstimator {
public:
    explicit ClosestPlaneMetricEstimator(bool sparse = false) : sparse_(sparse) {};

    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline bool isBetter(float new_value, float old_value) const override {
        return new_value > old_value;
    }

    void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                          float &rmse, UniformRandIntGenerator &rand) const override;

    void buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                           std::vector<InlierPair> &inlier_pairs,
                                           float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    void setTargetCloud(const PointNCloud::ConstPtr &tgt) override;

    inline std::string getClassName() const override {
        return "ClosestPlaneMetricEstimator";
    }

protected:
    pcl::KdTreeFLANN<PointN> tree_tgt_;
    bool sparse_;
};

class WeightedClosestPlaneMetricEstimator : public MetricEstimator {
public:
    WeightedClosestPlaneMetricEstimator() = delete;

    WeightedClosestPlaneMetricEstimator(std::string weight_id, float curvature_radius, bool sparse = false)
            : weight_id_(std::move(weight_id)), curvature_radius_(curvature_radius), sparse_(sparse) {}


    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline bool isBetter(float new_value, float old_value) const override {
        return new_value > old_value;
    }

    void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                          float &rmse, UniformRandIntGenerator &rand) const override;

    void buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                           std::vector<InlierPair> &inlier_pairs,
                                           float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    void setSourceCloud(const PointNCloud::ConstPtr &src) override;

    void setTargetCloud(const PointNCloud::ConstPtr &tgt) override;

    inline std::string getClassName() const override {
        return "WeightedClosestPlaneMetricEstimator";
    }

protected:
    pcl::KdTreeFLANN<PointN> tree_tgt_;
    std::string weight_id_;
    std::vector<float> weights_;
    float curvature_radius_;
    float weights_sum_ = 0.f;
    bool sparse_;
};

class CombinationMetricEstimator : public MetricEstimator {
public:
    explicit CombinationMetricEstimator(bool sparse = false) : closest_plane_estimator(sparse) {}

    float getInitialMetric() const override {
        return 0.0;
    }

    inline bool isBetter(float new_value, float old_value) const override {
        return new_value > old_value;
    }

    void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                          float &rmse, UniformRandIntGenerator &rand) const override;

    void buildInlierPairsAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                           std::vector<InlierPair> &inlier_pairs,
                                           float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    void setCorrespondences(const pcl::CorrespondencesConstPtr &correspondences) override;

    void setSourceCloud(const PointNCloud::ConstPtr &src) override;

    void setTargetCloud(const PointNCloud::ConstPtr &tgt) override;

    void setInlierThreshold(float inlier_threshold) override;

    inline std::string getClassName() const override {
        return "CombinationMetricEstimator";
    }

protected:
    CorrespondencesMetricEstimator correspondences_estimator;
    ClosestPlaneMetricEstimator closest_plane_estimator;
};

MetricEstimator::Ptr getMetricEstimatorFromParameters(const AlignmentParameters &parameters, bool sparse = false);

#endif
