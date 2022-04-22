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

class MetricEstimator {
public:
    using Ptr = std::shared_ptr<MetricEstimator>;
    using ConstPtr = std::shared_ptr<const MetricEstimator>;

    virtual float getInitialMetric() const = 0;

    virtual bool isBetter(float new_value, float old_value) const = 0;

    virtual void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                                  float &rmse) const = 0;

    virtual void estimateMetric(const std::vector<InlierPair> &inlier_pairs, float &metric) const = 0;

    virtual int estimateMaxIterations(const Eigen::Matrix4f &transformation, float confidence, int nr_samples) const;

    virtual void buildCorrectInlierPairs(const std::vector<InlierPair> &inlier_pairs,
                                         std::vector<InlierPair> &correct_inlier_pairs,
                                         const Eigen::Matrix4f &transformation_gt) const;

    virtual inline void setCorrespondences(const pcl::Correspondences &correspondences) {
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

    virtual std::string getClassName() = 0;

protected:
    pcl::Correspondences correspondences_;
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
                          float &rmse) const override;

    void estimateMetric(const std::vector<InlierPair> &inlier_pairs, float &metric) const override;

    inline std::string getClassName() override {
        return "CorrespondencesMetricEstimator";
    }
};

class ClosestPointMetricEstimator : public MetricEstimator {
public:
    explicit ClosestPointMetricEstimator(std::vector<float> weights) : weights_(std::move(weights)) {}

    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline bool isBetter(float new_value, float old_value) const override {
        return new_value > old_value;
    }

    void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                          float &rmse) const override;

    void estimateMetric(const std::vector<InlierPair> &inlier_pairs, float &metric) const override;

    void setSourceCloud(const PointNCloud::ConstPtr &src) override;

    void setTargetCloud(const PointNCloud::ConstPtr &tgt) override;

    inline std::string getClassName() override {
        return "ClosestPointMetricEstimator";
    }

protected:
    pcl::KdTreeFLANN<PointN> tree_tgt_;
    std::vector<float> weights_;
    bool weighted_ = false;
};

MetricEstimator::Ptr getMetricEstimator(const std::string &metric_id, const std::vector<float> &weights);

#endif
