#ifndef REGISTRATION_METRIC_H
#define REGISTRATION_METRIC_H

#include <memory>
#include <numeric>
#include <vector>
#include <Eigen/Core>

#include <pcl/types.h>
#include <pcl/kdtree/kdtree_flann.h>

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

    virtual inline void setCorrespondences(const std::vector<MultivaluedCorrespondence> &correspondences) {
        correspondences_ = correspondences;
    }

    virtual inline void setSourceCloud(const PointCloudTN::ConstPtr &src) {
        src_ = src;
    }

    virtual inline void setTargetCloud(const PointCloudTN::ConstPtr &tgt) {
        tgt_ = tgt;
    }

    virtual inline void setInlierThreshold(float inlier_threshold) {
        inlier_threshold_ = inlier_threshold;
    }

protected:
    std::vector<MultivaluedCorrespondence> correspondences_;
    PointCloudTN::ConstPtr src_, tgt_;
    float inlier_threshold_;
};

class CorrespondencesMetricEstimator : public MetricEstimator {
public:
    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline bool isBetter(float new_value, float old_value) const override {
        return new_value > old_value;
    }

    void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                          float &rmse) const override;

    void estimateMetric(const std::vector<InlierPair> &inlier_pairs, float &metric) const override;
};

class ClosestPointMetricEstimator : public MetricEstimator {
public:
    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline bool isBetter(float new_value, float old_value) const override {
        return new_value > old_value;
    }

    void buildInlierPairs(const Eigen::Matrix4f &transformation, std::vector<InlierPair> &inlier_pairs,
                          float &rmse) const override;

    void estimateMetric(const std::vector<InlierPair> &inlier_pairs, float &metric) const override;

    void setTargetCloud(const PointCloudTN::ConstPtr &tgt) override;

protected:
    pcl::KdTreeFLANN<PointTN> tree_tgt_;
};

MetricEstimator::Ptr getMetricEstimator(const std::string &metric_id);

#endif
