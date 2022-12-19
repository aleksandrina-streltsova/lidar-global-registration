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

enum ScoreFunction {
    Constant, MAE, MSE, EXP
};

class MetricEstimator {
public:
    using Ptr = std::shared_ptr<MetricEstimator>;
    using ConstPtr = std::shared_ptr<const MetricEstimator>;

    explicit MetricEstimator(ScoreFunction score_function = ScoreFunction::Constant)
            : score_function_(score_function) {};

    virtual float getInitialMetric() const = 0;

    virtual float getMinTolerableMetric() const = 0;

    virtual void buildInliers(const Eigen::Matrix4f &transformation, Correspondences &inliers,
                              float &rmse, UniformRandIntGenerator &rand) const = 0;

    virtual void buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                               Correspondences &inliers,
                                               float &rmse, float &metric, UniformRandIntGenerator &rand) const = 0;

    virtual int estimateMaxIterations(const Eigen::Matrix4f &transformation, float confidence, int nr_samples) const;

    virtual void buildCorrectInliers(const Correspondences &inliers,
                                     Correspondences &correct_inliers,
                                     const Eigen::Matrix4f &transformation_gt) const;

    virtual inline void setCorrespondences(const CorrespondencesConstPtr &correspondences) {
        correspondences_ = correspondences;
    }

    virtual inline void setSourceCloud(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &kps_src) {
        src_ = src;
        kps_src_ = kps_src;
    }

    virtual inline void setTargetCloud(const PointNCloud::ConstPtr &tgt, const PointNCloud::ConstPtr &kps_tgt) {
        tgt_ = tgt;
        kps_tgt_ = kps_tgt;
    }

    virtual std::string getClassName() const = 0;

protected:
    CorrespondencesConstPtr correspondences_;
    PointNCloud::ConstPtr src_, tgt_;
    PointNCloud::ConstPtr kps_src_, kps_tgt_;
    ScoreFunction score_function_;
};

class CorrespondencesMetricEstimator : public MetricEstimator {
public:
    explicit CorrespondencesMetricEstimator(ScoreFunction score_function = ScoreFunction::Constant)
            : MetricEstimator(score_function) {};

    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline float getMinTolerableMetric() const override {
        return 0.0;
    }

    void buildInliers(const Eigen::Matrix4f &transformation, Correspondences &inliers,
                      float &rmse, UniformRandIntGenerator &rand) const override;

    void buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                       Correspondences &inliers,
                                       float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    inline std::string getClassName() const override {
        return "CorrespondencesMetricEstimator";
    }
};

class UniformityMetricEstimator : public CorrespondencesMetricEstimator {
public:
    UniformityMetricEstimator() : CorrespondencesMetricEstimator(ScoreFunction::Constant) {};

    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline float getMinTolerableMetric() const override {
        return 0.3;
    }

    void setSourceCloud(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &kps_src) override;

    void buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                       Correspondences &inliers,
                                       float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    inline std::string getClassName() const override {
        return "CorrespondencesMetricEstimator";
    }

protected:
    std::pair<PointN, PointN> bbox_;
};

class ClosestPlaneMetricEstimator : public MetricEstimator {
public:
    explicit ClosestPlaneMetricEstimator(bool sparse = false, ScoreFunction score_function = ScoreFunction::Constant)
            : sparse_(sparse), MetricEstimator(score_function) {};

    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline float getMinTolerableMetric() const override {
        return 0.0;
    }

    void buildInliers(const Eigen::Matrix4f &transformation, Correspondences &inliers,
                      float &rmse, UniformRandIntGenerator &rand) const override;

    void buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                       Correspondences &inliers,
                                       float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    void setTargetCloud(const PointNCloud::ConstPtr &tgt, const PointNCloud::ConstPtr &kps_tgt) override;

    inline std::string getClassName() const override {
        return "ClosestPlaneMetricEstimator";
    }

protected:
    pcl::KdTreeFLANN<PointN> tree_tgt_;
    bool sparse_;
    float inlier_threshold_;
};

class WeightedClosestPlaneMetricEstimator : public MetricEstimator {
public:
    WeightedClosestPlaneMetricEstimator() = delete;

    WeightedClosestPlaneMetricEstimator(std::string weight_id, int nr_points, bool sparse = false,
                                        ScoreFunction score_function = ScoreFunction::Constant)
            : weight_id_(std::move(weight_id)), nr_points_(nr_points), sparse_(sparse),
              MetricEstimator(score_function) {}


    inline float getInitialMetric() const override {
        return 0.0;
    }

    inline float getMinTolerableMetric() const override {
        return 0.0;
    }

    void buildInliers(const Eigen::Matrix4f &transformation, Correspondences &inliers,
                      float &rmse, UniformRandIntGenerator &rand) const override;

    void buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                       Correspondences &inliers,
                                       float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    void setSourceCloud(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &kps_src) override;

    void setTargetCloud(const PointNCloud::ConstPtr &tgt, const PointNCloud::ConstPtr &kps_tgt) override;

    inline std::string getClassName() const override {
        return "WeightedClosestPlaneMetricEstimator";
    }

protected:
    pcl::KdTreeFLANN<PointN> tree_tgt_;
    std::string weight_id_;
    std::vector<float> weights_;
    int nr_points_;
    float weights_sum_ = 0.f, inlier_threshold_;
    bool sparse_;
};

class CombinationMetricEstimator : public MetricEstimator {
public:
    explicit CombinationMetricEstimator(bool sparse = false, ScoreFunction score_function = ScoreFunction::Constant)
            : closest_plane_estimator(sparse, score_function) {}

    float getInitialMetric() const override {
        return 0.0;
    }

    inline float getMinTolerableMetric() const override {
        return 0.0;
    }

    void buildInliers(const Eigen::Matrix4f &transformation, Correspondences &inliers,
                      float &rmse, UniformRandIntGenerator &rand) const override;

    void buildInliersAndEstimateMetric(const Eigen::Matrix4f &transformation,
                                       Correspondences &inliers,
                                       float &rmse, float &metric, UniformRandIntGenerator &rand) const override;

    void setCorrespondences(const CorrespondencesConstPtr &correspondences) override;

    void setSourceCloud(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &kps_src) override;

    void setTargetCloud(const PointNCloud::ConstPtr &tgt, const PointNCloud::ConstPtr &kps_tgt) override;

    inline std::string getClassName() const override {
        return "CombinationMetricEstimator";
    }

protected:
    CorrespondencesMetricEstimator correspondences_estimator;
    ClosestPlaneMetricEstimator closest_plane_estimator;
};

MetricEstimator::Ptr getMetricEstimatorFromParameters(const AlignmentParameters &parameters, bool sparse = false);

#endif
