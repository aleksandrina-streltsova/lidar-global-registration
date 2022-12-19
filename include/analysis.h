#ifndef REGISTRATION_ANALYSIS_H
#define REGISTRATION_ANALYSIS_H

#include <utility>

#include <Eigen/Core>

#include "common.h"
#include "metric.h"

std::pair<float, float> calculateRotationAndTranslationDifferences(const Eigen::Matrix4f &tn1,
                                                                   const Eigen::Matrix4f &tn2);

float calculatePointCloudRmse(const PointNCloud::ConstPtr &pcd,
                              const Eigen::Matrix4f &transformation, const Eigen::Matrix4f &transformation_gt);

float calculateOverlapRmse(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                           const Eigen::Matrix4f &transformation,
                           const Eigen::Matrix4f &transformation_gt,
                           float inlier_threshold);

float calculateCorrespondenceUniformity(const PointNCloud::ConstPtr &src,
                                        const Correspondences &correct_correspondences);

float calculateCorrespondenceUniformity(const PointNCloud::ConstPtr &src, const std::pair<PointN, PointN> &bbox,
                                        const Correspondences &correct_correspondences);

float calculateNormalDifference(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                float distance_thr, const Eigen::Matrix4f &transformation_gt);

std::pair<int, float>
getReproducedKeyPointsAndCalculateRmse(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                       const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                                       float distance_thr, const Eigen::Matrix4f &transformation_gt);

void buildCorrectCorrespondences(const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                                 const Correspondences &correspondences,
                                 Correspondences &correct_correspondences,
                                 const Eigen::Matrix4f &transformation_gt);

class AlignmentAnalysis {
public:
    AlignmentAnalysis() {}

    AlignmentAnalysis(AlignmentResult result, AlignmentParameters parameters);

    void start(const std::optional<Eigen::Matrix4f> &transformation_gt, const std::string &testname);

    inline bool alignmentHasConverged() const {
        return result_.converged;
    }

    inline Eigen::Matrix4f getTransformation() const {
        return transformation_;
    }

    inline float getRotationError() const {
        return r_error_;
    }

    inline float getTranslationError() const {
        return t_error_;
    }

    inline float getOverlapError() const {
        return overlap_error_;
    }

    inline float getPointCloudError() const {
        return pcd_error_;
    }

    inline float getRunningTime() const {
        return result_.time_cs + result_.time_te;
    }

    inline MetricEstimator::Ptr getMetricEstimator() {
        return metric_estimator_;
    }

    friend std::ostream &operator<<(std::ostream &stream, const AlignmentAnalysis &analysis);

private:
    PointNCloud::ConstPtr src_, tgt_;
    PointNCloud::ConstPtr kps_src_, kps_tgt_;
    AlignmentParameters parameters_;
    AlignmentResult result_;
    MetricEstimator::Ptr metric_estimator_;
    Eigen::Matrix4f transformation_;
    std::optional<Eigen::Matrix4f> transformation_gt_;
    Correspondences correspondences_, correct_correspondences_;
    Correspondences inliers_, correct_inliers_;
    float metric_{std::numeric_limits<float>::quiet_NaN()}, rmse_{std::numeric_limits<float>::quiet_NaN()};
    float pcd_error_{std::numeric_limits<float>::quiet_NaN()}, overlap_error_{std::numeric_limits<float>::quiet_NaN()};
    float r_error_{std::numeric_limits<float>::quiet_NaN()}, t_error_{std::numeric_limits<float>::quiet_NaN()};
    float normal_diff_{std::numeric_limits<float>::quiet_NaN()};
    float corr_uniformity_{std::numeric_limits<float>::quiet_NaN()};
    float overlap_{std::numeric_limits<float>::quiet_NaN()}, overlap_area_{std::numeric_limits<float>::quiet_NaN()};
    std::string testname_;

    void print();

    void save(const std::string &testname);
};

void printAnalysisHeader(std::ostream &out);

#endif
