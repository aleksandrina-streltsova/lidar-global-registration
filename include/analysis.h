#ifndef REGISTRATION_ANALYSIS_H
#define REGISTRATION_ANALYSIS_H

#include <utility>

#include <Eigen/Core>

#include "common.h"
#include "metric.h"

std::pair<float, float> calculate_rotation_and_translation_errors(const Eigen::Matrix4f &transformation,
                                                                  const Eigen::Matrix4f &transformation_gt);

float calculate_point_cloud_rmse(const PointNCloud::ConstPtr &pcd,
                                 const Eigen::Matrix4f &transformation, const Eigen::Matrix4f &transformation_gt);

float calculate_overlap_rmse(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                             const Eigen::Matrix4f &transformation,
                             const Eigen::Matrix4f &transformation_gt,
                             float inlier_threshold);

float calculate_correspondence_uniformity(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                          const pcl::Correspondences &correct_correspondences,
                                          const AlignmentParameters &parameters,
                                          const Eigen::Matrix4f &transformation_gt);

float calculate_normal_difference(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                  const AlignmentParameters &parameters, const Eigen::Matrix4f &transformation_gt);

void buildCorrectCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                 const pcl::Correspondences &correspondences,
                                 pcl::Correspondences &correct_correspondences,
                                 const Eigen::Matrix4f &transformation_gt, float error_threshold);

class AlignmentAnalysis {
public:
    AlignmentAnalysis() {}

    AlignmentAnalysis(AlignmentResult result, AlignmentParameters parameters);

    void start(const Eigen::Matrix4f &transformation_gt, const std::string &testname);

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
        return result_.time;
    }

    inline MetricEstimator::Ptr getMetricEstimator() {
        return metric_estimator_;
    }

    friend std::ostream &operator<<(std::ostream &stream, const AlignmentAnalysis &analysis);

private:
    PointNCloud::ConstPtr src_, tgt_;
    AlignmentParameters parameters_;
    AlignmentResult result_;
    MetricEstimator::Ptr metric_estimator_;
    Eigen::Matrix4f transformation_, transformation_gt_;
    pcl::Correspondences correspondences_, correct_correspondences_;
    std::vector<InlierPair> inlier_pairs_, correct_inlier_pairs_;
    float fitness_, rmse_;
    float pcd_error_, overlap_error_, r_error_, t_error_;
    float normal_diff_;
    float corr_uniformity_;
    std::string testname_;

    void print();

    void save(const std::string &testname);
};

void printAnalysisHeader(std::ostream &out);

#endif
