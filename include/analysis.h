#ifndef REGISTRATION_ANALYSIS_H
#define REGISTRATION_ANALYSIS_H

#include <utility>

#include <Eigen/Core>

#include "common.h"
#include "metric.h"

std::pair<float, float> calculate_rotation_and_translation_errors(const Eigen::Matrix4f &transformation,
                                                                  const Eigen::Matrix4f &transformation_gt);

float calculate_point_cloud_mean_error(const PointNCloud::ConstPtr &pcd,
                                       const Eigen::Matrix4f &transformation, const Eigen::Matrix4f &transformation_gt);

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

    AlignmentAnalysis(const AlignmentParameters &parameters,
                      const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                      const pcl::Correspondences &correspondences,
                      int iterations, const Eigen::Matrix4f &transformation);

    void start(const Eigen::Matrix4f &transformation_gt, const std::string &testname);

    inline bool alignmentHasConverged() const {
        return has_converged_;
    }

    inline Eigen::Matrix4f getTransformation() const {
        return transformation_;
    }

    friend std::ostream &operator<<(std::ostream &stream, const AlignmentAnalysis &analysis);

private:
    AlignmentParameters parameters_;
    MetricEstimator::Ptr metric_estimator_;
    int iterations_;
    PointNCloud::ConstPtr src_, tgt_;
    Eigen::Matrix4f transformation_, transformation_gt_;
    pcl::Correspondences correspondences_, correct_correspondences_;
    std::vector<InlierPair> inlier_pairs_, correct_inlier_pairs_;
    float fitness_, rmse_;
    float pcd_error_, r_error_, t_error_;
    float normal_diff_;
    float corr_uniformity_;
    std::string testname_;
    bool has_converged_ = false;

    void print();

    void save(const std::string &testname);
};

void printAnalysisHeader(std::ostream &out);

#endif
