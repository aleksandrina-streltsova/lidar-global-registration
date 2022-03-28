#ifndef REGISTRATION_ANALYSIS_H
#define REGISTRATION_ANALYSIS_H

#include <utility>

#include <Eigen/Core>

#include "common.h"
#include "metric.h"

std::pair<float, float> calculate_rotation_and_translation_errors(const Eigen::Matrix4f &transformation,
                                                                  const Eigen::Matrix4f &transformation_gt);

float calculate_point_cloud_mean_error(const PointCloudTN::ConstPtr &pcd,
                                       const Eigen::Matrix4f &transformation, const Eigen::Matrix4f &transformation_gt);

float calculate_correspondence_uniformity(const PointCloudTN::ConstPtr &src, const PointCloudTN::ConstPtr &tgt,
                                          const std::vector<MultivaluedCorrespondence> &correct_correspondences,
                                          const AlignmentParameters &parameters,
                                          const Eigen::Matrix4f &transformation_gt);

float calculate_normal_difference(const PointCloudTN::ConstPtr &src, const PointCloudTN::ConstPtr &tgt,
                                  const AlignmentParameters &parameters, const Eigen::Matrix4f &transformation_gt);

class AlignmentAnalysis {
public:
    AlignmentAnalysis() {}

    AlignmentAnalysis(AlignmentParameters parameters, MetricEstimator::ConstPtr metric_estimator,
                      PointCloudTN::ConstPtr src, PointCloudTN::ConstPtr tgt, std::vector<InlierPair> inlier_pairs,
                      std::vector<MultivaluedCorrespondence> correspondences, float rmse,
                      int iterations, Eigen::Matrix4f transformation) : parameters_(std::move(parameters)),
                                                                        metric_estimator_(std::move(metric_estimator)),
                                                                        src_(std::move(src)), tgt_(std::move(tgt)),
                                                                        inlier_pairs_(std::move(inlier_pairs)),
                                                                        correspondences_(std::move(correspondences)),
                                                                        rmse_(rmse), iterations_(iterations),
                                                                        transformation_(std::move(transformation)),
                                                                        has_converged_(true) {}

    void start(const Eigen::Matrix4f &transformation_gt, const std::string &testname);

    void saveFilesForDebug(const PointCloudTN::Ptr &src_fullsize, const AlignmentParameters &parameters);

    inline bool alignmentHasConverged() const {
        return has_converged_;
    }

    friend std::ostream &operator<<(std::ostream &stream, const AlignmentAnalysis &analysis);

private:
    AlignmentParameters parameters_;
    MetricEstimator::ConstPtr metric_estimator_;
    int iterations_;
    PointCloudTN::ConstPtr src_, tgt_;
    Eigen::Matrix4f transformation_, transformation_gt_;
    std::vector<MultivaluedCorrespondence> correspondences_, correct_correspondences_;
    std::vector<InlierPair> inlier_pairs_, correct_inlier_pairs_;
    float fitness_, rmse_;
    float pcd_error_, r_error_, t_error_;
    float normal_diff_;
    float corr_uniformity_;
    std::string testname_;
    bool has_converged_ = false;

    void buildCorrectCorrespondences(std::vector<MultivaluedCorrespondence> &correct_correspondences,
                                     const Eigen::Matrix4f &transformation_gt, float error_threshold);

    void print();

    void save(const std::string &testname);

    void saveTransformation();
};

void printAnalysisHeader(std::ostream &out);

#endif
