#ifndef REGISTRATION_ANALYSIS_H
#define REGISTRATION_ANALYSIS_H

#include <utility>

#include <Eigen/Core>

#include "common.h"

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

    AlignmentAnalysis(AlignmentParameters parameters,
                      PointCloudTN::ConstPtr src, PointCloudTN::ConstPtr tgt, pcl::Indices inliers,
                      std::vector<MultivaluedCorrespondence> correspondences, float rmse,
                      int iterations, Eigen::Matrix4f transformation) : parameters_(std::move(parameters)),
                                                                        src_(std::move(src)), tgt_(std::move(tgt)),
                                                                        inliers_(std::move(inliers)),
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
    int iterations_;
    PointCloudTN::ConstPtr src_, tgt_;
    Eigen::Matrix4f transformation_, transformation_gt_;
    std::vector<MultivaluedCorrespondence> correct_correspondences_;
    int inlier_count_, correct_inlier_count_;
    int correspondence_count_, correct_correspondence_count_;
    float fitness_, rmse_;
    float pcd_error_, r_error_, t_error_;
    float normal_diff_;
    float corr_uniformity_;
    std::vector<MultivaluedCorrespondence> correspondences_;
    pcl::Indices inliers_;
    std::string testname_;
    bool has_converged_ = false;

    std::vector<MultivaluedCorrespondence> getCorrectCorrespondences(const Eigen::Matrix4f &transformation_gt,
                                                                     float error_threshold, bool check_inlier = false);

    inline int countCorrectCorrespondences(const Eigen::Matrix4f &transformation_gt,
                                           float error_threshold, bool check_inlier = false) {
        return getCorrectCorrespondences(transformation_gt, error_threshold, check_inlier).size();
    };

    void print();

    void save(const std::string &testname);
};

void printAnalysisHeader(std::ostream &out);

#endif
