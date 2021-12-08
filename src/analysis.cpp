#include <fstream>
#include <filesystem>

#include <Eigen/Geometry>

#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>

#include "analysis.h"
#include "filter.h"

std::pair<float, float> calculate_rotation_and_translation_errors(const Eigen::Matrix4f &transformation,
                                                                  const Eigen::Matrix4f &transformation_gt) {
    Eigen::Matrix3f rotation_diff = transformation.block<3, 3>(0, 0).inverse() * transformation_gt.block<3, 3>(0, 0);
    Eigen::Vector3f translation_diff = transformation.block<3, 1>(0, 3) - transformation_gt.block<3, 1>(0, 3);
    float rotation_error = Eigen::AngleAxisf(rotation_diff).angle();
    float translation_error = translation_diff.norm();
    return {rotation_error, translation_error};
}

float calculate_point_cloud_mean_error(const PointCloudT::ConstPtr &pcd,
                                       const Eigen::Matrix4f &transformation,
                                       const Eigen::Matrix4f &transformation_gt) {
    PointCloudT::Ptr pcd_transformed(new PointCloudT);
    Eigen::Matrix4f transformation_diff = transformation.inverse() * transformation_gt;
    pcl::transformPointCloud(*pcd, *pcd_transformed, transformation_diff);

    float error = 0.f;
    for (int i = 0; i < pcd->size(); ++i) {
        error += pcl::L2_Norm(pcd->points[i].data, std::as_const(pcd_transformed->points[i].data), 3);
    }
    error /= pcd->size();
    return error;
}

AlignmentAnalysis::AlignmentAnalysis(const SampleConsensusPrerejectiveOMP<PointT, PointT, FeatureT> &align,
                                     const YamlConfig &config) : align_(align) {
    src_ = align_.getInputSource();
    tgt_ = align_.getInputTarget();
    voxel_size_ = config.get<float>("voxel_size").value();
    edge_thr_coef_ = config.get<float>("edge_thr").value();
    distance_thr_coef_ = config.get<float>("distance_thr_coef").value();
    normal_radius_coef_ = config.get<float>("normal_radius_coef").value();
    feature_radius_coef_ = config.get<float>("feature_radius_coef").value();
    iterations_ = align_.getRANSACIterations();
    reciprocal_ = config.get<bool>("reciprocal").value();
    randomness_ = config.get<int>("randomness").value();
    func_id_ = config.get<std::string>("filter", "");
    transformation_ = align_.getFinalTransformation();
}

void AlignmentAnalysis::start(const Eigen::Matrix4f &transformation_gt, const std::string &testname) {
    float error_thr = distance_thr_coef_ * voxel_size_;
    transformation_gt_ = transformation_gt;

    correct_correspondences_ = align_.getCorrectCorrespondences(transformation_gt_, error_thr);
    inlier_count_ = align_.getInliers().size();
    correct_inlier_count_ = align_.countCorrectCorrespondences(transformation_gt_, error_thr, true);
    correspondence_count_ = align_.getCorrespondences().size();
    correct_correspondence_count_ = correct_correspondences_.size();
    fitness_ = (float) inlier_count_ / (float) correspondence_count_;
    rmse_ = align_.getRMSEScore();
    pcd_error_ = calculate_point_cloud_mean_error(src_, transformation_, transformation_gt_);
    std::tie(r_error_, t_error_) = calculate_rotation_and_translation_errors(transformation_, transformation_gt_);

    print();
    save(testname);
}

void AlignmentAnalysis::print() {
    // Print results
    printf("\n");
    printTransformation(transformation_);
    printTransformation(transformation_gt_);
    pcl::console::print_info("fitness: %0.7f\n", fitness_);
    pcl::console::print_info("inliers_rmse: %0.7f\n", rmse_);
    pcl::console::print_info("inliers: %i/%i\n", inlier_count_, correspondence_count_);
    pcl::console::print_info("correct inliers: %i/%i\n", correct_inlier_count_, inlier_count_);
    pcl::console::print_info("correct correspondences: %i/%i\n",
                             correct_correspondence_count_, correspondence_count_);
    pcl::console::print_info("rotation error: %0.7f\n", r_error_);
    pcl::console::print_info("translation error: %0.7f\n", t_error_);
    pcl::console::print_info("point cloud mean error: %0.7f\n", pcd_error_);
}

void AlignmentAnalysis::save(const std::string &testname) {
    // Save test parameters and results
    std::string filepath = constructPath("test", "results", "csv", false);
    bool file_exists = std::filesystem::exists(filepath);
    std::fstream fout;
    if (!file_exists) {
        fout.open(filepath, std::ios_base::out);
    } else {
        fout.open(filepath, std::ios_base::app);
    }
    if (fout.is_open()) {
        if (!file_exists) {
            fout << "version,testname,fitness,rmse,correspondences,correct_correspondences,inliers,correct_inliers,";
            fout << "voxel_size,normal_radius_coef,feature_radius_coef,distance_thr_coef,edge_thr,";
            fout << "iteration,reciprocal,randomness,filter,threshold,n_random,r_err,t_err,pcd_err\n";
        }
        fout << VERSION << "," << testname << "," << fitness_ << "," << rmse_ << ",";
        fout << correspondence_count_ << "," << correct_correspondence_count_ << ",";
        fout << inlier_count_ << "," << correct_inlier_count_ << ",";
        fout << voxel_size_ << ",";
        fout << normal_radius_coef_ << ",";
        fout << feature_radius_coef_ << ",";
        fout << distance_thr_coef_ << ",";
        fout << edge_thr_coef_ << ",";
        fout << iterations_ << ",";
        fout << reciprocal_ << ",";
        fout << randomness_ << ",";
        auto func = getUniquenessFunction(func_id_);
        if (func != nullptr) {
            fout << func_id_ << "," << UNIQUENESS_THRESHOLD << "," << N_RANDOM_FEATURES << ",";
        } else {
            fout << ",,,";
        }
        fout << r_error_ << "," << t_error_ << "," << pcd_error_ << "\n";
    } else {
        perror(("error while opening file " + filepath).c_str());
    }
}

void AlignmentAnalysis::saveFilesForDebug(const PointCloudT::Ptr &src_fullsize, const std::string &testname) {
    PointCloudT::Ptr src_fullsize_aligned(new PointCloudT), src_fullsize_aligned_gt(new PointCloudT);
    saveCorrespondences(src_, tgt_, align_.getCorrespondences(), transformation_gt_, testname);
    saveCorrespondences(src_, tgt_, align_.getCorrespondences(), transformation_gt_, testname, true);
    saveCorrespondenceDistances(src_, tgt_, align_.getCorrespondences(), transformation_gt_, voxel_size_, testname);
    saveColorizedPointCloud(src_, align_.getCorrespondences(), correct_correspondences_, align_.getInliers(), testname);

    pcl::transformPointCloud(*src_fullsize, *src_fullsize_aligned, transformation_);
    pcl::transformPointCloud(*src_fullsize, *src_fullsize_aligned_gt, transformation_gt_);
    pcl::io::savePLYFileBinary(constructPath(testname, "aligned"), *src_fullsize_aligned);
    pcl::io::savePLYFileBinary(constructPath(testname, "aligned_gt"), *src_fullsize_aligned_gt);
}
