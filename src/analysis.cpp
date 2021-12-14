#include <fstream>
#include <filesystem>

#include <Eigen/Geometry>
#include <utility>

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

std::vector<MultivaluedCorrespondence> AlignmentAnalysis::getCorrectCorrespondences(
        const Eigen::Matrix4f &transformation_gt, float error_threshold, bool check_inlier
) {
    PointCloudT input_transformed;
    input_transformed.resize(src_->size());
    pcl::transformPointCloud(*src_, input_transformed, transformation_gt);

    std::set<int> inliers(inliers_.begin(), inliers_.end());
    std::vector<MultivaluedCorrespondence> correct_correspondences;
    for (const auto &correspondence: correspondences_) {
        int query_idx = correspondence.query_idx;
        if (!check_inlier || (check_inlier && inliers.find(query_idx) != inliers.end())) {
            int match_idx = correspondence.match_indices[0];
            PointT source_point(input_transformed.points[query_idx]);
            PointT target_point(tgt_->points[match_idx]);
            float e = pcl::L2_Norm(source_point.data, target_point.data, 3);
            if (e < error_threshold) {
                correct_correspondences.push_back(correspondence);
            }
        }
    }
    return correct_correspondences;
}

void AlignmentAnalysis::start(const Eigen::Matrix4f &transformation_gt, const std::string &testname) {
    float error_thr = parameters_.distance_thr_coef * parameters_.voxel_size;
    transformation_gt_ = transformation_gt;

    correct_correspondences_ = getCorrectCorrespondences(transformation_gt_, error_thr);
    inlier_count_ = inliers_.size();
    correct_inlier_count_ = countCorrectCorrespondences(transformation_gt_, error_thr, true);
    correspondence_count_ = correspondences_.size();
    correct_correspondence_count_ = correct_correspondences_.size();
    fitness_ = (float) inlier_count_ / (float) correspondence_count_;
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
            fout
                    << "version,descriptor,testname,fitness,rmse,correspondences,correct_correspondences,inliers,correct_inliers,";
            fout << "voxel_size,normal_radius_coef,feature_radius_coef,distance_thr_coef,edge_thr,";
            fout << "iteration,reciprocal,randomness,filter,threshold,n_random,r_err,t_err,pcd_err\n";
        }
        fout << VERSION << "," << parameters_.descriptor_id << "," << testname << "," << fitness_ << "," << rmse_ << ",";
        fout << correspondence_count_ << "," << correct_correspondence_count_ << ",";
        fout << inlier_count_ << "," << correct_inlier_count_ << ",";
        fout << parameters_.voxel_size << ",";
        fout << parameters_.normal_radius_coef << ",";
        fout << parameters_.feature_radius_coef << ",";
        fout << parameters_.distance_thr_coef << ",";
        fout << parameters_.edge_thr_coef << ",";
        fout << iterations_ << ",";
        fout << parameters_.reciprocal << ",";
        fout << parameters_.randomness << ",";
        auto func = getUniquenessFunction(parameters_.func_id);
        if (func != nullptr) {
            fout << parameters_.func_id << "," << UNIQUENESS_THRESHOLD << "," << N_RANDOM_FEATURES << ",";
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
    saveCorrespondences(src_, tgt_, correspondences_, transformation_gt_, testname);
    saveCorrespondences(src_, tgt_, correspondences_, transformation_gt_, testname, true);
    saveCorrespondenceDistances(src_, tgt_, correspondences_, transformation_gt_, parameters_.voxel_size, testname);
    saveColorizedPointCloud(src_, correspondences_, correct_correspondences_, inliers_, testname);

    pcl::transformPointCloud(*src_fullsize, *src_fullsize_aligned, transformation_);
    pcl::transformPointCloud(*src_fullsize, *src_fullsize_aligned_gt, transformation_gt_);
    pcl::io::savePLYFileBinary(constructPath(testname, "aligned"), *src_fullsize_aligned);
    pcl::io::savePLYFileBinary(constructPath(testname, "aligned_gt"), *src_fullsize_aligned_gt);
}
