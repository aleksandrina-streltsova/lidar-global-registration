#include <fstream>
#include <filesystem>

#include <Eigen/Geometry>
#include <utility>

#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include "analysis.h"
#include "filter.h"

#define N_BINS 4

namespace fs = std::filesystem;

std::pair<float, float> calculate_rotation_and_translation_errors(const Eigen::Matrix4f &transformation,
                                                                  const Eigen::Matrix4f &transformation_gt) {
    Eigen::Matrix3f rotation_diff = transformation.block<3, 3>(0, 0).inverse() * transformation_gt.block<3, 3>(0, 0);
    Eigen::Vector3f translation_diff = transformation.block<3, 1>(0, 3) - transformation_gt.block<3, 1>(0, 3);
    float rotation_error = Eigen::AngleAxisf(rotation_diff).angle();
    float translation_error = translation_diff.norm();
    return {rotation_error, translation_error};
}

float calculate_point_cloud_mean_error(const PointNCloud::ConstPtr &pcd,
                                       const Eigen::Matrix4f &transformation,
                                       const Eigen::Matrix4f &transformation_gt) {
    PointNCloud::Ptr pcd_transformed(new PointNCloud);
    Eigen::Matrix4f transformation_diff = transformation.inverse() * transformation_gt;
    pcl::transformPointCloud(*pcd, *pcd_transformed, transformation_diff);

    float error = 0.f;
    for (int i = 0; i < pcd->size(); ++i) {
        error += pcl::L2_Norm(pcd->points[i].data, std::as_const(pcd_transformed->points[i].data), 3);
    }
    error /= pcd->size();
    return error;
}

float calculate_correspondence_uniformity(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                          const pcl::Correspondences &correct_correspondences,
                                          const AlignmentParameters &parameters,
                                          const Eigen::Matrix4f &transformation_gt) {
    PointNCloud::Ptr src_aligned(new PointNCloud);
    pcl::transformPointCloud(*src, *src_aligned, transformation_gt);
    pcl::KdTreeFLANN<PointN>::Ptr tree(new pcl::KdTreeFLANN<PointN>());
    tree->setInputCloud(tgt);

    pcl::Indices indices;
    std::vector<float> distances;
    // calculate bbox of overlapping area
    PointN min_point, max_point;
    float error_thr = parameters.distance_thr_coef * parameters.voxel_size;
    for (int i = 0; i < src_aligned->size(); ++i) {
        (float) tree->nearestKSearch(*src_aligned, i, 1, indices, distances);
        const PointN &p_src(src_aligned->points[i]), p_tgt(tgt->points[indices[0]]);
        if (std::sqrt(distances[0]) < error_thr && std::isfinite(p_src.normal_x) && std::isfinite(p_tgt.normal_x)) {
            min_point.x = std::min(min_point.x, p_src.x);
            min_point.y = std::min(min_point.y, p_src.y);
            min_point.z = std::min(min_point.z, p_src.z);
            max_point.x = std::max(max_point.x, p_src.x);
            max_point.y = std::max(max_point.y, p_src.y);
            max_point.z = std::max(max_point.z, p_src.z);
        }
    }
    int count[3][N_BINS][N_BINS]{0};

    for (auto const &corr: correct_correspondences) {
        const auto &point = src_aligned->points[corr.index_query];
        if (pointInBoundingBox(point, min_point, max_point)) {
            int bin[3];
            bin[0] = std::floor((point.x - min_point.x) / (max_point.x - min_point.x) * N_BINS);
            bin[1] = std::floor((point.y - min_point.y) / (max_point.y - min_point.y) * N_BINS);
            bin[2] = std::floor((point.z - min_point.z) / (max_point.z - min_point.z) * N_BINS);
            // count 3D points projected to YZ, ZX, XY and fallen in 2D bin
            for (int k = 0; k < 3; ++k) {
                count[k][bin[(k + 1) % 3]][bin[(k + 2) % 3]]++;
            }
        }
    }
    float entropy[3]{0.f};
    float n = correct_correspondences.size();
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < N_BINS; ++i) {
            for (int j = 0; j < N_BINS; ++j) {
                float p = (float) count[k][i][j] / n;
                if (p == 0.f) continue;
                entropy[k] -= p * std::log(p);
            }
        }
        entropy[k] /= std::log((float) (N_BINS * N_BINS));
    }
    return std::cbrt(entropy[0] * entropy[1] * entropy[2]);
}

float calculate_normal_difference(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                  const AlignmentParameters &parameters, const Eigen::Matrix4f &transformation_gt) {
    PointNCloud::Ptr src_aligned(new PointNCloud);
    pcl::transformPointCloud(*src, *src_aligned, transformation_gt);
    pcl::KdTreeFLANN<PointN>::Ptr tree(new pcl::KdTreeFLANN<PointN>());
    tree->setInputCloud(tgt);

    pcl::Indices indices;
    std::vector<float> distances;
    float error_thr = parameters.distance_thr_coef * parameters.voxel_size;
    float difference = 0.f;
    int n_points_overlap = 0;
    for (int i = 0; i < src_aligned->size(); ++i) {
        (float) tree->nearestKSearch(*src_aligned, i, 1, indices, distances);
        const PointN &p_src(src_aligned->points[i]), p_tgt(tgt->points[indices[0]]);
        if (std::sqrt(distances[0]) < error_thr && std::isfinite(p_src.normal_x) && std::isfinite(p_tgt.normal_x)) {
            float cos = std::clamp(p_src.normal_x * p_tgt.normal_x + p_src.normal_y * p_tgt.normal_y +
                                   p_src.normal_z * p_tgt.normal_z, -1.f, 1.f);
            difference += std::abs(std::acos(cos));
            n_points_overlap++;
        }
    }
    return difference / (float) n_points_overlap;
}

void buildCorrectCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                 const pcl::Correspondences &correspondences,
                                 pcl::Correspondences &correct_correspondences,
                                 const Eigen::Matrix4f &transformation_gt, float error_threshold) {
    correct_correspondences.clear();
    correct_correspondences.reserve(correspondences.size());

    PointNCloud input_transformed;
    input_transformed.resize(src->size());
    pcl::transformPointCloud(*src, input_transformed, transformation_gt);

    for (const auto &correspondence: correspondences) {
        PointN source_point(input_transformed.points[correspondence.index_query]);
        PointN target_point(tgt->points[correspondence.index_match]);
        float e = pcl::L2_Norm(source_point.data, target_point.data, 3);
        if (e < error_threshold) {
            correct_correspondences.push_back(correspondence);
        }
    }
}

AlignmentAnalysis::AlignmentAnalysis(const AlignmentParameters &parameters,
                                     const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                     const pcl::Correspondences &correspondences,
                                     int iterations, const Eigen::Matrix4f &transformation) {
    parameters_ = parameters;
    src_ = src;
    tgt_ = tgt;
    correspondences_ = correspondences;
    iterations_ = iterations;
    transformation_ = transformation;
    has_converged_ = true;
    metric_estimator_ = getMetricEstimator(parameters);
    metric_estimator_->setSourceCloud(src);
    metric_estimator_->setTargetCloud(tgt);
    metric_estimator_->setCorrespondences(correspondences);
    metric_estimator_->setInlierThreshold(parameters.distance_thr_coef * parameters.voxel_size);
}

void AlignmentAnalysis::start(const Eigen::Matrix4f &transformation_gt, const std::string &testname) {
    testname_ = testname;
    float error_thr = parameters_.distance_thr_coef * parameters_.voxel_size;
    transformation_gt_ = transformation_gt;

    buildCorrectCorrespondences(src_, tgt_, correspondences_, correct_correspondences_, transformation_gt_, error_thr);
    metric_estimator_->buildInlierPairsAndEstimateMetric(transformation_, inlier_pairs_, rmse_, fitness_);
    metric_estimator_->buildCorrectInlierPairs(inlier_pairs_, correct_inlier_pairs_, transformation_gt_);
    pcd_error_ = calculate_point_cloud_mean_error(src_, transformation_, transformation_gt_);
    normal_diff_ = calculate_normal_difference(src_, tgt_, parameters_, transformation_gt_);
    corr_uniformity_ = calculate_correspondence_uniformity(src_, tgt_, correct_correspondences_,
                                                           parameters_, transformation_gt_);
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
    pcl::console::print_info("correct inliers: %i/%i\n", correct_inlier_pairs_.size(), inlier_pairs_.size());
    pcl::console::print_info("correct correspondences: %i/%i\n",
                             correct_correspondences_.size(), correspondences_.size());
    pcl::console::print_info("rotation error: %0.7f\n", r_error_);
    pcl::console::print_info("translation error: %0.7f\n", t_error_);
    pcl::console::print_info("point cloud mean error: %0.7f\n", pcd_error_);
    pcl::console::print_info("normal mean difference: %0.7f\n", normal_diff_);
    pcl::console::print_info("uniformity of correct correspondences' distribution: %0.7f\n", corr_uniformity_);
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
            printAnalysisHeader(fout);
        }
        fout << *this;
        fout.close();
    } else {
        perror(("error while opening file " + filepath).c_str());
    }
}

void printAnalysisHeader(std::ostream &out) {
    out << "version,descriptor,testname,fitness,rmse,correspondences,correct_correspondences,inliers,correct_inliers,";
    out << "voxel_size,normal_radius_coef,feature_radius_coef,distance_thr_coef,edge_thr,";
    out << "iteration,matching,randomness,filter,threshold,n_random,r_err,t_err,pcd_err,use_normals,";
    out << "normal_diff,corr_uniformity,lrf,metric\n";
}

std::ostream &operator<<(std::ostream &stream, const AlignmentAnalysis &analysis) {
    stream << VERSION << "," << analysis.parameters_.descriptor_id << "," << analysis.testname_ << ","
           << analysis.fitness_ << "," << analysis.rmse_ << ",";
    stream << analysis.correspondences_.size() << "," << analysis.correct_correspondences_.size() << ",";
    stream << analysis.inlier_pairs_.size() << "," << analysis.correct_inlier_pairs_.size() << ",";
    stream << analysis.parameters_.voxel_size << ",";
    stream << analysis.parameters_.normal_radius_coef << ",";
    stream << analysis.parameters_.feature_radius_coef << ",";
    stream << analysis.parameters_.distance_thr_coef << ",";
    stream << analysis.parameters_.edge_thr_coef << ",";
    stream << analysis.iterations_ << ",";
    stream << analysis.parameters_.matching_id << ",";
    stream << analysis.parameters_.randomness << ",";
    auto func = getUniquenessFunction(analysis.parameters_.func_id);
    if (func != nullptr) {
        stream << analysis.parameters_.func_id << "," << UNIQUENESS_THRESHOLD << "," << N_RANDOM_FEATURES << ",";
    } else {
        stream << ",,,";
    }
    stream << analysis.r_error_ << "," << analysis.t_error_ << "," << analysis.pcd_error_ << ",";
    stream << analysis.parameters_.use_normals << "," << analysis.normal_diff_ << ",";
    stream << analysis.corr_uniformity_ << "," << analysis.parameters_.lrf_id << ",";
    stream << analysis.parameters_.metric_id << "\n";
    return stream;
}
