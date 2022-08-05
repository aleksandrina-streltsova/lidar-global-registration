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

inline float dist2(const PointN &p1, const PointN &p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
};

float calculate_point_cloud_rmse(const PointNCloud::ConstPtr &pcd,
                                 const Eigen::Matrix4f &transformation,
                                 const Eigen::Matrix4f &transformation_gt) {
    PointNCloud pcd_transformed;
    Eigen::Matrix4f transformation_diff = transformation.inverse() * transformation_gt;
    pcl::transformPointCloud(*pcd, pcd_transformed, transformation_diff);

    float rmse = 0.f;
    for (int i = 0; i < pcd->size(); ++i) {
        rmse += dist2(pcd->points[i], pcd_transformed.points[i]);
    }
    rmse = std::sqrt(rmse / (float) pcd->size());
    return rmse;
}

float calculate_overlap_rmse(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                             const Eigen::Matrix4f &transformation,
                             const Eigen::Matrix4f &transformation_gt,
                             float inlier_threshold) {
    PointNCloud src_aligned, src_aligned_gt;
    pcl::transformPointCloud(*src, src_aligned, transformation);
    pcl::transformPointCloud(*src, src_aligned_gt, transformation_gt);

    pcl::KdTreeFLANN<PointN> tree_tgt;
    tree_tgt.setInputCloud(tgt);

    int overlap_size = 0;
    float rmse = 0.f, dist_to_plane;
    PointN nearest_point;
    pcl::Indices nn_indices;
    std::vector<float> nn_sqr_dists;
    for (int i = 0; i < src->size(); ++i) {
        tree_tgt.nearestKSearch(src_aligned_gt[i], 1, nn_indices, nn_sqr_dists);
        if (nn_sqr_dists[0] < inlier_threshold * inlier_threshold) {    // ith point in overlap
            nearest_point = tgt->points[nn_indices[0]];
            dist_to_plane = std::fabs(nearest_point.getNormalVector3fMap().transpose() *
                                      (nearest_point.getVector3fMap() - src_aligned[i].getVector3fMap()));
            dist_to_plane = std::isfinite(dist_to_plane) ? dist_to_plane : nn_sqr_dists[0];
            rmse += dist_to_plane * dist_to_plane;
            overlap_size++;
        }
    }
    if (overlap_size != 0) {
        rmse = std::sqrt(rmse / (float) overlap_size);
    } else {
        rmse = std::numeric_limits<float>::quiet_NaN();
    }
    return rmse;
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
    float error_thr = parameters.distance_thr;
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
    float error_thr = parameters.distance_thr;
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

AlignmentAnalysis::AlignmentAnalysis(AlignmentResult result, AlignmentParameters parameters)
        : src_(result.src), tgt_(result.tgt), parameters_(std::move(parameters)), result_(std::move(result)) {
    transformation_ = result_.transformation;
    metric_estimator_ = getMetricEstimatorFromParameters(parameters_);
    metric_estimator_->setSourceCloud(src_);
    metric_estimator_->setTargetCloud(tgt_);
    metric_estimator_->setCorrespondences(result_.correspondences);
    metric_estimator_->setInlierThreshold(parameters_.distance_thr);
    correspondences_ = *result_.correspondences;
}

void AlignmentAnalysis::start(const std::optional<Eigen::Matrix4f> &transformation_gt, const std::string &testname) {
    testname_ = testname;
    UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
    float error_thr = parameters_.distance_thr;
    transformation_gt_ = transformation_gt;
    metric_estimator_->buildInlierPairsAndEstimateMetric(transformation_, inlier_pairs_, rmse_, fitness_, rand);

    if (transformation_gt_.has_value()) {
        buildCorrectCorrespondences(src_, tgt_, correspondences_, correct_correspondences_,
                                    transformation_gt_.value(), error_thr);
        metric_estimator_->buildCorrectInlierPairs(inlier_pairs_, correct_inlier_pairs_, transformation_gt_.value());
        pcd_error_ = calculate_point_cloud_rmse(src_, transformation_, transformation_gt_.value());
        overlap_error_ = calculate_overlap_rmse(src_, tgt_, transformation_, transformation_gt_.value(), error_thr);
        normal_diff_ = calculate_normal_difference(src_, tgt_, parameters_, transformation_gt_.value());
        corr_uniformity_ = calculate_correspondence_uniformity(src_, tgt_, correct_correspondences_,
                                                               parameters_, transformation_gt_.value());
        std::tie(r_error_, t_error_) = calculate_rotation_and_translation_errors(transformation_, transformation_gt_.value());
    }
    print();
    save(testname);
}

void AlignmentAnalysis::print() {
    // Print results
    printf("\n Estimated transformation:\n");
    printTransformation(transformation_);
    if (transformation_gt_.has_value()) {
        printf("\n Ground truth transformation:\n");
        printTransformation(transformation_gt_.value());
    }
    pcl::console::print_info("fitness: %0.7f\n", fitness_);
    pcl::console::print_info("inliers_rmse: %0.7f\n", rmse_);
    if (transformation_gt_.has_value()) {
        pcl::console::print_info("correct inliers: %i/%i\n", correct_inlier_pairs_.size(), inlier_pairs_.size());
        pcl::console::print_info("correct correspondences: %i/%i\n",
                                 correct_correspondences_.size(), correspondences_.size());
        pcl::console::print_info("rotation error (deg): %0.7f\n", 180.0 / M_PI * r_error_);
        pcl::console::print_info("translation error: %0.7f\n", t_error_);
        pcl::console::print_info("point cloud error: %0.7f\n", pcd_error_);
        pcl::console::print_info("normal mean difference (deg): %0.7f\n", 180.0 / M_PI * normal_diff_);
        pcl::console::print_info("uniformity of correct correspondences' distribution: %0.7f\n", corr_uniformity_);
    } else {
        pcl::console::print_info("inliers: %i\n", inlier_pairs_.size());
        pcl::console::print_info("correspondences: %i\n", correspondences_.size());
    }
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
    out << "voxel_size,nr_points,distance_thr,edge_thr,";
    out << "iteration,matching,randomness,filter,threshold,n_random,r_err,t_err,pcd_err,";
    out << "normal_diff,corr_uniformity,lrf,metric,time,overlap_rmse,alignment,keypoint,time_cs,time_te,score,iss_coef\n";
}

std::ostream &operator<<(std::ostream &stream, const AlignmentAnalysis &analysis) {
    std::string matching_id = analysis.parameters_.matching_id;
    if (matching_id == MATCHING_RATIO) matching_id += std::to_string(analysis.parameters_.ratio_parameter);
    stream << VERSION << "," << analysis.parameters_.descriptor_id << "," << analysis.testname_ << ","
           << analysis.fitness_ << "," << analysis.rmse_ << ",";
    stream << analysis.correspondences_.size() << "," << analysis.correct_correspondences_.size() << ",";
    stream << analysis.inlier_pairs_.size() << "," << analysis.correct_inlier_pairs_.size() << ",";
    stream << analysis.parameters_.distance_thr << ",";
    stream << analysis.parameters_.feature_nr_points << ",";
    stream << analysis.parameters_.distance_thr << ",";
    stream << analysis.parameters_.edge_thr_coef << ",";
    stream << analysis.result_.iterations << ",";
    stream << matching_id << "," << analysis.parameters_.randomness << ",";
    auto func = getUniquenessFunction(analysis.parameters_.func_id);
    if (func != nullptr) {
        stream << analysis.parameters_.func_id << "," << UNIQUENESS_THRESHOLD << "," << N_RANDOM_FEATURES << ",";
    } else {
        stream << ",,,";
    }
    stream << analysis.r_error_ << "," << analysis.t_error_ << "," << analysis.pcd_error_ << ",";
    stream << analysis.normal_diff_ << "," << analysis.corr_uniformity_ << "," << analysis.parameters_.lrf_id << ",";
    stream << analysis.parameters_.metric_id << ",";
    stream << analysis.result_.time_te + analysis.result_.time_cs + analysis.result_.time_ds_ne << ",";
    stream << analysis.overlap_error_ << ",";
    stream << analysis.parameters_.alignment_id << "," << analysis.parameters_.keypoint_id << ",";
    stream << analysis.result_.time_cs << "," << analysis.result_.time_te << ",";
    stream << analysis.parameters_.score_id << "," << analysis.parameters_.iss_coef << "\n";
    return stream;
}
