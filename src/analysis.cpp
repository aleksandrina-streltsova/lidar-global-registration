#include <fstream>
#include <filesystem>

#include <Eigen/Geometry>
#include <utility>

#include <pcl/common/time.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include "analysis.h"
#include "filter.h"

#define N_BINS 100

namespace fs = std::filesystem;

std::pair<float, float> calculateRotationAndTranslationDifferences(const Eigen::Matrix4f &tn1,
                                                                   const Eigen::Matrix4f &tn2) {
    Eigen::Matrix3f rotation_diff = tn1.block<3, 3>(0, 0).inverse() * tn2.block<3, 3>(0, 0);
    Eigen::Vector3f translation_diff = tn1.block<3, 1>(0, 3) - tn2.block<3, 1>(0, 3);
    return {Eigen::AngleAxisf(rotation_diff).angle(), translation_diff.norm()};
}

inline float dist2(const PointN &p1, const PointN &p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
};

float calculatePointCloudRmse(const PointNCloud::ConstPtr &pcd,
                              const Eigen::Matrix4f &transformation,
                              const Eigen::Matrix4f &transformation_gt) {
    PointNCloud pcd_transformed;
    Eigen::Matrix4f transformation_diff = transformation.inverse() * transformation_gt;
    pcl::transformPointCloudWithNormals(*pcd, pcd_transformed, transformation_diff);

    float rmse = 0.f;
    for (int i = 0; i < pcd->size(); ++i) {
        rmse += dist2(pcd->points[i], pcd_transformed.points[i]);
    }
    rmse = std::sqrt(rmse / (float) pcd->size());
    return rmse;
}

float calculateOverlapRmse(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                           const Eigen::Matrix4f &transformation,
                           const Eigen::Matrix4f &transformation_gt,
                           float inlier_threshold) {
    PointNCloud src_aligned, src_aligned_gt;
    pcl::transformPointCloudWithNormals(*src, src_aligned, transformation);
    pcl::transformPointCloudWithNormals(*src, src_aligned_gt, transformation_gt);

    pcl::KdTreeFLANN<PointN> tree_tgt;
    tree_tgt.setInputCloud(tgt);

    float rmse = 0.f;
    int overlap_size = 0;
    float search_radius = DIST_TO_PLANE_COEFFICIENT * inlier_threshold;
#pragma omp parallel default(none) firstprivate(inlier_threshold, search_radius) \
    shared(src_aligned_gt, src_aligned, tgt, tree_tgt) \
    reduction(+:overlap_size, rmse)
    {
        float distance;
        Eigen::Vector3f point, nearest_point, point_on_plane, normal;
        pcl::Indices nn_indices;
        std::vector<float> nn_sqr_dists;
#pragma omp for
        for (int i = 0; i < src_aligned_gt.size(); ++i) {
            tree_tgt.radiusSearch(src_aligned_gt[i], search_radius, nn_indices, nn_sqr_dists, 1);
            if (nn_indices.empty()) continue;
            point = src_aligned_gt[i].getVector3fMap();
            nearest_point = tgt->points[nn_indices[0]].getVector3fMap();
            normal = tgt->points[nn_indices[0]].getNormalVector3fMap();
            if (!normal.allFinite()) continue;
            point_on_plane = point - ((point - nearest_point).transpose() * normal) * normal;
            if ((point - point_on_plane).norm() > inlier_threshold) continue;
            distance = (src_aligned[i].getVector3fMap() - point_on_plane).norm();
            rmse += distance * distance;
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

float calculateCorrespondenceUniformity(const PointNCloud::ConstPtr &src,
                                        const Correspondences &correct_correspondences) {
    return calculateCorrespondenceUniformity(src, calculateBoundingBox<PointN>(src), correct_correspondences);
}

float calculateCorrespondenceUniformity(const PointNCloud::ConstPtr &src, const std::pair<PointN, PointN> &bbox,
                                        const Correspondences &correct_correspondences) {
    std::array<std::array<std::array<int, N_BINS>, N_BINS>, 3> count{};
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < N_BINS; ++i) {
            for (int j = 0; j < N_BINS; ++j) {
                count[k][i][j] = 0;
            }
        }
    }
    auto[min_point, max_point] = bbox;
    for (auto const &corr: correct_correspondences) {
        const auto &point = src->points[corr.index_query];
        int bin[3];
        bin[0] = std::min(std::floor((point.x - min_point.x) / (max_point.x - min_point.x) * N_BINS), N_BINS - 1.f);
        bin[1] = std::min(std::floor((point.y - min_point.y) / (max_point.y - min_point.y) * N_BINS), N_BINS - 1.f);
        bin[2] = std::min(std::floor((point.z - min_point.z) / (max_point.z - min_point.z) * N_BINS), N_BINS - 1.f);
        // count 3D points projected to YZ, ZX, XY and fallen in 2D bin
        for (int k = 0; k < 3; ++k) {
            count[k][bin[(k + 1) % 3]][bin[(k + 2) % 3]]++;
        }
    }
    float entropy[3] = {0.f, 0.f, 0.f};
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

float calculateNormalDifference(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                float distance_thr, const Eigen::Matrix4f &transformation_gt) {
    PointNCloud::Ptr src_aligned(new PointNCloud);
    pcl::transformPointCloudWithNormals(*src, *src_aligned, transformation_gt);
    pcl::KdTreeFLANN<PointN>::Ptr tree(new pcl::KdTreeFLANN<PointN>());
    tree->setInputCloud(tgt);

    std::vector<float> differences(src_aligned->size(), -1);
#pragma omp parallel default(none) firstprivate(distance_thr) \
    shared(src_aligned, tgt, tree, differences)
    {
        pcl::Indices nn_indices;
        std::vector<float> nn_sqr_distances;
#pragma omp for
        for (int i = 0; i < src_aligned->size(); ++i) {
            (float) tree->nearestKSearch(*src_aligned, i, 1, nn_indices, nn_sqr_distances);
            const PointN &p_src(src_aligned->points[i]), p_tgt(tgt->points[nn_indices[0]]);
            if (std::sqrt(nn_sqr_distances[0]) < distance_thr && std::isfinite(p_src.normal_x) &&
                std::isfinite(p_tgt.normal_x)) {
                float cos = std::clamp(p_src.normal_x * p_tgt.normal_x + p_src.normal_y * p_tgt.normal_y +
                                       p_src.normal_z * p_tgt.normal_z, -1.f, 1.f);
                differences[i] = std::abs(std::acos(cos));
            }
        }
    }
    int n_points_overlap = 0;
    std::vector<float> differences_filtered;
    for (float diff: differences) {
        if (diff >= 0.f) {
            differences_filtered.push_back(diff);
            n_points_overlap++;
        }
    }
    if (n_points_overlap == 0) return M_PI;
    std::nth_element(differences_filtered.begin(), differences_filtered.begin() + n_points_overlap / 2,
                     differences_filtered.end());
    float result = differences_filtered[n_points_overlap / 2];
    PCL_DEBUG("[calculateNormalDifference] median of normal differences (deg): %0.7f "
              "[distance threshold = %0.7f, points in overlap: %i (%i, %i)]\n",
              180 * result / M_PI, distance_thr, n_points_overlap, src->size(), tgt->size());
    return result;
}

void buildCorrectCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                 const Correspondences &correspondences,
                                 Correspondences &correct_correspondences,
                                 const Eigen::Matrix4f &transformation_gt) {
    correct_correspondences.clear();
    correct_correspondences.reserve(correspondences.size());

    PointNCloud input_transformed;
    input_transformed.resize(src->size());
    pcl::transformPointCloudWithNormals(*src, input_transformed, transformation_gt);

    for (const auto &correspondence: correspondences) {
        PointN source_point(input_transformed.points[correspondence.index_query]);
        PointN target_point(tgt->points[correspondence.index_match]);
        float e = pcl::L2_Norm(source_point.data, target_point.data, 3);
        if (e < correspondence.threshold) {
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
    correspondences_ = *result_.correspondences;
}

void AlignmentAnalysis::start(const std::optional<Eigen::Matrix4f> &transformation_gt, const std::string &testname) {
    pcl::ScopeTime t("Analysis");
    testname_ = testname;
    transformation_gt_ = transformation_gt;
    UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
    metric_estimator_->buildInliersAndEstimateMetric(transformation_, inliers_, rmse_, metric_, rand);
    if (transformation_gt_.has_value()) {
        PointNCloud::Ptr src_aligned(new PointNCloud), pcd_overlap(new PointNCloud);
        float error_thr = parameters_.distance_thr;
        pcl::transformPointCloudWithNormals(*src_, *src_aligned, parameters_.ground_truth.value());
        mergeOverlaps(src_aligned, tgt_, pcd_overlap, parameters_.distance_thr);
        overlap_ = (float) pcd_overlap->size() / (float) (src_->size() + tgt_->size());
        std::vector<float> ds_overlap = calculateSmoothedDensities(pcd_overlap);
        std::vector<float> ds_src = calculateSmoothedDensities(src_);
        auto acc_squared = [](float a, float b) { return a + b * b; };
        overlap_area_ = std::accumulate(ds_overlap.begin(), ds_overlap.end(), 0.f, acc_squared) /
                        std::accumulate(ds_src.begin(), ds_src.end(), 0.f, acc_squared);
        buildCorrectCorrespondences(src_, tgt_, correspondences_, correct_correspondences_, transformation_gt_.value());
        metric_estimator_->buildCorrectInliers(inliers_, correct_inliers_, transformation_gt_.value());
        pcd_error_ = calculatePointCloudRmse(src_, transformation_, transformation_gt_.value());
        overlap_error_ = calculateOverlapRmse(src_, tgt_, transformation_, transformation_gt_.value(), error_thr);
        normal_diff_ = calculateNormalDifference(src_, tgt_, error_thr, transformation_gt_.value());
        corr_uniformity_ = calculateCorrespondenceUniformity(src_, correct_correspondences_);
        std::tie(r_error_, t_error_) = calculateRotationAndTranslationDifferences(transformation_,
                                                                                  transformation_gt_.value());
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
    pcl::console::print_info("converged: %s\n", result_.converged ? "true" : "false");
    pcl::console::print_info("metric: %0.7f\n", metric_);
    pcl::console::print_info("inliers_rmse: %0.7f\n", rmse_);
    if (transformation_gt_.has_value()) {
        pcl::console::print_info("correct inliers: %i/%i\n", correct_inliers_.size(), inliers_.size());
        pcl::console::print_info("correct correspondences: %i/%i\n",
                                 correct_correspondences_.size(), correspondences_.size());
        pcl::console::print_info("rotation error (deg): %0.7f\n", 180.0 / M_PI * r_error_);
        pcl::console::print_info("translation error: %0.7f\n", t_error_);
        pcl::console::print_info("point cloud error: %0.7f\n", pcd_error_);
        pcl::console::print_info("median of normal differences (deg): %0.7f\n", 180.0 / M_PI * normal_diff_);
        pcl::console::print_info("uniformity of correct correspondences' distribution: %0.7f\n", corr_uniformity_);
    } else {
        pcl::console::print_info("inliers: %i\n", inliers_.size());
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
    out << "version,descriptor,testname,metric,rmse,correspondences,correct_correspondences,inliers,correct_inliers,";
    out << "voxel_size,nr_points,distance_thr,edge_thr,";
    out << "iteration,matching,randomness,filter,threshold,n_random,r_err,t_err,pcd_err,";
    out << "normal_diff,corr_uniformity,lrf,metric_type,time,overlap_rmse,alignment,keypoint,time_cs,time_te,score,";
    out << "iss_radius_src,normal_nr_points,reestimate,scale,cluster_k,feature_radius";
    out << "overlap,overlap_area,converged,iss_radius_tgt\n";
}

std::ostream &operator<<(std::ostream &stream, const AlignmentAnalysis &analysis) {
    std::string matching_id = analysis.parameters_.matching_id;
    if (matching_id == MATCHING_RATIO) matching_id += std::to_string(analysis.parameters_.ratio_k);
    stream << VERSION << "," << analysis.parameters_.descriptor_id << "," << analysis.testname_ << ","
           << analysis.metric_ << "," << analysis.rmse_ << ",";
    stream << analysis.correspondences_.size() << "," << analysis.correct_correspondences_.size() << ",";
    stream << analysis.inliers_.size() << "," << analysis.correct_inliers_.size() << ",";
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
    stream << analysis.result_.time_te + analysis.result_.time_cs << ",";
    stream << analysis.overlap_error_ << ",";
    stream << analysis.parameters_.alignment_id << "," << analysis.parameters_.keypoint_id << ",";
    stream << analysis.result_.time_cs << "," << analysis.result_.time_te << ",";
    stream << analysis.parameters_.score_id << "," << analysis.parameters_.iss_radius_src << ",";
    stream << analysis.parameters_.normal_nr_points << "," << analysis.parameters_.reestimate_frames << ",";
    stream << analysis.parameters_.scale_factor << "," << analysis.parameters_.cluster_k << ",";
    if (analysis.parameters_.feature_radius.has_value()) stream << analysis.parameters_.feature_radius.value();
    stream << "," << analysis.overlap_ << "," << analysis.overlap_area_ << "," << analysis.result_.converged << ",";
    stream << analysis.parameters_.iss_radius_tgt << "\n";
    return stream;
}
