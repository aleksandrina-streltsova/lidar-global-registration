#include <filesystem>

#include <pcl/common/transforms.h>

#include "hypotheses.h"
#include "analysis.h"

#define MIN_ANGLE (M_PI / 9)
#define MIN_DISTANCE_COEF 20
#define MIN_METRIC_COEF 0.1

namespace fs = std::filesystem;

void updateHypotheses(std::vector<Eigen::Matrix4f> &transformations, std::vector<float> &metrics,
                      const Eigen::Matrix4f &new_transformation, float new_metric,
                      const AlignmentParameters &parameters) {
    rassert(transformations.size() == metrics.size(), 45832351834023)
    float best_metric = transformations.empty() ? 0 : *std::max_element(metrics.begin(), metrics.end());
    // don't update hypotheses if new hypothesis is too bad
    if (new_metric < MIN_METRIC_COEF * best_metric) return;
    std::vector<int> indices_desc_similar;
    for (int i = transformations.size() - 1; i >= 0; --i) {
        auto[r_diff, t_diff] = calculateRotationAndTranslationDifferences(new_transformation, transformations[i]);
        bool is_similar = r_diff < MIN_ANGLE && t_diff < MIN_DISTANCE_COEF * parameters.distance_thr;
        if (is_similar) indices_desc_similar.push_back(i);
        // don't update if there already exists similar hypothesis better than new one
        if (is_similar && metrics[i] > new_metric) return;
    }
    auto index_to_remove = indices_desc_similar.begin();
    for (int i = transformations.size() - 1; i >= 0; --i) {
        if (index_to_remove != indices_desc_similar.end() && *index_to_remove == i) {
            transformations.erase(transformations.begin() + *index_to_remove);
            metrics.erase(metrics.begin() + *index_to_remove);
            index_to_remove++;
        }
    }
    transformations.push_back(new_transformation);
    metrics.push_back(new_metric);

    if (new_metric > best_metric) {
        for (int i = transformations.size() - 1; i >= 0; --i) {
            if (metrics[i] < MIN_METRIC_COEF * new_metric) {
                transformations.erase(transformations.begin() + i);
                metrics.erase(metrics.begin() + i);
            }
        }
    }
}

Eigen::Matrix4f chooseBestHypothesis(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                     const CorrespondencesConstPtr &correspondences,
                                     const AlignmentParameters &params, std::vector<Eigen::Matrix4f> &tns) {
    PCL_DEBUG("[chooseBestHypothesis] comparing %i saved hypotheses..\n", tns.size());
    std::string filepath = constructPath("test", "hypotheses", "csv", false);
    bool file_exists = std::filesystem::exists(filepath);
    std::fstream fout;
    if (!file_exists) {
        fout.open(filepath, std::ios_base::out);
    } else {
        fout.open(filepath, std::ios_base::app);
    }
    if (fout.is_open()) {
        if (!file_exists) {
            fout << "testname,id,r_err,t_err,";
            fout << "inliers,mse,inliers_area,uniformity,overlap,overlap_area\n";
        }
    } else {
        perror(("error while opening file " + filepath).c_str());
    }

    auto acc_squared = [](float a, float b) { return a + b * b; };
    CorrespondencesMetricEstimator corrs_metric(ScoreFunction::MSE);
    corrs_metric.setSourceCloud(src);
    corrs_metric.setTargetCloud(tgt);
    corrs_metric.setCorrespondences(correspondences);
    UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);

    float best_uniformity = 0;
    Eigen::Matrix4f best_tn = Eigen::Matrix4f::Identity();

    std::vector<Eigen::Matrix4f> analyzed_tns;
    std::vector<std::string> ids;
    if (params.ground_truth.has_value()) {
        analyzed_tns.push_back(params.ground_truth.value());
        ids.emplace_back("gt");
    }
    for (int i = 0; i < tns.size(); ++i) {
        analyzed_tns.push_back(tns[i]);
        ids.push_back(std::to_string(i + 1));
    }
    PointNCloud::Ptr src_aligned(new PointNCloud), pcd_inliers(new PointNCloud), pcd_overlap(new PointNCloud);
    pcd_inliers->reserve(correspondences->size());
    pcd_overlap->reserve(src->size() + tgt->size());
    for (int i = 0; i < analyzed_tns.size(); ++i) {
        fout << params.testname << "," << ids[i] << ",";
        if (params.ground_truth.has_value()) {
            auto[r_err, t_err] = calculateRotationAndTranslationDifferences(analyzed_tns[i],
                                                                            params.ground_truth.value());
            fout << r_err << "," << t_err << ",";
        } else {
            fout << ",,";
        }
        float metric, error;
        Correspondences inliers;
        // rand isn't actually used in correspondence metric estimator, TODO: fix signature of metric estimator's functions
        corrs_metric.buildInliersAndEstimateMetric(analyzed_tns[i], inliers, error, metric, rand);
        pcd_inliers->clear();
        pcd_overlap->clear();
        for (const auto &inlier: inliers) {
            pcd_inliers->push_back(src->points[inlier.index_match]);
        }
        std::vector<float> ds_inliers = calculateSmoothedDensities(pcd_inliers);
        float inliers_area = std::accumulate(ds_inliers.begin(), ds_inliers.end(), 0.f, acc_squared);
        float inliers_uniformity = calculateCorrespondenceUniformity(src, inliers);

        pcl::transformPointCloud(*src, *src_aligned, analyzed_tns[i]);
        mergeOverlaps(src_aligned, tgt, pcd_overlap, params.distance_thr);
        std::vector<float> ds_overlap = calculateSmoothedDensities(pcd_overlap);
        float overlap_area = std::accumulate(ds_overlap.begin(), ds_overlap.end(), 0.f, acc_squared);

        fout << inliers.size() << "," << metric << ",";
        fout << inliers_area << "," << inliers_uniformity << "," << pcd_overlap->size() << "," << overlap_area << "\n";
        if (ids[i] != "gt" && inliers_uniformity > best_uniformity) {
            best_uniformity = inliers_uniformity;
            best_tn = analyzed_tns[i];
        }
    }
    fout.close();
    return best_tn;
}