#include <filesystem>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

#include "common.h"
#include "alignment.h"
#include "analysis.h"

#define TMP_DIR "tmp"
#define CORNER_SIZE 100
#define SHIFT 5

namespace fs = std::filesystem;

void assertClose(const std::string &name, float expected, float actual, float eps = 1e-5) {
    if (std::fabs(actual - expected) > eps) {
        std::cerr << "[" << name << "] actual value " << actual << " differs from expected " << expected << std::endl;
        abort();
    }
}

void assertLess(const std::string &name, float value, float threshold) {
    if (value > threshold) {
        std::cerr << "[" << name << "] value " << value << " is greater than threshold " << threshold << std::endl;
        abort();
    }
}

int main() {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    for (int i = 0; i < CORNER_SIZE; ++i) {
        for (int j = 0; j < CORNER_SIZE; ++j) {
            src->points.emplace_back(PointN{0 * SHIFT + 2.f * (float) i, 0 * SHIFT + 2.f * (float) j, 0.f, 1.f});
            src->points.emplace_back(PointN{1 * SHIFT + 2.f * (float) i, 0.f, 1 * SHIFT + 2.f * (float) j, 1.f});
            src->points.emplace_back(PointN{0.f, 2 * SHIFT + 2.f * (float) i, 2 * SHIFT + 2.f * (float) j, 1.f});
            tgt->points.emplace_back(PointN{0 * SHIFT + 2.f * (float) i + 1.f, 0 * SHIFT + 2.f * (float) j, 0.f, 1.f});
            tgt->points.emplace_back(PointN{1 * SHIFT + 2.f * (float) i, 0.f, 1 * SHIFT + 2.f * (float) j + 1.f, 1.f});
            tgt->points.emplace_back(PointN{0.f, 2 * SHIFT + 2.f * (float) i + 1.f, 2 * SHIFT + 2.f * (float) j, 1.f});
        }
    }
    src->width = src->points.size();
    src->height = 1;
    tgt->width = tgt->points.size();
    tgt->height = 1;

    pcl::io::savePLYFileBinary("corner1.ply", *src);
    pcl::io::savePLYFileBinary("corner2.ply", *tgt);

    Eigen::Matrix4f transformation_gt;
    transformation_gt << 0.0803703, -0.996763, -0.00201846, 1.2143,
            0.996758, 0.080377, -0.00349969, -6.13404,
            0.00365057, -0.00173067, 0.999992, -1.17221,
            0, 0, 0, 1;
    pcl::transformPointCloudWithNormals(*src, *src, transformation_gt.inverse());
    AlignmentParameters parameters{
            .distance_thr = 1.0f,
            .iss_radius_src = 1.0f,
            .iss_radius_tgt = 1.0f,
            .bf_block_size = 200000,
            .alignment_id = ALIGNMENT_RANSAC,
            .keypoint_id = KEYPOINT_ANY,
            .metric_id = METRIC_CLOSEST_PLANE,
            .max_iterations = 1000000,
            .ground_truth = std::optional<Eigen::Matrix4f>(transformation_gt),
            .fix_seed = true,
            .normals_available = false,
            .dir_path = TMP_DIR
    };
    fs::create_directory(TMP_DIR);

    Correspondences inliers;
    float metric, error;
    auto alignment_result = alignPointClouds(src, tgt, parameters);
    auto alignment_analysis = AlignmentAnalysis(alignment_result, parameters);

    PointNCloud src_aligned;
    pcl::transformPointCloudWithNormals(*src, src_aligned, alignment_result.transformation);
    pcl::io::savePLYFileBinary("corner1_aligned.ply",  src_aligned);

    auto transformation = alignment_result.transformation;
    auto metric_estimator = alignment_analysis.getMetricEstimator();
    UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
    metric_estimator->buildInliersAndEstimateMetric(transformation, inliers, error, metric, rand);
    alignment_analysis.start(transformation_gt, "corners");

    assertClose("inlier ratio", 1.f, (float) inliers.size() / (float) src->size());
    assertLess("metric error", error, 2.f / 3.f);
    assertLess("overlap rmse", alignment_analysis.getOverlapError(), 2.f / 3.f);
//    assertClose("rotation error", 0.f, alignment_analysis.getRotationError());
//    assertClose("translation error", 0.f, alignment_analysis.getTranslationError());
//    assertClose("point cloud error", 0.f, alignment_analysis.getPointCloudError());

    fs::remove_all(TMP_DIR);
}
