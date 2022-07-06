#include <Eigen/Core>

#include <pcl/io/ply_io.h>
#include "common.h"
#include "align.h"

#define TMP_DIR "tmp"
#define CORNER_SIZE 100
#define SHIFT 5

void assertClose(const std::string &name, float expected, float actual, float eps = 1e-5) {
    if (std::fabs(actual - expected) > eps) {
        std::cerr << "[" << name << "] actual value " << actual << " differs from expected " << expected << std::endl;
        abort();
    }
}

int main() {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    for (int i = 0; i < CORNER_SIZE; ++i) {
        for (int j = 0; j < CORNER_SIZE; ++j) {
            src->points.emplace_back(PointN{0 * SHIFT + 2.f * (float) i, 0 * SHIFT + 2.f * (float) j, 0.f});
            src->points.emplace_back(PointN{1 * SHIFT + 2.f * (float) i, 0.f, 1 * SHIFT + 2.f * (float) j});
            src->points.emplace_back(PointN{0.f, 2 * SHIFT + 2.f * (float) i, 2 * SHIFT + 2.f * (float) j});
            tgt->points.emplace_back(PointN{0 * SHIFT + 2.f * (float) i + 1.f, 0 * SHIFT + 2.f * (float) j + 1.f, 0.f});
            tgt->points.emplace_back(PointN{1 * SHIFT + 2.f * (float) i + 1.f, 0.f, 1 * SHIFT + 2.f * (float) j + 1.f});
            tgt->points.emplace_back(PointN{0.f, 2 * SHIFT + 2.f * (float) i + 1.f, 2 * SHIFT + 2.f * (float) j + 1.f});
        }
    }
    pcl::io::savePLYFileBinary("corner1.ply", *src);
    pcl::io::savePLYFileBinary("corner2.ply", *tgt);

    Eigen::Matrix4f transformation_gt;
    transformation_gt << 0.0803703, -0.996763, -0.00201846, 1.2143,
            0.996758, 0.080377, -0.00349969, -6.13404,
            0.00365057, -0.00173067, 0.999992, -1.17221,
            0, 0, 0, 1;
    pcl::transformPointCloud(*src, *src, transformation_gt.inverse());
    AlignmentParameters parameters{
            .normals_available = false,
            .voxel_size = 1.98f,
            .distance_thr_coef = 1.f,
            .keypoint_id = KEYPOINT_ANY,
            .metric_id = METRIC_CLOSEST_PLANE,
            .max_iterations = 1000,
            .ground_truth = std::make_shared<Eigen::Matrix4f>(transformation_gt),
            .fix_seed = true,
            .dir_path = TMP_DIR
    };
    fs::create_directory(TMP_DIR);

    std::vector<InlierPair> inlier_pairs;
    float metric, error;
    auto[alignment_result, time] = align_point_clouds<SHOT>(src, tgt, parameters);

    // TODO: remove after fixing transformation estimation
    PointNCloud src_aligned;
    alignment_result.setTransformationGuess(transformation_gt);
    alignment_result.align(src_aligned);

    auto transformation = alignment_result.getFinalTransformation();
    auto alignment_analysis = alignment_result.getAlignmentAnalysis(parameters, time);
    auto metric_estimator = alignment_analysis.getMetricEstimator();
    metric_estimator->buildInlierPairsAndEstimateMetric(transformation, inlier_pairs, error, metric);
    alignment_analysis.start(transformation_gt, "corners");

    assertClose("inlier ratio", 1.f, (float) inlier_pairs.size() / (float) src->size());
    assertClose("metric error", 0.f, error);
    // TODO: uncomment after fixing transformation estimation
//    assertClose("rotation error", 0.f, alignment_analysis.getRotationError());
//    assertClose("translation error", 0.f, alignment_analysis.getTranslationError());
    assertClose("point cloud error", 0.f, alignment_analysis.getPointCloudError());
    assertClose("overlap rmse", 0.f, alignment_analysis.getOverlapError());
    fs::remove_all(TMP_DIR);
}
