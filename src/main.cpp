#include <Eigen/Core>
#include <string>
#include <filesystem>
#include <array>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

#include "analysis.h"
#include "feature_analysis.h"
#include "alignment.h"
#include "downsample.h"
#include "weights.h"

namespace fs = std::filesystem;

const std::string ALIGNMENT = "alignment";
const std::string METRIC_ANALYSIS = "metric";
const std::string DEBUG = "debug";

std::vector<AlignmentAnalysis> runTest(const YamlConfig &config) {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    std::optional<Eigen::Matrix4f> tn_gt;
    std::string testname;
    loadPointClouds(config, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(config, config.get<std::string>("ground_truth").value(), tn_gt);
    std::vector<AlignmentAnalysis> analyses;
    for (auto &parameters: getParametersFromConfig(config, src, tgt, fields_src, fields_tgt)) {
        parameters.testname = testname;
        parameters.ground_truth = std::optional<Eigen::Matrix4f>(tn_gt);
        pcl::console::print_highlight("Starting alignment...\n");
        AlignmentResult result = alignPointClouds(src, tgt, parameters);
        AlignmentAnalysis analysis(result, parameters);
        analysis.start(tn_gt, testname);
        analyses.push_back(analysis);
    }
    return analyses;
}

void estimateTestMetric(const YamlConfig &config) {
    std::string filepath = constructPath("test", "metrics", "csv", false);
    bool file_exists = std::filesystem::exists(filepath);
    std::fstream fout;
    if (!file_exists) {
        fout.open(filepath, std::ios_base::out);
    } else {
        fout.open(filepath, std::ios_base::app);
    }
    if (fout.is_open()) {
        if (!file_exists) {
            fout << "testname,metric_corr,metric_icp,inliers_corr,inliers_icp,";
            fout << "metric_corr_gt,metric_icp_gt,inliers_corr_gt,inliers_icp_gt\n";
        }
    } else {
        perror(("error while opening file " + filepath).c_str());
    }

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    std::optional<Eigen::Matrix4f> tn_gt;
    std::string testname;
    float min_voxel_size;
    loadPointClouds(config, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(config, config.get<std::string>("ground_truth").value(), tn_gt);
    if (!tn_gt) {
        PCL_ERROR("Failed to read ground truth for %s!\n", testname.c_str());
    }

    for (auto &params: getParametersFromConfig(config, src, tgt, fields_src, fields_tgt)) {
        params.testname = testname;
        auto tn_name = config.get<std::string>("transformation", constructName(params, "transformation"));
        auto tn = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV), tn_name);
        ScoreFunction score_function;
        if (params.score_id == METRIC_SCORE_MAE) {
            score_function = ScoreFunction::MAE;
        } else if (params.score_id == METRIC_SCORE_MSE) {
            score_function = ScoreFunction::MSE;
        } else if (params.score_id == METRIC_SCORE_EXP) {
            score_function = ScoreFunction::EXP;
        } else {
            score_function = ScoreFunction::Constant;
        }
        CorrespondencesMetricEstimator estimator_corr(score_function);
        ClosestPlaneMetricEstimator estimator_icp(false, score_function);
        CorrespondencesPtr correspondences;
        Correspondences inliers_corr, inliers_icp;
        float error, metric_icp, metric_corr;
        bool success = false;
        std::string corrs_path = constructPath(params, "correspondences", "csv", true, false, false);
        correspondences = readCorrespondencesFromCSV(corrs_path, success);
        if (!success) {
            pcl::console::print_error("Failed to read correspondences for %s!\n", params.testname.c_str());
            exit(1);
        }

        estimator_corr.setSourceCloud(src);
        estimator_corr.setTargetCloud(tgt);
        estimator_corr.setCorrespondences(correspondences);

        estimator_icp.setSourceCloud(src);
        estimator_icp.setTargetCloud(tgt);
        estimator_icp.setCorrespondences(correspondences);

        fout << constructName(params, "metric", true, true, false);
        std::array<Eigen::Matrix4f, 2> transformations{tn, tn_gt.value()};
        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
        for (auto &transformation: transformations) {
            estimator_corr.buildInliersAndEstimateMetric(transformation, inliers_corr, error, metric_corr, rand);
            estimator_icp.buildInliersAndEstimateMetric(transformation, inliers_icp, error, metric_icp, rand);
            fout << "," << metric_corr << "," << metric_icp;
            fout << "," << inliers_corr.size() << "," << inliers_icp.size();
        }
        fout << "\n";
    }
}

void calculateSurfaceArea(const PointNCloud::ConstPtr &pcd, const AlignmentParameters &parameters,
                          const std::string &name) {

    // Create search tree*
    pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>);
    tree->setInputCloud(pcd);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<PointN> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(5 * parameters.distance_thr);

    // Set typical values for the parameters
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
    gp3.setMinimumAngle(M_PI / 18); // 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud(pcd);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(triangles);

    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();

    pcl::io::savePLYFileBinary(constructPath(parameters.testname, name), triangles);
}

void compareOverlaps(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                     const Eigen::Matrix4f &transformation, const Eigen::Matrix4f &transformation_gt,
                     const AlignmentParameters &parameters) {
    std::vector<Eigen::Matrix4f> tns = {transformation, transformation_gt};
    pcl::KdTreeFLANN<PointN> tree_tgt;
    tree_tgt.setInputCloud(tgt);
    int count[2] = {0, 0};
    float weighted_count[2] = {0.f, 0.f};
    for (int idx_tn = 0; idx_tn < 2; ++idx_tn) {
        PointNCloud::Ptr src_aligned(new PointNCloud), overlap(new PointNCloud);
        auto tn = tns[idx_tn];
        pcl::transformPointCloudWithNormals(*src, *src_aligned, tn);
        pcl::KdTreeFLANN<PointN> tree_src_aligned;
        tree_src_aligned.setInputCloud(src_aligned);
        pcl::Indices nn_indices;
        std::vector<float> nn_sqr_dists;
        for (int is_source = 1; is_source >= 0; --is_source) {
            const auto &pcd1 = is_source ? src_aligned : tgt;
            const auto &pcd2 = is_source ? tgt : src_aligned;
            const auto &tree2 = is_source ? tree_tgt : tree_src_aligned;
            std::vector<bool> is_in_overlap(pcd1->size(), false);
            PointN nearest_point;
#pragma omp parallel for default(none) firstprivate(idx_tn, nn_indices, nn_sqr_dists, parameters, nearest_point) shared(tree2, pcd1, pcd2, is_in_overlap)
            for (int i = 0; i < pcd1->size(); ++i) {
                tree2.nearestKSearch(*pcd1, i, 1, nn_indices, nn_sqr_dists);
                nearest_point = pcd2->points[nn_indices[0]];
                float dist_to_plane = std::fabs(nearest_point.getNormalVector3fMap().transpose() *
                                                (nearest_point.getVector3fMap() -
                                                 pcd1->points[i].getVector3fMap()));
                // normal can be invalid
                dist_to_plane = std::isfinite(dist_to_plane) ? dist_to_plane : nn_sqr_dists[0];
                if (dist_to_plane < parameters.distance_thr) {
                    is_in_overlap[i] = true;
                }
            }
            for (int i = 0; i < pcd1->size(); ++i) {
                if (is_in_overlap[i]) {
                    overlap->push_back(pcd1->points[i]);
                }
            }
        }
        count[idx_tn] = overlap->size();
        auto densities = calculateSmoothedDensities(overlap);
        for (int i = 0; i < densities.size(); ++i) {
            weighted_count[idx_tn] += densities[i] * densities[i];
        }
        savePointCloudWithCorrespondences(overlap, {}, {}, {}, {}, parameters, Eigen::Matrix4f::Identity(),
                                          idx_tn == 0);
//        calculateSurfaceArea(overlap, parameters, idx_tn == 0 ? "incorrect" : "correct");
    }

    std::cerr << "\tincorrect hypothesis: " << count[0] << " points, " << weighted_count[0] << "weighted points\n";
    std::cerr << "\t  correct hypothesis: " << count[1] << " points, " << weighted_count[1] << "weighted points\n";
}

void compareHypotheses(const YamlConfig &config) {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    std::optional<Eigen::Matrix4f> tn_gt;
    std::string testname;
    float min_voxel_size;
    loadPointClouds(config, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(config, config.get<std::string>("ground_truth").value(), tn_gt);
    if (!tn_gt) {
        PCL_ERROR("Failed to read ground truth for %s!\n", testname.c_str());
    }

    for (auto &params: getParametersFromConfig(config, src, tgt, fields_src, fields_tgt)) {
        params.testname = testname;
        auto tn_name = constructName(params, "tn");
        auto tn = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV), tn_name);
        saveTemperatureMaps(src, tgt, "temperature_gt", params, params.distance_thr, tn_gt.value());
        saveTemperatureMaps(src, tgt, "temperature", params, params.distance_thr, tn);
        compareOverlaps(src, tgt, tn, tn_gt.value(), params);
    }
}

void generateDebugFiles(const YamlConfig &config) {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    PointNCloud::Ptr src_aligned(new PointNCloud), src_aligned_gt(new PointNCloud);
    CorrespondencesPtr correspondences(new Correspondences);
    CorrespondencesPtr correct_correspondences(new Correspondences);
    Correspondences inliers;
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f tn;
    std::optional<Eigen::Matrix4f> tn_gt;
    std::string testname;
    float min_voxel_size, error;
    loadPointClouds(config, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(config, config.get<std::string>("ground_truth").value(), tn_gt);

    for (auto &params: getParametersFromConfig(config, src, tgt, fields_src, fields_tgt)) {
        params.testname = testname;
        bool success = false;
        std::string corrs_path = constructPath(params, "correspondences", "csv", true, false, false);
        correspondences = readCorrespondencesFromCSV(corrs_path, success);
        if (!success) {
            pcl::console::print_error("Failed to read correspondences for %s!\n", params.testname.c_str());
            exit(1);
        }
        pcl::IndicesPtr indices_src{nullptr}, indices_tgt{nullptr};
        tn = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                               constructName(params, "tn"));
        float error_thr = params.distance_thr;
        indices_src = detectKeyPoints(src, params, params.iss_radius_src);
        indices_tgt = detectKeyPoints(tgt, params, params.iss_radius_tgt);

        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
        auto metric_estimator = getMetricEstimatorFromParameters(params);
        metric_estimator->setSourceCloud(src);
        metric_estimator->setTargetCloud(tgt);
        metric_estimator->setCorrespondences(correspondences);
        metric_estimator->buildInliers(tn, inliers, error, rand);
        if (tn_gt.has_value()) {
            buildCorrectCorrespondences(src, tgt, *correspondences, *correct_correspondences, tn_gt.value());
//            saveCorrespondences(src, tgt, *correspondences, tn_gt.value(), params);
//            saveCorrespondences(src, tgt, *correspondences, tn_gt.value(), params, true);
//            saveCorrespondenceDistances(src, tgt, *correspondences, tn_gt.value(), params);
            savePointCloudWithCorrespondences(src, indices_src, *correspondences, *correct_correspondences,
                                              inliers, params, tn_gt.value(), true);
//            saveTemperatureMaps(src, tgt, "temperature_gt", params, error_thr, tn_gt.value());
        }
        savePointCloudWithCorrespondences(tgt, indices_tgt, *correspondences, *correct_correspondences,
                                          inliers, params, Eigen::Matrix4f::Identity(), false);

        if (params.metric_id == METRIC_WEIGHTED_CLOSEST_PLANE) {
            WeightFunction weight_function = getWeightFunction(params.weight_id);
            auto weights = weight_function(NORMAL_NR_POINTS, src);
            saveColorizedWeights(src, weights, "weights", params, tn);
        }
        saveTemperatureMaps(src, tgt, "temperature", params, error_thr, tn);
    }
}

void analyzeKeyPoints(const YamlConfig &config) {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    PointNCloud::Ptr subvoxel_kps_src(new PointNCloud), subvoxel_kps_tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    std::optional<Eigen::Matrix4f> tn_gt;
    std::string testname;
    float min_voxel_size;
    loadPointClouds(config, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(config, config.get<std::string>("ground_truth").value(), tn_gt);
    if (!tn_gt) {
        PCL_ERROR("Failed to read ground truth for %s!\n", testname.c_str());
    }
    for (auto &params: getParametersFromConfig(config, src, tgt, fields_src, fields_tgt)) {
        params.testname = testname;
        estimateNormalsPoints(params.normal_nr_points, src, {nullptr}, params.vp_src, params.normals_available);
        estimateNormalsPoints(params.normal_nr_points, tgt, {nullptr}, params.vp_tgt, params.normals_available);
        auto indices_src = detectKeyPoints(src, params, params.iss_radius_src, subvoxel_kps_src, true);
        auto indices_tgt = detectKeyPoints(tgt, params, params.iss_radius_tgt, subvoxel_kps_tgt, true);
        saveColorizedPointCloud(subvoxel_kps_src, tn_gt.value(), COLOR_RED, constructPath(params, "subvoxel_kps_src"));
        saveColorizedPointCloud(subvoxel_kps_tgt, Eigen::Matrix4f::Identity(), COLOR_RED,
                                constructPath(params, "subvoxel_kps_tgt"));
        savePointCloudWithCorrespondences(src, indices_src, {}, {}, {}, params, tn_gt.value(), true);
        savePointCloudWithCorrespondences(tgt, indices_tgt, {}, {}, {}, params, Eigen::Matrix4f::Identity(), false);
    }
}

void measureTestResults(const YamlConfig &config) {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    std::string filepath = constructPath("test", "measurements", "csv", false);
    bool file_exists = std::filesystem::exists(filepath);
    std::fstream fout;
    if (!file_exists) {
        fout.open(filepath, std::ios_base::out);
    } else {
        fout.open(filepath, std::ios_base::app);
    }
    if (fout.is_open()) {
        if (!file_exists) {
            fout << "testname,success_rate,mae,sae,mte,ste,mrmse,srmse,mtime,stime\n";
        }
    } else {
        perror(("error while opening file " + filepath).c_str());
    }

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    std::optional<Eigen::Matrix4f> tn_gt;
    std::string testname;
    float min_voxel_size;
    int n_times = config.get<int>("n_times", 10);
    loadPointClouds(config, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(config, config.get<std::string>("ground_truth").value(), tn_gt);
    for (auto &parameters: getParametersFromConfig(config, src, tgt, fields_src, fields_tgt)) {
        parameters.fix_seed = false;
        parameters.testname = testname;
        parameters.ground_truth = std::optional<Eigen::Matrix4f>(tn_gt);
        if (parameters.save_features && tn_gt.has_value()) {
            saveExtractedPointIds(src, tgt, tn_gt.value(), parameters, tgt);
        }
        std::vector<float> rotation_errors;
        std::vector<float> translation_errors;
        std::vector<float> overlap_errors;
        std::vector<float> runtimes;
        int n_successful_times = 0;
        if (parameters.alignment_id != ALIGNMENT_RANSAC) n_times = 1;
        for (int i = 0; i < n_times; ++i) {
            pcl::console::print_highlight("Starting alignment...\n");
            AlignmentResult result = alignPointClouds(src, tgt, parameters);
            AlignmentAnalysis analysis(result, parameters);
            analysis.start(tn_gt, testname);
            bool success = analysis.alignmentHasConverged() && analysis.getOverlapError() < parameters.distance_thr;
            if (success) {
                rotation_errors.push_back(analysis.getRotationError());
                translation_errors.push_back(analysis.getTranslationError());
                overlap_errors.push_back(analysis.getOverlapError());
                n_successful_times++;

            }
            runtimes.push_back(analysis.getRunningTime());
        }
        float success_rate = (float) n_successful_times / (float) n_times;
        auto mean_rotation_error = calculateMean<float>(rotation_errors);
        auto std_rotation_error = calculateStandardDeviation<float>(rotation_errors);
        auto mean_translation_error = calculateMean<float>(translation_errors);
        auto std_translation_error = calculateStandardDeviation<float>(translation_errors);
        auto mean_overlap_error = calculateMean<float>(overlap_errors);
        auto std_overlap_error = calculateStandardDeviation<float>(overlap_errors);
        auto mean_running_time = calculateMean<float>(runtimes);
        auto std_running_time = calculateStandardDeviation<float>(runtimes);

        fout << constructName(parameters, "measure") << "," << success_rate << ","
             << mean_rotation_error << "," << std_rotation_error << ","
             << mean_translation_error << "," << std_translation_error << ","
             << mean_overlap_error << "," << std_overlap_error << ","
             << mean_running_time << "," << std_running_time << "\n";
    }
}

void processTests(const std::vector<YAML::Node> &tests, const std::string &command) {
    for (auto &test: tests) {
        YamlConfig config;
        config.config = (*test.begin()).second;
        auto test_type = (*test.begin()).first.as<std::string>();
        if (test_type == "test") {
            if (command == ALIGNMENT) {
                runTest(config);
            } else if (command == METRIC_ANALYSIS) {
                estimateTestMetric(config);
            } else if (command == DEBUG) {
                generateDebugFiles(config);
            }
        } else if (test_type == "compare") {
            compareHypotheses(config);
        } else if (test_type == "keypoint") {
            analyzeKeyPoints(config);
        } else if (test_type == "measure") {
            measureTestResults(config);
        } else {
            pcl::console::print_error("Test type %s isn't supported!\n", test_type.c_str());
        }
    }
}

int main(int argc, char **argv) {
    std::string command(argv[1]);
    if (argc != 3 && !(command == ALIGNMENT || command == METRIC_ANALYSIS || command == DEBUG)) {
        pcl::console::print_error(("Syntax is: [" +
                                   ALIGNMENT + ", " + METRIC_ANALYSIS + ", " + DEBUG +
                                   "] %s config.yaml\n").c_str(), argv[0]);
        exit(1);
    }
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    // Load parameters from config
    YamlConfig config;
    config.init(argv[2]);
    auto tests = config.get<std::vector<YAML::Node>>("tests");
    if (tests.has_value()) {
        processTests(tests.value(), command);
    } else if (command == ALIGNMENT) {
        runTest(config);
    } else if (command == METRIC_ANALYSIS) {
        estimateTestMetric(config);
    } else if (command == DEBUG) {
        generateDebugFiles(config);
    }
}

