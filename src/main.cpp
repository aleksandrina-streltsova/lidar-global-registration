#include <Eigen/Core>
#include <string>
#include <filesystem>
#include <array>
#include <pcl/common/io.h>

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
    std::optional<Eigen::Matrix4f> transformation_gt;
    std::string testname;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);
    std::vector<AlignmentAnalysis> analyses;
    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt)) {
        parameters.testname = testname;
        parameters.ground_truth = std::optional<Eigen::Matrix4f>(transformation_gt);
        if (parameters.save_features && transformation_gt.has_value()) {
            saveExtractedPointIds(src, tgt, transformation_gt.value(), parameters, tgt);
        }
        pcl::console::print_highlight("Starting alignment...\n");
        AlignmentResult result = alignPointClouds(src, tgt, parameters);
        AlignmentAnalysis analysis(result, parameters);
        if (analysis.alignmentHasConverged()) {
            analysis.start(transformation_gt, testname);
        }
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
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    std::optional<Eigen::Matrix4f> transformation_gt;
    std::string testname;
    float min_voxel_size;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);
    if (transformation_gt) {
        PCL_ERROR("Failed to read ground truth for %s!\n", testname.c_str());
    }

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt)) {
        parameters.testname = testname;
        std::vector<float> voxel_sizes;
        std::vector<std::string> matching_ids;
        PointNCloud::Ptr curr_src(new PointNCloud), curr_tgt(new PointNCloud);
        float voxel_size_src = FINE_VOXEL_SIZE_COEFFICIENT * calculatePointCloudDensity<PointN>(src);
        float voxel_size_tgt = FINE_VOXEL_SIZE_COEFFICIENT * calculatePointCloudDensity<PointN>(tgt);
        downsamplePointCloud(src, curr_src, voxel_size_src);
        downsamplePointCloud(tgt, curr_tgt, voxel_size_tgt);
        auto tn_name = config.get<std::string>("transformation", constructName(parameters, "transformation"));
        auto transformation = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV), tn_name);

        estimateNormalsPoints(NORMAL_NR_POINTS, curr_src, normals_src, parameters.normals_available);
        estimateNormalsPoints(NORMAL_NR_POINTS, curr_tgt, normals_tgt, parameters.normals_available);
        pcl::concatenateFields(*curr_src, *normals_src, *curr_src);
        pcl::concatenateFields(*curr_tgt, *normals_tgt, *curr_tgt);
        ScoreFunction score_function;
        if (parameters.score_id == METRIC_SCORE_MAE) {
            score_function = ScoreFunction::MAE;
        } else if (parameters.score_id == METRIC_SCORE_MSE) {
            score_function = ScoreFunction::MSE;
        } else if (parameters.score_id == METRIC_SCORE_EXP) {
            score_function = ScoreFunction::EXP;
        } else {
            score_function = ScoreFunction::Constant;
        }
        CorrespondencesMetricEstimator estimator_corr(score_function);
        ClosestPlaneMetricEstimator estimator_icp(false, score_function);
        pcl::CorrespondencesPtr correspondences;
        std::vector<InlierPair> inlier_pairs_corr, inlier_pairs_icp;
        float error, metric_icp, metric_corr;
        bool success = false;
        std::string corrs_path = constructPath(parameters, "correspondences", "csv", true, false, false);
        correspondences = readCorrespondencesFromCSV(corrs_path, success);
        if (!success) {
            pcl::console::print_error("Failed to read correspondences for %s!\n", parameters.testname.c_str());
            exit(1);
        }

        estimator_corr.setSourceCloud(curr_src);
        estimator_corr.setTargetCloud(curr_tgt);
        estimator_corr.setInlierThreshold(parameters.distance_thr);
        estimator_corr.setCorrespondences(correspondences);

        estimator_icp.setSourceCloud(curr_src);
        estimator_icp.setTargetCloud(curr_tgt);
        estimator_icp.setInlierThreshold(parameters.distance_thr);
        estimator_icp.setCorrespondences(correspondences);


        fout << constructName(parameters, "metric", true, true, false);
        std::array<Eigen::Matrix4f, 2> transformations{transformation, transformation_gt.value()};
        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
        for (auto &tn: transformations) {
            estimator_corr.buildInlierPairsAndEstimateMetric(tn, inlier_pairs_corr, error, metric_corr, rand);
            estimator_icp.buildInlierPairsAndEstimateMetric(tn, inlier_pairs_icp, error, metric_icp, rand);
            fout << "," << metric_corr << "," << metric_icp;
            fout << "," << inlier_pairs_corr.size() << "," << inlier_pairs_icp.size();
        }
        fout << "\n";
    }
}

void generateDebugFiles(const YamlConfig &config) {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    PointNCloud::Ptr src_aligned(new PointNCloud), src_aligned_gt(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    pcl::CorrespondencesPtr correct_correspondences(new pcl::Correspondences);
    std::vector<InlierPair> inlier_pairs;
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f transformation;
    std::optional<Eigen::Matrix4f> transformation_gt;
    std::string testname;
    float min_voxel_size, error;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt)) {
        parameters.testname = testname;
        bool success = false;
        std::string corrs_path = constructPath(parameters, "correspondences", "csv", true, false, false);
        correspondences = readCorrespondencesFromCSV(corrs_path, success);
        if (!success) {
            pcl::console::print_error("Failed to read correspondences for %s!\n", parameters.testname.c_str());
            exit(1);
        }
        PointNCloud::Ptr curr_src(new PointNCloud), curr_tgt(new PointNCloud);
        pcl::IndicesPtr indices_src{nullptr}, indices_tgt{nullptr};
        float voxel_size_src = FINE_VOXEL_SIZE_COEFFICIENT * calculatePointCloudDensity<PointN>(src);
        float voxel_size_tgt = FINE_VOXEL_SIZE_COEFFICIENT * calculatePointCloudDensity<PointN>(tgt);
        downsamplePointCloud(src, curr_src, voxel_size_src);
        downsamplePointCloud(tgt, curr_tgt, voxel_size_tgt);
        transformation = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                                           constructName(parameters, "transformation"));
        float error_thr = parameters.distance_thr;
        estimateNormalsPoints(NORMAL_NR_POINTS, curr_src, normals_src, parameters.normals_available);
        estimateNormalsPoints(NORMAL_NR_POINTS, curr_tgt, normals_tgt, parameters.normals_available);
        pcl::concatenateFields(*curr_src, *normals_src, *curr_src);
        pcl::concatenateFields(*curr_tgt, *normals_tgt, *curr_tgt);

        indices_src = detectKeyPoints(curr_src, parameters);
        indices_tgt = detectKeyPoints(curr_tgt, parameters);

        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
        auto metric_estimator = getMetricEstimatorFromParameters(parameters);
        metric_estimator->setSourceCloud(curr_src);
        metric_estimator->setTargetCloud(curr_tgt);
        metric_estimator->setInlierThreshold(parameters.distance_thr);
        metric_estimator->setCorrespondences(correspondences);
        metric_estimator->buildInlierPairs(transformation, inlier_pairs, error, rand);
        if (transformation_gt.has_value()) {
            buildCorrectCorrespondences(curr_src, curr_tgt, *correspondences, *correct_correspondences,
                                        transformation_gt.value(), error_thr);
            saveCorrespondences(curr_src, curr_tgt, *correspondences, transformation_gt.value(), parameters);
            saveCorrespondences(curr_src, curr_tgt, *correspondences, transformation_gt.value(), parameters, true);
            saveCorrespondenceDistances(curr_src, curr_tgt, *correspondences, transformation_gt.value(), parameters);
            saveColorizedPointCloud(curr_src, indices_src, *correspondences, *correct_correspondences, inlier_pairs,
                                    parameters, transformation_gt.value(), true);
            saveTemperatureMaps(curr_src, curr_tgt, "temperature_gt", parameters, error_thr, transformation_gt.value());
            saveTemperatureMaps(src, tgt, "temperature_gt_fullsize", parameters, error_thr, transformation_gt.value(), parameters.normals_available);
        }
        saveColorizedPointCloud(curr_tgt, indices_tgt, *correspondences, *correct_correspondences, inlier_pairs,
                                parameters, Eigen::Matrix4f::Identity(), false);

        if (parameters.metric_id == METRIC_WEIGHTED_CLOSEST_PLANE) {
            WeightFunction weight_function = getWeightFunction(parameters.weight_id);
            auto weights = weight_function(NORMAL_NR_POINTS, curr_src);
            saveColorizedWeights(curr_src, weights, "weights", parameters, transformation);
        }
        saveTemperatureMaps(curr_src, curr_tgt, "temperature", parameters, transformation);
        saveTemperatureMaps(src, tgt, "temperature_fullsize", parameters, transformation,parameters.normals_available);
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
    std::optional<Eigen::Matrix4f> transformation_gt;
    std::string testname;
    float min_voxel_size;
    int n_times = config.get<int>("n_times", 10);
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);
    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt)) {
        parameters.fix_seed = false;
        parameters.testname = testname;
        parameters.ground_truth = std::optional<Eigen::Matrix4f>(transformation_gt);
        if (parameters.save_features && transformation_gt.has_value()) {
            saveExtractedPointIds(src, tgt, transformation_gt.value(), parameters, tgt);
        }
        std::vector<float> rotation_errors;
        std::vector<float> translation_errors;
        std::vector<float> overlap_errors;
        std::vector<float> runtimes;
        int n_successful_times = 0;
        if (parameters.alignment_id != ALIGNMENT_DEFAULT) n_times = 1;
        for (int i = 0; i < n_times; ++i) {
            pcl::console::print_highlight("Starting alignment...\n");
            AlignmentResult result = alignPointClouds(src, tgt, parameters);
            AlignmentAnalysis analysis(result, parameters);
            if (analysis.alignmentHasConverged()) {
                analysis.start(transformation_gt, testname);
                bool success = analysis.getOverlapError() < parameters.distance_thr;
                if (success) {
                    rotation_errors.push_back(analysis.getRotationError());
                    translation_errors.push_back(analysis.getTranslationError());
                    overlap_errors.push_back(analysis.getOverlapError());
                    n_successful_times++;
                }
            }
            runtimes.push_back(analysis.getRunningTime());
        }
        float success_rate = (float) n_successful_times / (float) n_times;
        auto mean_rotation_error = calculate_mean<float>(rotation_errors);
        auto std_rotation_error = calculate_standard_deviation<float>(rotation_errors);
        auto mean_translation_error = calculate_mean<float>(translation_errors);
        auto std_translation_error = calculate_standard_deviation<float>(translation_errors);
        auto mean_overlap_error = calculate_mean<float>(overlap_errors);
        auto std_overlap_error = calculate_standard_deviation<float>(overlap_errors);
        auto mean_running_time = calculate_mean<float>(runtimes);
        auto std_running_time = calculate_standard_deviation<float>(runtimes);

        fout << constructName(parameters, "measure") << "," << success_rate << ","
             << mean_rotation_error << "," << std_rotation_error << ","
             << mean_translation_error << "," << std_translation_error << ","
             << mean_overlap_error << "," << std_overlap_error << ","
             << mean_running_time << "," << std_running_time << "\n";
    }
}

void runLoopTest(const YamlConfig &config) {
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    std::string filepath = constructPath("loop_test", "results", "csv", false);
    bool file_exists = std::filesystem::exists(filepath);
    std::fstream fout;
    if (!file_exists) {
        fout.open(filepath, std::ios_base::out);
    } else {
        fout.open(filepath, std::ios_base::app);
    }
    if (fout.is_open()) {
        if (!file_exists) {
            fout << "testname,r_err,t_err,pcd_err\n";
        }
    } else {
        perror(("error while opening file " + filepath).c_str());
    }
    for (auto &parameters: getParametersFromConfig(config, {}, {})) {
        PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
        std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
        std::string testname;
        Eigen::Matrix4f transformation_diff = Eigen::Matrix4f::Identity();
        std::vector<std::string> pcd_paths = config.getVector<std::string>("clouds").value();
        for (int i = 0; i < pcd_paths.size(); ++i) {
            pcl::console::print_highlight("Loop test %u/%u...\n", i + 1, pcd_paths.size());
            std::string src_path = pcd_paths[i];
            std::string tgt_path = pcd_paths[(i + 1) % pcd_paths.size()];
            std::optional<Eigen::Matrix4f> transformation_gt;
            loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt);
            loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(),
                                 transformation_gt);
            parameters.testname = testname;
            pcl::console::print_highlight("Starting alignment...\n");
            AlignmentResult result = alignPointClouds(src, tgt, parameters);
            AlignmentAnalysis analysis(result, parameters);
            if (analysis.alignmentHasConverged()) {
                analysis.start(transformation_gt, testname);
            }
            transformation_diff = analysis.getTransformation() * transformation_diff;
        }
        auto[r_err, t_err] = calculate_rotation_and_translation_errors(transformation_diff,
                                                                       Eigen::Matrix4f::Identity());
        float pcd_err = calculatePointCloudRmse(tgt, transformation_diff, Eigen::Matrix4f::Identity());
        fout << constructName(parameters, testname) << "," << r_err << "," << t_err << "," << pcd_err << "\n";
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
        } else if (test_type == "measure") {
            measureTestResults(config);
        } else if (test_type == "loop") {
            runLoopTest(config);
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

