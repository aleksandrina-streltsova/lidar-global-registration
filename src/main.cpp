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
    Eigen::Matrix4f transformation_gt;
    std::string testname;
    float min_voxel_size;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt,
                    config.get<float>("density"), min_voxel_size);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);
    std::vector<AlignmentAnalysis> analyses;
    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        parameters.testname = testname;
        parameters.ground_truth = std::optional<Eigen::Matrix4f>(transformation_gt);
        if (parameters.save_features) {
            saveExtractedPointIds(src, tgt, transformation_gt, parameters, tgt);
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

    PointNCloud::Ptr src_fullsize(new PointNCloud), tgt_fullsize(new PointNCloud);
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f transformation_gt;
    std::string testname;
    float min_voxel_size;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src_fullsize, tgt_fullsize, fields_src, fields_tgt,
                    config.get<float>("density"), min_voxel_size);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        parameters.testname = testname;
        if (parameters.coarse_to_fine) {
            downsamplePointCloud(src_fullsize, src, parameters);
            downsamplePointCloud(tgt_fullsize, tgt, parameters);
        } else {
            src = src_fullsize;
            tgt = tgt_fullsize;
        }
        std::vector<float> voxel_sizes;
        std::vector<std::string> matching_ids;
        getIterationsInfo(fs::path(DATA_DEBUG_PATH) / fs::path(ITERATIONS_CSV),
                          constructName(parameters, "iterations"), voxel_sizes, matching_ids);
        for (int i = 0; i < voxel_sizes.size(); ++i) {
            AlignmentParameters curr_parameters(parameters);
            curr_parameters.voxel_size = voxel_sizes[i];
            curr_parameters.matching_id = matching_ids[i];
            PointNCloud::Ptr curr_src(new PointNCloud), curr_tgt(new PointNCloud);
            downsamplePointCloud(src, curr_src, curr_parameters);
            downsamplePointCloud(tgt, curr_tgt, curr_parameters);
            float normal_radius = curr_parameters.normal_radius_coef * curr_parameters.voxel_size;
            auto tn_name = config.get<std::string>("transformation", constructName(curr_parameters, "transformation"));
            auto transformation = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV), tn_name);

            estimateNormalsRadius(normal_radius, curr_src, normals_src, false);
            estimateNormalsRadius(normal_radius, curr_tgt, normals_tgt, false);
            pcl::concatenateFields(*curr_src, *normals_src, *curr_src);
            pcl::concatenateFields(*curr_tgt, *normals_tgt, *curr_tgt);
            CorrespondencesMetricEstimator estimator_corr;
            ClosestPlaneMetricEstimator estimator_icp;
            pcl::CorrespondencesPtr correspondences;
            std::vector<InlierPair> inlier_pairs_corr, inlier_pairs_icp;
            float error, metric_icp, metric_corr;
            bool success = false;
            std::string corrs_path = constructPath(curr_parameters, "correspondences", "csv", true, false, false);
            correspondences = readCorrespondencesFromCSV(corrs_path, success);
            if (!success) {
                pcl::console::print_error("Failed to read correspondences for %s!\n", curr_parameters.testname.c_str());
                exit(1);
            }

            estimator_corr.setSourceCloud(curr_src);
            estimator_corr.setTargetCloud(curr_tgt);
            estimator_corr.setInlierThreshold(curr_parameters.voxel_size * curr_parameters.distance_thr_coef);
            estimator_corr.setCorrespondences(correspondences);

            estimator_icp.setSourceCloud(curr_src);
            estimator_icp.setTargetCloud(curr_tgt);
            estimator_icp.setInlierThreshold(curr_parameters.voxel_size * curr_parameters.distance_thr_coef);
            estimator_icp.setCorrespondences(correspondences);

            fout << constructName(curr_parameters, "metric", true, false, false);
            std::array<Eigen::Matrix4f, 2> transformations{transformation, transformation_gt};
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
}

void generateDebugFiles(const YamlConfig &config) {
    PointNCloud::Ptr src_fullsize(new PointNCloud), tgt_fullsize(new PointNCloud);
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    PointNCloud::Ptr src_fullsize_aligned(new PointNCloud), src_fullsize_aligned_gt(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    pcl::CorrespondencesPtr correspondences, correct_correspondences(new pcl::Correspondences);
    std::vector<InlierPair> inlier_pairs;
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f transformation, transformation_gt;
    std::string testname;
    float min_voxel_size, error;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src_fullsize, tgt_fullsize, fields_src, fields_tgt,
                    config.get<float>("density"), min_voxel_size);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        parameters.testname = testname;
        if (parameters.coarse_to_fine) {
            downsamplePointCloud(src_fullsize, src, parameters);
            downsamplePointCloud(tgt_fullsize, tgt, parameters);
        } else {
            src = src_fullsize;
            tgt = tgt_fullsize;
        }
        std::vector<float> voxel_sizes;
        std::vector<std::string> matching_ids;
        getIterationsInfo(fs::path(DATA_DEBUG_PATH) / fs::path(ITERATIONS_CSV),
                          constructName(parameters, "iterations"), voxel_sizes, matching_ids);
        for (int i = 0; i < voxel_sizes.size(); ++i) {
            AlignmentParameters curr_parameters(parameters);
            curr_parameters.voxel_size = voxel_sizes[i];
            curr_parameters.matching_id = matching_ids[i];
            bool success = false;
            std::string corrs_path = constructPath(curr_parameters, "correspondences", "csv", true, false, false);
            correspondences = readCorrespondencesFromCSV(corrs_path, success);
            if (!success) {
                pcl::console::print_error("Failed to read correspondences for %s!\n", curr_parameters.testname.c_str());
                exit(1);
            }
            PointNCloud::Ptr curr_src(new PointNCloud), curr_tgt(new PointNCloud);
            pcl::IndicesPtr indices_src{nullptr}, indices_tgt{nullptr};
            downsamplePointCloud(src, curr_src, curr_parameters);
            downsamplePointCloud(tgt, curr_tgt, curr_parameters);
            transformation = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                                               constructName(curr_parameters, "transformation"));
            float normal_radius = curr_parameters.normal_radius_coef * curr_parameters.voxel_size;
            float error_thr = curr_parameters.distance_thr_coef * curr_parameters.voxel_size;
            estimateNormalsRadius(normal_radius, curr_src, normals_src, false);
            estimateNormalsRadius(normal_radius, curr_tgt, normals_tgt, false);
            pcl::concatenateFields(*curr_src, *normals_src, *curr_src);
            pcl::concatenateFields(*curr_tgt, *normals_tgt, *curr_tgt);

            indices_src = detectKeyPoints(curr_src, parameters);
            indices_tgt = detectKeyPoints(curr_tgt, parameters);

            UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
            auto metric_estimator = getMetricEstimatorFromParameters(curr_parameters);
            metric_estimator->setSourceCloud(curr_src);
            metric_estimator->setTargetCloud(curr_tgt);
            metric_estimator->setInlierThreshold(curr_parameters.voxel_size * curr_parameters.distance_thr_coef);
            metric_estimator->setCorrespondences(correspondences);
            metric_estimator->buildInlierPairs(transformation, inlier_pairs, error, rand);
            buildCorrectCorrespondences(curr_src, curr_tgt, *correspondences, *correct_correspondences, transformation_gt,
                                        error_thr);
            saveCorrespondences(curr_src, curr_tgt, *correspondences, transformation_gt, curr_parameters);
            saveCorrespondences(curr_src, curr_tgt, *correspondences, transformation_gt, curr_parameters, true);
            saveCorrespondenceDistances(curr_src, curr_tgt, *correspondences, transformation_gt, curr_parameters.voxel_size, curr_parameters);
            saveColorizedPointCloud(curr_src, indices_src, *correspondences, *correct_correspondences, inlier_pairs,
                                    curr_parameters, transformation_gt, true);
            saveColorizedPointCloud(curr_tgt, indices_tgt, *correspondences, *correct_correspondences, inlier_pairs,
                                    curr_parameters, Eigen::Matrix4f::Identity(), false);
            saveCorrespondencesDebug(*correspondences, *correct_correspondences, curr_parameters);

            if (curr_parameters.metric_id == METRIC_WEIGHTED_CLOSEST_PLANE) {
                WeightFunction weight_function = getWeightFunction(curr_parameters.weight_id);
                auto weights = weight_function(2.f * curr_parameters.normal_radius_coef * curr_parameters.voxel_size,
                                               curr_src);
                saveColorizedWeights(curr_src, weights, "weights", curr_parameters, transformation_gt);
            }
            saveTemperatureMaps(curr_src, curr_tgt, "temperature", curr_parameters, transformation);
//            saveTemperatureMaps(curr_src, curr_tgt, "temperature_gt", curr_parameters, transformation_gt);
        }
        saveTemperatureMaps(src_fullsize, tgt_fullsize, "temperature_fullsize", parameters, transformation,
                            parameters.normals_available);
//        saveTemperatureMaps(src_fullsize, tgt_fullsize, "temperature_gt_fullsize", parameters, transformation_gt);
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
    Eigen::Matrix4f transformation_gt;
    std::string testname;
    float min_voxel_size;
    int n_times = config.get<int>("n_times", 10);
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt,
                    config.get<float>("density"), min_voxel_size);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);
    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        parameters.fix_seed = false;
        parameters.testname = testname;
        parameters.ground_truth = std::optional<Eigen::Matrix4f>(transformation_gt);
        if (parameters.save_features) {
            saveExtractedPointIds(src, tgt, transformation_gt, parameters, tgt);
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
                bool success = analysis.getOverlapError() < parameters.distance_thr_coef * parameters.voxel_size;
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
    for (auto &parameters: getParametersFromConfig(config, {}, {}, config.get<float>("density").value())) {
        PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
        std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
        std::string testname;
        Eigen::Matrix4f transformation_diff = Eigen::Matrix4f::Identity();
        std::vector<std::string> pcd_paths = config.getVector<std::string>("clouds").value();
        for (int i = 0; i < pcd_paths.size(); ++i) {
            pcl::console::print_highlight("Loop test %u/%u...\n", i + 1, pcd_paths.size());
            std::string src_path = pcd_paths[i];
            std::string tgt_path = pcd_paths[(i + 1) % pcd_paths.size()];
            Eigen::Matrix4f transformation_gt;
            float min_voxel_size;
            loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt,
                            config.get<float>("density"), min_voxel_size);
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
        float pcd_err = calculate_point_cloud_rmse(tgt, transformation_diff, Eigen::Matrix4f::Identity());
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

