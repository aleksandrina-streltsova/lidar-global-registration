#include <Eigen/Core>
#include <string>
#include <filesystem>

#include "io.h"
#include "config.h"
#include "align.h"
#include "filter.h"
#include "downsample.h"

namespace fs = std::filesystem;

std::vector<AlignmentAnalysis> runTest(const YamlConfig &config) {
    // Point clouds
    PointCloudTN::Ptr src(new PointCloudTN), tgt(new PointCloudTN);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;

    // Load src and tgt
    pcl::console::print_highlight("Loading point clouds...\n");
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    if (loadPLYFile<PointTN>(src_path, *src, fields_src) < 0 ||
        loadPLYFile<PointTN>(tgt_path, *tgt, fields_tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }
    std::vector<AlignmentParameters> parameters_container = getParametersFromConfig(config, fields_src, fields_tgt);
    filter_duplicate_points(src);
    filter_duplicate_points(tgt);
    float src_density = calculatePointCloudDensity<PointTN>(src);
    float tgt_density = calculatePointCloudDensity<PointTN>(tgt);
    PCL_DEBUG("[runTest] src density: %.5f, tgt density: %.5f.\n", src_density, tgt_density);

    // Read ground truth transformation
    std::string csv_path = config.get<std::string>("ground_truth").value();
    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    Eigen::Matrix4f transformation_gt = getTransformation(csv_path, src_filename, tgt_filename);

    std::string testname = src_filename.substr(0, src_filename.find_last_of('.')) + '_' +
                           tgt_filename.substr(0, tgt_filename.find_last_of('.'));

    std::vector<AlignmentAnalysis> analyses;
    for (auto &parameters: parameters_container) {
        parameters.testname = testname;
        parameters.ground_truth = std::make_shared<Eigen::Matrix4f>(transformation_gt);
        if (parameters.save_features) {
            saveExtractedPointIds(src, tgt, transformation_gt, parameters, tgt);
        }
        // Perform alignment
        pcl::console::print_highlight("Starting alignment...\n");

        AlignmentAnalysis analysis;
        auto descriptor_id = parameters.descriptor_id;
        if (descriptor_id == "fpfh") {
            analysis = align_point_clouds<FPFH>(src, tgt, parameters).getAlignmentAnalysis(parameters);
        } else if (descriptor_id == "usc") {
            analysis = align_point_clouds<USC>(src, tgt, parameters).getAlignmentAnalysis(parameters);
        } else if (descriptor_id == "rops") {
            analysis = align_point_clouds<RoPS135>(src, tgt, parameters).getAlignmentAnalysis(parameters);
        } else if (descriptor_id == "shot") {
            analysis = align_point_clouds<SHOT>(src, tgt, parameters).getAlignmentAnalysis(parameters);
        } else {
            pcl::console::print_error("Descriptor %s isn't supported!\n", descriptor_id.c_str());
        }
        if (analysis.alignmentHasConverged()) {
            analysis.start(transformation_gt, testname);
            if (config.get<bool>("debug", false)) {
                analysis.saveFilesForDebug(src, parameters);
            }
        }
        analyses.push_back(analysis);
    }
    return analyses;
}

void runTests(const std::vector<YAML::Node> &tests, const std::string &testname) {
    std::string filepath = constructPath(testname, "results", "csv");
    std::fstream fout;
    fout.open(filepath, std::ios_base::out);
    if (!fout.is_open()) {
        perror(("error while opening file " + filepath).c_str());
    }
    printAnalysisHeader(fout);
    for (auto &test: tests) {
        YamlConfig config;
        config.config = (*test.begin()).second;
        auto analyses = runTest(config);
        for (const auto &analysis: analyses) {
            fout << analysis;
        }
    }
    fout.close();
}

void estimateTestMetric(std::fstream &fout, const YamlConfig &config) {
    // Point clouds
    PointCloudTN::Ptr src_fullsize(new PointCloudTN), tgt_fullsize(new PointCloudTN);
    PointCloudTN::Ptr src(new PointCloudTN), tgt(new PointCloudTN);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;

    // Load src and tgt
    pcl::console::print_highlight("Loading point clouds...\n");
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    if (loadPLYFile<PointTN>(src_path, *src_fullsize, fields_src) < 0 ||
        loadPLYFile<PointTN>(tgt_path, *tgt_fullsize, fields_tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }
    std::vector<AlignmentParameters> parameters_container = getParametersFromConfig(config, fields_src, fields_tgt);
    filter_duplicate_points(src_fullsize);
    filter_duplicate_points(tgt_fullsize);

    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    std::string testname = src_filename.substr(0, src_filename.find_last_of('.')) + '_' +
                           tgt_filename.substr(0, tgt_filename.find_last_of('.'));
    Eigen::Matrix4f transformation = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV), config.get<std::string>("transformation").value());

    for (auto &parameters: parameters_container) {
        parameters.testname = testname;
        downsamplePointCloud(src_fullsize, src, parameters);
        downsamplePointCloud(tgt_fullsize, tgt, parameters);
        CorrespondencesMetricEstimator estimator_corr;
        ClosestPointMetricEstimator estimator_icp;
        std::vector<MultivaluedCorrespondence> correspondences;
        std::vector<InlierPair> inlier_pairs_corr, inlier_pairs_icp;
        float error, metric_icp, metric_corr;
        bool success = false;
        readCorrespondencesFromCSV(constructPath(parameters, "correspondences", "csv", true, false), correspondences, success);
        if (!success) {
            pcl::console::print_error("Failed to read correspondences for %s!\n", parameters.testname.c_str());
            exit(1);
        }

        estimator_corr.setSourceCloud(src);
        estimator_corr.setTargetCloud(tgt);
        estimator_corr.setInlierThreshold(parameters.voxel_size * parameters.distance_thr_coef);
        estimator_corr.setCorrespondences(correspondences);
        estimator_corr.buildInlierPairs(transformation, inlier_pairs_corr, error);
        estimator_corr.estimateMetric(inlier_pairs_corr, metric_corr);

        estimator_icp.setSourceCloud(src);
        estimator_icp.setTargetCloud(tgt);
        estimator_icp.setInlierThreshold(parameters.voxel_size * parameters.distance_thr_coef);
        estimator_icp.setCorrespondences(correspondences);
        estimator_icp.buildInlierPairs(transformation, inlier_pairs_icp, error);
        estimator_icp.estimateMetric(inlier_pairs_icp, metric_icp);

        fout << constructName(parameters, "metric", true, false) << ",";
        fout << metric_corr << "," << metric_icp << ",";
        fout << inlier_pairs_corr.size() << "," << inlier_pairs_icp.size() << "\n";
    }
}

void estimateTestsMetrics(const std::vector<YAML::Node> &tests) {
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
            fout << "testname,metric_corr,metric_icp,inliers_corr,inliers_icp\n";
        }
    } else {
        perror(("error while opening file " + filepath).c_str());
    }
    for (auto &test: tests) {
        YamlConfig config;
        config.config = (*test.begin()).second;
        estimateTestMetric(fout, config);
    }
    fout.close();
}

int main(int argc, char **argv) {
    if (argc != 3 && !(strcmp(argv[1], "alignment") == 0 || strcmp(argv[1], "metric") == 0)) {
        pcl::console::print_error("Syntax is: [alignment, analysis] %s config.yaml\n", argv[0]);
        exit(1);
    }
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    // Load parameters from config
    YamlConfig config;
    config.init(argv[2]);
    auto tests = config.get<std::vector<YAML::Node>>("tests");
    bool align = strcmp(argv[1], "alignment") == 0;
    if (tests.has_value()) {
        std::string filename = fs::path(argv[2]).filename();
        if (align) {
            runTests(tests.value(), filename.erase(filename.length() - 5));
        } else {
            estimateTestsMetrics(tests.value());
        }
    } else if (align) {
        runTest(config);
    }
}

