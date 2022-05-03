#include <Eigen/Core>
#include <string>
#include <filesystem>
#include <array>

#include "io.h"
#include "config.h"
#include "align.h"
#include "filter.h"
#include "downsample.h"

namespace fs = std::filesystem;

const std::string ALIGNMENT = "alignment";
const std::string METRIC_ANALYSIS = "metric";
const std::string DEBUG = "debug";

void loadPointClouds(const YamlConfig &config, std::string &testname,
                     PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                     std::vector<::pcl::PCLPointField> &fields_src, std::vector<::pcl::PCLPointField> &fields_tgt,
                     float &min_voxel_size) {
    pcl::console::print_highlight("Loading point clouds...\n");
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    if (loadPLYFile<PointN>(src_path, *src, fields_src) < 0 ||
        loadPLYFile<PointN>(tgt_path, *tgt, fields_tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }
    filter_duplicate_points(src);
    filter_duplicate_points(tgt);

    float src_density = calculatePointCloudDensity<PointN>(src);
    float tgt_density = calculatePointCloudDensity<PointN>(tgt);
    PCL_DEBUG("[loadPointClouds] src density: %.5f, tgt density: %.5f.\n", src_density, tgt_density);
    min_voxel_size = std::max(src_density, tgt_density);

    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    testname = src_filename.substr(0, src_filename.find_last_of('.')) + '_' +
               tgt_filename.substr(0, tgt_filename.find_last_of('.'));
}

void loadTransformationGt(const YamlConfig &config, Eigen::Matrix4f &transformation_gt) {
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    std::string csv_path = config.get<std::string>("ground_truth").value();
    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    transformation_gt = getTransformation(csv_path, src_filename, tgt_filename);
}

std::vector<AlignmentAnalysis> runTest(const YamlConfig &config) {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f transformation_gt;
    std::string testname;
    float min_voxel_size;

    loadPointClouds(config, testname, src, tgt, fields_src, fields_tgt, min_voxel_size);
    loadTransformationGt(config, transformation_gt);
    std::vector<AlignmentAnalysis> analyses;
    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        parameters.testname = testname;
        parameters.ground_truth = std::make_shared<Eigen::Matrix4f>(transformation_gt);
        if (parameters.save_features) {
            saveExtractedPointIds(src, tgt, transformation_gt, parameters, tgt);
        }
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
            saveTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                               constructName(parameters, "transformation"), analysis.getTransformation());
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
    NormalCloud::Ptr normals_src(new NormalCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f transformation_gt;
    std::string testname;
    float min_voxel_size;

    loadPointClouds(config, testname, src_fullsize, tgt_fullsize, fields_src, fields_tgt, min_voxel_size);
    loadTransformationGt(config, transformation_gt);

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        parameters.testname = testname;
        float normal_radius = parameters.normal_radius_coef * parameters.voxel_size;
        auto tn_name = config.get<std::string>("transformation", constructName(parameters, "transformation"));
        auto transformation = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV), tn_name);
        downsamplePointCloud(src_fullsize, src, parameters);
        downsamplePointCloud(tgt_fullsize, tgt, parameters);
        estimateNormals(normal_radius, src, normals_src, false);
        pcl::concatenateFields(*src, *normals_src, *src);
        CorrespondencesMetricEstimator estimator_corr;
        ClosestPointMetricEstimator estimator_icp;
        pcl::Correspondences correspondences;
        std::vector<InlierPair> inlier_pairs_corr, inlier_pairs_icp;
        float error, metric_icp, metric_corr;
        bool success = false;
        readCorrespondencesFromCSV(constructPath(parameters, "correspondences", "csv", true, false, false),
                                   correspondences, success);
        if (!success) {
            pcl::console::print_error("Failed to read correspondences for %s!\n", parameters.testname.c_str());
            exit(1);
        }

        estimator_corr.setSourceCloud(src);
        estimator_corr.setTargetCloud(tgt);
        estimator_corr.setInlierThreshold(parameters.voxel_size * parameters.distance_thr_coef);
        estimator_corr.setCorrespondences(correspondences);

        estimator_icp.setSourceCloud(src);
        estimator_icp.setTargetCloud(tgt);
        estimator_icp.setInlierThreshold(parameters.voxel_size * parameters.distance_thr_coef);
        estimator_icp.setCorrespondences(correspondences);

        fout << constructName(parameters, "metric", true, false, false);
        std::array<Eigen::Matrix4f, 2> transformations{transformation, transformation_gt};
        for (auto &tn: transformations) {
            estimator_corr.buildInlierPairsAndEstimateMetric(tn, inlier_pairs_corr, error, metric_corr);
            estimator_icp.buildInlierPairsAndEstimateMetric(tn, inlier_pairs_icp, error, metric_icp);
            fout << "," << metric_corr << "," << metric_icp;
            fout << "," << inlier_pairs_corr.size() << "," << inlier_pairs_icp.size();
        }
        fout << "\n";
    }
}

void generateDebugFiles(const YamlConfig &config) {
    PointNCloud::Ptr src_fullsize(new PointNCloud), tgt_fullsize(new PointNCloud);
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    PointNCloud::Ptr src_fullsize_aligned(new PointNCloud), src_fullsize_aligned_gt(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    pcl::Correspondences correspondences, correct_correspondences;
    std::vector<InlierPair> inlier_pairs;
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f transformation, transformation_gt;
    std::string testname;
    float min_voxel_size, error;

    loadPointClouds(config, testname, src_fullsize, tgt_fullsize, fields_src, fields_tgt, min_voxel_size);
    loadTransformationGt(config, transformation_gt);

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        parameters.testname = testname;
        transformation = getTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                                           constructName(parameters, "transformation"));
        float normal_radius = parameters.normal_radius_coef * parameters.voxel_size;
        float error_thr = parameters.distance_thr_coef * parameters.voxel_size;
        downsamplePointCloud(src_fullsize, src, parameters);
        downsamplePointCloud(tgt_fullsize, tgt, parameters);
        estimateNormals(normal_radius, src, normals_src, false);
        estimateNormals(normal_radius, tgt, normals_tgt, false);
        pcl::concatenateFields(*src, *normals_src, *src);
        pcl::concatenateFields(*tgt, *normals_tgt, *tgt);
        bool success = false;
        readCorrespondencesFromCSV(constructPath(parameters, "correspondences", "csv", true, false, false),
                                   correspondences, success);
        if (!success) {
            pcl::console::print_error("Failed to read correspondences for %s!\n", parameters.testname.c_str());
            exit(1);
        }
        auto metric_estimator = getMetricEstimator(parameters);
        metric_estimator->setSourceCloud(src);
        metric_estimator->setTargetCloud(tgt);
        metric_estimator->setInlierThreshold(parameters.voxel_size * parameters.distance_thr_coef);
        metric_estimator->setCorrespondences(correspondences);
        metric_estimator->buildInlierPairs(transformation, inlier_pairs, error);
        buildCorrectCorrespondences(src, tgt, correspondences, correct_correspondences, transformation_gt, error_thr);
        saveCorrespondences(src, tgt, correspondences, transformation_gt, parameters);
        saveCorrespondences(src, tgt, correspondences, transformation_gt, parameters, true);
        saveCorrespondenceDistances(src, tgt, correspondences, transformation_gt, parameters.voxel_size, parameters);
        saveColorizedPointCloud(src, correspondences, correct_correspondences, inlier_pairs, parameters, transformation_gt, true);
        saveColorizedPointCloud(tgt, correspondences, correct_correspondences, inlier_pairs, parameters, Eigen::Matrix4f::Identity(), false);
        saveCorrespondencesDebug(correspondences, correct_correspondences, parameters);

        if (parameters.metric_id == METRIC_WEIGHTED_CLOSEST_POINT) {
            WeightFunction weight_function = getWeightFunction(parameters.weight_id);
            auto weights = weight_function(2.f * parameters.normal_radius_coef * parameters.voxel_size, src);
            saveColorizedWeights(src, weights, "weights", parameters, transformation_gt);
        }

        pcl::transformPointCloud(*src_fullsize, *src_fullsize_aligned, transformation);
        pcl::transformPointCloud(*src_fullsize, *src_fullsize_aligned_gt, transformation_gt);
        pcl::io::savePLYFileBinary(constructPath(parameters, "aligned"), *src_fullsize_aligned);
        pcl::io::savePLYFileBinary(constructPath(parameters.testname, "aligned_gt", "ply", false),
                                   *src_fullsize_aligned_gt);
    }
}

void processTests(const std::vector<YAML::Node> &tests, const std::string &command) {
    for (auto &test: tests) {
        YamlConfig config;
        config.config = (*test.begin()).second;
        if (command == ALIGNMENT) {
            runTest(config);
        } else if (command == METRIC_ANALYSIS) {
            estimateTestMetric(config);
        } else if (command == DEBUG) {
            generateDebugFiles(config);
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

