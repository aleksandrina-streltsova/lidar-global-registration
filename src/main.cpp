#include <Eigen/Core>
#include <string>
#include <filesystem>

#include <pcl/io/ply_io.h>
#include <pcl/common/io.h>

#include "config.h"
#include "align.h"
#include "filter.h"
#include "downsample.h"

namespace fs = std::filesystem;

std::vector<AlignmentAnalysis> runTest(const YamlConfig &config) {
    // Point clouds
    PointCloudT::Ptr src(new PointCloudT), tgt(new PointCloudT);

    std::vector<AlignmentParameters> parameters_container = getParametersFromConfig(config);

    // Load src and tgt
    pcl::console::print_highlight("Loading point clouds...\n");
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    if (pcl::io::loadPLYFile<PointT>(src_path, *src) < 0 ||
        pcl::io::loadPLYFile<PointT>(tgt_path, *tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }
    filter_duplicate_points(src);
    filter_duplicate_points(tgt);

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
        } else if (descriptor_id == "shot"){
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

int main(int argc, char **argv) {
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    // Get input src and tgt
    if (argc != 2) {
        pcl::console::print_error("Syntax is: %s config.yaml\n", argv[0]);
        exit(1);
    }

    // Load parameters from config
    YamlConfig config;
    config.init(argv[1]);
    auto tests = config.get<std::vector<YAML::Node>>("tests");
    if (tests.has_value()) {
        std::string filename = fs::path(argv[1]).filename();
        runTests(tests.value(), filename.erase(filename.length() - 5));
    } else {
        runTest(config);
    }
}

