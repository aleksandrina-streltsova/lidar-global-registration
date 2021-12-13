#include <Eigen/Core>
#include <string>

#include <pcl/io/ply_io.h>
#include <pcl/common/io.h>

#include "config.h"
#include "align.h"
#include "filter.h"
#include "downsample.h"

int main(int argc, char **argv) {
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    // Point clouds
    PointCloudT::Ptr src_fullsize(new PointCloudT), src(new PointCloudT), tgt(new PointCloudT);

    // Get input src and tgt
    if (argc != 2) {
        pcl::console::print_error("Syntax is: %s config.yaml\n", argv[0]);
        exit(1);
    }

    // Load parameters from config
    YamlConfig config;
    config.init(argv[1]);
    float voxel_size = config.get<float>("voxel_size").value();

    // Load src and tgt
    pcl::console::print_highlight("Loading point clouds...\n");
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    if (pcl::io::loadPLYFile<PointT>(src_path, *src_fullsize) < 0 ||
        pcl::io::loadPLYFile<PointT>(tgt_path, *tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }

    // Downsample
    if (config.get<bool>("downsample", true)) {
        pcl::console::print_highlight("Downsampling...\n");
        downsamplePointCloud(src_fullsize, src, voxel_size);
        downsamplePointCloud(tgt, tgt, voxel_size);
    } else {
        pcl::console::print_highlight("Filtering duplicate points...\n");
        filter_duplicate_points(src_fullsize);
        pcl::copyPointCloud(*src_fullsize, *src);
    }

    // Read ground truth transformation
    std::string csv_path = config.get<std::string>("ground_truth").value();
    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    Eigen::Matrix4f transformation_gt = getTransformation(csv_path, src_filename, tgt_filename);

    std::string testname = src_filename.substr(0, src_filename.find_last_of('.')) + '_' +
                           tgt_filename.substr(0, tgt_filename.find_last_of('.'));

    // Perform alignment
    pcl::console::print_highlight("Starting alignment...\n");

    AlignmentAnalysis analysis;
    auto descriptor_id = config.get<std::string>("descriptor", DEFAULT_DESCRIPTOR);
    if (descriptor_id == "fpfh") {
        analysis = align_point_clouds<pcl::FPFHSignature33>(src, tgt, config).getAlignmentAnalysis(config);
    } else if (descriptor_id == "usc") {
        analysis = align_point_clouds<pcl::UniqueShapeContext1960>(src, tgt, config).getAlignmentAnalysis(config);
    } else if (descriptor_id == "rops") {
        analysis = align_point_clouds<RoPS135>(src, tgt, config).getAlignmentAnalysis(config);
    } else {
        pcl::console::print_error("Descriptor isn't supported!\n");
    }
    analysis.start(transformation_gt, testname);
    if (config.get<bool>("debug", false)) {
        analysis.saveFilesForDebug(src_fullsize, testname);
    }
}

