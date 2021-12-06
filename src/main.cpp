#include <Eigen/Core>
#include <string>
#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/common/io.h>

#include "config.h"
#include "align.h"
#include "filter.h"
#include "downsample.h"
#include "feature_analysis.h"

void startAlignment(PointCloudT::Ptr &src_fullsize, PointCloudT::Ptr &src, PointCloudT::Ptr &tgt,
                    FeatureCloudT::Ptr &features_src, FeatureCloudT::Ptr &features_tgt,
                    const Eigen::Matrix4f &transformation_gt, const std::string &testname, const YamlConfig &config) {
    // Filter point clouds
    auto func_id = config.get<std::string>("filter", "");
    auto func = getUniquenessFunction(func_id);
    if (func != nullptr) {
        std::cout << "Point cloud downsampled after filtration (" << func_id << ") from " << src->size();
        filterPointCloud(func, func_id, src, features_src, src, features_src, transformation_gt, testname, true);
        std::cout << " to " << src->size() << "\n";
        std::cout << "Point cloud downsampled after filtration (" << func_id << ") from " << tgt->size();
        filterPointCloud(func, func_id, tgt, features_tgt, tgt, features_tgt, transformation_gt, testname, false);
        std::cout << " to " << tgt->size() << "\n";
    }

    // Perform alignment
    pcl::console::print_highlight("Starting alignment...\n");
    auto align = align_point_clouds(src, tgt, features_src, features_tgt, config);
    analyzeAlignment(src_fullsize, src, tgt, align, transformation_gt, config, testname);
}

void saveFeatureHistograms(const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                           const FeatureCloudT::Ptr &features_src, const FeatureCloudT::Ptr &features_tgt,
                           const PointCloudN::Ptr &normals_src, const PointCloudN::Ptr &normals_tgt,
                           float feature_radius, const Eigen::Matrix4f &transformation_gt,
                           const std::string &testname) {
    saveFeatures(feature_radius, src, normals_src, testname, true);
    saveFeatures(feature_radius, tgt, normals_tgt, testname, false);
    saveNormals(src, normals_src, transformation_gt, true, testname);
    saveNormals(tgt, normals_tgt, Eigen::Matrix4f::Identity(), false, testname);
    saveHistograms(features_src, testname, true);
    saveHistograms(features_tgt, testname, false);
}

int main(int argc, char **argv) {
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    // Point clouds
    PointCloudT::Ptr src_fullsize(new PointCloudT), src(new PointCloudT), tgt(new PointCloudT);

    PointCloudN::Ptr normals_src(new PointCloudN), normals_tgt(new PointCloudN);
    FeatureCloudT::Ptr features_src(new FeatureCloudT), features_tgt(new FeatureCloudT);

    // Get input src and tgt
    if (argc != 2) {
        pcl::console::print_error("Syntax is: %s config.yaml\n", argv[0]);
        return (1);
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
        return (1);
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

    float normal_radius = config.get<float>("normal_radius_coef").value() * voxel_size;
    float feature_radius = config.get<float>("feature_radius_coef").value() * voxel_size;

    // Estimate normals
    pcl::console::print_highlight("Estimating normals...\n");
    estimateNormals(normal_radius, src, normals_src);
    estimateNormals(normal_radius, tgt, normals_tgt);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    estimateFeatures(feature_radius, src, normals_src, features_src);
    estimateFeatures(feature_radius, tgt, normals_tgt, features_tgt);

    // Read ground truth transformation
    std::string csv_path = config.get<std::string>("ground_truth").value();
    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    Eigen::Matrix4f transformation_gt = getTransformation(csv_path, src_filename, tgt_filename);

    std::string testname = src_filename.substr(0, src_filename.find_last_of('.')) + '_' +
                           tgt_filename.substr(0, tgt_filename.find_last_of('.'));

    startAlignment(src_fullsize, src, tgt, features_src, features_tgt, transformation_gt, testname, config);
    return (0);
}

