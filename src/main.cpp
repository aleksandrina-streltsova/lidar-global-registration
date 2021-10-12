#include <Eigen/Core>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

#include "config.h"
#include "align.h"

int main(int argc, char **argv) {
    // Point clouds
    PointCloudT::Ptr src_fullsize(new PointCloudT), src(new PointCloudT), tgt(new PointCloudT);
    PointCloudT::Ptr src_aligned_gt(new PointCloudT), src_aligned(new PointCloudT);

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
    int iteration = config.get<int>("iteration").value();


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
    pcl::console::print_highlight("Downsampling...\n");
    downsamplePointCloud(src_fullsize, src, voxel_size);
    downsamplePointCloud(tgt, tgt, voxel_size);

    float model_size = getAABBDiagonal(src);
    float normal_radius = config.get<float>("normal_radius").value() * model_size;
    float feature_radius = config.get<float>("feature_radius").value() * model_size;

    // Estimate normals
    pcl::console::print_highlight("Estimating normals...\n");
    estimateNormals(normal_radius, src, normals_src);
    estimateNormals(normal_radius, tgt, normals_tgt);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    estimateFeatures(feature_radius, src, normals_src, features_src);
    estimateFeatures(feature_radius, tgt, normals_tgt, features_tgt);

    if (config.get<bool>("reciprocal").value()) {
        pcl::console::print_highlight("Filtering features (reciprocal)...\n");
        filterReciprocalCorrespondences(src, features_src, tgt, features_tgt);
    }

    // Read ground truth transformation
    std::string csv_path = config.get<std::string>("ground_truth").value();
    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    Eigen::Matrix4f transformation_gt = getTransformation(csv_path, src_filename, tgt_filename);

    // Perform alignment
    pcl::console::print_highlight("Starting alignment...\n");
    std::cout << "    iteration: " << iteration << std::endl;
    std::cout << "    voxel size: " << voxel_size << std::endl;
    Eigen::Matrix4f transformation = align(src, tgt, features_src, features_tgt, transformation_gt, config);

    pcl::transformPointCloud(*src_fullsize, *src_aligned, transformation);
    pcl::transformPointCloud(*src_fullsize, *src_aligned_gt, transformation_gt);
    pcl::io::savePLYFileBinary("source_aligned.ply", *src_aligned);
    pcl::io::savePLYFileBinary("source_aligned_gt.ply", *src_aligned_gt);

    return (0);
}

