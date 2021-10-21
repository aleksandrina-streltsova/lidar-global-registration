#include <fstream>

#include "align.h"
#include "sac_prerejective_omp.h"
#include "csv_parser.h"

float getAABBDiagonal(const PointCloudT::Ptr &pcd) {
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(pcd);
    feature_extractor.compute();

    PointT min_pointAABB, max_point_AABB;

    feature_extractor.getAABB(min_pointAABB, max_point_AABB);

    Eigen::Vector3f min_point(min_pointAABB.x, min_pointAABB.y, min_pointAABB.z);
    Eigen::Vector3f max_point(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);

    return (max_point - min_point).norm();
}

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename) {
    std::ifstream file(csv_path);
    Eigen::Matrix4f src_position, tgt_position;

    CSVRow row;
    while (file >> row) {
        if (row[0] == src_filename) {
            for (int i = 0; i < 16; ++i) {
                src_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
        } else if (row[0] == tgt_filename) {
            for (int i = 0; i < 16; ++i) {
                tgt_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
        }
    }
    return tgt_position.inverse() * src_position;
}

void downsamplePointCloud(const PointCloudT::Ptr &pcd_fullsize, PointCloudT::Ptr &pcd_down, float voxel_size) {
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    grid.setInputCloud(pcd_fullsize);
    pcl::console::print_highlight("Point cloud downsampled from %zu...", pcd_fullsize->size());
    grid.filter(*pcd_down);
    pcl::console::print_highlight("to %zu\n", pcd_down->size());
}

void estimateNormals(float radius_search, const PointCloudT::Ptr &pcd, PointCloudN::Ptr &normals) {
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_est;
    normal_est.setRadiusSearch(radius_search);

    normal_est.setInputCloud(pcd);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    normal_est.setSearchMethod(tree);
    normal_est.compute(*normals);
}

void estimateFeatures(float radius_search, const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                      FeatureCloudT::Ptr &features) {
    FeatureEstimationT fest;
    fest.setRadiusSearch(radius_search);
    fest.setInputCloud(pcd);
    fest.setInputNormals(normals);
    fest.compute(*features);
}

Eigen::Matrix4f align(const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                      const FeatureCloudT::Ptr &features_src, const FeatureCloudT::Ptr &features_tgt,
                      const Eigen::Matrix4f &transformation_gt, const YamlConfig &config, const std::string &testname) {
    SampleConsensusPrerejectiveOMP<PointT, PointT, FeatureT> align;
    PointCloudT src_aligned;

    if (config.get<bool>("reciprocal").value()) {
        align.enableMutualFiltering();
    }

    align.setInputSource(src);
    align.setSourceFeatures(features_src);

    align.setInputTarget(tgt);
    align.setTargetFeatures(features_tgt);

    float voxel_size = config.get<float>("voxel_size").value();

    align.setMaximumIterations(config.get<int>("iteration").value()); // Number of RANSAC iterations
    align.setNumberOfSamples(
            config.get<int>("n_samples").value()); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(config.get<int>("randomness").value()); // Number of nearest features to use
    align.setSimilarityThreshold(config.get<float>("edge_thr").value()); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(config.get<float>("distance_thr").value() * voxel_size); // Inlier threshold
    align.setInlierFraction(
            config.get<float>("inlier_fraction").value()); // Required inlier fraction for accepting a pose hypothesis
    {
        pcl::ScopeTime t("Alignment");
        align.align(src_aligned);
    }

    float error_thr = config.get<float>("error_thr").value();

    if (align.hasConverged()) {
        // Print results
        printf("\n");
        printTransformation(align.getFinalTransformation());
        printTransformation(transformation_gt);
        pcl::console::print_info("fitness: %0.7f\n",
                                 (float) align.getInliers().size() / (float) align.getCorrespondences().size());
        pcl::console::print_info("inliers_rmse: %0.7f\n", align.getRMSEScore());
        pcl::console::print_info("inliers: %i/%i\n", align.getInliers().size(), align.getCorrespondences().size());
        pcl::console::print_info("correct inliers: %i/%i\n",
                                 align.countCorrectCorrespondences(transformation_gt, error_thr),
                                 align.getCorrespondences().size());
    } else {
        pcl::console::print_error("Alignment failed!\n");
        exit(1);
    }
    saveColorizedPointCloud(src, align.getCorrespondences(), align.getInliers(), testname);
    return align.getFinalTransformation();
}

