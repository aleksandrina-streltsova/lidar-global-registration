#include <fstream>

#include <pcl/console/print.h>
#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/time.h>
#include <pcl/registration/correspondence_estimation.h>

#include "align.h"
#include "csv_parser.h"

void printTransformation(const Eigen::Matrix4f &transformation) {
    pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1),
                             transformation(0, 2));
    pcl::console::print_info("R = | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1),
                             transformation(1, 2));
    pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1),
                             transformation(2, 2));
    pcl::console::print_info("\n");
    pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3),
                             transformation(2, 3));
    pcl::console::print_info("\n");
}

int countCorrectCorrespondences(const Eigen::Matrix4f &transformation_gt,
                                const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                                std::vector<int> inliers, float error_threshold) {
    PointCloudT::Ptr src_aligned(new PointCloudT);
    pcl::transformPointCloud(*src, *src_aligned, transformation_gt);

    int correct_correspondences = 0;
    for (int i = 0; i < inliers.size(); ++i) {
        int idx = inliers[i];
        float e = pcl::L1_Norm(src_aligned->points[idx].data, tgt->points[idx].data, 3);
        if (e < error_threshold) {
            correct_correspondences++;
        }
    }
    return correct_correspondences;
}

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

void filterReciprocalCorrespondences(PointCloudT::Ptr &src, FeatureCloudT::Ptr &features_src,
                                     PointCloudT::Ptr &tgt, FeatureCloudT::Ptr &features_tgt) {
    pcl::registration::CorrespondenceEstimation<FeatureT, FeatureT> est;
    est.setInputSource(features_src);
    est.setInputTarget(features_tgt);
    pcl::CorrespondencesPtr reciprocal_correspondences(new pcl::Correspondences);
    est.determineReciprocalCorrespondences(*reciprocal_correspondences);

    PointCloudT::Ptr src_reciprocal(new PointCloudT), tgt_reciprocal(new PointCloudT);
    FeatureCloudT::Ptr reciprocal_features_src(new FeatureCloudT), reciprocal_features_tgt(new FeatureCloudT);
    src_reciprocal->resize(reciprocal_correspondences->size());
    tgt_reciprocal->resize(reciprocal_correspondences->size());
    reciprocal_features_src->resize(reciprocal_correspondences->size());
    reciprocal_features_tgt->resize(reciprocal_correspondences->size());
    for (std::size_t i = 0; i < reciprocal_correspondences->size(); ++i) {
        pcl::Correspondence correspondence = (*reciprocal_correspondences)[i];
        src_reciprocal->points[i] = src->points[correspondence.index_query];
        tgt_reciprocal->points[i] = tgt->points[correspondence.index_match];
        reciprocal_features_src->points[i] = features_src->points[correspondence.index_query];
        reciprocal_features_tgt->points[i] = features_tgt->points[correspondence.index_match];
    }
    src = src_reciprocal;
    tgt = tgt_reciprocal;
    features_src = reciprocal_features_src;
    features_tgt = reciprocal_features_tgt;
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

Eigen::Matrix4f align(PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                      const FeatureCloudT::Ptr &features_src, const FeatureCloudT::Ptr &features_tgt,
                      const Eigen::Matrix4f &transformation_gt, const YamlConfig &config) {
    pcl::SampleConsensusPrerejective<PointT, PointT, FeatureT> align;

    align.setInputSource(src);
    align.setSourceFeatures(features_src);

    align.setInputTarget(tgt);
    align.setTargetFeatures(features_tgt);

    float voxel_size = config.get<float>("voxel_size").value();

    align.setMaximumIterations(config.get<int>("iteration").value()); // Number of RANSAC iterations
    align.setNumberOfSamples(config.get<int>("n_samples").value()); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(config.get<int>("randomness").value()); // Number of nearest features to use
    align.setSimilarityThreshold(config.get<float>("edge_thr").value()); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(config.get<float>("distance_thr").value() * voxel_size); // Inlier threshold
    align.setInlierFraction(config.get<float>("inlier_fraction").value()); // Required inlier fraction for accepting a pose hypothesis
    {
        pcl::ScopeTime t("Alignment");
        align.align(*src);
    }

    float error_thr = config.get<float>("error_thr").value();
    int correct_inliers = countCorrectCorrespondences(transformation_gt, src, tgt, align.getInliers(), error_thr);

    if (align.hasConverged()) {
        // Print results
        printf("\n");
        printTransformation(align.getFinalTransformation());
        printTransformation(transformation_gt);
        pcl::console::print_info("fitness: %0.7f\n", (float) align.getInliers().size() / (float) features_src->size());
        pcl::console::print_info("inliers_rmse: %0.7f\n", align.getFitnessScore());
        pcl::console::print_info("inliers: %i/%i\n", align.getInliers().size(), src->size());
        pcl::console::print_info("correct inliers: %i/%i\n", correct_inliers, src->size());
    } else {
        pcl::console::print_error("Alignment failed!\n");
        exit(1);
    }
    return align.getFinalTransformation();
}

