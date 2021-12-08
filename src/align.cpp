#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/time.h>

#include "align.h"
#include "csv_parser.h"

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

SampleConsensusPrerejectiveOMP<PointT, PointT, FeatureT> align_point_clouds(const PointCloudT::Ptr &src,
                                                                            const PointCloudT::Ptr &tgt,
                                                                            const FeatureCloudT::Ptr &features_src,
                                                                            const FeatureCloudT::Ptr &features_tgt,
                                                                            const YamlConfig &config) {
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
    int n_samples = config.get<int>("n_samples").value();
    int iteration_brute_force = calculate_combination_or_max<int>((int) std::min(src->size(), tgt->size()), n_samples);
    align.setMaximumIterations(config.get<int>("iteration", iteration_brute_force)); // Number of RANSAC iterations
    align.setNumberOfSamples(n_samples); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(config.get<int>("randomness").value()); // Number of nearest features to use
    align.setSimilarityThreshold(config.get<float>("edge_thr").value()); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(config.get<float>("distance_thr_coef").value() * voxel_size); // Inlier threshold
    align.setConfidence(config.get<float>("confidence").value()); // Confidence in adaptive RANSAC
    align.setInlierFraction(
            config.get<float>("inlier_fraction").value()); // Required inlier fraction for accepting a pose hypothesis
    std::cout << "    iteration: " << align.getMaximumIterations() << std::endl;
    std::cout << "    voxel size: " << voxel_size << std::endl;
    {
        pcl::ScopeTime t("Alignment");
        align.align(src_aligned);
    }

    return align;
}
