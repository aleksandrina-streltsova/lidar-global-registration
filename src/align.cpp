#include <Eigen/Core>
#include <fstream>
#include <string>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include "../include/config.h"
#include "../include/csv_parser.h"

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::Normal> PointCloudN;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimation<PointT, pcl::Normal, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

void printTransformation(Eigen::Matrix4f transformation) {
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

Eigen::Matrix4f getTransformation(const std::string& csv_path,
                                  const std::string& src_filename, const std::string& tgt_filename) {
    std::ifstream file(csv_path);
    Eigen::Matrix4f src_position, tgt_position;
    
    CSVRow row;
    while (file >> row) {
        if (row[0] == src_filename) {
            for (int i = 0; i < 16; ++i) {
                src_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
            std::cout << std::endl;
        } else if (row[0] == tgt_filename) {
            for (int i = 0; i < 16; ++i) {
                tgt_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
            std::cout << std::endl;
        }
    }
    return tgt_position * src_position.inverse();

}

void filterReciprocalCorrespondents(PointCloudT::Ptr &src, FeatureCloudT::Ptr &src_features,
                                    PointCloudT::Ptr &tgt, FeatureCloudT::Ptr &tgt_features) {
    pcl::registration::CorrespondenceEstimation<FeatureT, FeatureT> est;
    est.setInputCloud(src_features);
    est.setInputTarget(tgt_features);
    pcl::CorrespondencesPtr reciprocal_correspondences(new pcl::Correspondences);
    est.determineReciprocalCorrespondences(*reciprocal_correspondences);

    PointCloudT::Ptr src_reciprocal(new PointCloudT), tgt_reciprocal(new PointCloudT);
    FeatureCloudT::Ptr src_reciprocal_features(new FeatureCloudT), tgt_reciprocal_features(new FeatureCloudT);
    src_reciprocal->resize(reciprocal_correspondences->size());
    tgt_reciprocal->resize(reciprocal_correspondences->size());
    src_reciprocal_features->resize(reciprocal_correspondences->size());
    tgt_reciprocal_features->resize(reciprocal_correspondences->size());
    for (std::size_t i = 0; i < reciprocal_correspondences->size(); ++i) {
        pcl::Correspondence correspondence = (*reciprocal_correspondences)[i];
        src_reciprocal->points[i] = src->points[correspondence.index_query];
        tgt_reciprocal->points[i] = tgt->points[correspondence.index_match];
        src_reciprocal_features->points[i] = src_features->points[correspondence.index_query];
        tgt_reciprocal_features->points[i] = tgt_features->points[correspondence.index_match];
    }
    src = src_reciprocal;
    tgt = tgt_reciprocal;
    src_features = src_reciprocal_features;
    tgt_features = tgt_reciprocal_features;
}

int main(int argc, char **argv) {
    // Point clouds
    PointCloudT::Ptr src(new PointCloudT);
    PointCloudT::Ptr src_aligned(new PointCloudT);
    PointCloudT::Ptr tgt(new PointCloudT);
    FeatureCloudT::Ptr src_features(new FeatureCloudT);
    FeatureCloudT::Ptr tgt_features(new FeatureCloudT);

    // Get input src and tgt
    if (argc != 2) {
        pcl::console::print_error("Syntax is: %s config.yaml\n", argv[0]);
        return (1);
    }

    // Load parameters from config
    YamlConfig config;
    config.init(argv[1]);

    // Load src and tgt
    pcl::console::print_highlight("Loading point clouds...\n");
    std::string src_path = config.get<std::string>("source", "source.ply");
    std::string tgt_path = config.get<std::string>("target", "target.ply");

    if (pcl::io::loadPLYFile<PointT>(src_path, *src) < 0 ||
        pcl::io::loadPLYFile<PointT>(tgt_path, *tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        return (1);
    }
    float leaf = config.get<float>("voxel_size", 0.001f);

    // Downsample
    pcl::console::print_highlight("Downsampling...\n");
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(src);
    pcl::console::print_highlight("Point cloud downsampled from %zu...", src->size());
    grid.filter(*src);
    pcl::console::print_highlight("to %zu\n", src->size());
    grid.setInputCloud(tgt);
    grid.filter(*tgt);

    float model_size = getAABBDiagonal(src);

    // Estimate normals for tgt
    pcl::console::print_highlight("Estimating normals...\n");
    PointCloudN::Ptr normals_src(new PointCloudN), normals_tgt(new PointCloudN);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_est;
    normal_est.setRadiusSearch(config.get<float>("normal_radius", 0.13f) * model_size);

    normal_est.setInputCloud(src);
    pcl::search::KdTree<PointT>::Ptr tree_src(new pcl::search::KdTree<PointT>());
    normal_est.setSearchMethod(tree_src);
    normal_est.compute(*normals_src);

    normal_est.setInputCloud(tgt);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree<PointT>());
    normal_est.setSearchMethod(tree_tgt);
    normal_est.compute(*normals_tgt);

    // Estimate features
    pcl::console::print_highlight("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch(config.get<float>("feature_radius", 0.13f) * model_size);
    fest.setInputCloud(src);
    fest.setInputNormals(normals_src);
    fest.compute(*src_features);
    fest.setInputCloud(tgt);
    fest.setInputNormals(normals_tgt);
    fest.compute(*tgt_features);

    filterReciprocalCorrespondents(src, src_features, tgt, tgt_features);

    // Perform alignment
    pcl::console::print_highlight("Starting alignment...\n");
    pcl::SampleConsensusPrerejective<PointT, PointT, FeatureT> align;

    align.setInputSource(src);
    align.setSourceFeatures(src_features);

    align.setInputTarget(tgt);
    align.setTargetFeatures(tgt_features);

    align.setMaximumIterations(config.get<int>("iteration", 1e5)); // Number of RANSAC iterations
    align.setNumberOfSamples(config.get<int>("n_samples", 3)); // Number of points to sample for generating/prerejecting a pose
    align.setCorrespondenceRandomness(1); // Number of nearest features to use
    align.setSimilarityThreshold(config.get<float>("edge_thr", 0.95)); // Polygonal edge length similarity threshold
    align.setMaxCorrespondenceDistance(config.get<float>("distance_thr", 1.5) * leaf); // Inlier threshold
    align.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesis
    {
        pcl::ScopeTime t("Alignment");
        align.align(*src_aligned);
    }

    std::string csv_path = config.get<std::string>("ground_truth", "");
    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);

    if (align.hasConverged()) {
        // Print results
        printf("\n");
        printTransformation(align.getFinalTransformation());
        printTransformation(getTransformation(csv_path, src_filename, tgt_filename));

        pcl::console::print_info("fitness: %0.7f\n", (float)align.getInliers().size() / (float)src_features->size());
        pcl::console::print_info("inliers_rmse: %0.7f\n", align.getFitnessScore());
        pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), src->size());
        pcl::io::savePLYFileBinary("source_aligned.ply", *src_aligned);
    } else {
        pcl::console::print_error("Alignment failed!\n");
        return (1);
    }
    return (0);
}

