#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <unordered_set>

#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include "config.h"
#include "common.h"
#include "keypoints.h"
#include "matching.h"
#include "downsample.h"
#include "feature_analysis.h"

#define KEYPOINT "keypoint"

namespace fs = std::filesystem;

void detectKeyPointsISS(const PointNCloud::ConstPtr &pcd, pcl::IndicesPtr &indices,
                        const AlignmentParameters &parameters) {
    PointNCloud key_points;
    ISSKeypoint3DDebug iss_detector;
    iss_detector.setSalientRadius(1.f);
    iss_detector.setNonMaxRadius(1.f);
    iss_detector.setThreshold21(0.975f);
    iss_detector.setThreshold32(0.975f);
    iss_detector.setMinNeighbors(4);
    iss_detector.setMaxNeighbors(352);
    iss_detector.setInputCloud(pcd);
    iss_detector.setNormals(pcd);
    iss_detector.compute(key_points);
    indices = std::make_shared<pcl::Indices>(pcl::Indices());
    *indices = iss_detector.getKeypointsIndices()->indices;
    if (parameters.fix_seed) {
        std::sort(indices->begin(), indices->end());
    }
    PCL_DEBUG("%d key points detected\n", indices->size());
}

void saveDebugFeatures(const std::string &corrs_path,
                       const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                       const AlignmentParameters &parameters) {
    pcl::KdTreeFLANN<PointN>::Ptr tree_src(new pcl::KdTreeFLANN<PointN>), tree_tgt(new pcl::KdTreeFLANN<PointN>);
    tree_src->setInputCloud(src);
    tree_tgt->setInputCloud(tgt);
    pcl::IndicesPtr indices_src(new pcl::Indices), indices_tgt(new pcl::Indices);

    bool file_exists = std::filesystem::exists(corrs_path);
    pcl::Indices nn_indices;
    std::vector<float> nn_dists;
    if (file_exists) {
        std::ifstream fin(corrs_path);
        if (fin.is_open()) {
            std::string line;
            std::vector<std::string> tokens;
            std::getline(fin, line); // header
            while (std::getline(fin, line)) {
                // query_idx, match_idx, distance, x_s, y_s, z_s, x_t, y_t, z_t
                split(line, tokens, ",");
                PointN p_src(std::stof(tokens[3]), std::stof(tokens[4]), std::stof(tokens[5]));
                PointN p_tgt(std::stof(tokens[6]), std::stof(tokens[7]), std::stof(tokens[8]));
                tree_src->nearestKSearch(p_src, 1, nn_indices, nn_dists);
                indices_src->push_back(nn_indices[0]);
                tree_tgt->nearestKSearch(p_tgt, 1, nn_indices, nn_dists);
                indices_tgt->push_back(nn_indices[0]);
            }
        } else {
            perror(("error while opening file " + corrs_path).c_str());
        }
    }
    pcl::PointCloud<SHOT>::Ptr features_src(new pcl::PointCloud<SHOT>), features_tgt(new pcl::PointCloud<SHOT>);
    estimateFeatures<SHOT>(src, indices_src, features_src, parameters);
    estimateFeatures<SHOT>(tgt, indices_tgt, features_tgt, parameters);

    AlignmentParameters parameters_bf{parameters};
    parameters_bf.randomness = 2;
    auto correspondences_src = matchBF<SHOT>(features_src, features_src, parameters_bf);
    parameters_bf.randomness = 1;
    auto correspondences_tgt = matchBF<SHOT>(features_src, features_tgt, parameters_bf);
    std::vector<float> closest_dists_src(features_src->size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> closest_dists_tgt(features_src->size(), std::numeric_limits<float>::quiet_NaN());
    for (int i = 0; i < features_src->size(); ++i) {
        if (correspondences_src[i].distances.size() == 2)
            closest_dists_src[i] = std::max(correspondences_src[i].distances[0], correspondences_src[i].distances[1]);
        if (correspondences_tgt[i].distances.size() == 1)
            closest_dists_tgt[i] = correspondences_tgt[i].distances[0];
    }
    saveVector(closest_dists_src, constructPath(parameters, "closest_dists_src", "csv", true, false, false, true));
    saveVector(closest_dists_tgt, constructPath(parameters, "closest_dists_tgt", "csv", true, false, false, true));
    saveFeatures<SHOT>(features_src, {nullptr}, parameters, true);
    saveFeatures<SHOT>(features_tgt, {nullptr}, parameters, false);
}

void analyzeKeyPoints(const YamlConfig &config) {
    std::string filepath = constructPath("test", "keypoints", "csv", false);
    bool file_exists = fs::exists(filepath);
    std::fstream fout;
    if (!file_exists) {
        fout.open(filepath, std::ios_base::out);
    } else {
        fout.open(filepath, std::ios_base::app);
    }
    if (fout.is_open()) {
        if (!file_exists) {
            fout << "testname,feature_radius_coef,correspondences,correct_correspondences\n";
        }
    } else {
        perror(("error while opening file " + filepath).c_str());
    }

    PointNCloud::Ptr src(new PointNCloud), src_gt(new PointNCloud), tgt(new PointNCloud);
    PointNCloud::Ptr src_ds(new PointNCloud), tgt_ds(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f transformation_gt;
    std::string testname;
    float min_voxel_size;

    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();
    loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt, config.get<float>("density"),
                    min_voxel_size);
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        parameters.testname = testname;
        parameters.ground_truth = std::optional<Eigen::Matrix4f>{transformation_gt};
        std::string kps_path = constructPath(parameters, "kps", "csv", true, false, false, false);
        bool exists = fs::exists(kps_path);
        if (!exists) {
            pcl::IndicesPtr indices_src(new pcl::Indices);
            estimateNormalsPoints(NORMAL_NR_POINTS, src, normals_src, parameters.normals_available);
            estimateNormalsPoints(NORMAL_NR_POINTS, tgt, normals_tgt, parameters.normals_available);
            pcl::concatenateFields(*src, *normals_src, *src);
            pcl::concatenateFields(*tgt, *normals_tgt, *tgt);
            detectKeyPointsISS(src, indices_src, parameters);
            pcl::KdTreeFLANN<PointN> tree;
            pcl::transformPointCloud(*src, *src_gt, transformation_gt);
            tree.setInputCloud(tgt);
            pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
            pcl::Indices nn_indices;
            std::vector<float> nn_dists;
            for (int i = 0; i < indices_src->size(); ++i) {
                tree.radiusSearch(*src_gt, indices_src->operator[](i), 0.05, nn_indices, nn_dists, 1);
                if (!nn_indices.empty()) {
                    correspondences->push_back({indices_src->operator[](i), nn_indices[0], 0.f});
                }
            }
            saveCorrespondencesToCSV(kps_path, src, tgt, correspondences);
        }
        downsamplePointCloud(src, src_ds, parameters);
        downsamplePointCloud(tgt, tgt_ds, parameters);
        {
            pcl::ScopeTime t("Normal estimation");
            estimateNormalsPoints(NORMAL_NR_POINTS, src_ds, normals_src, parameters.normals_available);
            estimateNormalsPoints(NORMAL_NR_POINTS, tgt_ds, normals_tgt, parameters.normals_available);
            pcl::concatenateFields(*src_ds, *normals_src, *src_ds);
            pcl::concatenateFields(*tgt_ds, *normals_tgt, *tgt_ds);
        }
        saveDebugFeatures(kps_path, src_ds, tgt_ds, parameters);
    }
}

void processTests(const std::vector<YAML::Node> &tests, const std::string &command) {
    for (auto &test: tests) {
        YamlConfig config;
        config.config = (*test.begin()).second;
        if (command == KEYPOINT) {
            analyzeKeyPoints(config);
        }
    }
}

int main(int argc, char **argv) {
    std::string command(argv[1]);
    if (argc != 3 && !(command == KEYPOINT)) {
        pcl::console::print_error((std::string("Syntax is: %s ") +
                                   KEYPOINT +
                                   " config.yaml\n").c_str(), argv[0]);
        exit(1);
    }
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    // Load parameters from config
    YamlConfig config;
    config.init(argv[2]);
    auto tests = config.get<std::vector<YAML::Node>>("tests");
    if (tests.has_value()) {
        processTests(tests.value(), command);
    } else if (command == KEYPOINT) {
        analyzeKeyPoints(config);
    }
}