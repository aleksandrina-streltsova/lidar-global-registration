#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <unordered_set>

#include <pcl/console/print.h>
#include <pcl/io/ply_io.h>

#include "config.h"
#include "common.h"
#include "downsample.h"
#include "analysis.h"
#include "keypoints.h"
#include "io.h"

#define KEYPOINT "keypoint"

namespace fs = std::filesystem;

void detectKeyPointsISS(const PointNCloud::ConstPtr &pcd, pcl::IndicesPtr &indices,
                        const AlignmentParameters &parameters,
                        const Eigen::Matrix4f &tn, bool is_source) {
    PointNCloud key_points;
    double iss_salient_radius = 6 * parameters.voxel_size;
    double iss_non_max_radius = 4 * parameters.voxel_size;
    double iss_gamma_21(0.975);
    double iss_gamma_32(0.975);
    int iss_min_neighbors(4);
    pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
    ISSKeypoint3DDebug iss_detector;
    iss_detector.setSearchMethod(tree);
    iss_detector.setSalientRadius(iss_salient_radius);
    iss_detector.setNonMaxRadius(iss_non_max_radius);
    iss_detector.setThreshold21(iss_gamma_21);
    iss_detector.setThreshold32(iss_gamma_32);
    iss_detector.setMinNeighbors(iss_min_neighbors);
    iss_detector.setInputCloud(pcd);
    iss_detector.setNormals(pcd);
    iss_detector.compute(key_points);
    indices = std::make_shared<pcl::Indices>(pcl::Indices());
    *indices = iss_detector.getKeypointsIndices()->indices;
    if (parameters.fix_seed) {
        std::sort(indices->begin(), indices->end());
    }

    if (iss_detector.getBorderRadius() > 0.0) {
        auto edges = iss_detector.getBoundaryPointsDebug();
        saveColorizedWeights(pcd, edges, "iss_boundary_" + std::string(is_source ? "src" : "tgt"), parameters, tn);
    }

    auto third_evs = iss_detector.getThirdEigenValuesDebug();
    saveColorizedWeights(pcd, third_evs, "iss_third_ev_" + std::string(is_source ? "src" : "tgt"), parameters, tn);
}

void detectKeyPointsHarris(const PointNCloud::ConstPtr &pcd, pcl::IndicesPtr &indices,
                           const AlignmentParameters &parameters,
                           const Eigen::Matrix4f &tn, bool is_source) {
    HarrisKeypoint3DDebug harris;
    pcl::PointCloud<pcl::PointXYZI> key_points;

    double curvature_radius = 6 * parameters.voxel_size;
    float non_max_radius = 4 * parameters.voxel_size;
    harris.setInputCloud(pcd);
    harris.setNormals(pcd);
    harris.setRadiusSearch(curvature_radius);
    harris.setRadius(non_max_radius);
    harris.setNonMaxSupression(true);
    harris.setThreshold(1e-6);
    harris.setMethod(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN>::HARRIS);
    harris.compute(key_points);

    indices = std::make_shared<pcl::Indices>(pcl::Indices());
    *indices = harris.getKeypointsIndices()->indices;
    if (parameters.fix_seed) {
        std::sort(indices->begin(), indices->end());
    }
    auto response = harris.getResponseHarrisDebug();
    saveColorizedWeights(pcd, response, "response_harris_" + std::string(is_source ? "src" : "tgt"), parameters, tn);
}

std::string getCloudDownsizePath(const std::string &cloud_path, float voxel_size) {
    std::string filename = fs::path(cloud_path).filename();
    std::string filename_downsize = filename.substr(0, filename.rfind('.')) +
                                    "_" + std::to_string((int) std::round(1e4 * voxel_size)) + ".ply";
    return fs::path(cloud_path).parent_path() / filename_downsize;
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
            fout << "testname,correspondences,correct_correspondences,"
                 << "keypoints_src,correct_keypoints_src,ratio_src,"
                 << "keypoints_tgt,correct_keypoints_tgt,ratio_tgt\n";
        }
    } else {
        perror(("error while opening file " + filepath).c_str());
    }

    PointNCloud::Ptr src_fullsize(new PointNCloud), tgt_fullsize(new PointNCloud);
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
    pcl::Correspondences correspondences, correct_correspondences;
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    Eigen::Matrix4f transformation_gt;
    std::string testname;
    float min_voxel_size;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();
    std::string src_filename = fs::path(src_path).filename();
    std::string tgt_filename = fs::path(tgt_path).filename();
    testname = src_filename.substr(0, src_filename.find_last_of('.')) + '_' +
               tgt_filename.substr(0, tgt_filename.find_last_of('.'));

    bool need_to_load = true, need_to_downsample = true;
    loadTransformationGt(src_path, tgt_path, config.get<std::string>("ground_truth").value(), transformation_gt);

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
        std::string src_downsize_path = getCloudDownsizePath(src_path, parameters.voxel_size);
        std::string tgt_downsize_path = getCloudDownsizePath(tgt_path, parameters.voxel_size);
        need_to_downsample = !(fs::exists(src_downsize_path) && fs::exists(tgt_downsize_path)) ||
                             parameters.coarse_to_fine;
        if (need_to_downsample && need_to_load) {
            loadPointClouds(src_path, tgt_path, testname, src_fullsize, tgt_fullsize, fields_src, fields_tgt,
                            config.get<float>("density"), min_voxel_size);
            need_to_load = false;
        }

        parameters.testname = testname;
        parameters.keypoint_id = KEYPOINT_ANY;
        if (parameters.coarse_to_fine) {
            downsamplePointCloud(src_fullsize, src, parameters);
            downsamplePointCloud(tgt_fullsize, tgt, parameters);
        } else {
            src = src_fullsize;
            tgt = tgt_fullsize;
        }
        std::vector<float> voxel_sizes;
        std::vector<std::string> matching_ids;
        getIterationsInfo(fs::path(DATA_DEBUG_PATH) / fs::path(ITERATIONS_CSV),
                          constructName(parameters, "iterations"), voxel_sizes, matching_ids);
        for (int i = 0; i < voxel_sizes.size(); ++i) {
            AlignmentParameters curr_parameters(parameters);
            curr_parameters.voxel_size = voxel_sizes[i];
            curr_parameters.matching_id = matching_ids[i];
            bool success = false;
            std::string corrs_path = constructPath(curr_parameters, "correspondences", "csv", true, false, false);
            correspondences = *readCorrespondencesFromCSV(corrs_path, success);
            curr_parameters.keypoint_id = KEYPOINT_ISS;
            if (!success) {
                pcl::console::print_error("Failed to read correspondences for %s!\n", curr_parameters.testname.c_str());
                exit(1);
            }
            PointNCloud::Ptr curr_src(new PointNCloud), curr_tgt(new PointNCloud);
            pcl::IndicesPtr indices_src{nullptr}, indices_tgt{nullptr};
            if (need_to_downsample) {
                downsamplePointCloud(src, curr_src, curr_parameters);
                downsamplePointCloud(tgt, curr_tgt, curr_parameters);
                if (pcl::io::savePLYFileBinary(src_downsize_path, *curr_src) < 0 ||
                    pcl::io::savePLYFileBinary(tgt_downsize_path, *curr_tgt) < 0) {
                    pcl::console::print_error("Error saving src/tgt file!\n");
                    exit(1);
                }
            } else {
                if (loadPLYFile<PointN>(src_downsize_path, *curr_src, fields_src) < 0 ||
                    loadPLYFile<PointN>(tgt_downsize_path, *curr_tgt, fields_tgt) < 0) {
                    pcl::console::print_error("Error loading src/tgt file!\n");
                    exit(1);
                }
            }
            curr_parameters.normals_available = pointCloudHasNormals<PointN>(fields_src) &&
                                                pointCloudHasNormals<PointN>(fields_tgt);
            float normal_radius = curr_parameters.normal_radius_coef * curr_parameters.voxel_size;
            float error_thr = curr_parameters.distance_thr_coef * curr_parameters.voxel_size;
            estimateNormalsRadius(normal_radius, curr_src, normals_src, false);
            estimateNormalsRadius(normal_radius, curr_tgt, normals_tgt, false);
            pcl::concatenateFields(*curr_src, *normals_src, *curr_src);
            pcl::concatenateFields(*curr_tgt, *normals_tgt, *curr_tgt);

            detectKeyPointsHarris(curr_src, indices_src, curr_parameters, transformation_gt, true);
            detectKeyPointsHarris(curr_tgt, indices_tgt, curr_parameters, Eigen::Matrix4f::Identity(), false);

            saveColorizedPointCloud(curr_src, indices_src, {}, {}, {}, curr_parameters, transformation_gt, true);
            saveColorizedPointCloud(curr_tgt, indices_tgt, {}, {}, {}, curr_parameters, Eigen::Matrix4f::Identity(), false);

            buildCorrectCorrespondences(curr_src, curr_tgt, correspondences, correct_correspondences, transformation_gt,
                                        error_thr);

            std::unordered_set<int> indices_correct_src, indices_correct_tgt;
            std::transform(correct_correspondences.begin(), correct_correspondences.end(),
                           std::inserter(indices_correct_src, indices_correct_src.begin()),
                           [](const pcl::Correspondence &corr) { return corr.index_query; });
            std::transform(correct_correspondences.begin(), correct_correspondences.end(),
                           std::inserter(indices_correct_tgt, indices_correct_tgt.begin()),
                           [](const pcl::Correspondence &corr) { return corr.index_match; });
            int count_correct_src = 0, count_correct_tgt = 0;
            for (int idx_src: *indices_src) {
                if (indices_correct_src.contains(idx_src)) count_correct_src++;
            }
            for (int idx_tgt: *indices_tgt) {
                if (indices_correct_tgt.contains(idx_tgt)) count_correct_tgt++;
            }
            float ratio_src = (float) count_correct_src / (float) indices_src->size();
            float ratio_tgt = (float) count_correct_tgt / (float) indices_tgt->size();
            fout << constructName(curr_parameters, "keypoints", true, true, true, true) << ","
                 << correspondences.size() << "," << correct_correspondences.size() << ","
                 << indices_src->size() << "," << count_correct_src << "," << ratio_src << ","
                 << indices_tgt->size() << "," << count_correct_tgt << "," << ratio_tgt << "\n";
        }
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
        pcl::console::print_error((std::string("Syntax is: [") +
                                   KEYPOINT +
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
    } else if (command == KEYPOINT) {
        analyzeKeyPoints(config);
    }
}