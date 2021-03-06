#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_set>

#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>
#include <pcl/common/copy_point.h>
#include <pcl/common/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/shot_lrf.h>

#include "common.h"
#include "csv_parser.h"
#include "io.h"

#define RF_MIN_ANGLE_RAD 0.04f

namespace fs = std::filesystem;

const std::string DATA_DEBUG_PATH = fs::path("data") / fs::path("debug");
const std::string TRANSFORMATIONS_CSV = "transformations.csv";
const std::string ITERATIONS_CSV = "iterations.csv";
const std::string VERSION = "10";
const std::string SUBVERSION = "";
const std::string ALIGNMENT_DEFAULT = "default";
const std::string ALIGNMENT_GROR = "gror";
const std::string KEYPOINT_ANY = "any";
const std::string KEYPOINT_ISS = "iss";
const std::string DESCRIPTOR_FPFH = "fpfh";
const std::string DESCRIPTOR_SHOT = "shot";
const std::string DESCRIPTOR_ROPS = "rops";
const std::string DESCRIPTOR_USC = "usc";
const std::string DEFAULT_LRF = "default";
const std::string METRIC_CORRESPONDENCES = "correspondences";
const std::string METRIC_CLOSEST_PLANE = "closest_plane";
const std::string METRIC_WEIGHTED_CLOSEST_PLANE = "weighted_closest_plane";
const std::string METRIC_COMBINATION = "combination";
const std::string MATCHING_LEFT_TO_RIGHT = "lr";
const std::string MATCHING_RATIO = "ratio";
const std::string MATCHING_CLUSTER = "cluster";
const std::string METRIC_WEIGHT_CONSTANT = "constant";
const std::string METRIC_WEIGHT_EXP_CURVATURE = "exp_curvature";
const std::string METRIC_WEIGHT_CURVEDNESS = "curvedness";
const std::string METRIC_WEIGHT_HARRIS = "harris";
const std::string METRIC_WEIGHT_TOMASI = "tomasi";
const std::string METRIC_WEIGHT_CURVATURE = "curvature";

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
        }
        if (row[0] == tgt_filename) {
            for (int i = 0; i < 16; ++i) {
                tgt_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
        }
    }
    return tgt_position.inverse() * src_position;
}

Eigen::Matrix4f getTransformation(const std::string &csv_path, const std::string &transformation_name) {
    std::ifstream file(csv_path);
    Eigen::Matrix4f transformation;
    bool success = false;

    CSVRow row;
    while (file >> row) {
        if (row[0] == transformation_name) {
            for (int i = 0; i < 16; ++i) {
                transformation(i / 4, i % 4) = std::stof(row[i + 1]);
            }
            success = true;
            break;
        }
    }
    if (!success) {
        pcl::console::print_error("Failed to get transformation %s!\n", transformation_name.c_str());
        exit(1);
    }
    return transformation;
}

void saveTransformation(const std::string &csv_path, const std::string &transformation_name,
                        const Eigen::Matrix4f &transformation) {
    bool file_exists = std::filesystem::exists(csv_path);
    std::fstream fout;
    if (!file_exists) {
        fout.open(csv_path, std::ios_base::out);
    } else {
        fout.open(csv_path, std::ios_base::app);
    }
    if (fout.is_open()) {
        if (!file_exists) {
            fout << "reading,gT00,gT01,gT02,gT03,gT10,gT11,gT12,gT13,gT20,gT21,gT22,gT23,gT30,gT31,gT32,gT33\n";
        }
    } else {
        perror(("error while opening file " + csv_path).c_str());
    }
    fout << transformation_name;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            fout << "," << transformation(i, j);
        }
    }
    fout << "\n";
}

void getIterationsInfo(const std::string &csv_path, const std::string &name,
                       std::vector<float> &voxel_sizes,
                       std::vector<std::string> &matching_ids) {
    std::ifstream file(csv_path);
    Eigen::Matrix4f transformation;

    CSVRow row;
    voxel_sizes.clear();
    bool success = false;
    while (file >> row) {
        if (row[0] == name) {
            int n = std::stoi(row[1]);
            for (int i = 0; i < n; ++i) {
                voxel_sizes.push_back(std::stof(row[2 * i + 2]));
                matching_ids.push_back(row[2 * i + 3]);
            }
            success = true;
            break;
        }
    }
    if (!success) {
        pcl::console::print_error("Failed to get iterations for test %s!\n", name.c_str());
        exit(1);
    }
}

void saveIterationsInfo(const std::string &csv_path, const std::string &name,
                        const std::vector<float> &voxel_sizes,
                        const std::vector<std::string> &matching_ids) {
    bool file_exists = std::filesystem::exists(csv_path);
    std::fstream fout;
    if (!file_exists) {
        fout.open(csv_path, std::ios_base::out);
    } else {
        fout.open(csv_path, std::ios_base::app);
    }
    if (fout.is_open()) {
        fout << name << "," << voxel_sizes.size();
        for (int i = 0; i < voxel_sizes.size(); ++i) {
            fout << "," << voxel_sizes[i] << "," << matching_ids[i];
        }
        fout << "\n";
    } else {
        perror(("error while opening file " + csv_path).c_str());
    }

}

std::vector<AlignmentParameters> getParametersFromConfig(const YamlConfig &config,
                                                         const std::vector<::pcl::PCLPointField> &fields_src,
                                                         const std::vector<::pcl::PCLPointField> &fields_tgt,
                                                         float min_voxel_size) {
    std::vector<AlignmentParameters> parameters_container, new_parameters_container;
    AlignmentParameters parameters;
    parameters.coarse_to_fine = config.get<bool>("coarse_to_fine", ALIGNMENT_COARSE_TO_FINE);
    parameters.edge_thr_coef = config.get<float>("edge_thr", ALIGNMENT_EDGE_THR);
    parameters.max_iterations = config.get<int>("iteration", std::numeric_limits<int>::max());
    parameters.confidence = config.get<float>("confidence", ALIGNMENT_CONFIDENCE);
    parameters.inlier_fraction = config.get<float>("inlier_fraction", ALIGNMENT_INLIER_FRACTION);
    parameters.use_bfmatcher = config.get<bool>("bf", ALIGNMENT_USE_BFMATCHER);
    parameters.randomness = config.get<int>("randomness", ALIGNMENT_RANDOMNESS);
    parameters.n_samples = config.get<int>("n_samples", ALIGNMENT_N_SAMPLES);
    parameters.save_features = config.get<bool>("save_features", ALIGNMENT_SAVE_FEATURES);
    parameters.bf_block_size = config.get<int>("block_size", ALIGNMENT_BLOCK_SIZE);

    bool use_normals = config.get<bool>("use_normals", ALIGNMENT_USE_NORMALS);
    // TODO: use normals even with one point cloud missing normals
    bool normals_available = pointCloudHasNormals<PointN>(fields_src) && pointCloudHasNormals<PointN>(fields_tgt);
    if (use_normals && !normals_available) {
        PCL_WARN("Point cloud doesn't have normals.\n");
    }
    parameters.normals_available = normals_available;
    parameters.use_normals = use_normals && normals_available;
    parameters_container.push_back(parameters);

    auto alignment_ids = config.getVector<std::string>("alignment", ALIGNMENT_DEFAULT);
    for (const auto &id: alignment_ids) {
        for (auto ps: parameters_container) {
            ps.alignment_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto keypoint_ids = config.getVector<std::string>("keypoint", KEYPOINT_ISS);
    for (const auto &id: keypoint_ids) {
        for (auto ps: parameters_container) {
            ps.keypoint_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto distance_thr_coefs = config.getVector<float>("distance_thr_coef");
    if (distance_thr_coefs.has_value()) {
        for (float coef: distance_thr_coefs.value()) {
            for (auto ps: parameters_container) {
                ps.distance_thr_coef = coef;
                new_parameters_container.push_back(ps);
            }
        }
        std::swap(parameters_container, new_parameters_container);
        new_parameters_container.clear();
    } else {
        for (auto &ps: parameters_container) {
            ps.distance_thr_coef = (ps.alignment_id == ALIGNMENT_GROR || ps.keypoint_id != KEYPOINT_ANY) ? 2.0 : 1.5;
        }
    }

    auto gror_iss_coefs = config.getVector<float>("gror_iss_coef", GROR_ISS_COEF);
    for (float gic: gror_iss_coefs) {
        for (auto ps: parameters_container) {
            ps.gror_iss_coef = gic;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto voxel_sizes = config.getVector<float>("voxel_size").value();
    for (float vs: voxel_sizes) {
        for (auto ps: parameters_container) {
            ps.voxel_size = vs;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto normal_radius_coefs = config.getVector<float>("normal_radius_coef", ALIGNMENT_NORMAL_RADIUS_COEF);
    for (float nrc: normal_radius_coefs) {
        for (auto ps: parameters_container) {
            ps.normal_radius_coef = nrc;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto feature_radius_coefs = config.getVector<float>("feature_radius_coef", ALIGNMENT_FEATURE_RADIUS_COEF);
    for (float frc: feature_radius_coefs) {
        for (auto ps: parameters_container) {
            ps.feature_radius_coef = frc;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto func_ids = config.getVector<std::string>("filter", "");
    for (const auto &id: func_ids) {
        for (auto ps: parameters_container) {
            ps.func_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto descriptor_ids = config.getVector<std::string>("descriptor", DESCRIPTOR_SHOT);
    for (const auto &id: descriptor_ids) {
        for (auto ps: parameters_container) {
            ps.descriptor_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto lrf_ids = config.getVector<std::string>("lrf", DEFAULT_LRF);
    for (const auto &id: lrf_ids) {
        for (auto ps: parameters_container) {
            ps.lrf_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto metric_ids = config.getVector<std::string>("metric", METRIC_COMBINATION);
    for (const auto &id: metric_ids) {
        for (auto ps: parameters_container) {
            ps.metric_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto matching_ids = config.getVector<std::string>("matching", MATCHING_CLUSTER);
    for (const auto &id: matching_ids) {
        for (auto ps: parameters_container) {
            ps.matching_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto weight_ids = config.getVector<std::string>("weight", METRIC_WEIGHT_CONSTANT);
    for (const auto &id: weight_ids) {
        for (auto ps: parameters_container) {
            ps.weight_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    for (auto &parameters: parameters_container) {
        if (parameters.voxel_size < min_voxel_size) {
            PCL_WARN("[getParametersFromConfig] "
                     "voxel size %.5f is less than estimated point cloud density, setting voxel size to %.5f\n",
                     parameters.voxel_size, min_voxel_size);
            parameters.voxel_size = min_voxel_size;
        }
    }
    return parameters_container;
}

void loadPointClouds(const std::string &src_path, const std::string &tgt_path,
                     std::string &testname, PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                     std::vector<::pcl::PCLPointField> &fields_src, std::vector<::pcl::PCLPointField> &fields_tgt,
                     const std::optional<float> &density, float &min_voxel_size) {
    pcl::console::print_highlight("Loading point clouds...\n");

    if (loadPLYFile<PointN>(src_path, *src, fields_src) < 0 ||
        loadPLYFile<PointN>(tgt_path, *tgt, fields_tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }

    if (density.has_value()) {
        min_voxel_size = density.value();
    } else {
        float src_density = calculatePointCloudDensity<PointN>(src);
        float tgt_density = calculatePointCloudDensity<PointN>(tgt);
        PCL_DEBUG("[loadPointClouds] src density: %.5f, tgt density: %.5f.\n", src_density, tgt_density);
        min_voxel_size = std::max(src_density, tgt_density);
    }

    std::string src_filename = fs::path(src_path).filename();
    std::string tgt_filename = fs::path(tgt_path).filename();
    testname = src_filename.substr(0, src_filename.find_last_of('.')) + '_' +
               tgt_filename.substr(0, tgt_filename.find_last_of('.'));
}

void loadTransformationGt(const std::string &src_path, const std::string &tgt_path,
                          const std::string &csv_path, Eigen::Matrix4f &transformation_gt) {
    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    transformation_gt = getTransformation(csv_path, src_filename, tgt_filename);
}

void updateMultivaluedCorrespondence(MultivaluedCorrespondence &corr, int query_idx,
                                     int k_matches, int match_idx, float distance) {
    int pos = 0;
    while (corr.match_indices.begin() + pos != corr.match_indices.end() && corr.distances[pos] < distance) {
        pos++;
    }
    corr.match_indices.insert(corr.match_indices.begin() + pos, match_idx);
    corr.distances.insert(corr.distances.begin() + pos, distance);
    if (corr.match_indices.size() > k_matches) {
        corr.match_indices.erase(corr.match_indices.begin() + k_matches);
        corr.distances.erase(corr.distances.begin() + k_matches);
    }
    corr.query_idx = query_idx;
}

float getAABBDiagonal(const PointNCloud::Ptr &pcd) {
    auto[min_point_AABB, max_point_AABB] = calculateBoundingBox<PointN>(pcd);

    Eigen::Vector3f min_point(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
    Eigen::Vector3f max_point(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);

    return (max_point - min_point).norm();
}

void postprocessNormals(const PointNCloud::Ptr &pcd, NormalCloud::Ptr &normals, bool normals_available) {
    // use normals from point cloud to orient estimated normals and replace NaN normals
    if (normals_available) {
        for (int i = 0; i < pcd->size(); ++i) {
            auto &normal = normals->points[i];
            const auto &point = pcd->points[i];
            if (!std::isfinite(normal.normal_x) || !std::isfinite(normal.normal_y) || !std::isfinite(normal.normal_z)) {
                normal.normal_x = point.normal_x;
                normal.normal_y = point.normal_y;
                normal.normal_z = point.normal_z;
                normal.curvature = 0.f;
            } else if (normal.normal_x * point.normal_x + normal.normal_y * point.normal_y +
                       normal.normal_z * point.normal_z < 0) {
                normal.normal_x *= -1.f;
                normal.normal_y *= -1.f;
                normal.normal_z *= -1.f;
            }
        }
    }
    int nan_counter = 0;
    for (auto &normal: normals->points) {
        const Eigen::Vector4f &normal_vec = normal.getNormalVector4fMap();
        if (!std::isfinite(normal_vec[0]) ||
            !std::isfinite(normal_vec[1]) ||
            !std::isfinite(normal_vec[2])) {
            nan_counter++;
        } else {
            float norm = std::sqrt(normal.normal_x * normal.normal_x + normal.normal_y * normal.normal_y +
                                   normal.normal_z * normal.normal_z);
            normal.normal_x /= norm;
            normal.normal_y /= norm;
            normal.normal_z /= norm;
        }
    }
    PCL_DEBUG("[estimateNormals] %d NaN normals.\n", nan_counter);
}

void estimateNormalsRadius(float radius_search, const PointNCloud::Ptr &pcd, NormalCloud::Ptr &normals,
                           bool normals_available) {
    pcl::NormalEstimationOMP<PointN, pcl::Normal> normal_est;
    normal_est.setRadiusSearch(radius_search);

    normal_est.setInputCloud(pcd);
    pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
    normal_est.setSearchMethod(tree);
    normal_est.compute(*normals);
    postprocessNormals(pcd, normals, normals_available);
}

void estimateNormalsPoints(int k_points, const PointNCloud::Ptr &pcd, NormalCloud::Ptr &normals,
                           bool normals_available) {
    pcl::NormalEstimationOMP<PointN, pcl::Normal> normal_est;
    normal_est.setKSearch(k_points);

    normal_est.setInputCloud(pcd);
    pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
    normal_est.setSearchMethod(tree);
    normal_est.compute(*normals);
    postprocessNormals(pcd, normals, normals_available);
}

void smoothNormals(float radius_search, float voxel_size, const PointNCloud::Ptr &pcd) {
    NormalCloud::Ptr old_normals(new NormalCloud);
    pcl::KdTreeFLANN<PointN>::Ptr tree(new pcl::KdTreeFLANN<PointN>());
    tree->setInputCloud(pcd);

    std::vector<pcl::Indices> vector_of_indices(pcd->size());
    std::vector<float> distances;

    for (int i = 0; i < (int) std::ceil(radius_search / voxel_size); i++) {
        pcl::copyPointCloud(*pcd, *old_normals);
        for (int j = 0; j < pcd->size(); ++j) {
            pcl::Normal acc(0.0, 0.0, 0.0), old_normal(old_normals->points[j]);
            for (auto idx: vector_of_indices[j]) {
                float dot_product = old_normal.normal_x * old_normals->points[idx].normal_x +
                                    old_normal.normal_y * old_normals->points[idx].normal_y +
                                    old_normal.normal_z * old_normals->points[idx].normal_z;
                if (dot_product > 0.0) {
                    acc.normal_x += old_normals->points[idx].normal_x;
                    acc.normal_y += old_normals->points[idx].normal_y;
                    acc.normal_z += old_normals->points[idx].normal_z;
                }
            }
            float norm = std::sqrt(acc.normal_x * acc.normal_x + acc.normal_y * acc.normal_y +
                                   acc.normal_z * acc.normal_z);
            pcd->points[j].normal_x = acc.normal_x / norm;
            pcd->points[j].normal_y = acc.normal_y / norm;
            pcd->points[j].normal_z = acc.normal_z / norm;
        }
    }
}

pcl::IndicesPtr detectKeyPoints(const PointNCloud::ConstPtr &pcd, const AlignmentParameters &parameters) {
    pcl::IndicesPtr indices(new pcl::Indices);
    if (parameters.keypoint_id == KEYPOINT_ISS) {
        PointNCloud key_points;
        double iss_salient_radius_ = 4 * parameters.voxel_size;
        double iss_non_max_radius_ = 2 * parameters.voxel_size;
        double iss_gamma_21_(0.975);
        double iss_gamma_32_(0.975);
        int iss_min_neighbors_(4);
        pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
        pcl::ISSKeypoint3D<PointN, PointN, PointN> iss_detector;
        iss_detector.setSearchMethod(tree);
        iss_detector.setSalientRadius(iss_salient_radius_);
        iss_detector.setNonMaxRadius(iss_non_max_radius_);
        iss_detector.setThreshold21(iss_gamma_21_);
        iss_detector.setThreshold32(iss_gamma_32_);
        iss_detector.setMinNeighbors(iss_min_neighbors_);
        iss_detector.setInputCloud(pcd);
        iss_detector.setNormals(pcd);
        iss_detector.compute(key_points);
        indices = std::make_shared<pcl::Indices>(iss_detector.getKeypointsIndices()->indices);
        if (parameters.fix_seed) {
            std::sort(indices->begin(), indices->end());
        }
        PCL_DEBUG("[detectKeyPoints] %d key points\n", indices->size());
    } else {
        if (parameters.keypoint_id != KEYPOINT_ANY) {
            PCL_WARN("[detectKeyPoints] Detection method %s isn't supported, no detection method will be applied.\n",
                     parameters.keypoint_id.c_str());
        }
        indices->resize(pcd->size());
        for (int i = 0; i < pcd->size(); ++i) {
            indices->operator[](i) = i;
        }
    }
    return indices;
}

void estimateReferenceFrames(const PointNCloud::ConstPtr &pcd, const pcl::IndicesConstPtr &indices,
                             PointRFCloud::Ptr &frames_kps, const AlignmentParameters &parameters) {
    int nr_kps = indices ? indices->size() : pcd->size();
    std::string lrf_id = parameters.lrf_id;
    std::transform(lrf_id.begin(), lrf_id.end(), lrf_id.begin(), [](unsigned char c) { return std::tolower(c); });
    if (lrf_id == "gt") {
        Eigen::Matrix3f lrf_eigen = Eigen::Matrix3f::Identity();
        if (!parameters.ground_truth) {
            PCL_ERROR("[estimateReferenceFrames] ground truth wasn't provided!");
        } else {
            lrf_eigen = parameters.ground_truth->block<3, 3>(0, 0).inverse() * lrf_eigen;
        }
        PointRF lrf;
        for (int d = 0; d < 3; ++d) {
            lrf.x_axis[d] = lrf_eigen.col(0)[d];
            lrf.y_axis[d] = lrf_eigen.col(1)[d];
            lrf.z_axis[d] = lrf_eigen.col(2)[d];
        }
        frames_kps = std::make_shared<PointRFCloud>(PointRFCloud());
        frames_kps->resize(nr_kps, lrf);
    } else if (lrf_id == "gravity") {
        frames_kps = std::make_shared<PointRFCloud>(PointRFCloud());
        frames_kps->resize(nr_kps);

        pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
        tree->setInputCloud(pcd);
        tree->setSortedResults(true);

        pcl::SHOTLocalReferenceFrameEstimation<PointN, PointRF>::Ptr lrf_estimator(
                new pcl::SHOTLocalReferenceFrameEstimation<PointN, PointRF>());
        float lrf_radius = parameters.voxel_size * parameters.feature_radius_coef;
        lrf_estimator->setRadiusSearch(lrf_radius);
        lrf_estimator->setInputCloud(pcd);
        if (indices) lrf_estimator->setIndices(indices);
        lrf_estimator->setSearchMethod(tree);
        lrf_estimator->compute(*frames_kps);
        rassert(frames_kps->size() == nr_kps, 15946243)

        Eigen::Vector3f gravity(0, 0, 1);
        for (std::size_t i = 0; i < nr_kps; ++i) {
            int idx = indices ? indices->operator[](i) : i;
            PointRF &output_rf = frames_kps->points[i];
            const PointN &normal = pcd->points[idx];
            Eigen::Vector3f x_axis(normal.normal_x, normal.normal_y, normal.normal_z);
            if (std::acos(std::abs(std::clamp(x_axis.dot(gravity), -1.0f, 1.0f))) > RF_MIN_ANGLE_RAD) {
                Eigen::Vector3f y_axis = gravity.cross(x_axis);
                Eigen::Vector3f z_axis = x_axis.cross(y_axis);
                for (int d = 0; d < 3; ++d) {
                    output_rf.x_axis[d] = x_axis[d];
                    output_rf.y_axis[d] = y_axis[d];
                    output_rf.z_axis[d] = z_axis[d];
                }
            }
        }
    } else if (lrf_id != DEFAULT_LRF) {
        PCL_WARN("[estimateReferenceFrames] LRF %s isn't supported, default LRF will be used.\n", lrf_id.c_str());
    }
}

void saveColorizedPointCloud(const PointNCloud::ConstPtr &pcd,
                             const pcl::IndicesConstPtr &key_point_indices,
                             const pcl::Correspondences &correspondences,
                             const pcl::Correspondences &correct_correspondences,
                             const std::vector<InlierPair> &inlier_pairs, const AlignmentParameters &parameters,
                             const Eigen::Matrix4f &transformation_gt, bool is_source) {
    PointNCloud pcd_aligned;
    pcl::transformPointCloud(*pcd, pcd_aligned, transformation_gt);

    PointColoredNCloud dst;
    dst.resize(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        pcl::copyPoint(pcd_aligned.points[i], dst.points[i]);
        setPointColor(dst.points[i], key_point_indices ? COLOR_PARAKEET : COLOR_BEIGE);
    }
    if (key_point_indices) {
        for (int idx: *key_point_indices) {
            setPointColor(dst.points[idx], COLOR_BEIGE);
        }
    }
    for (const auto &correspondence: correspondences) {
        if (is_source) {
            setPointColor(dst.points[correspondence.index_query], COLOR_RED);
        } else {
            setPointColor(dst.points[correspondence.index_match], COLOR_RED);
        }
    }
    for (const auto &ip: inlier_pairs) {
        if (is_source) {
            setPointColor(dst.points[ip.idx_src], COLOR_BLUE);
        } else {
            setPointColor(dst.points[ip.idx_tgt], COLOR_BLUE);
        }
    }
    for (const auto &correspondence: correct_correspondences) {
        if (is_source) {
            mixPointColor(dst.points[correspondence.index_query], COLOR_WHITE);
        } else {
            mixPointColor(dst.points[correspondence.index_match], COLOR_WHITE);
        }
    }
    std::string filepath = constructPath(parameters, std::string("downsampled_") + (is_source ? "src" : "tgt"),
                                         "ply", true, true, true, true);
    pcl::io::savePLYFileBinary(filepath, dst);
}

int getColor(float v, float vmin, float vmax) {
    float r = 1.f, g = 1.f, b = 1.f;
    float dv = vmax - vmin;
    v = std::max(vmin, std::min(v, vmax));

    if (v < (vmin + dv / 3.f)) {
        b = 1.f - 3.f * (v - vmin) / dv;
    } else if (v < (vmin + 2.f * dv / 3.f)) {
        b = 0.f;
        g = 2.f - 3.f * (v - vmin) / dv;
    } else {
        b = 0.f;
        g = 0.f;
        r = 3.f - 3.f * (v - vmin) / dv;
    }
    int r8 = (std::uint8_t) (255.f * r), g8 = (std::uint8_t) (255.f * g), b8 = (std::uint8_t) (255.f * b);
    return (r8 << 16) + (g8 << 8) + b8;
}

void saveColorizedWeights(const PointNCloud::ConstPtr &pcd, std::vector<float> &weights, const std::string &name,
                          const AlignmentParameters &parameters, const Eigen::Matrix4f &transformation_gt) {
    PointColoredNCloud dst;
    dst.resize(pcd->size());
    float weights_min = quantile(0.01, weights);
    float weights_max = quantile(0.99, weights);
    for (int i = 0; i < pcd->size(); ++i) {
        pcl::copyPoint(pcd->points[i], dst.points[i]);
        setPointColor(dst.points[i], getColor(weights[i], weights_min, weights_max));
    }
    pcl::transformPointCloud(dst, dst, transformation_gt);
    std::string filepath = constructPath(parameters, name, "ply", true, true, true, true);
    pcl::io::savePLYFileBinary(filepath, dst);
}

void saveHistogram(const std::string &values_path, const std::string &hist_path) {
    std::string command = "python3 plots.py histogram " + values_path + " " + hist_path;
    system(command.c_str());
}

void calculateTemperatureMapDistances(PointColoredNCloud::Ptr &compared,
                                      PointColoredNCloud::Ptr &reference,
                                      std::vector<float> &distances,
                                      float distance_min, float distance_max) {
    pcl::KdTreeFLANN<PointColoredN> tree_reference;
    tree_reference.setInputCloud(reference);

    int n = compared->size();
    pcl::Indices nn_indices;
    std::vector<float> nn_dists;
    float dist_to_plane;
    PointColoredN nearest_point;

#pragma omp parallel default(none) \
    firstprivate(nn_indices, nn_dists, dist_to_plane, n, distance_min, distance_max, nearest_point) \
    shared(tree_reference, compared, reference, distances)
    for (int i = 0; i < n; ++i) {
        tree_reference.radiusSearch(*compared, i, distance_max, nn_indices, nn_dists, 1);
        if (nn_dists.empty()) {
            dist_to_plane = distance_max;
        } else {
            nearest_point = reference->points[nn_indices[0]];
            dist_to_plane = std::fabs(nearest_point.getNormalVector3fMap().transpose() *
                                      (nearest_point.getVector3fMap() - compared->points[i].getVector3fMap()));
            // normal can be invalid
            dist_to_plane = std::isfinite(dist_to_plane) ? dist_to_plane : nn_dists[0];
        }
        setPointColor(compared->points[i], getColor(dist_to_plane, distance_min, distance_max));
        distances[i] = dist_to_plane;
    }
}

void saveTemperatureMaps(PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                         const std::string &name, const AlignmentParameters &parameters,
                         const Eigen::Matrix4f &transformation, bool normals_available) {
    if (!normals_available) {
        NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);
        int k_points = 6;
        estimateNormalsPoints(k_points, src, normals_src, false);
        estimateNormalsPoints(k_points, tgt, normals_tgt, false);
        pcl::concatenateFields(*src, *normals_src, *src);
        pcl::concatenateFields(*tgt, *normals_tgt, *tgt);
    }
    int n_src = src->size(), n_tgt = tgt->size();
    PointColoredNCloud::Ptr src_colored(new PointColoredNCloud), tgt_colored(new PointColoredNCloud);
    std::vector<float> distances_src, distances_tgt;
    distances_src.resize(n_src);
    distances_tgt.resize(n_tgt);
    src_colored->resize(n_src);
    tgt_colored->resize(n_tgt);
    pcl::copyPointCloud(*src, *src_colored);
    pcl::copyPointCloud(*tgt, *tgt_colored);
    pcl::transformPointCloud(*src_colored, *src_colored, transformation);
    float distance_min = 0;
    float distance_max = parameters.voxel_size;
    calculateTemperatureMapDistances(src_colored, tgt_colored, distances_src, distance_min, distance_max);
    calculateTemperatureMapDistances(tgt_colored, src_colored, distances_tgt, distance_min, distance_max);

    distances_src.erase(std::remove_if(distances_src.begin(), distances_src.end(),
                                       [&distance_max](float d) { return d >= distance_max; }), distances_src.end());
    distances_tgt.erase(std::remove_if(distances_tgt.begin(), distances_tgt.end(),
                                       [&distance_max](float d) { return d >= distance_max; }), distances_tgt.end());
    std::string distances_path_src = constructPath(parameters, name + "_distances_src", "csv");
    std::string distances_path_tgt = constructPath(parameters, name + "_distances_tgt", "csv");
    saveVector(distances_src, distances_path_src);
    saveVector(distances_tgt, distances_path_tgt);

    saveHistogram(distances_path_src, constructPath(parameters, name + "_histogram_src", "png"));
    saveHistogram(distances_path_tgt, constructPath(parameters, name + "_histogram_tgt", "png"));

    pcl::io::savePLYFileBinary(constructPath(parameters, name + "_src"), *src_colored);
    pcl::io::savePLYFileBinary(constructPath(parameters, name + "_tgt"), *tgt_colored);
}

void writeFacesToPLYFileASCII(const PointColoredNCloud::Ptr &pcd, std::size_t match_offset,
                              const pcl::Correspondences &correspondences,
                              const std::string &filepath) {
    std::string filepath_tmp = filepath + "tmp";
    std::fstream fin(filepath, std::ios_base::in), fout(filepath_tmp, std::ios_base::out);
    if (!fin.is_open())
        perror(("error while opening file " + filepath).c_str());
    if (!fout.is_open())
        perror(("error while creating temporary file " + filepath).c_str());

    std::string line;
    std::string vertex_str = "element vertex";
    std::string face_str = "element face";
    bool header_ended = false, vertices_ended = false;
    int n_vertices = 0;
    long pos = fout.tellg();
    while (std::getline(fin, line)) {
        if (line.substr(0, vertex_str.length()) == vertex_str) {
            fout << vertex_str << " " << pcd->size() + correspondences.size() << "\n";
        } else if (line.substr(0, face_str.length()) != face_str) {
            fout << line << "\n";
            if (line == "end_header") {
                fout.seekg(pos);
                fout << face_str << " " << correspondences.size() << "\n";
                fout << "property list uint8 int32 vertex_index\n";
                fout << "end_header\n";
                header_ended = true;
            } else if (header_ended && !vertices_ended) {
                n_vertices++;
            }
        }
        if (!vertices_ended && n_vertices == pcd->size()) {
            vertices_ended = true;
            for (const auto &corr: correspondences) {
                const auto &p = pcd->points[corr.index_query];
                fout << p.x << " " << p.y << " " << p.z << " "
                     << (int) p.r << " " << (int) p.g << " " << (int) p.b << " "
                     << p.normal_x << " " << p.normal_y << " " << p.normal_z << " " << p.curvature << "\n";
            }
        }
        pos = fout.tellg();
    }
    std::size_t midpoint_offset = pcd->size();
    for (int i = 0; i < correspondences.size(); i++) {
        const auto &corr = correspondences[i];
        fout << "3 " << corr.index_query << " " << match_offset + corr.index_match << " " << midpoint_offset + i
             << "\n";
    }
    fin.close();
    fout.close();
    fs::remove(filepath);
    fs::rename(filepath_tmp, filepath);
}

void saveCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                         const pcl::Correspondences &correspondences,
                         const Eigen::Matrix4f &transformation_gt,
                         const AlignmentParameters &parameters, bool sparse) {
    PointColoredNCloud::Ptr dst(new PointColoredNCloud);
    PointNCloud::Ptr src_aligned_gt(new PointNCloud);
    pcl::transformPointCloud(*src, *src_aligned_gt, transformation_gt);
    float diagonal = getAABBDiagonal(src_aligned_gt);

    dst->resize(src_aligned_gt->size() + tgt->size());
    for (int i = 0; i < src_aligned_gt->size(); ++i) {
        pcl::copyPoint(src_aligned_gt->points[i], dst->points[i]);
        setPointColor(dst->points[i], COLOR_BEIGE);
    }
    for (int i = 0; i < tgt->size(); ++i) {
        pcl::copyPoint(tgt->points[i], dst->points[src_aligned_gt->size() + i]);
        dst->points[src_aligned_gt->size() + i].x += diagonal;
        setPointColor(dst->points[src_aligned_gt->size() + i], COLOR_PURPLE);
    }
    UniformRandIntGenerator rand_generator(0, 255);
    for (const auto &corr: correspondences) {
        std::uint8_t red = rand_generator(), green = rand_generator(), blue = rand_generator();
        setPointColor(dst->points[corr.index_query], red, green, blue);
        setPointColor(dst->points[src_aligned_gt->size() + corr.index_match], red, green, blue);
    }
    std::string filepath;
    if (sparse) {
        filepath = constructPath(parameters, "correspondences_sparse");
    } else {
        filepath = constructPath(parameters, "correspondences");
    }
    pcl::io::savePLYFileASCII(filepath, *dst);
    if (sparse) {
        std::vector correspondences_sparse(correspondences);
        std::shuffle(correspondences_sparse.begin(), correspondences_sparse.end(),
                     std::mt19937(std::random_device()()));
        correspondences_sparse.resize(std::min(correspondences_sparse.size(), DEBUG_N_EDGES));
        writeFacesToPLYFileASCII(dst, src->size(), correspondences_sparse, filepath);
    } else {
        writeFacesToPLYFileASCII(dst, src->size(), correspondences, filepath);
    }
}

void saveCorrectCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                const pcl::Correspondences &correspondences,
                                const pcl::Correspondences &correct_correspondences,
                                const Eigen::Matrix4f &transformation_gt,
                                const AlignmentParameters &parameters, bool sparse) {
    PointColoredNCloud::Ptr dst(new PointColoredNCloud);
    PointNCloud::Ptr src_aligned_gt(new PointNCloud);
    pcl::transformPointCloud(*src, *src_aligned_gt, transformation_gt);
    float half_diagonal = 0.5 * getAABBDiagonal(src_aligned_gt);

    dst->resize(src_aligned_gt->size() + tgt->size());
    for (int i = 0; i < src_aligned_gt->size(); ++i) {
        pcl::copyPoint(src_aligned_gt->points[i], dst->points[i]);
        setPointColor(dst->points[i], COLOR_BEIGE);
    }
    for (int i = 0; i < tgt->size(); ++i) {
        pcl::copyPoint(tgt->points[i], dst->points[src_aligned_gt->size() + i]);
        dst->points[src_aligned_gt->size() + i].x += half_diagonal;
        setPointColor(dst->points[src_aligned_gt->size() + i], COLOR_PURPLE);
    }
    UniformRandIntGenerator rand_generator(0, 255);
    for (const auto &corr: correspondences) {
        setPointColor(dst->points[corr.index_query], COLOR_ROSE);
        setPointColor(dst->points[src_aligned_gt->size() + corr.index_match], COLOR_ROSE);
    }
    for (const auto &corr: correct_correspondences) {
        setPointColor(dst->points[corr.index_query], COLOR_PARAKEET);
        setPointColor(dst->points[src_aligned_gt->size() + corr.index_match], COLOR_PARAKEET);
    }
    std::string filepath;
    if (sparse) {
        filepath = constructPath(parameters, "correspondences_sparse");
    } else {
        filepath = constructPath(parameters, "correspondences");
    }
    pcl::io::savePLYFileASCII(filepath, *dst);
    if (sparse) {
        std::vector correspondences_sparse(correspondences);
        std::shuffle(correspondences_sparse.begin(), correspondences_sparse.end(),
                     std::mt19937(std::random_device()()));
        correspondences_sparse.resize((int) (0.02 * correspondences_sparse.size()));
        writeFacesToPLYFileASCII(dst, src->size(), correspondences_sparse, filepath);
    } else {
        writeFacesToPLYFileASCII(dst, src->size(), correspondences, filepath);
    }
}

void saveCorrespondenceDistances(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                 const pcl::Correspondences &correspondences,
                                 const Eigen::Matrix4f &transformation_gt, float voxel_size,
                                 const AlignmentParameters &parameters) {
    std::string filepath = constructPath(parameters, "distances", "csv");
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());
    PointNCloud src_aligned_gt;
    pcl::transformPointCloud(*src, src_aligned_gt, transformation_gt);

    fout << "distance\n";
    for (const auto &correspondence: correspondences) {
        PointN source_point(src_aligned_gt.points[correspondence.index_query]);
        PointN target_point(tgt->points[correspondence.index_match]);
        float dist = pcl::L2_Norm(source_point.data, target_point.data, 3) / voxel_size;
        fout << dist << "\n";
    }
    fout.close();
}

void saveCorrespondencesDebug(const pcl::Correspondences &correspondences,
                              const pcl::Correspondences &correct_correspondences,
                              const AlignmentParameters &parameters) {
    std::string filepath = constructPath(parameters, "correspondences_debug", "csv");
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());

    std::unordered_set<int> correct_cs_set;
    std::transform(correct_correspondences.begin(), correct_correspondences.end(),
                   std::inserter(correct_cs_set, correct_cs_set.begin()),
                   [](const pcl::Correspondence &corr) { return corr.index_query; });
    fout << "index_query,index_match,distance,is_correct\n";
    for (const auto &corr: correspondences) {
        fout << corr.index_query << "," << corr.index_match << "," << corr.distance << ",";
        fout << correct_cs_set.contains(corr.index_query) << "\n";
    }
    fout.close();
}

void setPointColor(PointColoredN &point, int color) {
    point.r = (color >> 16) & 0xff;
    point.g = (color >> 8) & 0xff;
    point.b = (color >> 0) & 0xff;
}

void mixPointColor(PointColoredN &point, int color) {
    point.r = point.r / 2 + ((color >> 16) & 0xff) / 2;
    point.g = point.g / 2 + ((color >> 8) & 0xff) / 2;
    point.b = point.b / 2 + ((color >> 0) & 0xff) / 2;
}

void setPointColor(PointColoredN &point, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    point.r = red;
    point.g = green;
    point.b = blue;
}


std::string constructPath(const std::string &test, const std::string &name,
                          const std::string &extension, bool with_version, bool with_subversion) {
    std::string filename = test + "_" + name;
    if (with_version) {
        filename += "_" + VERSION;
    }
    if (with_subversion) {
        filename += SUBVERSION;
    }
    filename += "." + extension;
    return fs::path(DATA_DEBUG_PATH) / fs::path(filename);
}

std::string constructPath(const AlignmentParameters &parameters, const std::string &name, const std::string &extension,
                          bool with_version, bool with_metric, bool with_weights, bool with_subversion) {
    auto test_dir_path = fs::path(parameters.dir_path) / fs::path(parameters.testname);
    fs::create_directories(test_dir_path);

    std::string filename = constructName(parameters, name, with_version, with_metric, with_weights, with_subversion);
    filename += "." + extension;
    return test_dir_path / fs::path(filename);
}

std::string constructName(const AlignmentParameters &parameters, const std::string &name,
                          bool with_version, bool with_metric, bool with_weights, bool with_subversion) {
    with_weights = parameters.metric_id == METRIC_WEIGHTED_CLOSEST_PLANE &&
                   parameters.weight_id != METRIC_WEIGHT_CONSTANT && with_weights;
    std::string full_name = parameters.testname + "_" + name +
                            "_" + std::to_string((int) std::round(1e4 * parameters.voxel_size)) +
                            "_" + parameters.descriptor_id + "_" + (parameters.use_bfmatcher ? "bf" : "flann") +
                            "_" + std::to_string((int) parameters.normal_radius_coef) +
                            "_" + std::to_string((int) parameters.feature_radius_coef) +
                            "_" + parameters.alignment_id + "_" + parameters.keypoint_id +
                            "_" + parameters.lrf_id + (with_metric ? "_" + parameters.metric_id : "") +
                            "_" + parameters.matching_id + "_" + std::to_string(parameters.randomness) +
                            (parameters.coarse_to_fine ? "_ctf" : "") +
                            (with_weights ? "_" + parameters.weight_id : "") +
                            (parameters.use_normals ? "_normals" : "");
    if (with_version) {
        full_name += "_" + VERSION;
    }
    if (with_subversion) {
        full_name += SUBVERSION;
    }
    return full_name;
}

pcl::CorrespondencesPtr readCorrespondencesFromCSV(const std::string &filepath, bool &success) {
    bool file_exists = std::filesystem::exists(filepath);
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    if (file_exists) {
        std::ifstream fin(filepath);
        if (fin.is_open()) {
            std::string line;
            std::vector<std::string> tokens;
            std::getline(fin, line); // header
            while (std::getline(fin, line)) {
                // query_idx, match_idx, distance, x_s, y_s, z_s, x_t, y_t, z_t
                split(line, tokens, ",");
                pcl::Correspondence corr{std::stoi(tokens[0]), std::stoi(tokens[1]), std::stof(tokens[2])};
                correspondences->push_back(corr);
            }
            success = true;
        } else {
            perror(("error while opening file " + filepath).c_str());
        }
    }
    return correspondences;
}

void saveCorrespondencesToCSV(const std::string &filepath,
                              const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                              const pcl::CorrespondencesConstPtr &correspondences) {
    std::ofstream fout(filepath);
    if (fout.is_open()) {
        fout << "query_idx,match_idx,distance,x_s,y_s,z_s,x_t,y_t,z_t\n";
        for (const auto &corr: *correspondences) {
            fout << corr.index_query << "," << corr.index_match << "," << corr.distance << ",";
            fout << src->points[corr.index_query].x << ","
                 << src->points[corr.index_query].y << ","
                 << src->points[corr.index_query].z << ",";
            fout << tgt->points[corr.index_match].x << ","
                 << tgt->points[corr.index_match].y << ","
                 << tgt->points[corr.index_match].z << '\n';
        }
    } else {
        perror(("error while opening file " + filepath).c_str());
    }
}