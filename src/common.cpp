#include <filesystem>
#include <fstream>
#include <memory>
#include <random>

#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>
#include <pcl/common/copy_point.h>
#include <pcl/common/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/shot_lrf.h>

#include "common.h"
#include "pcl/iss_debug.h"
#include "csv_parser.h"
#include "io.h"
#include "downsample.h"

#define RF_MIN_ANGLE_RAD 0.04f

namespace fs = std::filesystem;

const std::string DATA_DEBUG_PATH = fs::path("data") / fs::path("debug");
const std::string TRANSFORMATIONS_CSV = "transformations.csv";
const std::string ITERATIONS_CSV = "iterations.csv";
const std::string VERSION = "15";
const std::string SUBVERSION = "";
const std::string ALIGNMENT_RANSAC = "ransac";
const std::string ALIGNMENT_GROR = "gror";
const std::string ALIGNMENT_TEASER = "teaser";
const std::string KEYPOINT_ANY = "any";
const std::string KEYPOINT_ISS = "iss";
const std::string DESCRIPTOR_FPFH = "fpfh";
const std::string DESCRIPTOR_SHOT = "shot";
const std::string DESCRIPTOR_ROPS = "rops";
const std::string DESCRIPTOR_USC = "usc";
const std::string DEFAULT_LRF = "default";
const std::string METRIC_CORRESPONDENCES = "correspondences";
const std::string METRIC_UNIFORMITY = "uniformity";
const std::string METRIC_CLOSEST_PLANE = "closest_plane";
const std::string METRIC_WEIGHTED_CLOSEST_PLANE = "weighted_closest_plane";
const std::string METRIC_COMBINATION = "combination";
const std::string MATCHING_LEFT_TO_RIGHT = "lr";
const std::string MATCHING_RATIO = "ratio";
const std::string MATCHING_CLUSTER = "cluster";
const std::string MATCHING_ONE_SIDED = "one_sided";
const std::string METRIC_WEIGHT_CONSTANT = "constant";
const std::string METRIC_WEIGHT_EXP_CURVATURE = "exp_curvature";
const std::string METRIC_WEIGHT_CURVEDNESS = "curvedness";
const std::string METRIC_WEIGHT_HARRIS = "harris";
const std::string METRIC_WEIGHT_TOMASI = "tomasi";
const std::string METRIC_WEIGHT_CURVATURE = "curvature";
const std::string METRIC_WEIGHT_NSS = "nss";
const std::string METRIC_SCORE_CONSTANT = "constant";
const std::string METRIC_SCORE_MAE = "mae";
const std::string METRIC_SCORE_MSE = "mse";
const std::string METRIC_SCORE_EXP = "exp";

pcl::Correspondences correspondencesToPCL(const Correspondences &correspondences) {
    pcl::Correspondences correspondences_pcl;
    correspondences_pcl.reserve(correspondences.size());
    for (const auto &corr: correspondences) {
        correspondences_pcl.push_back(pcl::Correspondence{corr.index_query, corr.index_match, corr.distance});
    }
    return correspondences_pcl;
}

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

std::optional<Eigen::Matrix4f> getTransformation(const std::string &csv_path,
                                                 const std::string &src_filename, const std::string &tgt_filename) {
    std::ifstream file(csv_path);
    Eigen::Matrix4f src_position, tgt_position;

    CSVRow row;
    bool success_src{false}, success_tgt{false};
    while (file >> row) {
        if (row[0] == src_filename) {
            for (int i = 0; i < 16; ++i) {
                src_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
            success_src = true;
        }
        if (row[0] == tgt_filename) {
            for (int i = 0; i < 16; ++i) {
                tgt_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
            success_tgt = true;
        }
    }
    return success_src && success_tgt ? std::optional<Eigen::Matrix4f>{tgt_position.inverse() * src_position}
                                      : std::nullopt;
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

float calculatePointCloudDensity(const PointNCloud::ConstPtr &pcd, float quantile) {
    rassert(quantile >= 0.f && quantile <= 1.f, 3487234892347);
    std::vector<float> densities = calculateSmoothedDensities(pcd, 8);
    int k = std::max(std::min((int) (quantile * (float) densities.size() - 1), (int) densities.size() - 1), 0);
    std::nth_element(densities.begin(), densities.begin() + k, densities.end());
    return densities[k];
}

std::vector<AlignmentParameters> getParametersFromConfig(const YamlConfig &config,
                                                         const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                                                         const std::vector<::pcl::PCLPointField> &fields_src,
                                                         const std::vector<::pcl::PCLPointField> &fields_tgt) {
    std::vector<AlignmentParameters> parameters_container, new_parameters_container;
    AlignmentParameters parameters;
    parameters.edge_thr_coef = config.get<float>("edge_thr", ALIGNMENT_EDGE_THR);
    parameters.max_iterations = config.get<int>("iteration", std::numeric_limits<int>::max());
    parameters.confidence = config.get<float>("confidence", ALIGNMENT_CONFIDENCE);
    parameters.use_bfmatcher = config.get<bool>("bf", ALIGNMENT_USE_BFMATCHER);
    parameters.randomness = config.get<int>("randomness", ALIGNMENT_RANDOMNESS);
    parameters.n_samples = config.get<int>("n_samples", ALIGNMENT_N_SAMPLES);
    parameters.save_features = config.get<bool>("save_features", ALIGNMENT_SAVE_FEATURES);
    parameters.bf_block_size = config.get<int>("block_size", ALIGNMENT_BLOCK_SIZE);

    float density_src = calculatePointCloudDensity(src);
    float density_tgt = calculatePointCloudDensity(tgt);
    bool normals_available = pointCloudHasNormals<PointN>(fields_src) && pointCloudHasNormals<PointN>(fields_tgt);
    parameters.normals_available = normals_available;

    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();
    loadViewpoint(config.get<std::string>("viewpoints"), src_path, parameters.vp_src);
    loadViewpoint(config.get<std::string>("viewpoints"), tgt_path, parameters.vp_tgt);

    parameters_container.push_back(parameters);

    auto alignment_ids = config.getVector<std::string>("alignment", ALIGNMENT_RANSAC);
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

    auto distance_thrs = config.getVector<float>("distance_thr");
    if (distance_thrs.has_value()) {
        for (float thr: distance_thrs.value()) {
            for (auto ps: parameters_container) {
                ps.distance_thr = thr;
                new_parameters_container.push_back(ps);
            }
        }
        std::swap(parameters_container, new_parameters_container);
        new_parameters_container.clear();
    } else {
        float distance_thr = 4 * std::max(density_src, density_tgt);
        pcl::console::print_highlight("Automatic distance threshold requested, choosing %0.7f.\n", distance_thr);
        for (auto &ps: parameters_container) {
            ps.distance_thr = distance_thr;
        }
    }

    auto feature_radii = config.getVector<float>("feature_radius", 0.f);
    for (float fr: feature_radii) {
        for (auto ps: parameters_container) {
            ps.feature_radius = fr <= 0 ? std::nullopt : std::optional(fr);
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto nrs = config.getVector<int>("feature_nr", FEATURE_NR_POINTS);
    for (int nr: nrs) {
        for (auto ps: parameters_container) {
            ps.feature_nr_points = nr;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto normal_nrs = config.getVector<int>("normal_nr", NORMAL_NR_POINTS);
    for (int normal_nr: normal_nrs) {
        for (auto ps: parameters_container) {
            ps.normal_nr_points = normal_nr;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto flags_reestimate = config.getVector<bool>("reestimate", FEATURES_REESTIMATE_FRAMES);
    for (int flag: flags_reestimate) {
        for (auto ps: parameters_container) {
            ps.reestimate_frames = flag;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto iss_radii = config.getVector<float>("iss_radius");
    if (iss_radii.has_value()) {
        for (float ir: iss_radii.value()) {
            for (auto ps: parameters_container) {
                ps.iss_radius_src = ir;
                ps.iss_radius_tgt = ir;
                new_parameters_container.push_back(ps);
            }
        }
        std::swap(parameters_container, new_parameters_container);
        new_parameters_container.clear();
    } else {
        for (auto &ps: parameters_container) {
            ps.iss_radius_src = 2.f * density_src;
            ps.iss_radius_tgt = 2.f * density_tgt;
            pcl::console::print_highlight("Automatic ISS radius requested, choosing %0.7f.\n", ps.iss_radius_src);
            pcl::console::print_highlight("Automatic ISS radius requested, choosing %0.7f.\n", ps.iss_radius_tgt);
        }
    }

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

    auto metric_ids = config.getVector<std::string>("metric", METRIC_UNIFORMITY);
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

    auto score_ids = config.getVector<std::string>("score", METRIC_SCORE_MSE);
    for (const auto &id: score_ids) {
        for (auto ps: parameters_container) {
            ps.score_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto scales = config.getVector<float>("scale", FEATURES_SCALE_FACTOR);
    for (const auto &scale: scales) {
        for (auto ps: parameters_container) {
            ps.scale_factor = scale;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto cluster_ks = config.getVector<int>("cluster_k", MATCHING_CLUSTER_K);
    for (const auto &k: cluster_ks) {
        for (auto ps: parameters_container) {
            ps.cluster_k = k;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();
    return parameters_container;
}

void filterDuplicatePoints(PointNCloud::Ptr &pcd) {
    pcl::console::print_highlight("Point cloud size changed from %zu...", pcd->size());
    std::unordered_set<PointN, PointHash, PointEqual<PointN>> unique_points;
    unique_points.reserve(pcd->size());
    std::copy(pcd->points.begin(), pcd->points.end(), std::inserter(unique_points, unique_points.begin()));
    pcd->points.clear();
    std::copy(unique_points.begin(), unique_points.end(), std::back_inserter(pcd->points));
    pcd->width = unique_points.size();
    pcd->height = 1;
    pcl::console::print_highlight("to %zu\n", pcd->size());
}

void loadPointClouds(const YamlConfig &config, std::string &testname, PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                     std::vector<::pcl::PCLPointField> &fields_src, std::vector<::pcl::PCLPointField> &fields_tgt) {
    pcl::console::print_highlight("Loading point clouds...\n");

    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();
    if (loadPLYFile<PointN>(src_path, *src, fields_src) < 0 ||
        loadPLYFile<PointN>(tgt_path, *tgt, fields_tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }

    filterDuplicatePoints(src);
    filterDuplicatePoints(tgt);

    // store weight of each point in intensity field
    // weights are re-calculated during downsampling
    for (auto &p: src->points) {
        p.intensity = 1.f;
    }
    for (auto &p: tgt->points) {
        p.intensity = 1.f;
    }

    float voxel_size_src = FINE_VOXEL_SIZE_COEFFICIENT * calculatePointCloudDensity(src);
    float voxel_size_tgt = FINE_VOXEL_SIZE_COEFFICIENT * calculatePointCloudDensity(tgt);
    downsamplePointCloud(src, src, voxel_size_src);
    downsamplePointCloud(tgt, tgt, voxel_size_tgt);

    bool normals_available = pointCloudHasNormals<PointN>(fields_src) && pointCloudHasNormals<PointN>(fields_tgt);
    std::optional<Eigen::Vector3f> vp_src, vp_tgt;
    loadViewpoint(config.get<std::string>("viewpoints"), src_path, vp_src);
    loadViewpoint(config.get<std::string>("viewpoints"), tgt_path, vp_tgt);
    pcl::console::print_highlight("Estimating normals...\n");
    estimateNormalsPoints(NORMAL_NR_POINTS, src, {nullptr}, vp_src, normals_available);
    estimateNormalsPoints(NORMAL_NR_POINTS, tgt, {nullptr}, vp_tgt, normals_available);

    std::string src_filename = fs::path(src_path).filename();
    std::string tgt_filename = fs::path(tgt_path).filename();
    testname = src_filename.substr(0, src_filename.find_last_of('.')) + '_' +
               tgt_filename.substr(0, tgt_filename.find_last_of('.'));
}

void loadTransformationGt(const YamlConfig &config, const std::optional<std::string> &csv_path,
                          std::optional<Eigen::Matrix4f> &transformation_gt) {
    if (!csv_path.has_value()) return;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();
    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);
    transformation_gt = getTransformation(csv_path.value(), src_filename, tgt_filename);
}

void loadViewpoint(const std::optional<std::string> &viewpoints_path,
                   const std::string &pcd_path, std::optional<Eigen::Vector3f> &viewpoint) {
    if (!viewpoints_path.has_value()) {
        viewpoint = std::nullopt;
        return;
    }
    std::string pcd_filename = pcd_path.substr(pcd_path.find_last_of("/\\") + 1);
    std::ifstream file(viewpoints_path.value());
    Eigen::Vector3f viewpoint_vector;
    bool success = false;
    CSVRow row;
    while (file >> row) {
        if (row[0] == pcd_filename) {
            for (int i = 0; i < 3; ++i) {
                viewpoint_vector(i) = std::stof(row[i + 1]);
            }
            success = true;
            pcl::console::print_debug("Using vp from %s\n", viewpoints_path.value().c_str());
            viewpoint = std::optional<Eigen::Vector3f>(viewpoint_vector);
            break;
        }
    }
    if (!success) {
        pcl::console::print_warn("Failed to get view for %s!\n", pcd_filename.c_str());
    }
}

std::ostream &operator<<(std::ostream &out, const MultivaluedCorrespondence &corr) {
    for (int i = 0; i < corr.match_indices.size(); ++i) {
        out << ',' << corr.match_indices[i];
        out << ',' << corr.distances[i];
    }
    return out;
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
}

std::vector<float> calculateSmoothedDensities(const PointNCloud::ConstPtr &pcd, int k) {
    rassert(pcd->size() > 1 && k >= 2, 3458240390587502);
    pcl::KdTreeFLANN<PointN> tree;
    tree.setInputCloud(pcd);
    std::vector<float> densities(pcd->size());

    pcl::Indices nn_indices;
    std::vector<float> nn_sqr_dists;
#pragma omp parallel for default(none) firstprivate(nn_indices, nn_sqr_dists, k) shared(pcd, tree, densities)
    for (int i = 0; i < pcd->size(); ++i) {
        tree.nearestKSearch(*pcd, i, k, nn_indices, nn_sqr_dists);
        densities[i] = std::sqrt(nn_sqr_dists[k - 1]);
        tree.nearestKSearch(*pcd, nn_indices[1], k, nn_indices, nn_sqr_dists);
        densities[i] = std::min(densities[i], std::sqrt(nn_sqr_dists[k - 1]));
    }
    return densities;
}

float getAABBDiagonal(const PointNCloud::Ptr &pcd) {
    auto[min_point_AABB, max_point_AABB] = calculateBoundingBox<PointN>(pcd);

    Eigen::Vector3f min_point(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
    Eigen::Vector3f max_point(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);

    return (max_point - min_point).norm();
}

void mergeOverlaps(const PointNCloud::ConstPtr &pcd1, const PointNCloud::ConstPtr &pcd2, PointNCloud::Ptr &dst,
                   float distance_thr) {
    pcl::Indices nn_indices;
    std::vector<float> nn_sqr_dists;
    dst->clear();
    for (int first_is_reference = 0; first_is_reference < 2; ++first_is_reference) {
        const auto &compared = first_is_reference ? pcd2 : pcd1;
        const auto &reference = first_is_reference ? pcd1 : pcd2;
        pcl::KdTreeFLANN<PointN> tree;
        tree.setInputCloud(reference);
        std::vector<bool> is_in_overlap(compared->size(), false);
        PointN nearest_point;
        float search_radius = DIST_TO_PLANE_COEFFICIENT * distance_thr;
#pragma omp parallel for firstprivate(nn_indices, nn_sqr_dists, distance_thr, nearest_point, search_radius)\
                         shared(tree, compared, reference, is_in_overlap) default(none)
        for (int i = 0; i < compared->size(); ++i) {
            tree.radiusSearch(*compared, i, search_radius, nn_indices, nn_sqr_dists, 1);
            if (nn_indices.empty()) continue;
            nearest_point = reference->points[nn_indices[0]];
            float dist_to_plane = std::fabs(nearest_point.getNormalVector3fMap().transpose() *
                                            (nearest_point.getVector3fMap() - compared->points[i].getVector3fMap()));
            // normal can be invalid
            dist_to_plane = std::isfinite(dist_to_plane) ? dist_to_plane : nn_sqr_dists[0];
            if (dist_to_plane < distance_thr) {
                is_in_overlap[i] = true;
            }
        }
        for (int i = 0; i < compared->size(); ++i) {
            if (is_in_overlap[i]) {
                dst->push_back(compared->points[i]);
            }
        }
    }
}

void postprocessNormals(PointNCloud::Ptr &pcd, bool normals_available) {
    // use normals from point cloud to orient estimated normals and replace NaN normals
    if (normals_available) {
        for (int i = 0; i < pcd->size(); ++i) {
            auto &normal = pcd->points[i];
            const auto &point = pcd->points[i];
            if (!std::isfinite(normal.normal_x) || !std::isfinite(normal.normal_y) || !std::isfinite(normal.normal_z)) {
                normal.normal_x = point.normal_x;
                normal.normal_y = point.normal_y;
                normal.normal_z = point.normal_z;
            } else if (normal.normal_x * point.normal_x + normal.normal_y * point.normal_y +
                       normal.normal_z * point.normal_z < 0) {
                normal.normal_x *= -1.f;
                normal.normal_y *= -1.f;
                normal.normal_z *= -1.f;
            }
        }
    }
    int nan_normals_counter = 0, nan_curvatures_counter = 0;
    for (auto &normal: pcd->points) {
        const Eigen::Vector4f &normal_vec = normal.getNormalVector4fMap();
        if (!std::isfinite(normal_vec[0]) || !std::isfinite(normal_vec[1]) || !std::isfinite(normal_vec[2])) {
            nan_normals_counter++;
        } else {
            float norm = std::sqrt(normal.normal_x * normal.normal_x + normal.normal_y * normal.normal_y +
                                   normal.normal_z * normal.normal_z);
            normal.normal_x /= norm;
            normal.normal_y /= norm;
            normal.normal_z /= norm;
        }
        if (!std::isfinite(normal_vec[3])) {
            nan_curvatures_counter++;
        }
    }
    PCL_DEBUG("[estimateNormals] %d NaN normals, %d NaN curvatures.\n", nan_normals_counter, nan_curvatures_counter);
}

void estimateNormalsRadius(float radius_search, PointNCloud::Ptr &pcd, const PointNCloud::ConstPtr &surface,
                           const std::optional<Eigen::Vector3f> &vp, bool normals_available) {
    pcl::console::print_highlight("Estimating normals..\n");
    pcl::NormalEstimationOMP<PointN, PointN> normal_est;
    normal_est.setRadiusSearch(radius_search);
    normal_est.setInputCloud(pcd);
    if (surface) normal_est.setSearchSurface(surface);
    if (vp.has_value()) normal_est.setViewPoint(vp.value().x(), vp.value().y(), vp.value().z());
    pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
    normal_est.setSearchMethod(tree);
    normal_est.compute(*pcd);
    postprocessNormals(pcd, normals_available);
}

void estimateNormalsPoints(int k_points, PointNCloud::Ptr &pcd, const PointNCloud::ConstPtr &surface,
                           const std::optional<Eigen::Vector3f> &vp, bool normals_available) {
    pcl::NormalEstimationOMP<PointN, PointN> normal_est;
    normal_est.setKSearch(k_points);
    normal_est.setInputCloud(pcd);
    if (surface) normal_est.setSearchSurface(surface);
    if (vp.has_value()) normal_est.setViewPoint(vp.value().x(), vp.value().y(), vp.value().z());
    pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
    normal_est.setSearchMethod(tree);
    normal_est.compute(*pcd);
    postprocessNormals(pcd, normals_available);
}

PointNCloud::ConstPtr detectKeyPoints(const PointNCloud::ConstPtr &pcd, const AlignmentParameters &parameters,
                                      float iss_radius, bool debug, bool subvoxel) {
    PointNCloud::Ptr key_points(new PointNCloud);
    pcl::IndicesPtr indices(new pcl::Indices);
    if (parameters.keypoint_id == KEYPOINT_ISS) {
        pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
        ISSKeypoint3DDebug iss_detector;
        iss_detector.setSearchMethod(tree);
        iss_detector.setSalientRadius(iss_radius);
        iss_detector.setNonMaxRadius(iss_radius);
        iss_detector.setThreshold21(0.975);
        iss_detector.setThreshold32(0.975);
        iss_detector.setMinNeighbors(4);
        iss_detector.setInputCloud(pcd);
        iss_detector.setNormals(pcd);
        iss_detector.compute(*key_points);
        indices = std::make_shared<pcl::Indices>(iss_detector.getKeypointsIndices()->indices);
        if (parameters.fix_seed) {
            std::sort(indices->begin(), indices->end());
            for (int i = 0; i < indices->size(); ++i) {
                key_points->points[i] = pcd->points[indices->operator[](i)];
            }
        }
        PCL_DEBUG("[detectKeyPoints] %d key points\n", indices->size());
        if (debug) iss_detector.saveEigenValues(parameters);
        if (subvoxel) iss_detector.estimateSubVoxelKeyPoints(key_points);
    } else {
        if (parameters.keypoint_id != KEYPOINT_ANY) {
            PCL_WARN("[detectKeyPoints] Detection method %s isn't supported, no detection method will be applied.\n",
                     parameters.keypoint_id.c_str());
        }
        return pcd;
    }
    return key_points;
}

void estimateReferenceFrames(const PointNCloud::ConstPtr &pcd, const PointNCloud::ConstPtr &surface,
                             PointRFCloud::Ptr &frames, float radius_search, const AlignmentParameters &parameters) {
    std::string lrf_id = parameters.lrf_id;
    std::transform(lrf_id.begin(), lrf_id.end(), lrf_id.begin(), [](unsigned char c) { return std::tolower(c); });
    if (lrf_id == "gt") {
        Eigen::Matrix3f lrf_eigen = Eigen::Matrix3f::Identity();
        if (!parameters.ground_truth.has_value()) {
            PCL_ERROR("[estimateReferenceFrames] ground truth wasn't provided!");
        } else {
            lrf_eigen = parameters.ground_truth.value().block<3, 3>(0, 0).inverse() * lrf_eigen;
        }
        PointRF lrf;
        for (int d = 0; d < 3; ++d) {
            lrf.x_axis[d] = lrf_eigen.col(0)[d];
            lrf.y_axis[d] = lrf_eigen.col(1)[d];
            lrf.z_axis[d] = lrf_eigen.col(2)[d];
        }
        frames = std::make_shared<PointRFCloud>(PointRFCloud());
        frames->resize(pcd->size(), lrf);
    } else if (lrf_id == "gravity") {
        pcl::IndicesPtr indices_failing(new pcl::Indices);
        PointRFCloud::Ptr frames_failing(new PointRFCloud);
        frames = std::make_shared<PointRFCloud>(PointRFCloud());
        frames->resize(pcd->size());

        Eigen::Vector3f gravity(0, 0, 1);
        for (int i = 0; i < pcd->size(); ++i) {
            PointRF &output_rf = frames->points[i];
            const PointN &normal = pcd->points[i];
            Eigen::Vector3f z_axis(normal.normal_x, normal.normal_y, normal.normal_z);
            if (std::acos(std::abs(std::clamp(z_axis.dot(gravity), -1.0f, 1.0f))) > RF_MIN_ANGLE_RAD) {
                Eigen::Vector3f y_axis = gravity.cross(z_axis);
                Eigen::Vector3f x_axis = y_axis.cross(z_axis);
                for (int d = 0; d < 3; ++d) {
                    output_rf.x_axis[d] = x_axis[d];
                    output_rf.y_axis[d] = y_axis[d];
                    output_rf.z_axis[d] = z_axis[d];
                }
            } else {
                indices_failing->push_back(i);
            }
        }
        pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
        tree->setInputCloud(pcd);
        tree->setSortedResults(true);

        pcl::SHOTLocalReferenceFrameEstimation<PointN, PointRF>::Ptr lrf_estimator(
                new pcl::SHOTLocalReferenceFrameEstimation<PointN, PointRF>());
        lrf_estimator->setRadiusSearch(radius_search);
        lrf_estimator->setInputCloud(pcd);
        lrf_estimator->setIndices(indices_failing);
        lrf_estimator->setSearchSurface(surface);
        lrf_estimator->setSearchMethod(tree);
        lrf_estimator->compute(*frames_failing);
        for (int i = 0; i < indices_failing->size(); ++i) {
            frames->points[indices_failing->operator[](i)] = frames_failing->points[i];
        }
        PCL_DEBUG("[estimateReferenceFrames] %i/%i frames are estimated using SHOT.\n",
                  frames_failing->size(), frames->size());
    } else if (lrf_id != DEFAULT_LRF) {
        PCL_WARN("[estimateReferenceFrames] LRF %s isn't supported, default LRF will be used.\n", lrf_id.c_str());
    }
}

void saveColorizedPointCloud(const PointNCloud::ConstPtr &pcd, const Eigen::Matrix4f &transformation_gt,
                             int color, const std::string &filepath) {
    PointNCloud pcd_aligned;
    pcl::transformPointCloudWithNormals(*pcd, pcd_aligned, transformation_gt);

    PointColoredNCloud dst;
    dst.resize(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        pcl::copyPoint(pcd_aligned.points[i], dst.points[i]);
        setPointColor(dst.points[i], color);
    }
    pcl::io::savePLYFileBinary(filepath, dst);
}

void savePointCloudWithCorrespondences(const PointNCloud::ConstPtr &pcd,
                                       const PointNCloud::ConstPtr &kps,
                                       const Correspondences &correspondences,
                                       const Correspondences &correct_correspondences,
                                       const Correspondences &inliers,
                                       const AlignmentParameters &parameters,
                                       const Eigen::Matrix4f &transformation_gt, bool is_source) {
    int n = pcd->size();
    bool kps_any = pcd->size() == kps->size();

    PointNCloud pcd_aligned, kps_aligned;
    pcl::transformPointCloudWithNormals(*pcd, pcd_aligned, transformation_gt);

    PointColoredNCloud dst;
    dst.resize(n);
    for (int i = 0; i < n; ++i) {
        pcl::copyPoint(pcd_aligned.points[i], dst.points[i]);
        setPointColor(dst.points[i], kps_any ? COLOR_BEIGE : COLOR_PARAKEET);
    }
    if (!kps_any) {
        pcl::transformPointCloudWithNormals(*kps, kps_aligned, transformation_gt);
        dst.points.resize(n + kps->size());

        for (int i = 0; i < kps->size(); ++i) {
            pcl::copyPoint(kps_aligned.points[i], dst.points[n + i]);
            setPointColor(dst.points[n + i], COLOR_BEIGE);
        }
    }
    for (const auto &correspondence: correspondences) {
        if (is_source) {
            int index_query = kps_any ? correspondence.index_query : n + correspondence.index_query;
            setPointColor(dst.points[index_query], COLOR_RED);
        } else {
            int index_match = kps_any ? correspondence.index_match : n + correspondence.index_match;
            setPointColor(dst.points[index_match], COLOR_RED);
        }
    }
    for (const auto &inlier: inliers) {
        if (is_source) {
            int index_query = kps_any ? inlier.index_query : n + inlier.index_query;
            setPointColor(dst.points[index_query], COLOR_BLUE);
        } else {
            int index_match = kps_any ? inlier.index_match : n + inlier.index_match;
            setPointColor(dst.points[index_match], COLOR_BLUE);
        }
    }
    for (const auto &correspondence: correct_correspondences) {
        if (is_source) {
            int index_query = kps_any ? correspondence.index_query : n + correspondence.index_query;
            mixPointColor(dst.points[index_query], COLOR_WHITE);
        } else {
            int index_match = kps_any ? correspondence.index_match : n + correspondence.index_match;
            mixPointColor(dst.points[index_match], COLOR_WHITE);
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
    pcl::transformPointCloudWithNormals(dst, dst, transformation_gt);
    std::string filepath = constructPath(parameters, name, "ply", true, true, true, true);
    pcl::io::savePLYFileBinary(filepath, dst);
}

void saveHistogram(const std::string &values_path, const std::string &hist_path) {
    std::string command = "python3 plots.py histogram " + values_path + " " + hist_path;
    system(command.c_str());
}

void calculateTemperatureMap(PointColoredNCloud::Ptr &compared,
                             PointColoredNCloud::Ptr &reference,
                             TemperatureType type, std::vector<float> &temperatures,
                             float temperature_min, float temperature_max, float distance_max) {
    pcl::KdTreeFLANN<PointColoredN> tree_reference;
    tree_reference.setInputCloud(reference);

    int n = compared->size();
    pcl::Indices nn_indices;
    std::vector<float> nn_sqr_dists;
    float temperature;
    PointColoredN nearest_point;
    float search_radius = DIST_TO_PLANE_COEFFICIENT * distance_max;

#pragma omp parallel for default(none) \
    firstprivate(nn_indices, nn_sqr_dists, temperature, n, distance_max, nearest_point, \
                 temperature_min, temperature_max, type, search_radius) \
    shared(tree_reference, compared, reference, temperatures)
    for (int i = 0; i < n; ++i) {
        tree_reference.radiusSearch(*compared, i, search_radius, nn_indices, nn_sqr_dists, 1);
        float dist_to_plane = distance_max;
        if (!nn_indices.empty()) {
            nearest_point = reference->points[nn_indices[0]];
            dist_to_plane = std::fabs(nearest_point.getNormalVector3fMap().transpose() *
                                      (nearest_point.getVector3fMap() -
                                       compared->points[i].getVector3fMap()));
            // normal can be invalid
            dist_to_plane = std::isfinite(dist_to_plane) ? dist_to_plane : nn_sqr_dists[0];
        }
        if (dist_to_plane < distance_max) {
            if (type == TemperatureType::NormalDifference) {
                float cos_normal_diff = nearest_point.getNormalVector3fMap().dot(
                        compared->points[i].getNormalVector3fMap());
                float normal_diff = std::fabs(std::acos(std::max(std::min(cos_normal_diff, 1.f), -1.f)));
                normal_diff = std::min(normal_diff, temperature_max);
                normal_diff = std::isfinite(normal_diff) ? normal_diff : temperature_max;
                temperature = normal_diff;
            } else {
                temperature = dist_to_plane;
            }
            setPointColor(compared->points[i], getColor(temperature, temperature_min, temperature_max));
            temperatures[i] = temperature;
        } else {
            setPointColor(compared->points[i], getColor(temperature_max, temperature_min, temperature_max));
            temperatures[i] = temperature_max;
        }
    }
}

void saveTemperatureMaps(PointNCloud::Ptr &src, PointNCloud::Ptr &tgt,
                         const std::string &name, const AlignmentParameters &params, float distance_thr,
                         const Eigen::Matrix4f &transformation, bool normals_available) {
    if (!normals_available) {
        estimateNormalsPoints(params.normal_nr_points, src, {nullptr}, params.vp_src, false);
        estimateNormalsPoints(params.normal_nr_points, tgt, {nullptr}, params.vp_tgt, false);
    }
    int n_src = src->size(), n_tgt = tgt->size();
    PointColoredNCloud::Ptr src_colored(new PointColoredNCloud), tgt_colored(new PointColoredNCloud);
    std::vector<float> temperatures_src, temperatures_tgt;
    temperatures_src.resize(n_src);
    temperatures_tgt.resize(n_tgt);
    src_colored->resize(n_src);
    tgt_colored->resize(n_tgt);
    pcl::copyPointCloud(*src, *src_colored);
    pcl::copyPointCloud(*tgt, *tgt_colored);
    pcl::transformPointCloudWithNormals(*src_colored, *src_colored, transformation);
    float distance_min = 0;
    float distance_max = distance_thr;
    calculateTemperatureMap(src_colored, tgt_colored, TemperatureType::Distance, temperatures_src, distance_min,
                            distance_max, distance_max);
    calculateTemperatureMap(tgt_colored, src_colored, TemperatureType::Distance, temperatures_tgt, distance_min,
                            distance_max, distance_max);

    temperatures_src.erase(std::remove_if(temperatures_src.begin(), temperatures_src.end(),
                                          [&distance_max](float d) { return d >= distance_max; }),
                           temperatures_src.end());
    temperatures_tgt.erase(std::remove_if(temperatures_tgt.begin(), temperatures_tgt.end(),
                                          [&distance_max](float d) { return d >= distance_max; }),
                           temperatures_tgt.end());
    std::string distances_path_src = constructPath(params, name + "_distances_src", "csv");
    std::string distances_path_tgt = constructPath(params, name + "_distances_tgt", "csv");
    saveVector(temperatures_src, distances_path_src);
    saveVector(temperatures_tgt, distances_path_tgt);

    saveHistogram(distances_path_src, constructPath(params, name + "_histogram_src", "png"));
    saveHistogram(distances_path_tgt, constructPath(params, name + "_histogram_tgt", "png"));


    pcl::io::savePLYFileASCII(constructPath(params, name + "_dists_src"), *src_colored);
    pcl::io::savePLYFileASCII(constructPath(params, name + "_dists_tgt"), *tgt_colored);

    float normal_diff_min = 0;
    float normal_diff_max = M_PI / 2;
    calculateTemperatureMap(src_colored, tgt_colored, TemperatureType::NormalDifference, temperatures_src,
                            normal_diff_min, normal_diff_max, distance_max);
    calculateTemperatureMap(tgt_colored, src_colored, TemperatureType::NormalDifference, temperatures_tgt,
                            normal_diff_min, normal_diff_max, distance_max);
//    std::string normal_diffs_path_src = constructPath(params, name + "_normal_diffs_src", "csv");
//    std::string normal_diffs_path_tgt = constructPath(params, name + "_normal_diffs_tgt", "csv");
//    saveVector(temperatures_src, normal_diffs_path_src);
//    saveVector(temperatures_tgt, normal_diffs_path_tgt);

    pcl::io::savePLYFileBinary(constructPath(params, name + "_normal_diffs_src"), *src_colored);
    pcl::io::savePLYFileBinary(constructPath(params, name + "_normal_diffs_tgt"), *tgt_colored);
}

void writeFacesToPLYFileASCII(const PointColoredNCloud::Ptr &pcd, std::size_t match_offset,
                              const Correspondences &correspondences,
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

// TODO: fix
//void saveCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
//                         const Correspondences &correspondences,
//                         const Eigen::Matrix4f &transformation_gt,
//                         const AlignmentParameters &parameters, bool sparse) {
//    PointColoredNCloud::Ptr dst(new PointColoredNCloud);
//    PointNCloud::Ptr src_aligned_gt(new PointNCloud);
//    pcl::transformPointCloudWithNormals(*src, *src_aligned_gt, transformation_gt);
//    float diagonal = getAABBDiagonal(src_aligned_gt);
//
//    dst->resize(src_aligned_gt->size() + tgt->size());
//    for (int i = 0; i < src_aligned_gt->size(); ++i) {
//        pcl::copyPoint(src_aligned_gt->points[i], dst->points[i]);
//        setPointColor(dst->points[i], COLOR_BEIGE);
//    }
//    for (int i = 0; i < tgt->size(); ++i) {
//        pcl::copyPoint(tgt->points[i], dst->points[src_aligned_gt->size() + i]);
//        dst->points[src_aligned_gt->size() + i].x += diagonal;
//        setPointColor(dst->points[src_aligned_gt->size() + i], COLOR_PURPLE);
//    }
//    UniformRandIntGenerator rand_generator(0, 255);
//    for (const auto &corr: correspondences) {
//        std::uint8_t red = rand_generator(), green = rand_generator(), blue = rand_generator();
//        setPointColor(dst->points[corr.index_query], red, green, blue);
//        setPointColor(dst->points[src_aligned_gt->size() + corr.index_match], red, green, blue);
//    }
//    std::string filepath;
//    if (sparse) {
//        filepath = constructPath(parameters, "correspondences_sparse");
//    } else {
//        filepath = constructPath(parameters, "correspondences");
//    }
//    pcl::io::savePLYFileASCII(filepath, *dst);
//    if (sparse) {
//        std::vector correspondences_sparse(correspondences);
//        std::shuffle(correspondences_sparse.begin(), correspondences_sparse.end(),
//                     std::mt19937(std::random_device()()));
//        correspondences_sparse.resize(std::min(correspondences_sparse.size(), DEBUG_N_EDGES));
//        writeFacesToPLYFileASCII(dst, src->size(), correspondences_sparse, filepath);
//    } else {
//        writeFacesToPLYFileASCII(dst, src->size(), correspondences, filepath);
//    }
//}
//
//void saveCorrectCorrespondences(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
//                                const Correspondences &correspondences,
//                                const Correspondences &correct_correspondences,
//                                const Eigen::Matrix4f &transformation_gt,
//                                const AlignmentParameters &parameters, bool sparse) {
//    PointColoredNCloud::Ptr dst(new PointColoredNCloud);
//    PointNCloud::Ptr src_aligned_gt(new PointNCloud);
//    pcl::transformPointCloudWithNormals(*src, *src_aligned_gt, transformation_gt);
//    float half_diagonal = 0.5 * getAABBDiagonal(src_aligned_gt);
//
//    dst->resize(src_aligned_gt->size() + tgt->size());
//    for (int i = 0; i < src_aligned_gt->size(); ++i) {
//        pcl::copyPoint(src_aligned_gt->points[i], dst->points[i]);
//        setPointColor(dst->points[i], COLOR_BEIGE);
//    }
//    for (int i = 0; i < tgt->size(); ++i) {
//        pcl::copyPoint(tgt->points[i], dst->points[src_aligned_gt->size() + i]);
//        dst->points[src_aligned_gt->size() + i].x += half_diagonal;
//        setPointColor(dst->points[src_aligned_gt->size() + i], COLOR_PURPLE);
//    }
//    UniformRandIntGenerator rand_generator(0, 255);
//    for (const auto &corr: correspondences) {
//        setPointColor(dst->points[corr.index_query], COLOR_ROSE);
//        setPointColor(dst->points[src_aligned_gt->size() + corr.index_match], COLOR_ROSE);
//    }
//    for (const auto &corr: correct_correspondences) {
//        setPointColor(dst->points[corr.index_query], COLOR_PARAKEET);
//        setPointColor(dst->points[src_aligned_gt->size() + corr.index_match], COLOR_PARAKEET);
//    }
//    std::string filepath;
//    if (sparse) {
//        filepath = constructPath(parameters, "correspondences_sparse");
//    } else {
//        filepath = constructPath(parameters, "correspondences");
//    }
//    pcl::io::savePLYFileASCII(filepath, *dst);
//    if (sparse) {
//        std::vector correspondences_sparse(correspondences);
//        std::shuffle(correspondences_sparse.begin(), correspondences_sparse.end(),
//                     std::mt19937(std::random_device()()));
//        correspondences_sparse.resize((int) (0.02 * correspondences_sparse.size()));
//        writeFacesToPLYFileASCII(dst, src->size(), correspondences_sparse, filepath);
//    } else {
//        writeFacesToPLYFileASCII(dst, src->size(), correspondences, filepath);
//    }
//}

void saveCorrespondenceDistances(const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                                 const Correspondences &correspondences,
                                 const Eigen::Matrix4f &transformation_gt, const AlignmentParameters &parameters) {
    std::string filepath = constructPath(parameters, "distances", "csv");
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());
    PointNCloud kps_src_aligned_gt;
    pcl::transformPointCloudWithNormals(*kps_src, kps_src_aligned_gt, transformation_gt);

    fout << "distance\n";
    for (const auto &correspondence: correspondences) {
        PointN point_src(kps_src_aligned_gt.points[correspondence.index_query]);
        PointN point_tgt(kps_tgt->points[correspondence.index_match]);
        float dist = pcl::L2_Norm(point_src.data, point_tgt.data, 3) / parameters.distance_thr;
        fout << dist << "\n";
    }
    fout.close();
}

void saveCorrespondencesDebug(const Correspondences &correspondences,
                              const Correspondences &correct_correspondences,
                              const AlignmentParameters &parameters) {
    std::string filepath = constructPath(parameters, "correspondences_debug", "csv");
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());

    std::unordered_set<int> correct_cs_set;
    std::transform(correct_correspondences.begin(), correct_correspondences.end(),
                   std::inserter(correct_cs_set, correct_cs_set.begin()),
                   [](const Correspondence &corr) { return corr.index_query; });
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
    std::string matching_id = parameters.matching_id;
    if (matching_id == MATCHING_RATIO) matching_id += std::to_string(parameters.ratio_k);
    std::string full_name = parameters.testname + "_" + name +
                            "_" + std::to_string(parameters.feature_nr_points) +
                            "_" + parameters.descriptor_id + "_" + (parameters.use_bfmatcher ? "bf" : "flann") +
                            (with_metric ? "_" + parameters.alignment_id : "") +
                            "_" + parameters.keypoint_id + "_" + parameters.lrf_id +
                            (with_metric ? "_" + parameters.metric_id + "_" + parameters.score_id : "") +
                            "_" + matching_id + "_" + std::to_string(parameters.randomness) +
                            (with_weights ? "_" + parameters.weight_id : "") +
                            "_" + std::to_string(parameters.normal_nr_points) +
                            "_" + std::to_string(parameters.reestimate_frames) +
                            "_" + std::to_string(parameters.iss_radius_src) +
                            "_" + std::to_string(parameters.iss_radius_tgt) +
                            "_" + std::to_string(parameters.scale_factor) +
                            "_" + std::to_string(parameters.cluster_k);
    if (parameters.feature_radius.has_value()) {
        full_name += "_" + std::to_string(parameters.feature_radius.value());
    }
    if (with_version) {
        full_name += "_" + VERSION;
    }
    if (with_subversion) {
        full_name += SUBVERSION;
    }
    return full_name;
}

void readKeyPointsAndCorrespondences(CorrespondencesConstPtr &correspondences,
                                     PointNCloud::ConstPtr &kps_src, PointNCloud::ConstPtr &kps_tgt,
                                     const AlignmentParameters &params, bool &success) {
    std::string filepath_corrs = constructPath(params, "correspondences", "csv", true, false, false);

    bool file_exists = std::filesystem::exists(filepath_corrs);
    CorrespondencesPtr correspondences_ = std::make_shared<Correspondences>();
    PointNCloud::Ptr kps_src_(new PointNCloud), kps_tgt_(new PointNCloud);
    if (file_exists) {
        std::ifstream fin(filepath_corrs);
        if (fin.is_open()) {
            std::string line;
            std::vector<std::string> tokens;
            std::getline(fin, line); // header
            while (std::getline(fin, line)) {
                // query_idx, match_idx, distance, x_s, y_s, z_s, x_t, y_t, z_t
                split(line, tokens, ",");
                Correspondence corr{std::stoi(tokens[0]), std::stoi(tokens[1]),
                                    std::stof(tokens[2]), std::stof(tokens[3])};
                correspondences_->push_back(corr);
            }
            correspondences = correspondences_;
        } else {
            perror(("error while opening file " + filepath_corrs).c_str());
        }

        if (pcl::io::loadPLYFile(constructPath(params, "key_points_src", "ply", true, false, false), *kps_src_) < 0 ||
            pcl::io::loadPLYFile(constructPath(params, "key_points_tgt", "ply", true, false, false), *kps_tgt_) < 0) {
            perror("error loading src/tgt key points file!");
        }
        kps_src = kps_src_;
        kps_tgt = kps_tgt_;

        success = true;
    }
}

void saveKeyPointsAndCorrespondences(const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                                     const CorrespondencesConstPtr &correspondences,
                                     const AlignmentParameters &params) {
    std::string filepath_corrs = constructPath(params, "correspondences", "csv", true, false, false);

    std::ofstream fout(filepath_corrs);
    if (fout.is_open()) {
        fout << "query_idx,match_idx,distance,threshold,x_s,y_s,z_s,x_t,y_t,z_t\n";
        for (const auto &corr: *correspondences) {
            fout << corr.index_query << "," << corr.index_match << "," << corr.distance << "," << corr.threshold << ",";
            fout << kps_src->points[corr.index_query].x << ","
                 << kps_src->points[corr.index_query].y << ","
                 << kps_src->points[corr.index_query].z << ",";
            fout << kps_tgt->points[corr.index_match].x << ","
                 << kps_tgt->points[corr.index_match].y << ","
                 << kps_tgt->points[corr.index_match].z << '\n';
        }
        fout.close();
    } else {
        perror(("error while opening file " + filepath_corrs).c_str());
    }
    pcl::io::savePLYFileBinary(constructPath(params, "key_points_src", "ply", true, false, false), *kps_src);
    pcl::io::savePLYFileBinary(constructPath(params, "key_points_tgt", "ply", true, false, false), *kps_tgt);
}