#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_set>

#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>
#include <pcl/common/copy_point.h>
#include <pcl/io/ply_io.h>

#include "common.h"

namespace fs = std::filesystem;

const std::string DATA_DEBUG_PATH = fs::path("data") / fs::path("debug");
const std::string VERSION = "04";
const std::string DEFAULT_DESCRIPTOR = "fpfh";
const std::string DEFAULT_LRF = "default";
const std::string DEFAULT_METRIC = "correspondences";

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

std::vector<AlignmentParameters> getParametersFromConfig(const YamlConfig &config,
                                                         const std::vector<::pcl::PCLPointField> &fields_src,
                                                         const std::vector<::pcl::PCLPointField> &fields_tgt) {
    std::vector<AlignmentParameters> parameters_container, new_parameters_container;
    AlignmentParameters parameters;
    parameters.downsample = config.get<bool>("downsample", true);
    parameters.edge_thr_coef = config.get<float>("edge_thr").value();
    parameters.distance_thr_coef = config.get<float>("distance_thr_coef").value();
    parameters.max_iterations = config.get<int>("iteration");
    parameters.confidence = config.get<float>("confidence").value();
    parameters.inlier_fraction = config.get<float>("inlier_fraction").value();
    parameters.reciprocal = config.get<bool>("reciprocal").value();
    parameters.use_bfmatcher = config.get<bool>("bf", false);
    parameters.randomness = config.get<int>("randomness").value();
    parameters.n_samples = config.get<int>("n_samples").value();
    parameters.save_features = config.get<bool>("save_features", false);
    parameters.bf_block_size = config.get<int>("block_size", 10000);

    bool use_normals = config.get<bool>("use_normals", false);
    // TODO: use normals even with one point cloud missing normals
    bool normals_available = pointCloudHasNormals<PointTN>(fields_src) && pointCloudHasNormals<PointTN>(fields_tgt);
    if (use_normals && !normals_available) {
        PCL_WARN("Point cloud doesn't have normals.\n");
    }
    parameters.normals_available = normals_available;
    parameters.use_normals = use_normals && normals_available;
    parameters_container.push_back(parameters);

    auto voxel_sizes = config.getVector<float>("voxel_size").value();
    for (float vs: voxel_sizes) {
        for (auto ps: parameters_container) {
            ps.voxel_size = vs;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto normal_radius_coefs = config.getVector<float>("normal_radius_coef").value();
    for (float nrc: normal_radius_coefs) {
        for (auto ps: parameters_container) {
            ps.normal_radius_coef = nrc;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    auto feature_radius_coefs = config.getVector<float>("feature_radius_coef").value();
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

    auto descriptor_ids = config.getVector<std::string>("descriptor", DEFAULT_DESCRIPTOR);
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

    auto metric_ids = config.getVector<std::string>("metric", DEFAULT_METRIC);
    for (const auto &id: metric_ids) {
        for (auto ps: parameters_container) {
            ps.metric_id = id;
            new_parameters_container.push_back(ps);
        }
    }
    std::swap(parameters_container, new_parameters_container);
    new_parameters_container.clear();

    return parameters_container;
}

float getAABBDiagonal(const PointCloudTN::Ptr &pcd) {
    auto[min_point_AABB, max_point_AABB] = calculateBoundingBox<PointTN>(pcd);

    Eigen::Vector3f min_point(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
    Eigen::Vector3f max_point(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);

    return (max_point - min_point).norm();
}

void saveColorizedPointCloud(const PointCloudTN::ConstPtr &pcd,
                             const std::vector<MultivaluedCorrespondence> &correspondences,
                             const std::vector<MultivaluedCorrespondence> &correct_correspondences,
                             const std::vector<InlierPair> &inlier_pairs, const AlignmentParameters &parameters,
                             const Eigen::Matrix4f &transformation_gt, bool is_source) {
    PointCloudTN pcd_aligned;
    pcl::transformPointCloud(*pcd, pcd_aligned, transformation_gt);

    PointCloudColoredTN dst;
    dst.resize(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        pcl::copyPoint(pcd_aligned.points[i], dst.points[i]);
        setPointColor(dst.points[i], COLOR_BEIGE);
    }
    for (const auto &correspondence: correspondences) {
        if (is_source) {
            setPointColor(dst.points[correspondence.query_idx], COLOR_RED);
        } else {
            setPointColor(dst.points[correspondence.match_indices[0]], COLOR_RED);
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
            mixPointColor(dst.points[correspondence.query_idx], COLOR_WHITE);
        } else {
            mixPointColor(dst.points[correspondence.match_indices[0]], COLOR_WHITE);
        }
    }
    std::string filepath = constructPath(parameters, std::string("downsampled_") + (is_source ? "src" : "tgt"));
    pcl::io::savePLYFileBinary(filepath, dst);
}

void writeFacesToPLYFileASCII(const PointCloudColoredTN::Ptr &pcd, std::size_t match_offset,
                              const std::vector<MultivaluedCorrespondence> &correspondences,
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
                const auto &p = pcd->points[corr.query_idx];
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
        fout << "3 " << corr.query_idx << " " << match_offset + corr.match_indices[0] << " " << midpoint_offset + i
             << "\n";
    }
    fin.close();
    fout.close();
    fs::remove(filepath);
    fs::rename(filepath_tmp, filepath);
}

void saveCorrespondences(const PointCloudTN::ConstPtr &src, const PointCloudTN::ConstPtr &tgt,
                         const std::vector<MultivaluedCorrespondence> &correspondences,
                         const Eigen::Matrix4f &transformation_gt,
                         const AlignmentParameters &parameters, bool sparse) {
    PointCloudColoredTN::Ptr dst(new PointCloudColoredTN);
    PointCloudTN::Ptr src_aligned_gt(new PointCloudTN);
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
        setPointColor(dst->points[corr.query_idx], red, green, blue);
        setPointColor(dst->points[src_aligned_gt->size() + corr.match_indices[0]], red, green, blue);
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

void saveCorrespondenceDistances(const PointCloudTN::ConstPtr &src, const PointCloudTN::ConstPtr &tgt,
                                 const std::vector<MultivaluedCorrespondence> &correspondences,
                                 const Eigen::Matrix4f &transformation_gt, float voxel_size,
                                 const AlignmentParameters &parameters) {
    std::string filepath = constructPath(parameters, "distances", "csv");
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());
    PointCloudTN src_aligned_gt;
    pcl::transformPointCloud(*src, src_aligned_gt, transformation_gt);

    fout << "distance\n";
    for (const auto &correspondence: correspondences) {
        PointTN source_point(src_aligned_gt.points[correspondence.query_idx]);
        PointTN target_point(tgt->points[correspondence.match_indices[0]]);
        float dist = pcl::L2_Norm(source_point.data, target_point.data, 3) / voxel_size;
        fout << dist << "\n";
    }
    fout.close();
}

// TODO: why did I need this function?
void saveInlierIds(const std::vector<MultivaluedCorrespondence> &correspondences,
                   const std::vector<MultivaluedCorrespondence> &correct_correspondences,
                   const pcl::Indices &inliers, const AlignmentParameters &parameters) {
    std::string filepath = constructPath(parameters, "inliers", "csv");
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());

    std::unordered_set<int> inliers_set, correct_cs_set;
    std::copy(inliers.begin(), inliers.end(), std::inserter(inliers_set, inliers_set.begin()));
    std::transform(correct_correspondences.begin(), correct_correspondences.end(),
                   std::inserter(correct_cs_set, correct_cs_set.begin()),
                   [](const MultivaluedCorrespondence &corr) { return corr.query_idx; });
    fout << "id_src,id_tgt,is_correct,is_inlier\n";
    for (const auto &correspondence: correspondences) {
        fout << correspondence.query_idx << "," << correspondence.match_indices[0] << ",";
        fout << correct_cs_set.contains(correspondence.query_idx) << ",";
        fout << inliers_set.contains(correspondence.query_idx) << "\n";
    }
    fout.close();
}

void setPointColor(PointColoredTN &point, int color) {
    point.r = (color >> 16) & 0xff;
    point.g = (color >> 8) & 0xff;
    point.b = (color >> 0) & 0xff;
}

void mixPointColor(PointColoredTN &point, int color) {
    point.r = point.r / 2 + ((color >> 16) & 0xff) / 2;
    point.g = point.g / 2 + ((color >> 8) & 0xff) / 2;
    point.b = point.b / 2 + ((color >> 0) & 0xff) / 2;
}

void setPointColor(PointColoredTN &point, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    point.r = red;
    point.g = green;
    point.b = blue;
}


std::string
constructPath(const std::string &test, const std::string &name, const std::string &extension, bool with_version) {
    std::string filename = test + "_" + name;
    if (with_version) {
        filename += "_" + VERSION;
    }
    filename += "." + extension;
    return fs::path(DATA_DEBUG_PATH) / fs::path(filename);
}

std::string
constructPath(const AlignmentParameters &parameters, const std::string &name, const std::string &extension,
              bool with_version) {
    std::string filename = parameters.testname + "_" + name +
                           "_" + std::to_string((int) std::round(1e4 * parameters.voxel_size)) +
                           "_" + parameters.descriptor_id + "_" + (parameters.use_bfmatcher ? "bf" : "flann") +
                           "_" + std::to_string((int) parameters.normal_radius_coef) +
                           "_" + std::to_string((int) parameters.feature_radius_coef) +
                           "_" + parameters.lrf_id + "_" + parameters.metric_id +
                           (parameters.use_normals ? "_normals" : "");
    if (with_version) {
        filename += "_" + VERSION;
    }
    filename += "." + extension;
    return fs::path(DATA_DEBUG_PATH) / fs::path(filename);
}