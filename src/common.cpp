#include <filesystem>
#include <fstream>
#include <random>

#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>
#include <pcl/io/ply_io.h>

#include "common.h"

namespace fs = std::filesystem;

const std::string DATA_DEBUG_PATH = fs::path("data") / fs::path("debug");
const std::string VERSION = "02";

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

std::pair<PointT, PointT> calculateBoundingBox(const PointCloudT::Ptr &pcd) {
    float min = std::numeric_limits<float>::min(), max = std::numeric_limits<float>::max();
    PointT min_point_AABB(max, max, max);
    PointT max_point_AABB(min, min, min);
    for (auto p: pcd->points) {
        min_point_AABB.x = std::min(min_point_AABB.x, p.x);
        min_point_AABB.y = std::min(min_point_AABB.y, p.y);
        min_point_AABB.z = std::min(min_point_AABB.z, p.z);
        max_point_AABB.x = std::max(max_point_AABB.x, p.x);
        max_point_AABB.y = std::max(max_point_AABB.y, p.y);
        max_point_AABB.z = std::max(max_point_AABB.z, p.z);
    }
    return {min_point_AABB, max_point_AABB};
}

float getAABBDiagonal(const PointCloudT::Ptr &pcd) {
    auto [min_point_AABB, max_point_AABB] = calculateBoundingBox(pcd);

    Eigen::Vector3f min_point(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
    Eigen::Vector3f max_point(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);

    return (max_point - min_point).norm();
}

void saveColorizedPointCloud(const PointCloudT::Ptr &src,
                             const std::vector<MultivaluedCorrespondence> &correspondences,
                             const std::vector<MultivaluedCorrespondence> &correct_correspondences,
                             const pcl::Indices &inliers,
                             const std::string &testname) {
    PointCloudColoredT dst;
    dst.resize(src->size());
    for (int i = 0; i < src->size(); ++i) {
        dst.points[i].x = src->points[i].x;
        dst.points[i].y = src->points[i].y;
        dst.points[i].z = src->points[i].z;
        setPointColor(dst.points[i], COLOR_BEIGE);
    }
    for (const auto &correspondence: correspondences) {
        setPointColor(dst.points[correspondence.query_idx], COLOR_RED);
    }
    for (const auto &idx: inliers) {
        setPointColor(dst.points[idx], COLOR_BLUE);
    }
    for (const auto &correspondence: correct_correspondences) {
        mixPointColor(dst.points[correspondence.query_idx], COLOR_WHITE);
    }
    pcl::io::savePLYFileBinary(constructPath(testname, "downsampled"), dst);
}

void writeEdgesToPLYFileASCII(const std::vector<MultivaluedCorrespondence> &correspondences,
                              std::size_t match_offset, const std::string &filepath) {
    std::string filepath_tmp = filepath + "tmp";
    std::fstream fin(filepath, std::ios_base::in), fout(filepath_tmp, std::ios_base::out);
    if (!fin.is_open())
        perror(("error while opening file " + filepath).c_str());
    if (!fout.is_open())
        perror(("error while creating temporary file " + filepath).c_str());

    std::string line;
    long pos = fout.tellg();
    while (std::getline(fin, line)) {
        fout << line << "\n";
        if (line == "end_header") {
            fout.seekg(pos);
            fout << "element edge " << correspondences.size() << "\n";
            fout << "property int vertex1\n" << "property int vertex2\n";
            fout << "end_header\n";
        }
        pos = fout.tellg();
    }
    for (const auto &correspondence: correspondences) {
        fout << correspondence.query_idx << " " << match_offset + correspondence.match_indices[0] << "\n";
    }
    fin.close();
    fout.close();
    fs::remove(filepath);
    fs::rename(filepath_tmp, filepath);
}

void saveCorrespondences(const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                         const std::vector<MultivaluedCorrespondence> &correspondences,
                         const Eigen::Matrix4f &transformation_gt,
                         const std::string &testname, bool sparse) {
    PointCloudColoredT dst;
    PointCloudT::Ptr src_aligned_gt(new PointCloudT);
    pcl::transformPointCloud(*src, *src_aligned_gt, transformation_gt);
    float diagonal = getAABBDiagonal(src_aligned_gt);

    dst.resize(src_aligned_gt->size() + tgt->size());
    for (int i = 0; i < src_aligned_gt->size(); ++i) {
        dst.points[i].x = src_aligned_gt->points[i].x;
        dst.points[i].y = src_aligned_gt->points[i].y;
        dst.points[i].z = src_aligned_gt->points[i].z;
        setPointColor(dst.points[i], COLOR_BEIGE);
    }
    for (int i = 0; i < tgt->size(); ++i) {
        dst.points[src_aligned_gt->size() + i].x = tgt->points[i].x + diagonal;
        dst.points[src_aligned_gt->size() + i].y = tgt->points[i].y;
        dst.points[src_aligned_gt->size() + i].z = tgt->points[i].z;
        setPointColor(dst.points[src_aligned_gt->size() + i], COLOR_PURPLE);
    }
    UniformRandIntGenerator rand_generator(0, 255);
    for (const auto &corr: correspondences) {
        std::uint8_t red = rand_generator(), green = rand_generator(), blue = rand_generator();
        setPointColor(dst.points[corr.query_idx], red, green, blue);
        setPointColor(dst.points[src_aligned_gt->size() + corr.match_indices[0]], red, green, blue);
    }
    std::string filepath;
    if (sparse) {
        filepath = constructPath(testname, "correspondences_sparse");
    } else {
        filepath = constructPath(testname, "correspondences");
    }
    pcl::io::savePLYFileASCII(filepath, dst);
    if (sparse) {
        std::vector correspondences_sparse(correspondences);
        std::shuffle(correspondences_sparse.begin(), correspondences_sparse.end(), std::mt19937(std::random_device()()));
        correspondences_sparse.resize(std::min(correspondences_sparse.size(), DEBUG_N_EDGES));
        writeEdgesToPLYFileASCII(correspondences_sparse, src_aligned_gt->size(), filepath);
    } else {
        writeEdgesToPLYFileASCII(correspondences, src_aligned_gt->size(), filepath);
    }
}

void saveCorrespondenceDistances(const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                                 const std::vector<MultivaluedCorrespondence> &correspondences,
                                 const Eigen::Matrix4f &transformation_gt, float voxel_size,
                                 const std::string &testname) {
    std::string filepath = constructPath(testname, "distances", "csv");
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());
    PointCloudT src_aligned_gt;
    pcl::transformPointCloud(*src, src_aligned_gt, transformation_gt);

    fout << "distance\n";
    for (const auto &correspondence: correspondences) {
        PointT source_point(src_aligned_gt.points[correspondence.query_idx]);
        PointT target_point(tgt->points[correspondence.match_indices[0]]);
        float dist = pcl::L2_Norm(source_point.data, target_point.data, 3) / voxel_size;
        fout << dist << "\n";
    }
    fout.close();
}

void setPointColor(PointColoredT &point, int color) {
    point.r = (color >> 16) & 0xff;
    point.g = (color >> 8) & 0xff;
    point.b = (color >> 0) & 0xff;
}

void mixPointColor(PointColoredT &point, int color) {
    point.r = point.r / 2 + ((color >> 16) & 0xff) / 2;
    point.g = point.g / 2 + ((color >> 8) & 0xff) / 2;
    point.b = point.b / 2 + ((color >> 0) & 0xff) / 2;
}

void setPointColor(PointColoredT &point, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    point.r = red;
    point.g = green;
    point.b = blue;
}


std::string constructPath(const std::string &test, const std::string &name, const std::string &extension, bool with_version) {
    std::string filename =  test + "_" + name;
    if (with_version) {
        filename += "_" + VERSION;
    }
    filename += "." + extension;
    return fs::path(DATA_DEBUG_PATH) / fs::path(filename);
}