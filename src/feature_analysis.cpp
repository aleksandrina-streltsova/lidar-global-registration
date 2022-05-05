#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "feature_analysis.h"
#include "downsample.h"

void saveNormals(const PointNCloud::Ptr &pcd,
                 const Eigen::Matrix4f &transformation_gt, bool is_source,  const AlignmentParameters &parameters) {
    pcl::console::print_highlight("Saving %s normals...\n", is_source ? "source" : "target");
    PointNCloud pcd_aligned;
    pcl::transformPointCloudWithNormals(*pcd, pcd_aligned, transformation_gt);
    std::string filepath = constructPath(parameters,  std::string("normals_")+ (is_source ? "src" : "tgt"));
    pcl::io::savePLYFileBinary(filepath, pcd_aligned);
}

std::vector<int> getPointIds(const PointNCloud::Ptr &all_points, const PointNCloud::Ptr &extracted_points) {
    int n = extracted_points->size();
    pcl::KdTreeFLANN<PointN> tree(new pcl::KdTreeFLANN<PointN>);
    tree.setInputCloud(all_points);

    std::vector<int> ids(n);
    std::vector<int> indices(1);
    std::vector<float> distances(1);

    for (int i = 0; i < n; ++i) {
        tree.nearestKSearch(*extracted_points, i, 1, indices, distances);
        ids[i] = indices[0];
    }
    return ids;
}

void saveExtractedPointIds(const PointNCloud::Ptr &src_fullsize, const PointNCloud::Ptr &tgt_fullsize,
                           const Eigen::Matrix4f &transformation_gt,
                           const AlignmentParameters &parameters, const PointNCloud::Ptr &extracted_points) {
    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    // Downsample
    pcl::console::print_highlight("Downsampling...\n");
    downsamplePointCloud(src_fullsize, src, parameters);
    downsamplePointCloud(tgt_fullsize, tgt, parameters);

    PointNCloud::Ptr src_aligned_gt(new PointNCloud);

    pcl::transformPointCloud(*src, *src_aligned_gt, transformation_gt);
    std::string filepath = constructPath(parameters, "ids", "csv");
    std::fstream fout(filepath, std::ios_base::out);
    std::vector<int> src_ids = getPointIds(src_aligned_gt, extracted_points);
    std::vector<int> tgt_ids = getPointIds(tgt, extracted_points);
    fout << "id_src,id_tgt,x_src,x_tgt,y_src,y_tgt,z_src,z_tgt\n";
    for (int i = 0; i < extracted_points->size(); ++i) {
        int src_id = src_ids[i];
        int tgt_id = tgt_ids[i];
        fout << src_id << "," << tgt_id << ",";
        fout << src_aligned_gt->points[src_id].x << "," << tgt->points[tgt_id].x << ",";
        fout << src_aligned_gt->points[src_id].y << "," << tgt->points[tgt_id].y << ",";
        fout << src_aligned_gt->points[src_id].z << "," << tgt->points[tgt_id].z << "\n";
    }
    fout.close();
}

void saveExtractedPointIds(const PointNCloud::Ptr &src_fullsize, const PointNCloud::Ptr &tgt_fullsize,
                           const Eigen::Matrix4f &transformation_gt,
                           const AlignmentParameters &parameters, const std::string &extracted_path) {
    PointNCloud::Ptr extracted_points(new PointNCloud);
    if (pcl::io::loadPLYFile<PointN>(extracted_path, *extracted_points) < 0) {
        pcl::console::print_error("Error loading file with extracted point!\n");
        exit(1);
    }
    saveExtractedPointIds(src_fullsize, tgt_fullsize, transformation_gt, parameters, extracted_points);
}