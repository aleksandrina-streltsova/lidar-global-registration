#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "feature_analysis.h"
#include "downsample.h"

void saveNormals(const PointCloudTN::Ptr &pcd,
                 const Eigen::Matrix4f &transformation_gt, bool is_source,  const AlignmentParameters &parameters) {
    pcl::console::print_highlight("Saving %s normals...\n", is_source ? "source" : "target");
    PointCloudTN pcd_aligned;
    pcl::transformPointCloudWithNormals(*pcd, pcd_aligned, transformation_gt);
    std::string filepath = constructPath(parameters,  std::string("normals_")+ (is_source ? "src" : "tgt"));
    pcl::io::savePLYFileBinary(filepath, pcd_aligned);
}

std::vector<int> getPointIds(const PointCloudT::Ptr &all_points, const PointCloudT::Ptr &extracted_points) {
    int n = extracted_points->size();
    pcl::KdTreeFLANN<PointT> tree(new pcl::KdTreeFLANN<PointT>);
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

void saveExtractedPointIds(const PointCloudT::Ptr &src_fullsize, const PointCloudT::Ptr &tgt_fullsize,
                           const Eigen::Matrix4f &transformation_gt,
                           const AlignmentParameters &parameters, const PointCloudT::Ptr &extracted_points) {
    PointCloudT::Ptr src(new PointCloudT), tgt(new PointCloudT);
    // Downsample
    if (parameters.downsample) {
        pcl::console::print_highlight("Downsampling...\n");
        downsamplePointCloud(src_fullsize, src, parameters.voxel_size);
        downsamplePointCloud(tgt_fullsize, tgt, parameters.voxel_size);
    }

    PointCloudT::Ptr src_aligned_gt(new PointCloudT);

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

void saveExtractedPointIds(const PointCloudT::Ptr &src_fullsize, const PointCloudT::Ptr &tgt_fullsize,
                           const Eigen::Matrix4f &transformation_gt,
                           const AlignmentParameters &parameters, const std::string &extracted_path) {
    PointCloudT::Ptr extracted_points(new PointCloudT);
    if (pcl::io::loadPLYFile<PointT>(extracted_path, *extracted_points) < 0) {
        pcl::console::print_error("Error loading file with extracted point!\n");
        exit(1);
    }
    saveExtractedPointIds(src_fullsize, tgt_fullsize, transformation_gt, parameters, extracted_points);
}