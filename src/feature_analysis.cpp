#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>

#include "feature_analysis.h"

typedef pcl::PointXYZLNormal PointTN;
typedef pcl::PointCloud<PointTN> PointCloudTN;

void saveNormals(const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                 const Eigen::Matrix4f &transformation_gt, bool is_source, const std::string &testname) {
    pcl::console::print_highlight("Saving %s normals...\n", is_source ? "source" : "target");
    PointCloudTN pcd_with_normals;
    pcl::concatenateFields(*pcd, *normals, pcd_with_normals);
    pcl::transformPointCloudWithNormals(pcd_with_normals, pcd_with_normals, transformation_gt);
    std::string filepath = constructPath(testname,  std::string("normals_")+ (is_source ? "src" : "tgt"));
    pcl::io::savePLYFileBinary(filepath, pcd_with_normals);
}

std::vector<int> getPointIds(const PointCloudT::Ptr &all_points, const PointCloudT::Ptr &extracted_points) {
    int n = extracted_points->size();
    pcl::KdTreeFLANN<PointT> tree(new pcl::KdTreeFLANN<PointT>);
    tree.setInputCloud(all_points);

    std::vector<int> ids(n);
    std::vector<int> indices(1);
    std::vector<float> distances(1);

    for (int i = 0; i < n; ++i) {
        tree.nearestKSearch(*extracted_points, i,1,indices,distances);
        ids[i] = indices[0];
    }
    return ids;
}

void saveExtractedPointIds(const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                           const Eigen::Matrix4f &transformation_gt,
                           const std::string &testname,  const YamlConfig &config) {
    PointCloudT::Ptr src_aligned_gt(new PointCloudT), extracted_points(new PointCloudT);
    std::string extracted_path = config.get<std::string>("extracted").value();

    if (pcl::io::loadPLYFile<PointT>(extracted_path, *extracted_points) < 0) {
        pcl::console::print_error("Error loading file with extracted point!\n");
        exit(1);
    }

    pcl::transformPointCloud(*src, *src_aligned_gt, transformation_gt);
    std::string filepath = constructPath(testname,  "ids", "csv");
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