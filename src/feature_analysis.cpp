#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

#include "feature_analysis.h"

typedef pcl::PointXYZLNormal PointTN;
typedef pcl::PointCloud<PointTN> PointCloudTN;

void saveHistograms(const FeatureCloudT::Ptr &features, const std::string &testname, bool is_source) {
    pcl::console::print_highlight("Saving %s histograms...\n", is_source ? "source" : "target");
    std::string filepath = constructPath(testname,  std::string("histograms_")+ (is_source ? "src" : "tgt"), "csv");
    std::fstream fout(filepath, std::ios_base::out);
    int n = features->size(), m = features->points[0].descriptorSize();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fout << features->points[i].histogram[j];
            if (j != m - 1) {
                fout << ",";
            }
        }
        fout << "\n";
    }
    fout.close();
}

void saveFeatures(float radius_search, const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                  const std::string &testname, bool is_source) {
    pcl::console::print_highlight("Saving %s features...\n", is_source ? "source" : "target");
    std::string filepath = constructPath(testname,  std::string("features_")+ (is_source ? "src" : "tgt"), "csv");
    std::fstream fout(filepath, std::ios_base::out);
    FeatureEstimationT fest;
    fest.setRadiusSearch(radius_search);
    fest.setInputCloud(pcd);
    fest.setInputNormals(normals);
    fout << "f1,f2,f3,f4\n";
    int n = pcd->size();
    for (int p_idx = 0; p_idx < n; ++p_idx) {
        for (int q_idx = 0; q_idx < p_idx; ++q_idx) {
            float f1, f2, f3, f4;
            fest.computePairFeatures(*pcd, *normals, p_idx, q_idx, f1, f2, f3, f4);
            fout << f1 << "," << f2 << "," << f3 << "," << f4 << "\n";
        }
    }
    fout.close();
}

void saveNormals(const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                 const Eigen::Matrix4f &transformation_gt, bool is_source, const std::string &testname) {
    pcl::console::print_highlight("Saving %s normals...\n", is_source ? "source" : "target");
    PointCloudTN pcd_with_normals;
    int n = pcd->size();
    for (int i = 0; i < n; ++i) {
        auto p = pcd->points[i];
        auto n = normals->points[i];
        PointTN pn;
        pn.x = p.x;
        pn.y = p.y;
        pn.z = p.z;
        pn.normal_x = n.normal_x;
        pn.normal_y = n.normal_y;
        pn.normal_z = n.normal_z;
        pcd_with_normals.points.push_back(pn);
    }
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