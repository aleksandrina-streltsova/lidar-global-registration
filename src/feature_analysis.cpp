#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

#include "common.h"

typedef pcl::PointXYZLNormal PointTN;
typedef pcl::PointCloud<PointTN> PointCloudTN;

void saveHistograms(const FeatureCloudT::Ptr &features, const std::string &testname, bool is_source) {
    std::string filepath = constructPath(testname,  std::string("histograms_")+ (is_source ? "src" : "tgt"), "csv");
    std::fstream fout(filepath, std::ios_base::out);
    int n = features->size(), m = features->points[0].descriptorSize();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fout << features->points[i].histogram[j] << ",";
        }
        fout << "\n";
    }
    fout.close();
}

void saveFeatures(float radius_search, const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                  const std::string &testname, bool is_source) {
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
