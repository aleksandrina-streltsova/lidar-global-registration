#ifndef REGISTRATION_FEATURE_ANALYSIS_H
#define REGISTRATION_FEATURE_ANALYSIS_H

#include <Eigen/Core>
#include <pcl/point_cloud.h>

#include "common.h"
#include "config.h"

template <typename FeatureT>
void saveHistograms(const typename pcl::PointCloud<FeatureT>::Ptr &features, const std::string &testname, bool is_source) {
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

template <typename FeatureT>
void saveDescriptors(const typename pcl::PointCloud<FeatureT>::Ptr &features, const std::string &testname, bool is_source) {
    pcl::console::print_highlight("Saving %s descriptors...\n", is_source ? "source" : "target");
    std::string filepath = constructPath(testname,  std::string("descriptors_")+ (is_source ? "src" : "tgt"), "csv");
    std::fstream fout(filepath, std::ios_base::out);
    int n = features->size(), m = features->points[0].descriptorSize();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fout << features->points[i].descriptor[j];
            if (j != m - 1) {
                fout << ",";
            }
        }
        fout << "\n";
    }
    fout.close();
}

void saveNormals(const PointCloudT::Ptr &pcd, const PointCloudN::Ptr &normals,
                 const Eigen::Matrix4f &transformation_gt, bool is_source, const std::string &testname);

void saveExtractedPointIds(const PointCloudT::Ptr &src, const PointCloudT::Ptr &tgt,
                           const Eigen::Matrix4f &transformation_gt,
                           const std::string &testname,  const YamlConfig &config);
#endif
