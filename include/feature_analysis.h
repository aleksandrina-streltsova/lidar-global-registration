#ifndef REGISTRATION_FEATURE_ANALYSIS_H
#define REGISTRATION_FEATURE_ANALYSIS_H

#include <Eigen/Core>
#include <pcl/point_cloud.h>

#include "common.h"

template <typename FeatureT>
void saveFeatures(const typename pcl::PointCloud<FeatureT>::Ptr &features,  const AlignmentParameters &parameters, bool is_source) {
    pcl::console::print_highlight("Saving %s histograms...\n", is_source ? "source" : "target");
    std::string filepath = constructPath(parameters,  std::string("histograms_")+ (is_source ? "src" : "tgt"), "csv");
    std::fstream fout(filepath, std::ios_base::out);
    int n = features->size(), m = features->points[0].descriptorSize();
    for (int i = 0; i < n; ++i) {
        auto *data = reinterpret_cast<const float *> (&features->points[i]);
        for (int j = 0; j < m; ++j) {
            fout << data[j];
            if (j != m - 1) {
                fout << ",";
            }
        }
        fout << "\n";
    }
    fout.close();
}

void saveNormals(const PointNCloud::Ptr &pcd,
                 const Eigen::Matrix4f &transformation_gt, bool is_source, const AlignmentParameters &parameters);

void saveExtractedPointIds(const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                           const Eigen::Matrix4f &transformation_gt,
                           const AlignmentParameters &parameters, const PointNCloud::Ptr &extracted_points);

void saveExtractedPointIds(const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                           const Eigen::Matrix4f &transformation_gt,
                           const AlignmentParameters &parameters, const std::string &extracted_path);
#endif
