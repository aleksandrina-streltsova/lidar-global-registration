#include <pcl/features/normal_3d_omp.h>

#include "align.h"
#include "csv_parser.h"
#include "filter.h"

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename) {
    std::ifstream file(csv_path);
    Eigen::Matrix4f src_position, tgt_position;

    CSVRow row;
    while (file >> row) {
        if (row[0] == src_filename) {
            for (int i = 0; i < 16; ++i) {
                src_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
        }
        if (row[0] == tgt_filename) {
            for (int i = 0; i < 16; ++i) {
                tgt_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
        }
    }
    return tgt_position.inverse() * src_position;
}

void estimateNormals(float radius_search, const PointCloudT::Ptr &pcd, PointCloudN::Ptr &normals) {
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_est;
    normal_est.setRadiusSearch(radius_search);

    normal_est.setInputCloud(pcd);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    normal_est.setSearchMethod(tree);
    normal_est.compute(*normals);
    int nan_counter = 0;
    for (const auto &normal: normals->points) {
        const Eigen::Vector4f &normal_vec = normal.getNormalVector4fMap();
        if (!std::isfinite(normal_vec[0]) ||
            !std::isfinite(normal_vec[1]) ||
            !std::isfinite(normal_vec[2])) {
            nan_counter++;
        }
    }
    PCL_DEBUG("[estimateNormals] %d NaN normals.\n", nan_counter);
}
