#include <Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/common/norms.h>

#include "analysis.h"

std::pair<float, float> calculate_rotation_and_translation_errors(const Eigen::Matrix4f &transformation,
                                                                  const Eigen::Matrix4f &transformation_gt) {
    Eigen::Matrix3f rotation_diff = transformation.block<3, 3>(0, 0).inverse() * transformation_gt.block<3, 3>(0, 0);
    Eigen::Vector3f translation_diff = transformation.block<3, 1>(0, 3) - transformation_gt.block<3, 1>(0, 3);
    float rotation_error = Eigen::AngleAxisf(rotation_diff).angle();
    float translation_error = translation_diff.norm();
    return {rotation_error, translation_error};
}

float calculate_point_cloud_mean_error(const PointCloudT::Ptr &pcd,
                                       const Eigen::Matrix4f &transformation, const Eigen::Matrix4f &transformation_gt) {
    PointCloudT::Ptr pcd_transformed(new PointCloudT);
    Eigen::Matrix4f transformation_diff = transformation.inverse() * transformation_gt;
    pcl::transformPointCloud(*pcd, *pcd_transformed, transformation_diff);

    float error = 0.f;
    for (int i = 0; i < pcd->size(); ++i) {
        error += pcl::L2_Norm(pcd->points[i].data, pcd_transformed->points[i].data, 3);
    }
    error /= pcd->size();
    return error;
}
