#ifndef REGISTRATION_DOWNSAMPLE_H
#define REGISTRATION_DOWNSAMPLE_H

#include "common.h"

class AccumulatedPoint {
public:
    void AddPoint(const PointNCloud::Ptr &pcd, int index) {
        point_.x += pcd->points[index].x;
        point_.y += pcd->points[index].y;
        point_.z += pcd->points[index].z;
        point_.normal_x += pcd->points[index].normal_x;
        point_.normal_y += pcd->points[index].normal_y;
        point_.normal_z += pcd->points[index].normal_z;
        num_of_points_++;
    }

    PointN GetAveragePoint() const {
        pcl::Normal n(point_.normal_x / num_of_points_, point_.normal_y / num_of_points_,
                           point_.normal_z / num_of_points_);
        float norm = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y + n.normal_z * n.normal_z);
        norm = norm < 1e-5 ? 1.f : norm;
        return PointN(point_.x / num_of_points_, point_.y / num_of_points_, point_.z / num_of_points_,
                       n.normal_x / norm, n.normal_y / norm, n.normal_z / norm);
    }

private:
    float num_of_points_ = 0;
    PointN point_ = PointN(0, 0, 0, 0, 0, 0);
};

void downsamplePointCloud(const PointNCloud::Ptr &pcd_fullsize, PointNCloud::Ptr &pcd_down,
                          const AlignmentParameters &parameters);

#endif
