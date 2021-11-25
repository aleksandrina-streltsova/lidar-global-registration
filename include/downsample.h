#ifndef REGISTRATION_DOWNSAMPLE_H
#define REGISTRATION_DOWNSAMPLE_H

#include "common.h"

class AccumulatedPoint {
public:
    void AddPoint(const PointCloudT::Ptr &pcd, int index) {
        point_.x += pcd->points[index].x;
        point_.y += pcd->points[index].y;
        point_.z += pcd->points[index].z;
        num_of_points_++;
    }

    PointT GetAveragePoint() const {
        return PointT(point_.x / num_of_points_, point_.y / num_of_points_, point_.z / num_of_points_);
    }

private:
    float num_of_points_ = 0;
    PointT point_ = PointT(0, 0, 0);
};

void downsamplePointCloud(const PointCloudT::Ptr &pcd_fullsize, PointCloudT::Ptr &pcd_down, float voxel_size);

#endif
