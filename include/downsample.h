#ifndef REGISTRATION_DOWNSAMPLE_H
#define REGISTRATION_DOWNSAMPLE_H

#include "common.h"

class AccumulatedPoint {
public:
    void AddPoint(const PointNCloud::ConstPtr &pcd, int index) {
        float weight = pcd->points[index].intensity;
        point_.x += weight * pcd->points[index].x;
        point_.y += weight * pcd->points[index].y;
        point_.z += weight * pcd->points[index].z;
        point_.intensity += weight;
        point_.normal_x += weight * pcd->points[index].normal_x;
        point_.normal_y += weight * pcd->points[index].normal_y;
        point_.normal_z += weight * pcd->points[index].normal_z;
    }

    PointN GetAveragePoint() const {
        float weight = point_.intensity;
        pcl::Normal n(point_.normal_x / weight, point_.normal_y / weight,point_.normal_z / weight);
        float norm = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y + n.normal_z * n.normal_z);
        norm = norm < 1e-5 ? 1.f : norm;
        return PointN(point_.x / weight, point_.y / weight, point_.z / weight, weight,
                      n.normal_x / norm, n.normal_y / norm, n.normal_z / norm);
    }

private:
    PointN point_ = PointN(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
};

void downsamplePointCloud(const PointNCloud::ConstPtr &pcd_fullsize, PointNCloud::Ptr &pcd_down, float voxel_size);

#endif
