#ifndef REGISTRATION_KEYPOINTS_H
#define REGISTRATION_KEYPOINTS_H

#include <vector>

#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>

#include "common.h"

class ISSKeypoint3DDebug : public pcl::ISSKeypoint3D<PointN, PointN, PointN> {
public:
    std::vector<float> getBoundaryPointsDebug();

    std::vector<float> getThirdEigenValuesDebug();

    inline double getBorderRadius() {
        return this->border_radius_;
    }
};

class HarrisKeypoint3DDebug : public pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> {
public:
    std::vector<float> getResponseHarrisDebug();
};

#endif