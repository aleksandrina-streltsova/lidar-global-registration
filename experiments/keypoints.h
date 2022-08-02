#ifndef REGISTRATION_KEYPOINTS_H
#define REGISTRATION_KEYPOINTS_H

#include <vector>

#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>

#include "common.h"

class ISSKeypoint3DDebug : public pcl::ISSKeypoint3D<PointN, PointN, PointN> {
public:
    bool initCompute() override;

    inline void setMaxNeighbors(int max_neighbors) {
        max_neighbors_ = max_neighbors;
    }

protected:
    int max_neighbors_ = 0;
};

#endif