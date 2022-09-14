#ifndef REGISTRATION_SHOT_DEBUG_H
#define REGISTRATION_SHOT_DEBUG_H

#include <array>
#include <vector>
#include <pcl/features/shot_omp.h>

class SHOTEstimationDebug : public pcl::SHOTEstimationOMP<pcl::PointXYZINormal, pcl::PointXYZINormal, pcl::SHOT352> {
public:
    void setInputCloud(const PointCloudConstPtr &cloud) override;

    inline std::vector<std::array<int, 32>> getVolumesDebugInfo() {
        return volumes_;
    }

    void computePointSHOT(const int index,
                          const pcl::Indices &indices,
                          const std::vector<float> &sqr_dists,
                          Eigen::VectorXf &shot) override;

protected:
    void interpolateSingleChannelDebug(const pcl::Indices &indices,
                                       const std::vector<float> &sqr_dists,
                                       const int index,
                                       std::vector<double> &binDistance,
                                       const int nr_bins,
                                       Eigen::VectorXf &shot);

    std::vector<std::array<int, 32>> volumes_;
};

#endif
