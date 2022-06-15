#include "keypoints.h"

std::vector<float> ISSKeypoint3DDebug::getBoundaryPointsDebug() {
    int n = this->input_->size();
    std::vector<float> weights;
    std::transform(this->edge_points_, this->edge_points_ + n, std::back_inserter(weights),
                   [](bool is_edge) { return is_edge ? 1.f : 0.f; });
    return weights;
}

std::vector<float> ISSKeypoint3DDebug::getThirdEigenValuesDebug() {
    int n = this->input_->size();
    std::vector<float> weights;
    std::copy(this->third_eigen_value_, this->third_eigen_value_ + n, std::back_inserter(weights));
    return weights;
}

std::vector<float> HarrisKeypoint3DDebug::getResponseHarrisDebug() {
    pcl::PointCloud<pcl::PointXYZI> output;
    this->responseHarris(output);
    std::vector<float> weights;
    std::transform(output.begin(), output.end(), std::back_inserter(weights),
                   [](const pcl::PointXYZI &point) { return point.intensity; });
    return weights;
}