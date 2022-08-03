#include <fstream>

#include <pcl/features/principal_curvatures.h>
#include <pcl/keypoints/harris_3d.h>

#include <weights.h>

#define NS_BIN_SIZE 8

std::vector<float> computeWeightConstant(int nr_points, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightExponential(int nr_points, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightLogCurvedness(int nr_points, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightHarris(int nr_points, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightTomasi(int nr_points, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightCurvature(int nr_points, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightNSS(int nr_points, const PointNCloud::ConstPtr &pcd);

WeightFunction getWeightFunction(const std::string &identifier) {
    if (identifier == METRIC_WEIGHT_EXP_CURVATURE)
        return computeWeightExponential;
    else if (identifier == METRIC_WEIGHT_CURVEDNESS)
        return computeWeightLogCurvedness;
    else if (identifier == METRIC_WEIGHT_HARRIS)
        return computeWeightHarris;
    else if (identifier == METRIC_WEIGHT_TOMASI)
        return computeWeightTomasi;
    else if (identifier == METRIC_WEIGHT_CURVATURE)
        return computeWeightCurvature;
    else if (identifier == METRIC_WEIGHT_NSS)
        return computeWeightNSS;
    else if (identifier != METRIC_WEIGHT_CONSTANT)
        PCL_WARN("[getWeightFunction] weight function %s isn't supported, constant weights will be used\n",
                 identifier.c_str());
    return computeWeightConstant;
}

std::vector<float> computeWeightConstant(int nr_points, const PointNCloud::ConstPtr &pcd) {
    return std::vector<float>(pcd->size(), 1.0);
}

void estimateCurvatures(int nr_points, const PointNCloud::ConstPtr &pcd,
                        const pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr &pcs) {
    pcl::PrincipalCurvaturesEstimation<PointN, PointN, pcl::PrincipalCurvatures> estimation;
    estimation.setKSearch(nr_points);
    estimation.setInputCloud(pcd);
    estimation.setInputNormals(pcd);
    estimation.compute(*pcs);
}

std::vector<float> computeWeightExponential(int nr_points, const PointNCloud::ConstPtr &pcd) {
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr pcs(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    estimateCurvatures(nr_points, pcd, pcs);

    std::vector<float> max_pcs(pcs->size());
    std::vector<float> weights(pcs->size());
    for (int i = 0; i < pcd->size(); ++i) {
        bool is_finite = std::isfinite(pcs->points[i].pc1) && std::isfinite(pcs->points[i].pc2);
        max_pcs[i] = is_finite ? std::max(pcs->points[i].pc1, pcs->points[i].pc2) : 0.f;
    }
    float q = quantile(0.8, max_pcs);
    float lambda = logf(1.05f) * q;
    for (int i = 0; i < pcd->size(); ++i) {
        if (max_pcs[i] == 0.f) {
            weights[i] = 0.f;
        } else {
            weights[i] = std::exp(-lambda / max_pcs[i]);
        }
    }
    return weights;
}

std::vector<float> computeWeightLogCurvedness(int nr_points, const PointNCloud::ConstPtr &pcd) {
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr pcs(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    estimateCurvatures(nr_points, pcd, pcs);

    std::vector<float> weights(pcs->size());
    for (int i = 0; i < pcd->size(); ++i) {
        float pc1 = pcs->points[i].pc1, pc2 = pcs->points[i].pc2;
        bool is_finite = std::isfinite(pc1) && std::isfinite(pc2);
        weights[i] = is_finite ? std::log(std::sqrt((pc1 * pc1 + pc2 * pc2) / 2.f) + 1.f) : 0.f;
    }
    return weights;
}

void initializeHarrisKeypoint3D(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> &harris,
                                int nr_points, const PointNCloud::ConstPtr &pcd) {
    harris.setKSearch(nr_points);
    harris.setThreshold(0.f);
    harris.setInputCloud(pcd);
    harris.setNormals(pcd);
    harris.setNonMaxSupression(false);
}

std::vector<float> computeWeightHarris(int nr_points, const PointNCloud::ConstPtr &pcd) {
    pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> harris;
    pcl::PointCloud<pcl::PointXYZI> keypoints;

    initializeHarrisKeypoint3D(harris, nr_points, pcd);
    harris.setMethod(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN>::HARRIS);
    harris.compute(keypoints);

    std::vector<float> weights(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        weights[i] = keypoints[i].intensity;
    }
    return weights;
}

std::vector<float> computeWeightTomasi(int nr_points, const PointNCloud::ConstPtr &pcd) {
    pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> harris;
    pcl::PointCloud<pcl::PointXYZI> keypoints;

    initializeHarrisKeypoint3D(harris, nr_points, pcd);
    harris.setMethod(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN>::TOMASI);
    harris.compute(keypoints);

    std::vector<float> weights(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        weights[i] = keypoints[i].intensity;
    }
    return weights;
}

std::vector<float> computeWeightCurvature(int nr_points, const PointNCloud::ConstPtr &pcd) {
    pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> harris;
    pcl::PointCloud<pcl::PointXYZI> keypoints;

    initializeHarrisKeypoint3D(harris, nr_points, pcd);
    harris.setMethod(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN>::CURVATURE);
    harris.compute(keypoints);

    std::vector<float> weights(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        weights[i] = std::isfinite(keypoints[i].intensity) ? keypoints[i].intensity : 0.f;
    }
    return weights;
}

int findBin(const PointN &normal) {
    // polar angle in [0, pi]
    float theta = std::acos(normal.normal_z);
    // azimuthal angle in [0, 2pi]
    float phi = std::fmod(std::atan2(normal.normal_y, normal.normal_x) + 2.f * M_PI, 2.f * M_PI);

    theta = std::min(std::max(theta, 0.f), (float) M_PI);
    phi = std::min(std::max(phi, 0.f), (float) (2.f * M_PI));
    if (theta == M_PI) theta = 0.f;
    if (phi == 2.f * M_PI) phi = 0.f;

    return (int) (std::floor(theta * NS_BIN_SIZE) * NS_BIN_SIZE + std::floor(phi * NS_BIN_SIZE));
}

// weights based on Normal Space Sampling
std::vector<float> computeWeightNSS(int, const PointNCloud::ConstPtr &pcd) {
    std::vector<int> hist(NS_BIN_SIZE * NS_BIN_SIZE);
    for (const auto point: pcd->points) {
        bool is_finite =
                std::isfinite(point.normal_x) && std::isfinite(point.normal_y) && std::isfinite(point.normal_z);
        if (!is_finite) continue;
        int bin = findBin(point);
        hist[bin] += 1;
    }

    std::vector<float> weights(pcd->size(), 0.f);
    for (int i = 0; i < pcd->size(); ++i) {
        const auto &point = pcd->points[i];
        bool is_finite =
                std::isfinite(point.normal_x) && std::isfinite(point.normal_y) && std::isfinite(point.normal_z);
        if (!is_finite) continue;
        int bin = findBin(point);
        weights[i] = 1.f / (float) hist[bin] / (float) (NS_BIN_SIZE * NS_BIN_SIZE);
    }
    return weights;
}