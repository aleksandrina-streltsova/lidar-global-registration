#include <fstream>

#include <pcl/features/principal_curvatures.h>
#include <pcl/keypoints/harris_3d.h>

#include <weights.h>

std::vector<float> computeWeightConstant(float curvature_radius, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightExponential(float curvature_radius, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightLogCurvedness(float curvature_radius, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightHarris(float curvature_radius, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightTomasi(float curvature_radius, const PointNCloud::ConstPtr &pcd);

std::vector<float> computeWeightCurvature(float curvature_radius, const PointNCloud::ConstPtr &pcd);

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
    else if (identifier != METRIC_WEIGHT_CONSTANT)
        PCL_WARN("[getWeightFunction] weight function %s isn't supported, constant weights will be used\n",
                 identifier.c_str());
    return computeWeightConstant;
}

std::vector<float> computeWeightConstant(float curvature_radius, const PointNCloud::ConstPtr &pcd) {
    return std::vector<float>(pcd->size(), 1.0);
}

void estimateCurvatures(float curvature_radius, const PointNCloud::ConstPtr &pcd,
                        const pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr &pcs) {
    pcl::PrincipalCurvaturesEstimation<PointN, PointN, pcl::PrincipalCurvatures> estimation;
    estimation.setRadiusSearch(curvature_radius);
    estimation.setInputCloud(pcd);
    estimation.setInputNormals(pcd);
    estimation.compute(*pcs);
}

std::vector<float> computeWeightExponential(float curvature_radius, const PointNCloud::ConstPtr &pcd) {
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr pcs(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    estimateCurvatures(curvature_radius, pcd, pcs);

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

std::vector<float> computeWeightLogCurvedness(float curvature_radius, const PointNCloud::ConstPtr &pcd) {
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr pcs(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    estimateCurvatures(curvature_radius, pcd, pcs);

    std::vector<float> weights(pcs->size());
    for (int i = 0; i < pcd->size(); ++i) {
        float pc1 = pcs->points[i].pc1, pc2 = pcs->points[i].pc2;
        bool is_finite = std::isfinite(pc1) && std::isfinite(pc2);
        weights[i] = is_finite ? std::log(std::sqrt((pc1 * pc1 + pc2 * pc2) / 2.f) + 1.f) : 0.f;
    }
    return weights;
}

void initializeHarrisKeypoint3D(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> &harris,
                                float curvature_radius, const PointNCloud::ConstPtr &pcd) {
    harris.setRadiusSearch(curvature_radius);
    harris.setThreshold(0.f);
    harris.setInputCloud(pcd);
    harris.setNormals(pcd);
    harris.setNonMaxSupression(false);
}

std::vector<float> computeWeightHarris(float curvature_radius, const PointNCloud::ConstPtr &pcd) {
    pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> harris;
    pcl::PointCloud<pcl::PointXYZI> keypoints;

    initializeHarrisKeypoint3D(harris, curvature_radius, pcd);
    harris.setMethod(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN>::HARRIS);
    harris.compute(keypoints);

    std::vector<float> weights(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        weights[i] = keypoints[i].intensity;
    }
    return weights;
}

std::vector<float> computeWeightTomasi(float curvature_radius, const PointNCloud::ConstPtr &pcd) {
    pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> harris;
    pcl::PointCloud<pcl::PointXYZI> keypoints;

    initializeHarrisKeypoint3D(harris, curvature_radius, pcd);
    harris.setMethod(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN>::TOMASI);
    harris.compute(keypoints);

    std::vector<float> weights(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        weights[i] = keypoints[i].intensity;
    }
    return weights;
}

std::vector<float> computeWeightCurvature(float curvature_radius, const PointNCloud::ConstPtr &pcd) {
    pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN> harris;
    pcl::PointCloud<pcl::PointXYZI> keypoints;

    initializeHarrisKeypoint3D(harris, curvature_radius, pcd);
    harris.setMethod(pcl::HarrisKeypoint3D<PointN, pcl::PointXYZI, PointN>::CURVATURE);
    harris.compute(keypoints);

    std::vector<float> weights(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        weights[i] = std::isfinite(keypoints[i].intensity) ? keypoints[i].intensity : 0.f;
    }
    return weights;
}