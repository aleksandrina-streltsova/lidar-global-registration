#include <fstream>

#include <pcl/features/principal_curvatures.h>

#include <weights.h>

std::vector<float> computeWeightConstant(float radius_search, const PointNCloud::ConstPtr &pcd, const NormalCloud::ConstPtr &normals);

std::vector<float> computeWeightExponential(float radius_search, const PointNCloud::ConstPtr &pcd, const NormalCloud::ConstPtr &normals);

WeightFunction getWeightFunction(const std::string &identifier) {
    if (identifier == METRIC_WEIGHT_EXPONENTIAL)
        return computeWeightExponential;
    else if (identifier != METRIC_WEIGHT_CONSTANT)
        PCL_WARN("[getWeightFunction] weight function %s isn't supported, constant weights will be used\n",
                 identifier.c_str());
    return computeWeightConstant;
}

std::vector<float> computeWeightConstant(float radius_search, const PointNCloud::ConstPtr &pcd, const NormalCloud::ConstPtr &normals) {
    return std::vector<float>(pcd->size(), 1.0);
}

std::vector<float> computeWeightExponential(float radius_search, const PointNCloud::ConstPtr &pcd, const NormalCloud::ConstPtr &normals) {
    pcl::PrincipalCurvaturesEstimation<PointN, pcl::Normal, pcl::PrincipalCurvatures> estimation;
    pcl::PointCloud<pcl::PrincipalCurvatures> pcs;
    estimation.setRadiusSearch(radius_search);
    estimation.setInputCloud(pcd);
    estimation.setInputNormals(normals);
    estimation.compute(pcs);

    std::vector<float> max_pcs(pcs.size());
    std::vector<float> weights(pcs.size());
    for (int i = 0; i < pcd->size(); ++i) {
        bool is_finite = std::isfinite(pcs[i].pc1) && std::isfinite(pcs[i].pc2);
        max_pcs[i] = is_finite ? std::max(pcs[i].pc1, pcs[i].pc2) : std::numeric_limits<float>::max();
    }
    float lambda = logf(2.f) * quantile(0.8, max_pcs);
    for (int i = 0; i < pcd->size(); ++i) {
        weights[i] = std::exp(-lambda / max_pcs[i]);
    }
//    std::string filepath = "curvatures.csv";
//    std::ofstream fout(filepath);
//    fout << "normal_x,normal_y,normal_z,k1,k2\n";
//    if (fout.is_open()) {
//        for (int i = 0; i < curvatures.size(); ++i) {
//            fout << normals->points[i].normal_x << ",";
//            fout << normals->points[i].normal_y << ",";
//            fout << normals->points[i].normal_z << ",";
//            fout << curvatures[i].pc1 << "," << curvatures[i].pc2 << "\n";
//        }
//    } else {
//        perror(("error while opening file " + filepath).c_str());
//    }
    return weights;
}