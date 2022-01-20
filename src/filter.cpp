#include <algorithm>
#include <fstream>
#include <unordered_set>

#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>

#include "filter.h"
#include "utils.h"
#include "common.h"

std::vector<float> computeDistanceNN(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features);

std::vector<float> computeDistanceMean(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features);

std::vector<float> computeDistanceRandom(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features);

void saveUniquenesses(const PointCloudT::Ptr &pcd, const std::vector<float> &uniquenesses, const pcl::Indices &indices,
                      const AlignmentParameters &parameters, const std::string &func_identifier,
                      bool is_source, const Eigen::Matrix4f &transformation_gt);

UniquenessFunction getUniquenessFunction(const std::string &identifier) {
    if (identifier == "nn")
        return computeDistanceNN;
    if (identifier == "mean")
        return computeDistanceMean;
    if (identifier == "random")
        return computeDistanceRandom;
    return nullptr;
}

void filterPointCloud(UniquenessFunction func, const std::string &func_identifier,
                      const PointCloudT::Ptr &pcd, const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features,
                      PointCloudT::Ptr &dst_pcd, pcl::PointCloud<pcl::FPFHSignature33>::Ptr &dst_features,
                      const Eigen::Matrix4f &transformation_gt,
                      const AlignmentParameters &parameters, bool is_source) {
    std::vector<float> uniquenesses = func(features);
    float threshold = UNIQUENESS_THRESHOLD;
    pcl::Indices indices;
    indices.reserve(pcd->size());
    for (int i = 0; i < uniquenesses.size(); ++i) {
        if (uniquenesses[i] > threshold) {
            indices.push_back(i);
        }
    }
    pcl::PointIndices::Ptr point_indices(new pcl::PointIndices);
    point_indices->indices = indices;

    saveUniquenesses(pcd, uniquenesses, indices, parameters, func_identifier, is_source, transformation_gt);

    // Filter point cloud
    pcl::ExtractIndices<PointT> pcd_filter;
    pcd_filter.setInputCloud(pcd);
    pcd_filter.setIndices(point_indices);
    pcd_filter.setNegative(false);
    pcd_filter.filter(*dst_pcd);

    // Filter features
    pcl::ExtractIndices<pcl::FPFHSignature33> features_filter;
    features_filter.setInputCloud(features);
    features_filter.setIndices(point_indices);
    features_filter.setNegative(false);
    features_filter.filter(*dst_features);
}

std::vector<float> computeDistanceNN(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features) {
    std::vector<float> distances(features->size());
    pcl::KdTreeFLANN<pcl::FPFHSignature33> feature_tree(new pcl::KdTreeFLANN<pcl::FPFHSignature33>);
    feature_tree.setInputCloud(features);
    for (int i = 0; i < features->size(); ++i) {
        std::vector<int> match_indices(1);
        std::vector<float> match_distances(1);
        feature_tree.nearestKSearch(*features, i, 2, match_indices, match_distances);
        distances[i] = match_distances[1];
    }
    return distances;
}

std::vector<float> computeDistanceMean(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features) {
    std::string filepath = "distances.csv";
    std::fstream fout(filepath, std::ios_base::out);

    int feature_dim = features->points[0].descriptorSize();
    int n_features = (int) features->size();
    Eigen::MatrixXf fs(n_features, feature_dim);
    for (int i = 0; i < n_features; ++i) {
        fs.row(i) = Eigen::Map<Eigen::VectorXf>(features->points[i].histogram, feature_dim);
    }

    auto m = fs.colwise().mean();

    std::vector<float> distances(features->size());
    for (int i = 0; i < n_features; ++i) {
        auto f = fs.row(i);
        distances[i] = (f - m).norm();
    }
    return distances;
}

std::vector<float> computeDistanceRandom(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &features) {
    int feature_dim = features->points[0].descriptorSize();

    std::vector<Eigen::VectorXf> fs(features->size());
    for (int i = 0; i < fs.size(); ++i) {
        fs[i] = Eigen::Map<Eigen::VectorXf>(features->points[i].histogram, feature_dim);
    }

    std::vector<float> distances(features->size());

#pragma omp parallel default(none) shared(distances, fs)
    {
        UniformRandIntGenerator rand_generator(0, (int) distances.size() - 1);
#pragma omp for
        for (int i = 0; i < distances.size(); ++i) {
            float min_dist = std::numeric_limits<float>::max();
            for (int j = 0; j < N_RANDOM_FEATURES; ++j) {
                int idx = rand_generator();
                if (i == idx) {
                    continue;
                }
                min_dist = std::min(min_dist, (fs[i] - fs[idx]).norm());
            }
            distances[i] = min_dist;
        }
    }
    return distances;
}

void saveUniquenesses(const PointCloudT::Ptr &pcd, const std::vector<float> &uniquenesses, const pcl::Indices &indices,
                      const AlignmentParameters &parameters, const std::string &func_identifier,
                      bool is_source, const Eigen::Matrix4f &transformation_gt) {
    std::string name = std::string("uniquenesses_") + (is_source ? "src_" : "tgt_") + func_identifier;
    std::string filepath = constructPath(parameters, name, "csv");
    std::fstream fout(filepath, std::ios_base::out);
    if (!fout.is_open())
        perror(("error while opening file " + filepath).c_str());

    fout << "uniqueness\n";
    for (const auto &uniqueness: uniquenesses) {
        fout << uniqueness << "\n";
    }
    fout.close();

    PointCloudColoredT dst;
    dst.resize(pcd->size());
    for (int i = 0; i < pcd->size(); ++i) {
        dst.points[i].x = pcd->points[i].x;
        dst.points[i].y = pcd->points[i].y;
        dst.points[i].z = pcd->points[i].z;
        setPointColor(dst.points[i], COLOR_BEIGE);
    }
    for (int i: indices) {
        setPointColor(dst.points[i], COLOR_BROWN);
    }
    if (is_source) {
        pcl::transformPointCloud(dst, dst, transformation_gt);
    }
    filepath = constructPath(parameters, name);
    pcl::io::savePLYFileASCII(filepath, dst);
}

void filter_duplicate_points(PointCloudT::Ptr &pcd) {
    std::unordered_set<PointT, PointHash, PointEqual> unique_points;
    std::copy(pcd->points.begin(), pcd->points.end(), std::inserter(unique_points, unique_points.begin()));
    pcd->points.clear();
    std::copy(unique_points.begin(), unique_points.end(), std::back_inserter(pcd->points));
}