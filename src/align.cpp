#include <pcl/features/normal_3d_omp.h>

#include "align.h"
#include "csv_parser.h"
#include "filter.h"
#include "pcl/features/shot_lrf.h"

#define RF_MIN_ANGLE_RAD 0.04f

Eigen::Matrix4f getTransformation(const std::string &csv_path,
                                  const std::string &src_filename, const std::string &tgt_filename) {
    std::ifstream file(csv_path);
    Eigen::Matrix4f src_position, tgt_position;

    CSVRow row;
    while (file >> row) {
        if (row[0] == src_filename) {
            for (int i = 0; i < 16; ++i) {
                src_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
        }
        if (row[0] == tgt_filename) {
            for (int i = 0; i < 16; ++i) {
                tgt_position(i / 4, i % 4) = std::stof(row[i + 1]);
            }
        }
    }
    return tgt_position.inverse() * src_position;
}

void estimateNormals(float radius_search, const PointCloudTN::Ptr &pcd, PointCloudN::Ptr &normals,
                     bool normals_available) {
    pcl::NormalEstimationOMP<PointTN, pcl::Normal> normal_est;
    normal_est.setRadiusSearch(radius_search);

    normal_est.setInputCloud(pcd);
    pcl::search::KdTree<PointTN>::Ptr tree(new pcl::search::KdTree<PointTN>());
    normal_est.setSearchMethod(tree);
    normal_est.compute(*normals);
    // use normals from point cloud to orient estimated normals and replace NaN normals
    if (normals_available) {
        for (int i = 0; i < pcd->size(); ++i) {
            auto &normal = normals->points[i];
            const auto &point = pcd->points[i];
            if (!std::isfinite(normal.normal_x) || !std::isfinite(normal.normal_y) || !std::isfinite(normal.normal_z)) {
                normal.normal_x = point.normal_x;
                normal.normal_y = point.normal_y;
                normal.normal_z = point.normal_z;
                normal.curvature = 0.f;
            } else if (normal.normal_x * point.normal_x + normal.normal_y * point.normal_y +
                       normal.normal_z * point.normal_z < 0) {
                normal.normal_x *= -1.f;
                normal.normal_y *= -1.f;
                normal.normal_z *= -1.f;
            }
        }
    }
    int nan_counter = 0;
    for (const auto &normal: normals->points) {
        const Eigen::Vector4f &normal_vec = normal.getNormalVector4fMap();
        if (!std::isfinite(normal_vec[0]) ||
            !std::isfinite(normal_vec[1]) ||
            !std::isfinite(normal_vec[2])) {
            nan_counter++;
        }
    }
    PCL_DEBUG("[estimateNormals] %d NaN normals.\n", nan_counter);
}

void smoothNormals(float radius_search, float voxel_size, const PointCloudTN::Ptr &pcd) {
    PointCloudN::Ptr old_normals(new PointCloudN);
    pcl::KdTreeFLANN<PointTN>::Ptr tree(new pcl::KdTreeFLANN<PointTN>());
    tree->setInputCloud(pcd);

    std::vector<pcl::Indices> vector_of_indices(pcd->size());
    std::vector<float> distances;

    for (int i = 0; i < (int) std::ceil(radius_search / voxel_size); i++) {
        pcl::copyPointCloud(*pcd, *old_normals);
        for (int j = 0; j < pcd->size(); ++j) {
            pcl::Normal acc(0.0, 0.0, 0.0), old_normal(old_normals->points[j]);
            for (auto idx: vector_of_indices[j]) {
                float dot_product = old_normal.normal_x * old_normals->points[idx].normal_x +
                                    old_normal.normal_y * old_normals->points[idx].normal_y +
                                    old_normal.normal_z * old_normals->points[idx].normal_z;
                if (dot_product > 0.0) {
                    acc.normal_x += old_normals->points[idx].normal_x;
                    acc.normal_y += old_normals->points[idx].normal_y;
                    acc.normal_z += old_normals->points[idx].normal_z;
                }
            }
            float norm = std::sqrt(acc.normal_x * acc.normal_x + acc.normal_y * acc.normal_y +
                                   acc.normal_z * acc.normal_z);
            pcd->points[j].normal_x = acc.normal_x / norm;
            pcd->points[j].normal_y = acc.normal_y / norm;
            pcd->points[j].normal_z = acc.normal_z / norm;
        }
    }
}

void estimateReferenceFrames(const PointCloudTN::Ptr &pcd, const PointCloudN::Ptr &normals,
                             PointCloudRF::Ptr &frames, const AlignmentParameters &parameters, bool is_source) {
    std::string lrf_id = parameters.lrf_id;
    std::transform(lrf_id.begin(), lrf_id.end(), lrf_id.begin(), [](unsigned char c) { return std::tolower(c); });
    if (lrf_id == "gt") {
        Eigen::Matrix3f lrf_eigen = Eigen::Matrix3f::Identity();
        if (is_source) {
            if (!parameters.ground_truth) {
                PCL_ERROR("[estimateReferenceFrames] ground truth wasn't provided!");
            } else {
                lrf_eigen = parameters.ground_truth->block<3, 3>(0, 0).inverse() * lrf_eigen;
            }
        }
        PointRF lrf;
        for (int d = 0; d < 3; ++d) {
            lrf.x_axis[d] = lrf_eigen.row (0)[d];
            lrf.y_axis[d] = lrf_eigen.row (1)[d];
            lrf.z_axis[d] = lrf_eigen.row (2)[d];
        }
        frames = std::make_shared<PointCloudRF>(PointCloudRF());
        frames->resize(pcd->size(), lrf);
    }  else if (lrf_id == "gravity") {
        frames = std::make_shared<PointCloudRF>(PointCloudRF());
        frames->resize(pcd->size());

        pcl::search::KdTree<PointTN>::Ptr tree(new pcl::search::KdTree<PointTN>());
        tree->setInputCloud(pcd);
        tree->setSortedResults(true);

        pcl::SHOTLocalReferenceFrameEstimation<PointTN, PointRF>::Ptr lrf_estimator(new pcl::SHOTLocalReferenceFrameEstimation<PointTN, PointRF>());
        float lrf_radius = parameters.voxel_size * parameters.feature_radius_coef;
        lrf_estimator->setRadiusSearch(lrf_radius);
        lrf_estimator->setInputCloud(pcd);
        lrf_estimator->setSearchMethod(tree);
        lrf_estimator->compute(*frames);

        Eigen::Vector3f gravity(0, 0, 1);
        for (std::size_t i = 0; i < pcd->size(); ++i) {
            PointRF &output_rf = frames->points[i];
            const pcl::Normal &normal = normals->points[i];
            Eigen::Vector3f x_axis(normal.normal_x, normal.normal_y, normal.normal_z);
            if (std::acos(std::abs(std::clamp(x_axis.dot(gravity), -1.0f, 1.0f))) > RF_MIN_ANGLE_RAD) {
                Eigen::Vector3f y_axis = gravity.cross(x_axis);
                Eigen::Vector3f z_axis = x_axis.cross(y_axis);
                for (int d = 0; d < 3; ++d) {
                    output_rf.x_axis[d] = x_axis[d];
                    output_rf.y_axis[d] = y_axis[d];
                    output_rf.z_axis[d] = z_axis[d];
                }
            }
        }
   }
}