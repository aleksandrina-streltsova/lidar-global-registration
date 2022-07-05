#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/keypoints/iss_3d.h>

#include "align.h"
#include "csv_parser.h"
#include "filter.h"
#include "utils.h"

#define RF_MIN_ANGLE_RAD 0.04f

void detectKeyPoints(const PointNCloud::ConstPtr &pcd, const NormalCloud::ConstPtr &normals, pcl::IndicesPtr &indices,
                     const AlignmentParameters &parameters) {
    if (parameters.keypoint_id == KEYPOINT_ISS) {
        PointNCloud key_points;
        double iss_salient_radius_ = 4 * parameters.voxel_size;
        double iss_non_max_radius_ = 2 * parameters.voxel_size;
        double iss_gamma_21_(0.975);
        double iss_gamma_32_(0.975);
        int iss_min_neighbors_(4);
        pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
        pcl::ISSKeypoint3D<PointN, PointN, PointN> iss_detector;
        iss_detector.setSearchMethod(tree);
        iss_detector.setSalientRadius(iss_salient_radius_);
        iss_detector.setNonMaxRadius(iss_non_max_radius_);
        iss_detector.setThreshold21(iss_gamma_21_);
        iss_detector.setThreshold32(iss_gamma_32_);
        iss_detector.setMinNeighbors(iss_min_neighbors_);
        iss_detector.setInputCloud(pcd);
        iss_detector.setNormals(pcd);
        iss_detector.compute(key_points);
        indices = std::make_shared<pcl::Indices>(pcl::Indices());
        *indices = iss_detector.getKeypointsIndices()->indices;
        if (parameters.fix_seed) {
            std::sort(indices->begin(), indices->end());
        }
        PCL_DEBUG("[detectKeyPoints] %d key points\n", indices->size());
    } else {
        if (parameters.keypoint_id != KEYPOINT_ANY) {
            PCL_WARN("[detectKeyPoints] Detection method %s isn't supported, no detection method will be applied.\n",
                     parameters.keypoint_id.c_str());
        }
    }
}

void estimateNormals(float radius_search, const PointNCloud::Ptr &pcd, NormalCloud::Ptr &normals,
                     bool normals_available) {
    pcl::NormalEstimationOMP<PointN, pcl::Normal> normal_est;
    normal_est.setRadiusSearch(radius_search);

    normal_est.setInputCloud(pcd);
    pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
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

void smoothNormals(float radius_search, float voxel_size, const PointNCloud::Ptr &pcd) {
    NormalCloud::Ptr old_normals(new NormalCloud);
    pcl::KdTreeFLANN<PointN>::Ptr tree(new pcl::KdTreeFLANN<PointN>());
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

void estimateReferenceFrames(const PointNCloud::Ptr &pcd, const NormalCloud::Ptr &normals,
                             const pcl::IndicesPtr &indices, PointRFCloud::Ptr &frames_kps,
                             const AlignmentParameters &parameters, bool is_source) {
    int nr_kps = indices ? indices->size() : pcd->size();
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
            lrf.x_axis[d] = lrf_eigen.col (0)[d];
            lrf.y_axis[d] = lrf_eigen.col (1)[d];
            lrf.z_axis[d] = lrf_eigen.col (2)[d];
        }
        frames_kps = std::make_shared<PointRFCloud>(PointRFCloud());
        frames_kps->resize(nr_kps, lrf);
    }  else if (lrf_id == "gravity") {
        frames_kps = std::make_shared<PointRFCloud>(PointRFCloud());
        frames_kps->resize(nr_kps);

        pcl::search::KdTree<PointN>::Ptr tree(new pcl::search::KdTree<PointN>());
        tree->setInputCloud(pcd);
        tree->setSortedResults(true);

        pcl::SHOTLocalReferenceFrameEstimation<PointN, PointRF>::Ptr lrf_estimator(new pcl::SHOTLocalReferenceFrameEstimation<PointN, PointRF>());
        float lrf_radius = parameters.voxel_size * parameters.feature_radius_coef;
        lrf_estimator->setRadiusSearch(lrf_radius);
        lrf_estimator->setInputCloud(pcd);
        if (indices) lrf_estimator->setIndices(indices);
        lrf_estimator->setSearchMethod(tree);
        lrf_estimator->compute(*frames_kps);
        rassert(frames_kps->size() == nr_kps, 15946243)

        Eigen::Vector3f gravity(0, 0, 1);
        for (std::size_t i = 0; i < nr_kps; ++i) {
            int idx = indices ? indices->operator[](i) : i;
            PointRF &output_rf = frames_kps->points[i];
            const pcl::Normal &normal = normals->points[idx];
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
    } else if (lrf_id != DEFAULT_LRF){
        PCL_WARN("[estimateReferenceFrames] LRF %s isn't supported, default LRF will be used.\n", lrf_id.c_str());
    }
}