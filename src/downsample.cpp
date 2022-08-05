#include "downsample.h"

void downsamplePointCloud(const PointNCloud::ConstPtr &pcd_fullsize, PointNCloud::Ptr &pcd_down, float voxel_size) {
    pcl::console::print_highlight("Downsampling [voxel_size = %.5f]...\n", voxel_size);
    if (voxel_size <= 0.0) {
        PCL_ERROR("[downsamplePointCloud] voxel_size <= 0.");
    }
    std::vector<PointN> points;
    Eigen::Vector3f voxel_size3 = Eigen::Vector3f(voxel_size, voxel_size, voxel_size);
    auto[min_point_AABB, max_point_AABB] = calculateBoundingBox<PointN>(pcd_fullsize);
    Eigen::Vector3f voxel_min_bound =
            Eigen::Vector3f(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z) - voxel_size3 * 0.5;
    Eigen::Vector3f voxel_max_bound =
            Eigen::Vector3f(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z) - voxel_size3 * 0.5;
    if (voxel_size * (float) std::numeric_limits<int>::max() < (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        PCL_ERROR("[downsamplePointCloud] voxel_size is too small.");
    }
    pcl::console::print_highlight("Point cloud downsampled with voxel_size = %f, from %zu...",
                                  voxel_size, pcd_fullsize->size());
    std::unordered_map<Eigen::Vector3i, AccumulatedPoint, HashEigen<Eigen::Vector3i>> voxelindex_to_accpoint;
    Eigen::Vector3f ref_coord;
    Eigen::Vector3i voxel_index;
    auto point_representation = pcl::DefaultPointRepresentation<PointN>();
    for (int i = 0; i < (int) pcd_fullsize->size(); i++) {
        if (!point_representation.isValid(pcd_fullsize->points[i])) continue;
        ref_coord = Eigen::Vector3f(pcd_fullsize->points[i].x, pcd_fullsize->points[i].y, pcd_fullsize->points[i].z);
        ref_coord = (ref_coord - voxel_min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))), int(floor(ref_coord(2)));
        voxelindex_to_accpoint[voxel_index].AddPoint(pcd_fullsize, i);
    }

    for (const auto &accpoint: voxelindex_to_accpoint) {
        points.push_back(accpoint.second.GetAveragePoint());
    }
    pcd_down->points.clear();
    std::copy(points.begin(), points.end(), std::back_inserter(pcd_down->points));
    pcd_down->width = points.size();
    pcd_down->height = 1;
    pcd_down->is_dense = true;
    pcl::console::print_highlight("to %zu\n", pcd_down->size());
}

