#include "common.h"

void printTransformation(const Eigen::Matrix4f &transformation) {
    pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1),
                             transformation(0, 2));
    pcl::console::print_info("R = | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1),
                             transformation(1, 2));
    pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1),
                             transformation(2, 2));
    pcl::console::print_info("\n");
    pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3),
                             transformation(2, 3));
    pcl::console::print_info("\n");
}

void saveColorizedPointCloud(const PointCloudT::Ptr &src,
                             const std::vector<MultivaluedCorrespondence> &correspondences,
                             const pcl::Indices &inliers,
                             const std::string &testname) {
    PointCloudColoredT dst;
    dst.resize(src->size());
    for (int i = 0; i < src->size(); ++i) {
        dst.points[i].x = src->points[i].x;
        dst.points[i].y = src->points[i].y;
        dst.points[i].z = src->points[i].z;
        setPointColor(dst.points[i], COLOR_BEIGE);
    }
    for (const auto &correspondence: correspondences) {
        setPointColor(dst.points[correspondence.query_idx], COLOR_BROWN);
    }
    for (const auto &idx: inliers) {
        setPointColor(dst.points[idx], COLOR_PURPLE);
    }
    pcl::io::savePLYFileBinary(testname + "_downsampled.ply", dst);
}

void setPointColor(PointColoredT &point, int color) {
    point.r = (color >> 16) & 0xff;
    point.g = (color >> 8) & 0xff;
    point.b = (color >> 0) & 0xff;
}
