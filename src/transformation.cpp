#include "transformation.h"

// http://nghiaho.com/?page_id=671
void estimateOptimalRigidTransformation(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                        const std::vector<InlierPair> &inlier_pairs, Eigen::Matrix4f &transformation) {
    Eigen::Vector3f centroid_src{0.0, 0.0, 0.0}, centroid_tgt{0.0, 0.0, 0.0};
    int n = inlier_pairs.size();
    for (const auto &ip: inlier_pairs) {
        centroid_src += src->points[ip.idx_src].getVector3fMap();
        centroid_tgt += tgt->points[ip.idx_tgt].getVector3fMap();
    }
    centroid_src /= (float) n;
    centroid_tgt /= (float) n;

    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();

    // H = (A - c_A) (B - c_B)^T, size of A and B -- (3 x N)
    for (const auto &ip: inlier_pairs) {
        const Eigen::Vector3f &p = src->points[ip.idx_src].getVector3fMap();
        const Eigen::Vector3f &q = tgt->points[ip.idx_tgt].getVector3fMap();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                H(i, j) += (p(i) - centroid_src(i)) * (q(j) - centroid_tgt(j));
            }
        }
    }
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3f V = svd.matrixV();
        V.block<3, 1>(0, 2) *= -1;
        R = V * svd.matrixU().transpose();
    }
    Eigen::Vector3f t = centroid_tgt - R * centroid_src;
    transformation = Eigen::Matrix4f::Identity();
    transformation.block<3, 3>(0, 0) = R;
    transformation.block<3, 1>(0, 3) = t;
}

