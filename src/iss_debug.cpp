#include "iss_debug.h"
#include "quadric.h"

void ISSKeypoint3DDebug::saveEigenValues(const AlignmentParameters &parameters) {
    int n = this->input_->size();
    std::vector<float> eigen_values(n);
    for (int i = 0; i < n; ++i) {
        eigen_values[i] = this->third_eigen_value_[i];
    }
    // TODO: use correct transformation
    saveColorizedWeights(this->input_, eigen_values, "weights", parameters, Eigen::Matrix4f::Identity());
}

bool ISSKeypoint3DDebug::initCompute() {
    if (!PCLBase<PointN>::initCompute())
        return (false);

    // Initialize the spatial locator
    if (!this->tree_) {
        if (input_->isOrganized())
            this->tree_.reset(new pcl::search::OrganizedNeighbor<PointN>());
        else
            this->tree_.reset(new pcl::search::KdTree<PointN>(false));
    }

    // If no search surface has been defined, use the input dataset as the search surface itself
    if (!this->surface_)
        this->surface_ = this->input_;

    // Send the surface dataset to the spatial locator
    this->tree_->setInputCloud(this->surface_);

    // Do a fast check to see if the search parameters are well defined
    if (this->search_radius_ != 0.0) {
        if (this->k_ != 0) {
            PCL_ERROR (
                    "[pcl::%s::initCompute] Both radius (%f) and K (%d) defined! Set one of them to zero first and then re-run compute ().\n",
                    getClassName().c_str(), this->search_radius_, this->k_);
            return (false);
        }

        // Use the radiusSearch () function
        this->search_parameter_ = this->search_radius_;
        if (this->surface_ == this->input_)       // if the two surfaces are the same
        {
            // Declare the search locator definition
            this->search_method_ = [this](pcl::index_t index, double radius, pcl::Indices &k_indices,
                                          std::vector<float> &k_distances) {
                int nr_neighbors = this->tree_->radiusSearch(index, radius, k_indices, k_distances, max_neighbors_);
                if (nr_neighbors < min_required_neighbors_) {
                    return this->tree_->nearestKSearch(index, min_required_neighbors_, k_indices, k_distances);
                }
                return nr_neighbors;
            };
        } else {
            // Declare the search locator definition
            this->search_method_surface_ = [this](const PointCloudIn &cloud, pcl::index_t index, double radius,
                                                  pcl::Indices &k_indices, std::vector<float> &k_distances) {
                int nr_neighbors = this->tree_->radiusSearch(cloud, index, radius, k_indices, k_distances,
                                                             max_neighbors_);
                if (nr_neighbors < min_required_neighbors_) {
                    return this->tree_->nearestKSearch(cloud, index, min_required_neighbors_, k_indices, k_distances);
                }
                return nr_neighbors;
            };
        }
    } else {
        if (this->k_ != 0)         // Use the nearestKSearch () function
        {
            this->search_parameter_ = this->k_;
            if (this->surface_ == this->input_)       // if the two surfaces are the same
            {
                // Declare the search locator definition
                this->search_method_ = [this](pcl::index_t index, int k, pcl::Indices &k_indices,
                                              std::vector<float> &k_distances) {
                    return this->tree_->nearestKSearch(index, k, k_indices, k_distances);
                };
            } else {
                // Declare the search locator definition
                this->search_method_surface_ = [this](const PointCloudIn &cloud, pcl::index_t index, int k,
                                                      pcl::Indices &k_indices, std::vector<float> &k_distances) {
                    return this->tree_->nearestKSearch(cloud, index, k, k_indices, k_distances);
                };
            }
        } else {
            PCL_ERROR (
                    "[pcl::%s::initCompute] Neither radius nor K defined! Set one of them to a positive number first and then re-run compute ().\n",
                    getClassName().c_str());
            return (false);
        }
    }

    this->keypoints_indices_.reset(new pcl::PointIndices);
    this->keypoints_indices_->indices.reserve(input_->size());

    if (this->salient_radius_ <= 0) {
        PCL_ERROR ("[pcl::%s::initCompute] : the salient radius (%f) must be strict positive!\n",
                   this->name_.c_str(), this->salient_radius_);
        return (false);
    }
    if (this->non_max_radius_ <= 0) {
        PCL_ERROR ("[pcl::%s::initCompute] : the non maxima radius (%f) must be strict positive!\n",
                   this->name_.c_str(), this->non_max_radius_);
        return (false);
    }
    if (this->gamma_21_ <= 0) {
        PCL_ERROR (
                "[pcl::%s::initCompute] : the threshold on the ratio between the 2nd and the 1rst eigenvalue (%f) must be strict positive!\n",
                this->name_.c_str(), this->gamma_21_);
        return (false);
    }
    if (this->gamma_32_ <= 0) {
        PCL_ERROR (
                "[pcl::%s::initCompute] : the threshold on the ratio between the 3rd and the 2nd eigenvalue (%f) must be strict positive!\n",
                this->name_.c_str(), this->gamma_32_);
        return (false);
    }
    if (this->min_neighbors_ <= 0) {
        PCL_ERROR ("[pcl::%s::initCompute] : the minimum number of neighbors (%f) must be strict positive!\n",
                   this->name_.c_str(), this->min_neighbors_);
        return (false);
    }

    delete[] this->third_eigen_value_;

    this->third_eigen_value_ = new double[this->input_->size()];
    memset(this->third_eigen_value_, 0, sizeof(double) * input_->size());

    delete[] this->edge_points_;

    if (this->border_radius_ > 0.0) {
        if (this->normals_->empty()) {
            if (this->normal_radius_ <= 0.) {
                PCL_ERROR (
                        "[pcl::%s::initCompute] : the radius used to estimate surface normals (%f) must be positive!\n",
                        this->name_.c_str(), this->normal_radius_);
                return (false);
            }

            PointCloudNPtr normal_ptr(new PointCloudN());
            if (this->input_->height == 1) {
                pcl::NormalEstimation<PointN, PointN> normal_estimation;
                normal_estimation.setInputCloud(this->surface_);
                normal_estimation.setRadiusSearch(this->normal_radius_);
                normal_estimation.compute(*normal_ptr);
            } else {
                pcl::IntegralImageNormalEstimation<PointN, PointN> normal_estimation;
                normal_estimation.setNormalEstimationMethod(
                        pcl::IntegralImageNormalEstimation<PointN, PointN>::SIMPLE_3D_GRADIENT);
                normal_estimation.setInputCloud(this->surface_);
                normal_estimation.setNormalSmoothingSize(5.0);
                normal_estimation.compute(*normal_ptr);
            }
            this->normals_ = normal_ptr;
        }
        if (this->normals_->size() != this->surface_->size()) {
            PCL_ERROR (
                    "[pcl::%s::initCompute] normals given, but the number of normals does not match the number of input points!\n",
                    this->name_.c_str());
            return (false);
        }
    } else if (this->border_radius_ < 0.0) {
        PCL_ERROR (
                "[pcl::%s::initCompute] : the border radius used to estimate boundary points (%f) must be positive!\n",
                this->name_.c_str(), this->border_radius_);
        return (false);
    }
    return (true);
}

void ISSKeypoint3DDebug::estimateSubVoxelKeyPoints(PointNCloud::Ptr &subvoxel_kps) {
    rassert(subvoxel_kps, 2730954205409203)
    pcl::NormalEstimation<PointN, PointN> normal_estimation;
    if (!this->surface_) {  // in method compute surface_.reset is called, we need to set it again
        this->surface_ = this->input_;
    }
    auto &indices = this->keypoints_indices_->indices;
    std::sort(indices.begin(), indices.end());
    indices.resize(std::min(10, (int) indices.size()));
    subvoxel_kps->resize(indices.size());
    pcl::Indices nn_indices;
    std::vector<float> nn_sqr_dists;
    int count = 0;
    for (int i = 0; i < indices.size(); ++i) {
//        this->searchForNeighbors(indices[i], this->salient_radius_, nn_indices, nn_sqr_dists);
//        if (nn_indices.size() < 6) {
//            PCL_DEBUG("[%s::estimateSubVoxelKeyPoints] key point has %i neighbors, "
//                      "quadric coefficients can't be calculated",
//                      this->getClassName().c_str(), nn_indices.size());
//            subvoxel_kps->operator[](i) = this->input_->points[indices[i]];
//        }
        this->tree_->nearestKSearch(*this->input_, indices[i], 6, nn_indices, nn_sqr_dists);
        Eigen::Vector3f normal;
        float curvature;
        normal_estimation.computePointNormal(*this->input_, nn_indices, normal.x(), normal.y(), normal.z(), curvature);
        Eigen::MatrixX3d points;
        Eigen::VectorXd values;
        points.resize(nn_indices.size(), 3);
        values.resize(nn_indices.size(), 1);
        for (int j = 0; j < nn_indices.size(); ++j) {
            points(j, 0) = this->input_->points[nn_indices[j]].x;
            points(j, 1) = this->input_->points[nn_indices[j]].y;
            points(j, 2) = this->input_->points[nn_indices[j]].z;
            values(j) = this->third_eigen_value_[nn_indices[j]];
        }
        Eigen::Vector3f kp = estimateMaximumPoint(points, normal.cast<double>(), values, i).cast<float>();
        if ((kp - this->input_->points[indices[i]].getVector3fMap()).norm() < this->salient_radius_) {
            subvoxel_kps->operator[](i) = PointN{kp.x(), kp.y(), kp.z()};
        } else {
            subvoxel_kps->operator[](i) = PointN{1.0, 1.0, 1.0};
            count++;
        }
        // TODO: normal estimation?
    }
    PCL_DEBUG("%i/%i unsuccessful attempts!!\n", count, indices.size());
    if (this->input_ == this->surface_) {
        this->surface_.reset();
    }
}