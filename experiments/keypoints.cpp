#include "keypoints.h"

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
                return this->tree_->radiusSearch(index, radius, k_indices, k_distances, max_neighbors_);
            };
        } else {
            // Declare the search locator definition
            this->search_method_surface_ = [this](const PointCloudIn &cloud, pcl::index_t index, double radius,
                                                  pcl::Indices &k_indices, std::vector<float> &k_distances) {
                return this->tree_->radiusSearch(cloud, index, radius, k_indices, k_distances, max_neighbors_);
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

std::vector<float> ISSKeypoint3DDebug::getThirdEigenValuesDebug() {
    int n = this->input_->size();
    std::vector<float> weights;
    std::copy(this->third_eigen_value_, this->third_eigen_value_ + n, std::back_inserter(weights));
    return weights;
}

std::vector<float> HarrisKeypoint3DDebug::getResponseHarrisDebug() {
    pcl::PointCloud<pcl::PointXYZI> output;
    this->responseHarris(output);
    std::vector<float> weights;
    std::transform(output.begin(), output.end(), std::back_inserter(weights),
                   [](const pcl::PointXYZI &point) { return point.intensity; });
    return weights;
}