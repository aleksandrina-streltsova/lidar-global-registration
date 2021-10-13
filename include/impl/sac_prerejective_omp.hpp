#ifndef REGISTRATION_SAC_PREREJECTIVE_OMP_HPP
#define REGISTRATION_SAC_PREREJECTIVE_OMP_HPP

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::computeTransformation(PointCloudSource &output,
                                                                                               const Eigen::Matrix4f &guess) {
    // Some sanity checks first
    if (!this->input_features_) {
        PCL_ERROR("[pcl::%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR(
                "No source features were given! Call setSourceFeatures before aligning.\n");
        return;
    }
    if (!this->target_features_) {
        PCL_ERROR("[pcl::%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR(
                "No target features were given! Call setTargetFeatures before aligning.\n");
        return;
    }

    if (this->input_->size() != this->input_features_->size()) {
        PCL_ERROR("[pcl::%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("The source points and source feature points need to be in a one-to-one "
                  "relationship! Current input cloud sizes: %ld vs %ld.\n",
                  this->input_->size(),
                  this->input_features_->size());
        return;
    }

    if (this->target_->size() != this->target_features_->size()) {
        PCL_ERROR("[pcl::%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("The target points and target feature points need to be in a one-to-one "
                  "relationship! Current input cloud sizes: %ld vs %ld.\n",
                  this->target_->size(),
                  this->target_features_->size());
        return;
    }

    if (this->inlier_fraction_ < 0.0f || this->inlier_fraction_ > 1.0f) {
        PCL_ERROR("[pcl::%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("Illegal inlier fraction %f, must be in [0,1]!\n", this->inlier_fraction_);
        return;
    }

    const float similarity_threshold =
            this->correspondence_rejector_poly_->getSimilarityThreshold();
    if (similarity_threshold < 0.0f || similarity_threshold >= 1.0f) {
        PCL_ERROR("[pcl::%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("Illegal prerejection similarity threshold %f, must be in [0,1[!\n",
                  similarity_threshold);
        return;
    }

    if (this->k_correspondences_ <= 0) {
        PCL_ERROR("[pcl::%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("Illegal correspondence randomness %d, must be > 0!\n",
                  this->k_correspondences_);
        return;
    }

    // Initialize prerejector (similarity threshold already set to default value in
    // constructor)
    this->correspondence_rejector_poly_->setInputSource(this->input_);
    this->correspondence_rejector_poly_->setInputTarget(this->target_);
    this->correspondence_rejector_poly_->setCardinality(this->nr_samples_);
    int num_rejections = 0; // For debugging

    // Initialize results
    this->final_transformation_ = guess;
    this->inliers_.clear();
    float lowest_error = std::numeric_limits<float>::max();
    this->converged_ = false;

    // Temporaries
    pcl::Indices inliers;
    float inlier_fraction;
    float error;

    // If guess is not the Identity matrix we check it
    if (!guess.isApprox(Eigen::Matrix4f::Identity(), 0.01f)) {
        this->getFitness(inliers, error);
        inlier_fraction =
                static_cast<float>(inliers.size()) / static_cast<float>(this->input_->size());

        if (inlier_fraction >= this->inlier_fraction_ && error < lowest_error) {
            this->inliers_ = inliers;
            lowest_error = error;
            this->converged_ = true;
        }
    }

    // Feature correspondence cache
    std::vector<pcl::Indices> similar_features(this->input_->size());

    // Start
    for (int i = 0; i < this->max_iterations_; ++i) {
        // Temporary containers
        pcl::Indices sample_indices;
        pcl::Indices corresponding_indices;

        // Draw nr_samples_ random samples
        this->selectSamples(*(this->input_), this->nr_samples_, sample_indices);

        // Find corresponding features in the target cloud
        this->findSimilarFeatures(sample_indices, similar_features, corresponding_indices);

        // Apply prerejection
        if (!this->correspondence_rejector_poly_->thresholdPolygon(sample_indices,
                                                                   corresponding_indices)) {
            ++num_rejections;
            continue;
        }

        // Estimate the transform from the correspondences, write to transformation_
        this->transformation_estimation_->estimateRigidTransformation(
                *(this->input_), sample_indices, *(this->target_), corresponding_indices, this->transformation_);

        // Take a backup of previous result
        const Matrix4 final_transformation_prev = this->final_transformation_;

        // Set final result to current transformation
        this->final_transformation_ = this->transformation_;

        // Transform the input and compute the error (uses input_ and final_transformation_)
        this->getFitness(inliers, error);

        // Restore previous result
        this->final_transformation_ = final_transformation_prev;

        // If the new fit is better, update results
        inlier_fraction =
                static_cast<float>(inliers.size()) / static_cast<float>(this->input_->size());

        // Update result if pose hypothesis is better
        if (inlier_fraction >= this->inlier_fraction_ && error < lowest_error) {
            this->inliers_ = inliers;
            lowest_error = error;
            this->converged_ = true;
            this->final_transformation_ = this->transformation_;
        }
    }

    // Apply the final transformation
    if (this->converged_)
        transformPointCloud(*(this->input_), output, this->final_transformation_);

    // Debug output
    PCL_DEBUG("[pcl::%s::computeTransformation] Rejected %i out of %i generated pose "
              "hypotheses.\n",
              this->getClassName().c_str(),
              num_rejections,
              this->max_iterations_);
}

#endif