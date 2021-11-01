#ifndef REGISTRATION_SAC_PREREJECTIVE_OMP_HPP
#define REGISTRATION_SAC_PREREJECTIVE_OMP_HPP

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::setNumberOfThreads(unsigned int nr_threads) {
    if (nr_threads == 0)
#ifdef _OPENMP
    {
        threads_ = omp_get_num_procs();
        PCL_DEBUG (
                "[SampleConsensusPrerejectiveOMP::setNumberOfThreads] Automatic number of threads requested, choosing %i threads.\n",
                threads_);
    }
#else
        threads_ = 1;
#endif
    else
        threads_ = nr_threads;
}

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::buildIndices(
        const pcl::Indices &sample_indices,
        const std::vector<MultivaluedCorrespondence> &correspondences,
        pcl::Indices &source_indices,
        pcl::Indices &target_indices) {
    // Allocate results
    source_indices.resize(sample_indices.size());
    target_indices.resize(sample_indices.size());

    // Loop over the sampled features
    for (std::size_t j = 0; j < sample_indices.size(); ++j) {
        // Current correspondence index
        const auto &idx = sample_indices[j];
        source_indices[j] = correspondences[idx].query_idx;

        // Select one at random and add it to target_indices
        if (this->k_correspondences_ == 1)
            target_indices[j] = correspondences[idx].match_indices[0];
        else
            target_indices[j] = correspondences[idx].match_indices[this->getRandomIndex(this->k_correspondences_)];
    }
}

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::getRMSE(pcl::Indices &inliers,
                                                                                 const std::vector<MultivaluedCorrespondence> &correspondences,
                                                                                 const Matrix4 &transformation,
                                                                                 float &rmse_score) {
    // Initialize variables
    inliers.clear();
    inliers.reserve(correspondences.size());
    rmse_score = 0.0f;

    // Transform the input dataset using the final transformation
    PointCloudSource input_transformed;
    input_transformed.resize(this->input_->size());
    transformPointCloud(*(this->input_), input_transformed, transformation);

    // For each point from correspondences in the source dataset
    for (int i = 0; i < correspondences.size(); ++i) {
        int query_idx = correspondences[i].query_idx;
        int match_idx = correspondences[i].match_indices[0];
        PointT source_point(input_transformed.points[query_idx]);
        PointT target_point(this->target_->points[match_idx]);

        // Calculate correspondence distance
        float dist = pcl::L2_Norm(source_point.data, target_point.data, 3);

        // Check if correspondence is an inlier
        if (dist < this->corr_dist_threshold_) {
            // Update inliers
            inliers.push_back(query_idx);

            // Update fitness score
            rmse_score += dist * dist;
        }
    }

    // Calculate RMSE
    if (!inliers.empty())
        rmse_score = std::sqrt(rmse_score / static_cast<float>(inliers.size()));
    else
        rmse_score = std::numeric_limits<float>::max();
}

template<typename PointSource, typename PointTarget, typename FeatureT>
float SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::getRMSEScore() {
    return rmse_;
}

template<typename PointSource, typename PointTarget, typename FeatureT>
int SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::countCorrectCorrespondences(
        const Eigen::Matrix4f &transformation_gt, float error_threshold, bool check_inlier) {
    PointCloudSource input_transformed;
    input_transformed.resize(this->input_->size());
    pcl::transformPointCloud(*(this->input_), input_transformed, transformation_gt);

    std::set<int> inliers(this->inliers_.begin(), this->inliers_.end());
    int correct_correspondences = 0;
    for (const auto &correspondence: this->multivalued_correspondences_) {
        int query_idx = correspondence.query_idx;
        if (!check_inlier || (check_inlier && inliers.find(query_idx) != inliers.end())) {
            int match_idx = correspondence.match_indices[0];
            PointT source_point(input_transformed.points[query_idx]);
            PointT target_point(this->target_->points[match_idx]);
            float e = pcl::L2_Norm(source_point.data, target_point.data, 3);
            if (e < error_threshold) {
                correct_correspondences++;
            }
        }
    }
    return correct_correspondences;
}

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::selectCorrespondences(
        int nr_correspondences,
        int nr_samples,
        pcl::Indices &sample_indices,
        UniformRandIntGenerator &rand_generator) {
    if (nr_samples > nr_correspondences) {
        PCL_ERROR("[%s::selectCorrespondences] ", this->getClassName().c_str());
        PCL_ERROR("The number of samples (%d) must not be greater than the number of correspondences (%zu)!\n",
                  nr_samples,
                  nr_correspondences);
        return;
    }

    sample_indices.resize(nr_samples);
    int temp_sample;

    // Draw random samples until n samples is reached
    for (int i = 0; i < nr_samples; i++) {
        // Select a random number
        sample_indices[i] = rand_generator();

        // Run trough list of numbers, starting at the lowest, to avoid duplicates
        for (int j = 0; j < i; j++) {
            // Move value up if it is higher than previous selections to ensure true
            // randomness
            if (sample_indices[i] >= sample_indices[j]) {
                sample_indices[i]++;
            } else {
                // The new number is lower, place it at the correct point and break for a sorted
                // list
                temp_sample = sample_indices[i];
                for (int k = i; k > j; k--)
                    sample_indices[k] = sample_indices[k - 1];

                sample_indices[j] = temp_sample;
                break;
            }
        }
    }
}

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::computeTransformation(PointCloudSource &output,
                                                                                               const Eigen::Matrix4f &guess) {
    // Some sanity checks first
    if (!this->input_features_) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR(
                "No source features were given! Call setSourceFeatures before aligning.\n");
        return;
    }
    if (!this->target_features_) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR(
                "No target features were given! Call setTargetFeatures before aligning.\n");
        return;
    }

    if (this->input_->size() != this->input_features_->size()) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("The source points and source feature points need to be in a one-to-one "
                  "relationship! Current input cloud sizes: %ld vs %ld.\n",
                  this->input_->size(),
                  this->input_features_->size());
        return;
    }

    if (this->target_->size() != this->target_features_->size()) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("The target points and target feature points need to be in a one-to-one "
                  "relationship! Current input cloud sizes: %ld vs %ld.\n",
                  this->target_->size(),
                  this->target_features_->size());
        return;
    }

    if (this->inlier_fraction_ < 0.0f || this->inlier_fraction_ > 1.0f) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("Illegal inlier fraction %f, must be in [0,1]!\n", this->inlier_fraction_);
        return;
    }

    const float similarity_threshold =
            this->correspondence_rejector_poly_->getSimilarityThreshold();
    if (similarity_threshold < 0.0f || similarity_threshold >= 1.0f) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("Illegal prerejection similarity threshold %f, must be in [0,1[!\n",
                  similarity_threshold);
        return;
    }

    if (this->k_correspondences_ <= 0) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
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
    this->rmse_ = std::numeric_limits<float>::max();
    float max_inlier_fraction = 0.0f;
    this->converged_ = false;

    int threads = threads_;
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
    if (threads_ == 0) {
        threads = omp_get_num_procs();
        PCL_DEBUG("[%s::computeTransformation] Automatic number of threads requested, choosing %i threads.\n",
                  this->getClassName().c_str(),
                  threads);
    }
#else
    // Parallelization desired, but not available
    PCL_WARN ("[SampleConsensusPrerejectiveOMP::computeTransformation] Parallelization is requested, but OpenMP 3.1 is not available! Continuing without parallelization.\n");
    threads = -1;
#endif

    // Feature correspondence set
    std::vector<MultivaluedCorrespondence> correspondences_ij(this->input_->size());

    // Build correspondences
    {
        pcl::ScopeTime t("Correspondence search");
#pragma omp parallel for num_threads(threads) default(none) shared(correspondences_ij)
        for (int i = 0; i < this->input_->size(); i++) {
            correspondences_ij[i].query_idx = i;
            pcl::Indices &match_indices = correspondences_ij[i].match_indices;
            match_indices.resize(this->k_correspondences_);
            std::vector<float> match_distances(this->k_correspondences_);
            this->feature_tree_->nearestKSearch(*(this->input_features_),
                                                i,
                                                this->k_correspondences_,
                                                match_indices,
                                                match_distances);
        }

        if (reciprocal_) {
            pcl::KdTreeFLANN<FeatureT> feature_tree_src_(new pcl::KdTreeFLANN<FeatureT>);
            feature_tree_src_.setInputCloud(this->input_features_);
            std::vector<MultivaluedCorrespondence> correspondences_ji(this->target_->size());

#pragma omp parallel for num_threads(threads) default(none) shared(correspondences_ji, feature_tree_src_)
            for (int j = 0; j < this->target_->size(); ++j) {
                correspondences_ji[j].query_idx = j;
                pcl::Indices &match_indices = correspondences_ji[j].match_indices;
                match_indices.resize(1);
                std::vector<float> match_distances(1);
                feature_tree_src_.nearestKSearch(*(this->target_features_),
                                                 j,
                                                 1,
                                                 match_indices,
                                                 match_distances);
            }

            std::vector<MultivaluedCorrespondence> correspondences_mutual;
            for (int i = 0; i < this->input_->size(); ++i) {
                bool reciprocal = false;
                MultivaluedCorrespondence corr = correspondences_ij[i];
                for (const int &j: corr.match_indices) {
                    if (correspondences_ji[j].match_indices[0] == i) {
                        reciprocal = true;
                    }
                }
                if (reciprocal) {
                    correspondences_mutual.emplace_back(corr);
                }
            }
            correspondences_ij = correspondences_mutual;
            PCL_DEBUG("[%s::computeTransformation] %i correspondences remain after mutual filter.\n",
                      this->getClassName().c_str(),
                      correspondences_mutual.size());
        }

        this->multivalued_correspondences_ = correspondences_ij;
    }

    {
        pcl::Indices inliers;
        float inlier_fraction, error;

        // If guess is not the Identity matrix we check it
        if (!guess.isApprox(Eigen::Matrix4f::Identity(), 0.01f)) {
            this->getRMSE(inliers, correspondences_ij, guess, error);
            inlier_fraction = static_cast<float>(inliers.size()) / static_cast<float>(this->input_->size());

            if (inlier_fraction >= max_inlier_fraction) {
                this->inliers_ = inliers;
                this->rmse_ = error;
                max_inlier_fraction = inlier_fraction;
                this->converged_ = true;
                this->final_transformation_ = guess;
            }
        }
    }

    // Start
    {
        pcl::ScopeTime t("RANSAC");
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
#pragma omp parallel \
    num_threads(threads) \
    default(none) \
    shared(correspondences_ij, max_inlier_fraction) \
    reduction(+:num_rejections)
#endif
        {
            // Local best results
            pcl::Indices best_inliers;
            float min_error_local = std::numeric_limits<float>::max();
            float max_inlier_fraction_local = 0.0f;
            Matrix4 best_transformation;

            // Temporaries
            pcl::Indices inliers;
            float inlier_fraction, error;

            UniformRandIntGenerator rand_generator(0, (int) correspondences_ij.size() - 1);

#pragma omp for nowait
            for (int i = 0; i < this->max_iterations_; ++i) {
                // Temporary containers
                pcl::Indices sample_indices;
                pcl::Indices source_indices;
                pcl::Indices target_indices;

                // Draw nr_samples_ random samples
                selectCorrespondences(correspondences_ij.size(), this->nr_samples_, sample_indices, rand_generator);

                // Find corresponding features in the target cloud
                buildIndices(sample_indices, correspondences_ij, source_indices, target_indices);

                // Apply prerejection
                if (!this->correspondence_rejector_poly_->thresholdPolygon(source_indices, target_indices)) {
                    ++num_rejections;
                    continue;
                }

                Matrix4 transformation;
                // Estimate the transform from the correspondences, write to transformation_
                this->transformation_estimation_->estimateRigidTransformation(*(this->input_), source_indices,
                                                                              *(this->target_),
                                                                              target_indices, transformation);

                // Transform the input and compute the error (uses input_filtered_)
                this->getRMSE(inliers, correspondences_ij, transformation, error);

                // If the new fit is better, update results
                inlier_fraction = static_cast<float>(inliers.size()) / static_cast<float>(correspondences_ij.size());

                if (inlier_fraction >= max_inlier_fraction_local) {
                    min_error_local = error;
                    max_inlier_fraction_local = inlier_fraction;
                    best_inliers = inliers;
                    best_transformation = transformation;
                }
            } // for
#pragma omp critical(registration_result)
            {
                if (max_inlier_fraction < max_inlier_fraction_local) {
                    this->inliers_ = best_inliers;
                    this->rmse_ = min_error_local;
                    max_inlier_fraction = max_inlier_fraction_local;
                    this->converged_ = true;
                    this->final_transformation_ = best_transformation;
                }
            }
        }
    }

    // Apply the final transformation
    if (this->converged_)
        transformPointCloud(*(this->input_), output, this->final_transformation_);

    // Debug output
    PCL_DEBUG("[%s::computeTransformation] Rejected %i out of %i generated pose "
              "hypotheses.\n",
              this->getClassName().c_str(),
              num_rejections,
              this->max_iterations_);
}

#endif