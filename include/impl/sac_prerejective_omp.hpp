#ifndef REGISTRATION_SAC_PREREJECTIVE_OMP_HPP
#define REGISTRATION_SAC_PREREJECTIVE_OMP_HPP

template<typename FeatureT>
void SampleConsensusPrerejectiveOMP<FeatureT>::setNumberOfThreads(unsigned int nr_threads) {
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

template<typename FeatureT>
void SampleConsensusPrerejectiveOMP<FeatureT>::buildIndices(const pcl::Indices &sample_indices,
                                                            pcl::Indices &source_indices,
                                                            pcl::Indices &target_indices) {
    // Allocate results
    source_indices.resize(sample_indices.size());
    target_indices.resize(sample_indices.size());

    // Loop over the sampled features
    for (std::size_t j = 0; j < sample_indices.size(); ++j) {
        // Current correspondence index
        const auto &idx = sample_indices[j];
        source_indices[j] = this->correspondences_->operator[](idx).index_query;
        target_indices[j] = this->correspondences_->operator[](idx).index_match;
    }
}

template<typename FeatureT>
const pcl::Indices &SampleConsensusPrerejectiveOMP<FeatureT>::getInliers() const {
    if (this->inliers_.empty()) {
        const auto& inlier_pairs = getInlierPairs();
        this->inliers_.clear();
        this->inliers.reserve(inlier_pairs.size());
        for (const auto &ip: inlier_pairs) {
            this->inliers_.push_back(ip.idx_src);
        }
    }
    return this->inliers_;
}

template<typename FeatureT>
void SampleConsensusPrerejectiveOMP<FeatureT>::readCorrespondences(const AlignmentParameters &parameters) {
    std::string filepath = constructPath(parameters, "correspondences", "csv", true, false, false);
    readCorrespondencesFromCSV(filepath, *(this->correspondences_), correspondence_ids_from_file);
}

template<typename FeatureT>
void SampleConsensusPrerejectiveOMP<FeatureT>::saveCorrespondences(const AlignmentParameters &parameters) {
    std::string filepath = constructPath(parameters, "correspondences", "csv", true, false, false);
    saveCorrespondencesToCSV(filepath, *(this->correspondences_));
}

template<typename FeatureT>
AlignmentAnalysis SampleConsensusPrerejectiveOMP<FeatureT>::getAlignmentAnalysis(
        const AlignmentParameters &parameters
) const {
    if (this->hasConverged()) {
        return AlignmentAnalysis(parameters, this->metric_estimator_, this->input_, this->target_,
                                 this->inlier_pairs_, *(this->correspondences_),
                                 this->getRMSEScore(), this->ransac_iterations_, this->final_transformation_);
    } else {
        pcl::console::print_error("Alignment failed!\n");
        return {};
    }
}

template<typename FeatureT>
void SampleConsensusPrerejectiveOMP<FeatureT>::setConfidence(float confidence) {
    if (confidence > 0.0 && confidence < 1.0) {
        confidence_ = confidence;
    } else {
        PCL_ERROR("The confidence must be greater than 0.0 and less than 1.0!\n");
    }
}

template<typename FeatureT>
void SampleConsensusPrerejectiveOMP<FeatureT>::selectCorrespondences(int nr_correspondences, int nr_samples,
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
                if (sample_indices[i] < nr_correspondences - 1) {
                    sample_indices[i]++;
                    continue;
                } else if (sample_indices[j] == 0) {
                    sample_indices[i] = 1;
                    continue;
                } else {
                    sample_indices[i] = 0;
                }
            }
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

template<typename FeatureT>
unsigned int SampleConsensusPrerejectiveOMP<FeatureT>::getNumberOfThreads() {
    unsigned int threads = threads_;
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
    return threads;
}

template<typename FeatureT>
void SampleConsensusPrerejectiveOMP<FeatureT>::computeTransformation(PointNCloud &output,
                                                                     const Eigen::Matrix4f &guess) {
    // Some sanity checks first
    if (!this->input_features_) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("No source features were given! Call setSourceFeatures before aligning.\n");
        return;
    }
    if (!this->target_features_) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("No target features were given! Call setTargetFeatures before aligning.\n");
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

    const float similarity_threshold = this->correspondence_rejector_poly_->getSimilarityThreshold();
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

    if (!this->metric_estimator_) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("No metric estimator was given! Call setMetricEstimator before aligning.\n");
        return;
    }

    // Initialize search trees
    if (this->target_cloud_updated_ && !this->force_no_recompute_) {
        this->tree_->setInputCloud(this->target_);
        this->target_cloud_updated_ = false;
    }
    if (this->source_cloud_updated_ && !this->force_no_recompute_) {
        this->tree_reciprocal_->setInputCloud(this->input_);
        this->source_cloud_updated_ = false;
    }

    // Initialize prerejector (similarity threshold already set to default value in
    // constructor)
    this->correspondence_rejector_poly_->setInputSource(this->input_);
    this->correspondence_rejector_poly_->setInputTarget(this->target_);
    this->correspondence_rejector_poly_->setCardinality(this->nr_samples_);
    int num_rejections = 0; // For debugging
    int ransac_iterations = 0;

    // Initialize results
    this->final_transformation_ = guess;
    this->inlier_pairs_.clear();
    rmse_ = std::numeric_limits<float>::max();
    float best_metric = metric_estimator_->getInitialMetric();
    this->converged_ = false;

    if (correspondence_ids_from_file) {
        PCL_DEBUG("[%s::computeTransformation] read correspondences from file\n", this->getClassName().c_str());
    } else {
        pcl::ScopeTime t("Correspondence search");
        *(this->correspondences_) = feature_matcher_->match(this->input_features_, this->target_features_,
                                                            this->tree_reciprocal_, this->tree_,
                                                            point_representation_, getNumberOfThreads());
    }

    // Initialize metric estimator
    metric_estimator_->setSourceCloud(this->input_);
    metric_estimator_->setTargetCloud(this->target_);
    metric_estimator_->setCorrespondences(*(this->correspondences_));
    metric_estimator_->setInlierThreshold(this->corr_dist_threshold_);

    this->max_iterations_ = std::min(
            calculate_combination_or_max<int>(this->correspondences_->size(), this->nr_samples_), this->max_iterations_);
    int estimated_iters = this->max_iterations_; // For debugging
    {
        std::vector<InlierPair> inlier_pairs;
        float metric, error;

        // If guess is not the Identity matrix we check it
        if (!guess.isApprox(Eigen::Matrix4f::Identity(), 0.01f)) {
            metric_estimator_->buildInlierPairs(guess, inlier_pairs, error);
            metric_estimator_->estimateMetric(inlier_pairs, metric);

            if (metric_estimator_->isBetter(metric, best_metric)) {
                this->inlier_pairs_ = inlier_pairs;
                rmse_ = error;
                best_metric = metric;
                this->converged_ = true;
                this->final_transformation_ = guess;
            }
        }
    }

    // Start
    {
        pcl::ScopeTime t("RANSAC");
        int threads = getNumberOfThreads();
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
#pragma omp parallel \
    num_threads(threads) \
    default(none) \
    shared(best_metric, estimated_iters) \
    reduction(+:num_rejections, ransac_iterations)
#endif
        {
            // Local best results
            std::vector<InlierPair> best_inlier_pairs;
            float min_error_local = std::numeric_limits<float>::max();
            float best_metric_local = metric_estimator_->getInitialMetric();
            int iters_local = this->max_iterations_;
            Matrix4 best_transformation;

            // Temporaries
            std::vector<InlierPair> inlier_pairs;
            float metric, error;

            UniformRandIntGenerator rand_generator(0, (int) this->correspondences_->size() - 1);

#pragma omp for nowait
            for (int i = 0; i < this->max_iterations_; ++i) {
                if (i >= iters_local) {
                    continue;
                }
                ++ransac_iterations;

                // Temporary containers
                pcl::Indices sample_indices;
                pcl::Indices source_indices;
                pcl::Indices target_indices;

                // Draw nr_samples_ random samples
                selectCorrespondences(this->correspondences_->size(), this->nr_samples_, sample_indices,
                                      rand_generator);

                // Find corresponding features in the target cloud
                buildIndices(sample_indices, source_indices, target_indices);

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
                // If the new fit is better, update results
                metric_estimator_->buildInlierPairs(transformation, inlier_pairs, error);
                metric_estimator_->estimateMetric(inlier_pairs, metric);

                if (metric_estimator_->isBetter(metric, best_metric_local)) {
                    min_error_local = error;
                    best_metric_local = metric;
                    best_inlier_pairs = inlier_pairs;
                    best_transformation = transformation;
                    iters_local = std::min(metric_estimator_->estimateMaxIterations(transformation, confidence_,
                                                                                    this->nr_samples_), iters_local);
                }
            } // for
#pragma omp critical(registration_result)
            {
                if (metric_estimator_->isBetter(best_metric_local, best_metric)) {
                    this->inlier_pairs_ = best_inlier_pairs;
                    rmse_ = min_error_local;
                    best_metric = best_metric_local;
                    this->converged_ = true;
                    this->final_transformation_ = best_transformation;
                }
                if (iters_local < estimated_iters) {
                    estimated_iters = iters_local;
                }
            }
        }
    }
    this->ransac_iterations_ = ransac_iterations;

    // Apply the final transformation
    if (this->converged_)
        transformPointCloud(*(this->input_), output, this->final_transformation_);

    // Debug output
    PCL_DEBUG("[%s::computeTransformation] RANSAC exits at %i-th iteration: "
              "rejected %i out of %i generated pose hypotheses.\n",
              this->getClassName().c_str(),
              this->ransac_iterations_,
              num_rejections,
              this->ransac_iterations_);
    PCL_DEBUG("[%s::computeTransformation] Minimum of estimated iterations: %i\n",
              this->getClassName().c_str(),
              estimated_iters);
}

#endif