#ifndef REGISTRATION_SAC_PREREJECTIVE_OMP_HPP
#define REGISTRATION_SAC_PREREJECTIVE_OMP_HPP

#include <opencv2/features2d.hpp>

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
        pcl::Indices &source_indices,
        pcl::Indices &target_indices) {
    // Allocate results
    source_indices.resize(sample_indices.size());
    target_indices.resize(sample_indices.size());

    // Loop over the sampled features
    for (std::size_t j = 0; j < sample_indices.size(); ++j) {
        // Current correspondence index
        const auto &idx = sample_indices[j];
        source_indices[j] = multivalued_correspondences_[idx].query_idx;

        // Select one at random and add it to target_indices
        if (this->k_correspondences_ == 1)
            target_indices[j] = multivalued_correspondences_[idx].match_indices[0];
        else
            target_indices[j] = multivalued_correspondences_[idx].match_indices[this->getRandomIndex(
                    this->k_correspondences_)];
    }
}

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::getRMSE(pcl::Indices &inliers,
                                                                                 const Matrix4 &transformation,
                                                                                 float &rmse_score) {
    // Initialize variables
    inliers.clear();
    inliers.reserve(multivalued_correspondences_.size());
    rmse_score = 0.0f;

    // Transform the input dataset using the final transformation
    PointCloudSource input_transformed;
    input_transformed.resize(this->input_->size());
    transformPointCloud(*(this->input_), input_transformed, transformation);

    // For each point from correspondences in the source dataset
    for (int i = 0; i < multivalued_correspondences_.size(); ++i) {
        int query_idx = multivalued_correspondences_[i].query_idx;
        int match_idx = multivalued_correspondences_[i].match_indices[0];
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
AlignmentAnalysis SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::getAlignmentAnalysis(
        const AlignmentParameters &parameters
) const {
    if (this->hasConverged()) {
        return AlignmentAnalysis(parameters, this->input_, this->target_,
                                 this->inliers_, this->multivalued_correspondences_,
                                 this->getRMSEScore(), this->ransac_iterations_, this->final_transformation_);
    } else {
        pcl::console::print_error("Alignment failed!\n");
        return {};
    }
}

template<typename PointSource, typename PointTarget, typename FeatureT>
int SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::estimateMaxIterations(float inlier_fraction) {
    if (inlier_fraction <= 0.0) {
        return std::numeric_limits<int>::max();
    }
    double iterations = std::log(1.0 - confidence_) / std::log(1.0 - std::pow(inlier_fraction, this->nr_samples_));
    return static_cast<int>(std::min((double) std::numeric_limits<int>::max(), iterations));
}

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::setConfidence(float confidence) {
    if (confidence > 0.0 && confidence < 1.0) {
        confidence_ = confidence;
    } else {
        PCL_ERROR("The confidence must be greater than 0.0 and less than 1.0!\n");
    }
}

template<typename PointSource, typename PointTarget, typename FeatureT>
float SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::getRMSEScore() const {
    return rmse_;
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

template<typename PointSource, typename PointTarget, typename FeatureT>
unsigned int SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::getNumberOfThreads() {
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

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::findCorrespondencesFlann() {
    unsigned int threads = getNumberOfThreads();

    // Feature correspondence set
    std::vector<MultivaluedCorrespondence> correspondences_ij(this->input_->size());

    float dists_sum = 0.f;
#pragma omp parallel for num_threads(threads) default(none) shared(correspondences_ij) reduction(+:dists_sum)
    for (int i = 0; i < this->input_->size(); i++) {
        if (point_representation_->isValid(this->input_features_->points[i])) {
            correspondences_ij[i].query_idx = i;
            pcl::Indices &match_indices = correspondences_ij[i].match_indices;
            match_indices.resize(this->k_correspondences_);
            std::vector<float> match_distances(this->k_correspondences_);
            this->feature_tree_->nearestKSearch(*(this->input_features_),
                                                i,
                                                this->k_correspondences_,
                                                match_indices,
                                                match_distances);
            dists_sum += match_distances[0];
        }
    }
    PCL_DEBUG("[%s::computeTransformation] average distance to nearest neighbour: %0.7f.\n",
              this->getClassName().c_str(),
              dists_sum / (float) this->input_->size());

    if (reciprocal_) {
        pcl::KdTreeFLANN<FeatureT> feature_tree_src_(new pcl::KdTreeFLANN<FeatureT>);
        feature_tree_src_.setInputCloud(this->input_features_);

        std::vector<MultivaluedCorrespondence> correspondences_ji(this->target_->size());

#pragma omp parallel for num_threads(threads) default(none) shared(correspondences_ji, feature_tree_src_)
        for (int j = 0; j < this->target_->size(); ++j) {
            if (point_representation_->isValid(this->target_features_->points[j])) {
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
        }

        std::vector<MultivaluedCorrespondence> correspondences_mutual;
        for (int i = 0; i < this->input_->size(); ++i) {
            bool reciprocal = false;
            MultivaluedCorrespondence corr = correspondences_ij[i];
            for (const int &j: corr.match_indices) {
                if (!correspondences_ji[j].match_indices.empty() && correspondences_ji[j].match_indices[0] == i) {
                    reciprocal = true;
                }
            }
            if (reciprocal) {
                correspondences_mutual.emplace_back(corr);
            }
        }
        correspondences_ij = correspondences_mutual;
        PCL_DEBUG("[%s::findCorrespondencesFlann] %i correspondences remain after mutual filter.\n",
                  this->getClassName().c_str(),
                  correspondences_mutual.size());
    }
    for (auto corr: correspondences_ij) {
        if (corr.query_idx >= 0) {
            multivalued_correspondences_.emplace_back(corr);
        }
    }
}

template<typename PointSource, typename PointTarget, typename FeatureT>
void SampleConsensusPrerejectiveOMP<PointSource, PointTarget, FeatureT>::findCorrespondencesBF() {
    // TODO: support k > 1, refactor correspondence search
    assert(this->k_correspondences_ == 1);
    cv::UMat features_src, features_tgt;
    int nr_dims = point_representation_->getNumberOfDimensions();
    pcl2cv<FeatureT>(nr_dims, this->input_features_, features_src);
    pcl2cv<FeatureT>(nr_dims, this->target_features_, features_tgt);
    auto matcher = cv::BFMatcher::create(cv::NORM_L2, reciprocal_);
    std::vector<cv::DMatch> matches;
    matcher->match(features_src, features_tgt, matches);

    std::vector<MultivaluedCorrespondence> correspondences_ij;
    for (int i = 0; i < this->input_->size(); ++i) {
        correspondences_ij.push_back(MultivaluedCorrespondence{matches[i].queryIdx, {matches[i].trainIdx}});
    }
    if (reciprocal_) {
        matches.clear();
        matcher->match(features_tgt, features_src, matches);
        std::vector<MultivaluedCorrespondence> correspondences_ji;
        for (int i = 0; i < this->target_->size(); ++i) {
            correspondences_ji.push_back(MultivaluedCorrespondence{matches[i].queryIdx, {matches[i].trainIdx}});
        }

        std::vector<MultivaluedCorrespondence> correspondences_mutual;
        for (int i = 0; i < this->input_->size(); ++i) {
            bool reciprocal = false;
            MultivaluedCorrespondence corr = correspondences_ij[i];
            for (const int &j: corr.match_indices) {
                if (!correspondences_ji[j].match_indices.empty() && correspondences_ji[j].match_indices[0] == i) {
                    reciprocal = true;
                }
            }
            if (reciprocal) {
                correspondences_mutual.emplace_back(corr);
            }
        }
        correspondences_ij = correspondences_mutual;
        PCL_DEBUG("[%s::findCorrespondencesBF] %i correspondences remain after mutual filter.\n",
                  this->getClassName().c_str(),
                  correspondences_mutual.size());
    }
    for (auto corr: correspondences_ij) {
        if (corr.query_idx >= 0) {
            multivalued_correspondences_.emplace_back(corr);
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
    int ransac_iterations = 0;

    // Initialize results
    this->final_transformation_ = guess;
    this->inliers_.clear();
    this->rmse_ = std::numeric_limits<float>::max();
    float max_inlier_fraction = 0.0f;
    this->converged_ = false;

    // Build correspondences
    {
        pcl::ScopeTime t("Correspondence search");
        if (use_bfmatcher_) {
            findCorrespondencesBF();
        } else {
            findCorrespondencesFlann();
        }
    }
    this->max_iterations_ = std::min(
            calculate_combination_or_max<int>(multivalued_correspondences_.size(), this->nr_samples_),
            this->max_iterations_);
    int estimated_iters = this->max_iterations_; // For debugging
    {
        pcl::Indices inliers;
        float inlier_fraction, error;

        // If guess is not the Identity matrix we check it
        if (!guess.isApprox(Eigen::Matrix4f::Identity(), 0.01f)) {
            this->getRMSE(inliers, guess, error);
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
        int threads = getNumberOfThreads();
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
#pragma omp parallel \
    num_threads(threads) \
    default(none) \
    shared(max_inlier_fraction, estimated_iters) \
    reduction(+:num_rejections, ransac_iterations)
#endif
        {
            // Local best results
            pcl::Indices best_inliers;
            float min_error_local = std::numeric_limits<float>::max();
            float max_inlier_fraction_local = 0.0f;
            int estimated_iters_local = this->max_iterations_;
            Matrix4 best_transformation;

            // Temporaries
            pcl::Indices inliers;
            float inlier_fraction, error;

            UniformRandIntGenerator rand_generator(0, (int) multivalued_correspondences_.size() - 1);

#pragma omp for nowait
            for (int i = 0; i < this->max_iterations_; ++i) {
                if (i >= estimated_iters_local) {
                    continue;
                }
                ++ransac_iterations;

                // Temporary containers
                pcl::Indices sample_indices;
                pcl::Indices source_indices;
                pcl::Indices target_indices;

                // Draw nr_samples_ random samples
                selectCorrespondences(multivalued_correspondences_.size(), this->nr_samples_, sample_indices,
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

                // Transform the input and compute the error (uses input_filtered_)
                this->getRMSE(inliers, transformation, error);

                // If the new fit is better, update results
                inlier_fraction = static_cast<float>(inliers.size()) /
                                  static_cast<float>(multivalued_correspondences_.size());

                if (inlier_fraction > max_inlier_fraction_local) {
                    min_error_local = error;
                    max_inlier_fraction_local = inlier_fraction;
                    best_inliers = inliers;
                    best_transformation = transformation;
                    estimated_iters_local = std::min(estimateMaxIterations(inlier_fraction), estimated_iters_local);
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
                if (estimated_iters_local < estimated_iters) {
                    estimated_iters = estimated_iters_local;
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