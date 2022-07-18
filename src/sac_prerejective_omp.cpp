#include <pcl/common/time.h>

#include "sac_prerejective_omp.h"
#include "transformation.h"

void SampleConsensusPrerejectiveOMP::buildIndices(const pcl::Indices &sample_indices,
                                                  pcl::Indices &source_indices,
                                                  pcl::Indices &target_indices) const {
    // Allocate results
    source_indices.resize(sample_indices.size());
    target_indices.resize(sample_indices.size());

    // Loop over the sampled features
    for (std::size_t j = 0; j < sample_indices.size(); ++j) {
        // Current correspondence index
        const auto &idx = sample_indices[j];
        source_indices[j] = correspondences_->operator[](idx).index_query;
        target_indices[j] = correspondences_->operator[](idx).index_match;
    }
}

void SampleConsensusPrerejectiveOMP::selectCorrespondences(int nr_correspondences, int nr_samples,
                                                           pcl::Indices &sample_indices,
                                                           UniformRandIntGenerator &rand_generator) const {
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

unsigned int SampleConsensusPrerejectiveOMP::getNumberOfThreads() const {
    unsigned int threads;
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
    threads = omp_get_num_procs();
    PCL_DEBUG("[%s::computeTransformation] Automatic number of threads requested, choosing %i threads.\n",
              this->getClassName().c_str(), threads);
#else
    // Parallelization desired, but not available
    PCL_WARN ("[SampleConsensusPrerejectiveOMP::computeTransformation] Parallelization is requested, but OpenMP 3.1 is not available! Continuing without parallelization.\n");
    threads = 1;
#endif
    return threads;
}

SampleConsensusPrerejectiveOMP::SampleConsensusPrerejectiveOMP(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                                                               pcl::CorrespondencesConstPtr correspondences,
                                                               AlignmentParameters parameters)
        : src_(std::move(src)), tgt_(std::move(tgt)),
          correspondences_(std::move(correspondences)),
          parameters_(std::move(parameters)) {
    // Some sanity checks first
    if (parameters_.inlier_fraction < 0.0f || parameters.inlier_fraction > 1.0f) {
        PCL_ERROR("[%s::%s] ", this->getClassName().c_str(), this->getClassName().c_str());
        PCL_ERROR("Illegal inlier fraction %f, must be in [0,1]!\n", parameters.inlier_fraction);
        return;
    }

    if (parameters.randomness <= 0) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("Illegal correspondence randomness %d, must be > 0!\n", parameters.randomness);
        return;
    }
    correspondence_rejector_poly_.setSimilarityThreshold(parameters_.edge_thr_coef);
    correspondence_rejector_poly_.setInputSource(src_);
    correspondence_rejector_poly_.setInputTarget(tgt_);
    correspondence_rejector_poly_.setCardinality(parameters.n_samples);
    metric_estimator_ = getMetricEstimatorFromParameters(parameters_, true);
    metric_estimator_->setSourceCloud(src_);
    metric_estimator_->setTargetCloud(tgt_);
    metric_estimator_->setCorrespondences(correspondences_);
    metric_estimator_->setInlierThreshold(parameters_.distance_thr_coef * parameters.voxel_size);
}

AlignmentResult SampleConsensusPrerejectiveOMP::align() {
    pcl::ScopeTime t("RANSAC");
    int num_rejections = 0; // For debugging
    int ransac_iterations = 0;

    // Initialize results
    Eigen::Matrix4f final_transformation = Eigen::Matrix4f::Identity();
    std::shared_ptr<std::vector<InlierPair>> final_inlier_pairs(new std::vector<InlierPair>);
    float final_rmse = std::numeric_limits<float>::max();
    float final_metric = metric_estimator_->getInitialMetric();
    bool converged = false;
    int max_iterations = std::min(calculate_combination_or_max<int>(correspondences_->size(), parameters_.n_samples),
                                  parameters_.max_iterations);
    int estimated_iters = max_iterations; // For debugging

    // If guess is available we check it
    if (parameters_.guess.has_value()) {
        std::vector<InlierPair> inlier_pairs;
        float metric, error;
        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
        Eigen::Matrix4f guess = parameters_.guess.value();
        metric_estimator_->buildInlierPairsAndEstimateMetric(guess, inlier_pairs, error, metric, rand);
        if (metric_estimator_->isBetter(metric, final_metric)) {
            *final_inlier_pairs = inlier_pairs;
            final_rmse = error;
            final_metric = metric;
            converged = true;
            final_transformation = parameters_.guess.value();
        }
    }

    // Start
    int threads = getNumberOfThreads();
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
#pragma omp parallel \
    num_threads(threads) \
    default(none) \
    firstprivate(max_iterations) \
    shared(final_transformation, final_inlier_pairs, final_rmse, final_metric, converged, estimated_iters) \
    reduction(+:num_rejections, ransac_iterations)
#endif
    {
        int omp_num_threads = omp_get_num_threads();

        // Local best results
        std::vector<InlierPair> best_inlier_pairs_local;
        float min_error_local = std::numeric_limits<float>::max();
        float best_metric_local = metric_estimator_->getInitialMetric();
        int iters_local = max_iterations;
        Eigen::Matrix4f best_transformation_local;

        // Temporaries
        std::vector<InlierPair> inlier_pairs;
        float metric, error;
        Eigen::Matrix4f transformation;

        int seed = parameters_.fix_seed ? SEED + omp_get_thread_num() : std::random_device{}();
        UniformRandIntGenerator rand(0, (int) this->correspondences_->size() - 1, seed);

#pragma omp for nowait
        for (int i = 0; i < max_iterations; ++i) {
            if (ransac_iterations * omp_num_threads >= iters_local) {
                continue;
            }
            ++ransac_iterations;

            // Temporary containers
            pcl::Indices sample_indices;
            pcl::Indices source_indices;
            pcl::Indices target_indices;

            // Draw nr_samples_ random samples
            selectCorrespondences(correspondences_->size(), parameters_.n_samples, sample_indices, rand);

            // Find corresponding features in the target cloud
            buildIndices(sample_indices, source_indices, target_indices);

            // Apply prerejection
            if (!correspondence_rejector_poly_.thresholdPolygon(source_indices, target_indices)) {
                ++num_rejections;
                continue;
            }


            // Estimate the transform from the correspondences, write to transformation_
            transformation_estimation_.estimateRigidTransformation(*src_, source_indices, *tgt_, target_indices,
                                                                   transformation);
            // If the new fit is better, update results
            metric_estimator_->buildInlierPairsAndEstimateMetric(transformation, inlier_pairs, error, metric, rand);
            if (metric_estimator_->isBetter(metric, best_metric_local)) {
                min_error_local = error;
                best_metric_local = metric;
                best_inlier_pairs_local = inlier_pairs;
                best_transformation_local = transformation;
                iters_local = std::min(metric_estimator_->estimateMaxIterations(
                        transformation, parameters_.confidence, parameters_.n_samples), iters_local);
            }
        } // for
#pragma omp critical(registration_result)
        {
            if (metric_estimator_->isBetter(best_metric_local, final_metric)) {
                *final_inlier_pairs = best_inlier_pairs_local;
                final_rmse = min_error_local;
                final_metric = best_metric_local;
                converged = true;
                final_transformation = best_transformation_local;
            }
            if (iters_local < estimated_iters) {
                estimated_iters = iters_local;
            }
        }
    }

    // Estimate optimal transformation using resulting inliers
    if (converged) {
        Eigen::Matrix4f transformation;
        std::vector<InlierPair> inlier_pairs;
        float metric, error;
        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
        estimateOptimalRigidTransformation(src_, tgt_, *final_inlier_pairs, transformation);
        metric_estimator_->buildInlierPairsAndEstimateMetric(transformation, inlier_pairs, error, metric, rand);
        if (metric_estimator_->isBetter(metric, final_metric)) {
            PCL_WARN("[%s::computeTransformation] number of inliers decreased "
                     "after estimating optimal rigid transformation.\n",
                     this->getClassName().c_str());
        }
        final_transformation = transformation;
        *final_inlier_pairs = inlier_pairs;
        final_rmse = error;
        final_metric = metric;
    }

    // Debug output
    PCL_DEBUG("[%s::computeTransformation] RANSAC exits at %i-th iteration: "
              "rejected %i out of %i generated pose hypotheses.\n",
              this->getClassName().c_str(), ransac_iterations, num_rejections, ransac_iterations);
    PCL_DEBUG("[%s::computeTransformation] Minimum of estimated iterations: %i\n",
              this->getClassName().c_str(), estimated_iters);
    return AlignmentResult{src_, tgt_, final_transformation, correspondences_, ransac_iterations, converged,
                           t.getTimeSeconds()};
}