#include <pcl/common/time.h>
#include <unordered_set>

#include "sac_prerejective_omp.h"
#include "transformation.h"
#include "analysis.h"

#define MIN_NR_INLIERS 10
#define MIN_NR_FINAL_INLIERS 20
#define MIN_INLIER_RATE 0.15
#define SAVE_MULTIPLE_HYPOTHESES false

#if SAVE_MULTIPLE_HYPOTHESES
#include "hypotheses.h"
#endif

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
        sample_indices[i] = rand_generator() % nr_correspondences;

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
                                                               CorrespondencesConstPtr correspondences,
                                                               AlignmentParameters parameters)
        : src_(std::move(src)), tgt_(std::move(tgt)),
          correspondences_(std::move(correspondences)),
          parameters_(std::move(parameters)) {
    // Some sanity checks first
    if (parameters_.inlier_fraction < 0.0f || parameters_.inlier_fraction > 1.0f) {
        PCL_ERROR("[%s::%s] ", this->getClassName().c_str(), this->getClassName().c_str());
        PCL_ERROR("Illegal inlier fraction %f, must be in [0,1]!\n", parameters_.inlier_fraction);
        return;
    }

    if (parameters_.randomness <= 0) {
        PCL_ERROR("[%s::computeTransformation] ", this->getClassName().c_str());
        PCL_ERROR("Illegal correspondence randomness %d, must be > 0!\n", parameters_.randomness);
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
}

AlignmentResult SampleConsensusPrerejectiveOMP::align() {
    pcl::ScopeTime t("RANSAC");
    int num_rejections = 0; // For debugging
    int ransac_iterations = 0;

    // Initialize results
    Correspondences largest_inlier_set;
#if SAVE_MULTIPLE_HYPOTHESES
    std::vector<Eigen::Matrix4f> final_tns;
    std::vector<float> final_metrics;
#else
    Eigen::Matrix4f final_tn = Eigen::Matrix4f::Identity();
    float final_metric = 0;
#endif

    int max_iterations = std::min(calculateCombinationOrMax<int>(correspondences_->size(), parameters_.n_samples),
                                  parameters_.max_iterations);
    int estimated_iters = max_iterations; // For debugging

    // If guess is available we check it
    if (parameters_.guess.has_value()) {
        Correspondences inliers;
        float metric, error;
        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
        Eigen::Matrix4f guess = parameters_.guess.value();
        metric_estimator_->buildInliersAndEstimateMetric(guess, inliers, error, metric, rand);
        largest_inlier_set = inliers;
#if SAVE_MULTIPLE_HYPOTHESES
        updateHypotheses(final_tns, final_metrics, guess, metric, parameters_);
#else
        final_tn = guess;
        final_metric = metric;
#endif
    }

    // Start
    int threads = getNumberOfThreads();
#if SAVE_MULTIPLE_HYPOTHESES
    std::vector<std::vector<Eigen::Matrix4f>> tns_local_all(threads);
    std::vector<std::vector<float>> metrics_local_all(threads);
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
#pragma omp parallel \
    num_threads(threads) \
    default(none) \
    firstprivate(max_iterations) \
    reduction(+:num_rejections, ransac_iterations) \
    shared(tns_local_all, metrics_local_all, estimated_iters, largest_inlier_set)
#endif
#else
#if OPENMP_AVAILABLE_RANSAC_PREREJECTIVE
#pragma omp parallel \
    num_threads(threads) \
    default(none) \
    firstprivate(max_iterations) \
    reduction(+:num_rejections, ransac_iterations) \
    shared(final_tn, final_metric, estimated_iters, largest_inlier_set)
#endif
#endif
    {
        int omp_num_threads = omp_get_num_threads();

        // Local best results
        Correspondences largest_inlier_set_local;
#if SAVE_MULTIPLE_HYPOTHESES
        std::vector<Eigen::Matrix4f> tns_local;
        std::vector<float> metrics_local;
#else
        Eigen::Matrix4f best_tn_local = Eigen::Matrix4f::Identity();
        float best_metric_local = metric_estimator_->getInitialMetric();
#endif
        int iters_local = max_iterations;

        // Temporaries
        Correspondences inliers;
        float metric, error;
        Eigen::Matrix4f tn;

        unsigned long seed = parameters_.fix_seed ? SEED + omp_get_thread_num() : std::random_device{}();
        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), seed);

#pragma omp for
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
            transformation_estimation_.estimateRigidTransformation(*src_, source_indices, *tgt_, target_indices, tn);
            // If the new fit is better, update hypotheses, re-estimate number of required iterations
            metric_estimator_->buildInliersAndEstimateMetric(tn, inliers, error, metric, rand);
            if (inliers.size() < MIN_NR_INLIERS) continue;
            if (largest_inlier_set_local.size() < inliers.size()) {
                largest_inlier_set_local = inliers;
                iters_local = std::min(metric_estimator_->estimateMaxIterations(tn, parameters_.confidence,
                                                                                parameters_.n_samples), iters_local);
            }
#if SAVE_MULTIPLE_HYPOTHESES
            updateHypotheses(tns_local, metrics_local, tn, metric, parameters_);
#else
            if (best_metric_local < metric) {
                best_tn_local = tn;
                best_metric_local = metric;
            }
#endif
        } // for
#if SAVE_MULTIPLE_HYPOTHESES
        tns_local_all[omp_get_thread_num()] = tns_local;
        metrics_local_all[omp_get_thread_num()] = metrics_local;
#endif
#pragma omp critical(registration_result)
        {
            if (largest_inlier_set.size() < largest_inlier_set_local.size()) {
                largest_inlier_set = largest_inlier_set_local;
            }
            if (iters_local < estimated_iters) {
                estimated_iters = iters_local;
            }
#if !SAVE_MULTIPLE_HYPOTHESES
            if (final_metric < best_metric_local) {
                final_metric = best_metric_local;
                final_tn = best_tn_local;
            }
#endif
        }
    }
#if SAVE_MULTIPLE_HYPOTHESES
    for (int i = 0; i < tns_local_all.size(); ++i) {
        for (int j = 0; j < tns_local_all[i].size(); ++j) {
            updateHypotheses(final_tns, final_metrics, tns_local_all[i][j], metrics_local_all[i][j], parameters_);
        }
    }
#endif
    bool converged = false;
#if !SAVE_MULTIPLE_HYPOTHESES
    std::vector<Eigen::Matrix4f> final_tns = {final_tn};
    std::vector<float> final_metrics = {final_metric};
#endif
    for (int i = 0; i < final_tns.size(); ++i) {
        Eigen::Matrix4f tn;
        Correspondences inliers;
        float metric, error;
        UniformRandIntGenerator rand(0, std::numeric_limits<int>::max(), SEED);
        // build inliers for best tn and then re-estimate optimal tn using these inliers
        metric_estimator_->buildInliersAndEstimateMetric(final_tns[i], inliers, error, metric, rand);
        bool enough_inliers = inliers.size() > MIN_NR_FINAL_INLIERS ||
                              (float) inliers.size() > MIN_INLIER_RATE * (float) correspondences_->size();
        if (enough_inliers && metric > metric_estimator_->getMinTolerableMetric()) {
            converged = true;
        }
        estimateOptimalRigidTransformation(src_, tgt_, inliers, tn);
        metric_estimator_->buildInliersAndEstimateMetric(tn, inliers, error, metric, rand);
        if (metric < final_metrics[i]) {
            PCL_WARN("[%s::computeTransformation] number of inliers decreased "
                     "after estimating optimal rigid transformation.\n",
                     this->getClassName().c_str());
        }
        final_tns[i] = tn;
        final_metrics[i] = metric;
    }
#if SAVE_MULTIPLE_HYPOTHESES
    Eigen::Matrix4f final_tn = chooseBestHypothesis(src_, tgt_, correspondences_, parameters_, final_tns);
#else
    final_tn = final_tns[0];
#endif
    // Debug output
    if (parameters_.ground_truth.has_value()) {
        Correspondences correct_inliers;
        metric_estimator_->buildCorrectInliers(largest_inlier_set, correct_inliers, parameters_.ground_truth.value());
        PCL_DEBUG("[%s::computeTransformation] %i/%i correct inliers in largest set of inliers\n",
                  this->getClassName().c_str(), correct_inliers.size(), largest_inlier_set.size());
    } else {
        PCL_DEBUG("[%s::computeTransformation] %i inliers in largest set of inliers\n",
                  this->getClassName().c_str(), largest_inlier_set.size());
    }
    PCL_DEBUG("[%s::computeTransformation] RANSAC exits at %i-th iteration: "
              "rejected %i out of %i generated pose hypotheses.\n",
              this->getClassName().c_str(), ransac_iterations, num_rejections, ransac_iterations);
    PCL_DEBUG("[%s::computeTransformation] Minimum of estimated iterations: %i\n",
              this->getClassName().c_str(), estimated_iters);

    return AlignmentResult{src_, tgt_, final_tn, correspondences_, ransac_iterations, converged, t.getTimeSeconds()};
}