#ifndef REGISTRATION_SAC_PREREJECTIVE_OMP_H
#define REGISTRATION_SAC_PREREJECTIVE_OMP_H

#include <filesystem>

#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/common/norms.h>
#include <pcl/common/time.h>
#include <pcl/point_representation.h>

#include "utils.h"
#include "common.h"
#include "analysis.h"
#include "matching.h"
#include "metric.h"

#ifdef _OPENMP

#include <omp.h>

#endif

#if defined _OPENMP && _OPENMP >= 201107 // We need OpenMP 3.1 for the atomic constructs
#define OPENMP_AVAILABLE_RANSAC_PREREJECTIVE true
#else
#define OPENMP_AVAILABLE_RANSAC_PREREJECTIVE false
#endif

template<typename FeatureT>
class SampleConsensusPrerejectiveOMP : public pcl::SampleConsensusPrerejective<PointTN, PointTN, FeatureT> {
public:
    using Matrix4 = typename pcl::Registration<PointTN, PointTN>::Matrix4;

    SampleConsensusPrerejectiveOMP() : point_representation_(new pcl::DefaultPointRepresentation<FeatureT>) {
        this->reg_name_ = "SampleConsensusPrerejectiveOMP";
        setNumberOfThreads(0);
    }

    void setConfidence(float confidence);

    inline void setMetricEstimator(const MetricEstimator::Ptr &metric_estimator) {
        metric_estimator_ = metric_estimator;
    }

    inline void setFeatureMatcher(const typename FeatureMatcher<FeatureT>::Ptr &feature_matcher) {
        feature_matcher_ = feature_matcher;
    }

    // TODO: fix
    inline void useBFMatcher() {
        use_bfmatcher_ = true;
    }

    inline void setBFBlockSize(int bf_block_size) {
        bf_block_size_ = bf_block_size;
    }

    inline float getRMSEScore() const {
        return rmse_;
    }

    inline const pcl::Correspondences &getCorrespondences() const {
        return this->correspondences_;
    }

    inline const std::vector<InlierPair> &getInlierPairs() const {
        return inlier_pairs_;
    }

    const pcl::Indices &getInliers() const;

    void readCorrespondences(const AlignmentParameters &parameters);

    void saveCorrespondences(const AlignmentParameters &parameters);


    AlignmentAnalysis getAlignmentAnalysis(const AlignmentParameters &parameters) const;

protected:
    void computeTransformation(PointCloudTN &output, const Eigen::Matrix4f &guess) override;

    void buildIndices(const pcl::Indices &sample_indices,
                      pcl::Indices &source_indices,
                      pcl::Indices &target_indices);

    void selectCorrespondences(int nr_correspondences,
                               int nr_samples,
                               pcl::Indices &sample_indices,
                               UniformRandIntGenerator &rand_generator);

    void setNumberOfThreads(unsigned int nr_threads = 0);

    std::vector<InlierPair> inlier_pairs_;
    bool correspondence_ids_from_file = false;
    bool use_bfmatcher_ = false;
    int bf_block_size_ = 10000;
    float rmse_ = std::numeric_limits<float>::max();
    float confidence_ = 0.999f;
    unsigned int threads_{};
    typename pcl::PointRepresentation<FeatureT>::Ptr point_representation_;
    MetricEstimator::Ptr metric_estimator_{nullptr};
    typename FeatureMatcher<FeatureT>::Ptr feature_matcher_{nullptr};

private:
    unsigned int getNumberOfThreads();
};

#include "impl/sac_prerejective_omp.hpp"

#endif
