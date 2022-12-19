#ifndef REGISTRATION_SAC_PREREJECTIVE_OMP_H
#define REGISTRATION_SAC_PREREJECTIVE_OMP_H

#include <pcl/registration/correspondence_rejection_poly.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include "metric.h"

#ifdef _OPENMP

#include <omp.h>

#endif

#if defined _OPENMP && _OPENMP >= 201107 // We need OpenMP 3.1 for the atomic constructs
#define OPENMP_AVAILABLE_RANSAC_PREREJECTIVE true
#else
#define OPENMP_AVAILABLE_RANSAC_PREREJECTIVE false
#endif

class SampleConsensusPrerejectiveOMP {
public:
    using CorrespondenceRejectorPoly = pcl::registration::CorrespondenceRejectorPoly<PointN, PointN>;
    using TransformationEstimation = pcl::registration::TransformationEstimationSVD<PointN, PointN>;

    SampleConsensusPrerejectiveOMP() = delete;

    SampleConsensusPrerejectiveOMP(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                                   PointNCloud::ConstPtr kps_src, PointNCloud::ConstPtr kps_tgt,
                                   CorrespondencesConstPtr correspondences, AlignmentParameters parameters);

    AlignmentResult align();

    inline std::string getClassName() const {
        return "SampleConsensusPrerejectiveOMP";
    }

protected:
    PointNCloud::ConstPtr src_, tgt_;
    PointNCloud::ConstPtr kps_src_, kps_tgt_;
    CorrespondencesConstPtr correspondences_;
    AlignmentParameters parameters_;

    CorrespondenceRejectorPoly correspondence_rejector_poly_{};
    TransformationEstimation transformation_estimation_{};
    MetricEstimator::Ptr metric_estimator_{nullptr};

    void buildIndices(const pcl::Indices &sample_indices,
                      pcl::Indices &source_indices,
                      pcl::Indices &target_indices) const;

    void selectCorrespondences(int nr_correspondences,
                               int nr_samples,
                               pcl::Indices &sample_indices,
                               UniformRandIntGenerator &rand_generator) const;

    unsigned int getNumberOfThreads() const;
};

#endif
