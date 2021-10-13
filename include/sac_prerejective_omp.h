#ifndef REGISTRATION_SAC_PREREJECTIVE_OMP_H
#define REGISTRATION_SAC_PREREJECTIVE_OMP_H

#include <pcl/registration/sample_consensus_prerejective.h>

template <typename PointSource, typename PointTarget, typename FeatureT>
class SampleConsensusPrerejectiveOMP : public pcl::SampleConsensusPrerejective<PointSource, PointTarget, FeatureT> {
public:
    using Matrix4 = typename pcl::Registration<PointSource, PointTarget>::Matrix4;
    using PointCloudSource = typename pcl::Registration<PointSource, PointTarget>::PointCloudSource;
protected:
    void computeTransformation(PointCloudSource& output, const Eigen::Matrix4f& guess) override;
};

#include "impl/sac_prerejective_omp.hpp"

#endif
