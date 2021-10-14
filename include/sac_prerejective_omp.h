#ifndef REGISTRATION_SAC_PREREJECTIVE_OMP_H
#define REGISTRATION_SAC_PREREJECTIVE_OMP_H

#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/common/norms.h>

#ifdef _OPENMP

#include <omp.h>

#endif

#if defined _OPENMP && _OPENMP >= 201107 // We need OpenMP 3.1 for the atomic constructs
#define OPENMP_AVAILABLE_RANSAC_PREREJECTIVE true
#else
#define OPENMP_AVAILABLE_RANSAC_PREREJECTIVE false
#endif

struct MultivaluedCorrespondence {
    int query_idx;
    pcl::Indices match_indices;
};

template<typename PointSource, typename PointTarget, typename FeatureT>
class SampleConsensusPrerejectiveOMP : public pcl::SampleConsensusPrerejective<PointSource, PointTarget, FeatureT> {
public:
    using Matrix4 = typename pcl::Registration<PointSource, PointTarget>::Matrix4;
    using PointCloudSource = typename pcl::Registration<PointSource, PointTarget>::PointCloudSource;

    SampleConsensusPrerejectiveOMP() {
        this->reg_name_ = "SampleConsensusPrerejectiveOMP";
        setNumberOfThreads(0);
    }

    void enableMutualFiltering() {
        reciprocal_ = true;
    }

    float getRMSEScore();

    inline const std::vector<MultivaluedCorrespondence> getCorrespondences() const
    {
        return multivalued_correspondences_;
    }

    int countCorrectCorrespondences(const Eigen::Matrix4f &transformation_gt, float error_threshold);

protected:
    void computeTransformation(PointCloudSource &output, const Eigen::Matrix4f &guess) override;

    void buildIndices(const pcl::Indices &sample_indices,
                      std::vector<MultivaluedCorrespondence> &correspondences,
                      pcl::Indices &source_indices,
                      pcl::Indices &target_indices);

    void selectCorrespondences(int nr_correspondences,
                               int nr_samples,
                               pcl::Indices &sample_indices);

    void setNumberOfThreads(unsigned int nr_threads = 0);

    void getRMSE(pcl::Indices &inliers, const std::vector<MultivaluedCorrespondence> &correspondences, float &rmse_score);

    bool reciprocal_ = false;
    float rmse_ = std::numeric_limits<float>::max();
    unsigned int threads_{};

    std::vector<MultivaluedCorrespondence> multivalued_correspondences_;
};

#include "impl/sac_prerejective_omp.hpp"

#endif
