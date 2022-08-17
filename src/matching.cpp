#include "matching.h"

void FeatureBasedMatcher::printDebugInfo(const std::vector<MultivaluedCorrespondence> &mv_correspondences) {
    float dists_sum = 0.f;
    int n_dists = 0;
    for (int i = 0; i < mv_correspondences.size(); i++) {
        if (!mv_correspondences[i].match_indices.empty()) {
            dists_sum += mv_correspondences[i].distances[0];
            n_dists++;
        }
    }
    if (n_dists == 0) {
        PCL_ERROR("[%s::match] no distances were calculated.\n", getClassName().c_str());
    } else {
        average_distance_ = dists_sum / (float) n_dists;
        PCL_DEBUG("[%s::match] average distance to nearest neighbour: %0.7f.\n",
                  getClassName().c_str(), average_distance_);
    }
}

FeatureBasedMatcher::Ptr getFeatureBasedMatcherFromParameters(const PointNCloud::ConstPtr &src,
                                                              const PointNCloud::ConstPtr &tgt,
                                                              const pcl::IndicesConstPtr &indices_src,
                                                              const pcl::IndicesConstPtr &indices_tgt,
                                                              const AlignmentParameters &parameters) {
    if (parameters.matching_id == MATCHING_RATIO) {
        PCL_ERROR("Ratio filtering currently isn't supported!\n");
        if (parameters.descriptor_id == DESCRIPTOR_FPFH) {
            auto ptr = std::make_shared<RatioMatcher<FPFH>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters.descriptor_id == DESCRIPTOR_ROPS) {
            auto ptr = std::make_shared<RatioMatcher<RoPS135>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters.descriptor_id == DESCRIPTOR_SHOT) {
            auto ptr = std::make_shared<RatioMatcher<SHOT>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else pcl::console::print_error("Descriptor %s isn't supported!\n", parameters.descriptor_id.c_str());
    } else if (parameters.matching_id == MATCHING_CLUSTER) {
        if (parameters.descriptor_id == DESCRIPTOR_FPFH) {
            auto ptr = std::make_shared<ClusterMatcher<FPFH>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters.descriptor_id == DESCRIPTOR_ROPS) {
            auto ptr = std::make_shared<ClusterMatcher<RoPS135>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters.descriptor_id == DESCRIPTOR_SHOT) {
            auto ptr = std::make_shared<ClusterMatcher<SHOT>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else pcl::console::print_error("Descriptor %s isn't supported!\n", parameters.descriptor_id.c_str());
    } else if (parameters.matching_id == MATCHING_ONE_SIDED) {
        if (parameters.descriptor_id == DESCRIPTOR_FPFH) {
            auto ptr = std::make_shared<OneSidedMatcher<FPFH>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters.descriptor_id == DESCRIPTOR_ROPS) {
            auto ptr = std::make_shared<OneSidedMatcher<RoPS135>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters.descriptor_id == DESCRIPTOR_SHOT) {
            auto ptr = std::make_shared<OneSidedMatcher<SHOT>>(src, tgt, indices_src, indices_tgt, parameters);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else pcl::console::print_error("Descriptor %s isn't supported!\n", parameters.descriptor_id.c_str());
    } else if (parameters.matching_id != MATCHING_LEFT_TO_RIGHT) {
        PCL_WARN("[getFeatureBasedMatcherFromParameters] feature matcher %s isn't supported, "
                 "left-to-right matcher will be used.",
                 parameters.matching_id.c_str());
    }
    if (parameters.descriptor_id == DESCRIPTOR_FPFH) {
        auto ptr = std::make_shared<LeftToRightMatcher<FPFH>>(src, tgt, indices_src, indices_tgt, parameters);
        return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
    } else if (parameters.descriptor_id == DESCRIPTOR_ROPS) {
        auto ptr = std::make_shared<LeftToRightMatcher<RoPS135>>(src, tgt, indices_src, indices_tgt, parameters);
        return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
    } else if (parameters.descriptor_id == DESCRIPTOR_SHOT) {
        auto ptr = std::make_shared<LeftToRightMatcher<SHOT>>(src, tgt, indices_src, indices_tgt, parameters);
        return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
    }
    PCL_ERROR("Descriptor %s isn't supported!\n", parameters.descriptor_id.c_str());
}