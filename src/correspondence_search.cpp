#include "correspondence_search.h"
#include <pcl/common/time.h>
pcl::CorrespondencesPtr FeatureBasedCorrespondenceSearch::calculateCorrespondences() {
    // Detect key points
    pcl::console::print_highlight("Detecting key points...\n");
    {
        pcl::ScopeTime t("Key point detection");
        indices_src_ = detectKeyPoints(src_, parameters_);
        indices_tgt_ = detectKeyPoints(tgt_, parameters_);
    }
    // Match key points
    pcl::console::print_highlight("Matching key points...\n");
    auto matcher = getFeatureBasedMatcher();
    return matcher->match();
}

FeatureBasedMatcher::Ptr
FeatureBasedCorrespondenceSearch::getFeatureBasedMatcher() const {
    if (parameters_.matching_id == MATCHING_RATIO) {
        if (parameters_.descriptor_id == DESCRIPTOR_FPFH) {
            auto ptr = std::make_shared<RatioMatcher<FPFH>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters_.descriptor_id == DESCRIPTOR_ROPS) {
            auto ptr = std::make_shared<RatioMatcher<RoPS135>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters_.descriptor_id == DESCRIPTOR_SHOT) {
            auto ptr = std::make_shared<RatioMatcher<SHOT>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else pcl::console::print_error("Descriptor %s isn't supported!\n", parameters_.descriptor_id.c_str());
    } else if (parameters_.matching_id == MATCHING_CLUSTER) {
        if (parameters_.descriptor_id == DESCRIPTOR_FPFH) {
            auto ptr = std::make_shared<ClusterMatcher<FPFH>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters_.descriptor_id == DESCRIPTOR_ROPS) {
            auto ptr = std::make_shared<ClusterMatcher<RoPS135>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else if (parameters_.descriptor_id == DESCRIPTOR_SHOT) {
            auto ptr = std::make_shared<ClusterMatcher<SHOT>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
            return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
        } else pcl::console::print_error("Descriptor %s isn't supported!\n", parameters_.descriptor_id.c_str());
    } else if (parameters_.matching_id != MATCHING_LEFT_TO_RIGHT) {
        PCL_WARN("[getFeatureMatcher] feature matcher %s isn't supported, left-to-right matcher will be used.",
                 parameters_.matching_id.c_str());
    }
    if (parameters_.descriptor_id == DESCRIPTOR_FPFH) {
        auto ptr = std::make_shared<LeftToRightMatcher<FPFH>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
        return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
    } else if (parameters_.descriptor_id == DESCRIPTOR_ROPS) {
        auto ptr = std::make_shared<LeftToRightMatcher<RoPS135>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
        return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
    } else if (parameters_.descriptor_id == DESCRIPTOR_SHOT) {
        auto ptr = std::make_shared<LeftToRightMatcher<SHOT>>(src_, tgt_, indices_src_, indices_tgt_, parameters_);
        return std::static_pointer_cast<FeatureBasedMatcher>(ptr);
    }
    PCL_ERROR("Descriptor %s isn't supported!\n", parameters_.descriptor_id.c_str());
}