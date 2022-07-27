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
    auto matcher = getFeatureBasedMatcherFromParameters(src_, tgt_, indices_src_, indices_tgt_, parameters_);
    return matcher->match();
}