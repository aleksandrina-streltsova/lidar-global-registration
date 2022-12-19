#include "correspondence_search.h"
#include <pcl/common/time.h>

CorrespondencesPtr FeatureBasedCorrespondenceSearch::calculateCorrespondences() {
    pcl::console::print_highlight("Matching key points...\n");
    auto matcher = getFeatureBasedMatcherFromParameters(src_, tgt_, kps_src_, kps_tgt_, parameters_);
    return matcher->match();
}