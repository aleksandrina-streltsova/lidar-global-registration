#include "matching.h"

void FeatureBasedMatcher::printDebugInfo(const std::vector<MultivaluedCorrespondence> &mv_correspondences) {
    float dists_sum = 0.f;
    int n_dists = 0;
    for (int i = 0; i < mv_correspondences.size(); i++) {
        if (mv_correspondences[i].query_idx >= 0) {
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