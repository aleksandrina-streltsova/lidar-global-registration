#ifndef REGISTRATION_CORRESPONDENCE_SEARCH_H
#define REGISTRATION_CORRESPONDENCE_SEARCH_H

#include <pcl/correspondence.h>

#include "common.h"
#include "matching.h"

class CorrespondenceSearch {
public:
    virtual pcl::CorrespondencesPtr calculateCorrespondences() = 0;
};

class FeatureBasedCorrespondenceSearch : CorrespondenceSearch {
public:
    FeatureBasedCorrespondenceSearch() = delete;

    FeatureBasedCorrespondenceSearch(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                                     AlignmentParameters parameters) :
            src_(std::move(src)), tgt_(std::move(tgt)), parameters_(std::move(parameters)) {}

    pcl::CorrespondencesPtr calculateCorrespondences() override;

protected:
    FeatureBasedMatcher::Ptr getFeatureBasedMatcher() const;

    PointNCloud::ConstPtr src_, tgt_;
    pcl::IndicesConstPtr indices_src_, indices_tgt_;
    AlignmentParameters parameters_;
};

#endif
