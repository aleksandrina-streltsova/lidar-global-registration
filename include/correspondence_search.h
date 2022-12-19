#ifndef REGISTRATION_CORRESPONDENCE_SEARCH_H
#define REGISTRATION_CORRESPONDENCE_SEARCH_H

#include <pcl/correspondence.h>

#include "common.h"
#include "matching.h"

class CorrespondenceSearch {
public:
    virtual CorrespondencesPtr calculateCorrespondences() = 0;
};

class FeatureBasedCorrespondenceSearch : CorrespondenceSearch {
public:
    FeatureBasedCorrespondenceSearch() = delete;

    FeatureBasedCorrespondenceSearch(PointNCloud::ConstPtr src, PointNCloud::ConstPtr tgt,
                                     PointNCloud::ConstPtr kps_src, PointNCloud::ConstPtr kps_tgt,
                                     AlignmentParameters parameters) :
            src_(std::move(src)), tgt_(std::move(tgt)),
            kps_src_(std::move(kps_src)), kps_tgt_(std::move(kps_tgt)), parameters_(std::move(parameters)) {}

    CorrespondencesPtr calculateCorrespondences() override;

protected:
    PointNCloud::ConstPtr src_, tgt_;
    PointNCloud::ConstPtr kps_src_, kps_tgt_;
    AlignmentParameters parameters_;
};

#endif
