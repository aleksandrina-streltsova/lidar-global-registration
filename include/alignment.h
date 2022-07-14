#ifndef REGISTRATION_ALIGNMENT_H
#define REGISTRATION_ALIGNMENT_H

#include "common.h"

AlignmentResult alignPointClouds(const PointNCloud::Ptr &src_fullsize,
                                 const PointNCloud::Ptr &tgt_fullsize,
                                 const AlignmentParameters &parameters);

#endif
