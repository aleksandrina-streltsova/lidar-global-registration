#ifndef REGISTRATION_ALIGNMENT_H
#define REGISTRATION_ALIGNMENT_H

#include "common.h"

AlignmentResult alignRansac(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                            const CorrespondencesPtr &correspondences,
                            const AlignmentParameters &parameters);

AlignmentResult alignGror(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                          const CorrespondencesPtr &correspondences,
                          const AlignmentParameters &parameters);

AlignmentResult alignTeaser(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                            const CorrespondencesPtr &correspondences,
                            const AlignmentParameters &parameters);

AlignmentResult alignPointClouds(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                 const AlignmentParameters &params);

#endif
