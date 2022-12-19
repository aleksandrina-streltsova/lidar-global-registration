#ifndef REGISTRATION_ALIGNMENT_H
#define REGISTRATION_ALIGNMENT_H

#include "common.h"

AlignmentResult alignRansac(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                            const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                            const CorrespondencesConstPtr &correspondences,
                            const AlignmentParameters &parameters);

AlignmentResult alignGror(const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                          const CorrespondencesConstPtr &correspondences,
                          const AlignmentParameters &parameters);

AlignmentResult alignTeaser(const PointNCloud::ConstPtr &kps_src, const PointNCloud::ConstPtr &kps_tgt,
                            const CorrespondencesConstPtr &correspondences,
                            const AlignmentParameters &parameters);

AlignmentResult alignPointClouds(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                                 const AlignmentParameters &params);

#endif
