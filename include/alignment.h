#ifndef REGISTRATION_ALIGNMENT_H
#define REGISTRATION_ALIGNMENT_H

#include "common.h"

AlignmentResult alignRansac(const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                            const pcl::CorrespondencesPtr &correspondences,
                            const AlignmentParameters &parameters);

AlignmentResult alignGror(const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                          const pcl::CorrespondencesPtr &correspondences,
                          const AlignmentParameters &parameters);

AlignmentResult alignTeaser(const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                            const pcl::CorrespondencesPtr &correspondences,
                            const AlignmentParameters &parameters);

AlignmentResult alignPointClouds(const PointNCloud::Ptr &src_fullsize,
                                 const PointNCloud::Ptr &tgt_fullsize,
                                 const AlignmentParameters &parameters);

#endif
