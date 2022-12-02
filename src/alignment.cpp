#include <filesystem>
#include <pcl/common/time.h>
//#include <teaser/registration.h>

#include "alignment.h"
#include "sac_prerejective_omp.h"
#include "correspondence_search.h"
#include "downsample.h"

#include "gror/ia_gror.h"

namespace fs = std::filesystem;

AlignmentResult alignRansac(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                            const CorrespondencesPtr &correspondences,
                            const AlignmentParameters &parameters) {
    SampleConsensusPrerejectiveOMP ransac(src, tgt, correspondences, parameters);
    return ransac.align();
}

AlignmentResult alignGror(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                          const CorrespondencesPtr &correspondences,
                          const AlignmentParameters &parameters) {
    pcl::ScopeTime t("GROR");
    pcl::registration::GRORInitialAlignment<PointN, PointN, float> gror;
    pcl::PointCloud<PointN>::Ptr pcs(new pcl::PointCloud<PointN>);
    gror.setInputSource(src);
    gror.setInputTarget(tgt);
    gror.setResolution(parameters.distance_thr);
    gror.setOptimalSelectionNumber(800);
    gror.setNumberOfThreads(omp_get_num_procs());
    gror.setInputCorrespondences(std::make_shared<pcl::Correspondences>(correspondencesToPCL(*correspondences)));
    gror.align(*pcs);
    return AlignmentResult{src, tgt, gror.getFinalTransformation(), correspondences, 1, true, t.getTimeSeconds()};
}

AlignmentResult alignTeaser(const PointNCloud::ConstPtr &src, const PointNCloud::ConstPtr &tgt,
                            const CorrespondencesPtr &correspondences,
                            const AlignmentParameters &parameters) {
    throw std::runtime_error("Not implemented: support TEASER");
//    pcl::ScopeTime t("TEASER");
//
//    teaser::PointCloud src_teaser, tgt_teaser;
//    auto transform_pcd = [](const PointN &p) { return teaser::PointXYZ{p.x, p.y, p.z}; };
//    std::transform(src->points.begin(), src->points.end(), std::back_inserter(src_teaser), transform_pcd);
//    std::transform(tgt->points.begin(), tgt->points.end(), std::back_inserter(tgt_teaser), transform_pcd);
//
//    std::vector<std::pair<int, int>> corrs_teaser;
//    std::transform(correspondences->begin(), correspondences->end(), std::back_inserter(corrs_teaser),
//                   [](const Correspondence &corr) {
//                       return std::pair<int, int>{corr.index_query, corr.index_match};
//                   });
//
//    teaser::RobustRegistrationSolver::Params params;
//    params.noise_bound = parameters.distance_thr;
//    params.cbar2 = 1;
//    params.estimate_scaling = false;
//    params.rotation_max_iterations = 100;
//    params.rotation_gnc_factor = 1.4;
//    params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
//    params.rotation_cost_threshold = 0.005;
//
//    teaser::RobustRegistrationSolver solver(params);
//    solver.solve(src_teaser, tgt_teaser, corrs_teaser);
//    auto solution = solver.getSolution();
//    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
//    transformation.block<3, 3>(0, 0) = solution.rotation.cast<float>();
//    transformation.block<3, 1>(0, 3) = solution.translation.cast<float>();
//    return AlignmentResult{src, tgt, transformation, correspondences, 1, true, t.getTimeSeconds()};
}

AlignmentResult alignPointClouds(const PointNCloud::ConstPtr &src,
                                 const PointNCloud::ConstPtr &tgt,
                                 const AlignmentParameters &params) {
    pcl::ScopeTime t_alignment("Alignment");
    double time_correspondence_search = 0.0;
    bool success = false;
    std::string filepath = constructPath(params, "correspondences", "csv", true, false, false);
    CorrespondencesPtr correspondences;
//    correspondences = readCorrespondencesFromCSV(filepath, success);
    if (success) {
        PCL_DEBUG("[alignPointClouds] read correspondences from file\n");
    } else {
        pcl::ScopeTime t("Correspondence search");
        FeatureBasedCorrespondenceSearch corr_search(src, tgt, params);
        correspondences = corr_search.calculateCorrespondences();
        saveCorrespondencesToCSV(filepath, src, tgt, correspondences);
        time_correspondence_search = t.getTimeSeconds();
    }
    AlignmentResult alignment_result;
    if (params.alignment_id == ALIGNMENT_GROR) {
        alignment_result = alignGror(src, tgt, correspondences, params);
    } else if (params.alignment_id == ALIGNMENT_TEASER) {
        alignment_result = alignTeaser(src, tgt, correspondences, params);
    } else {
        if (params.alignment_id != ALIGNMENT_RANSAC) {
            PCL_WARN("[alignPointClouds] Transformation estimation method %s isn't supported,"
                     " RANSAC will be used.\n", params.alignment_id.c_str());
        }
        alignment_result = alignRansac(src, tgt, correspondences, params);
    }
    alignment_result.time_cs = time_correspondence_search;
    if (params.ground_truth.has_value()) {
        saveTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                           constructName(params, "transformation_gt"), params.ground_truth.value());
    }
    saveTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                       constructName(params, "transformation"), alignment_result.transformation);
    return alignment_result;
}