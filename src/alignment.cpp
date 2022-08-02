#include <filesystem>
#include <pcl/common/time.h>
#include <teaser/registration.h>

#include "alignment.h"
#include "sac_prerejective_omp.h"
#include "correspondence_search.h"
#include "downsample.h"

#include "gror/ia_gror.h"

#define INITIAL_STEP_VOXEL_SIZE_COEF 8
#define FINAL_STEP_EDG_THR_COEF 0.99
#define MATCH_SEARCH_RADIUS_COEF 4

namespace fs = std::filesystem;

AlignmentResult alignRansac(const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                            const pcl::CorrespondencesPtr &correspondences,
                            const AlignmentParameters &parameters) {
    SampleConsensusPrerejectiveOMP ransac(src, tgt, correspondences, parameters);
    return ransac.align();
}

AlignmentResult alignGror(const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                          const pcl::CorrespondencesPtr &correspondences,
                          const AlignmentParameters &parameters) {
    pcl::ScopeTime t("GROR");
    pcl::registration::GRORInitialAlignment<PointN, PointN, float> gror;
    pcl::PointCloud<PointN>::Ptr pcs(new pcl::PointCloud<PointN>);
    gror.setInputSource(src);
    gror.setInputTarget(tgt);
    gror.setResolution(parameters.voxel_size);
    gror.setOptimalSelectionNumber(800);
    gror.setNumberOfThreads(omp_get_num_procs());
    gror.setInputCorrespondences(correspondences);
    gror.align(*pcs);
    return AlignmentResult{src, tgt, gror.getFinalTransformation(), correspondences, 1, true, t.getTimeSeconds()};
}

AlignmentResult alignTeaser(const PointNCloud::Ptr &src, const PointNCloud::Ptr &tgt,
                            const pcl::CorrespondencesPtr &correspondences,
                            const AlignmentParameters &parameters) {
    pcl::ScopeTime t("TEASER");

    teaser::PointCloud src_teaser, tgt_teaser;
    auto transform_pcd = [](const PointN &p) { return teaser::PointXYZ{p.x, p.y, p.z}; };
    std::transform(src->points.begin(), src->points.end(), std::back_inserter(src_teaser), transform_pcd);
    std::transform(tgt->points.begin(), tgt->points.end(), std::back_inserter(tgt_teaser), transform_pcd);

    std::vector<std::pair<int, int>> corrs_teaser;
    std::transform(correspondences->begin(), correspondences->end(), std::back_inserter(corrs_teaser),
                   [](const pcl::Correspondence &corr) {
                       return std::pair<int, int>{corr.index_query, corr.index_match};
                   });

    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = parameters.distance_thr_coef * parameters.voxel_size;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;

    teaser::RobustRegistrationSolver solver(params);
    solver.solve(src_teaser, tgt_teaser, corrs_teaser);
    auto solution = solver.getSolution();
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    transformation.block<3, 3>(0, 0) = solution.rotation.cast<float>();
    transformation.block<3, 1>(0, 3) = solution.translation.cast<float>();
    return AlignmentResult{src, tgt, transformation, correspondences, 1, true, t.getTimeSeconds()};
}

AlignmentResult executeAlignmentStep(const PointNCloud::Ptr &src_final,
                                     const PointNCloud::Ptr &tgt_final,
                                     const AlignmentParameters &parameters) {
    double time_correspondence_search = 0.0, time_downsampling_and_normals = 0.0;
    bool is_initial = !parameters.guess.has_value();
    pcl::console::print_highlight("Starting%s alignment step...\n", std::string(is_initial ? " initial" : "").c_str());

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);

    float voxel_size = parameters.voxel_size;
    float feature_radius = parameters.feature_radius;
    {
        pcl::ScopeTime t("Downsampling and normal estimation");
        // Downsample
        pcl::console::print_highlight("Downsampling [voxel_size = %.5f]...\n", parameters.voxel_size);
        downsamplePointCloud(src_final, src, parameters);
        downsamplePointCloud(tgt_final, tgt, parameters);

        // Estimate normals
        pcl::console::print_highlight("Estimating normals...\n");
        estimateNormalsPoints(NORMAL_NR_POINTS, src, normals_src, parameters.normals_available);
        estimateNormalsPoints(NORMAL_NR_POINTS, tgt, normals_tgt, parameters.normals_available);
        pcl::concatenateFields(*src, *normals_src, *src);
        pcl::concatenateFields(*tgt, *normals_tgt, *tgt);
        time_downsampling_and_normals = t.getTimeSeconds();
    }

    bool success = false;
    std::string filepath = constructPath(parameters, "correspondences", "csv", true, false, false);
    pcl::CorrespondencesPtr correspondences;
//    correspondences = readCorrespondencesFromCSV(filepath, success);
    if (success) {
        PCL_DEBUG("[executeAlignmentStep] read correspondences from file\n");
    } else {
        pcl::ScopeTime t("Correspondence search");
        FeatureBasedCorrespondenceSearch corr_search(src, tgt, parameters);
        correspondences = corr_search.calculateCorrespondences();
        saveCorrespondencesToCSV(filepath, src, tgt, correspondences);
        time_correspondence_search = t.getTimeSeconds();
    }
    AlignmentResult alignment_result;
    if (parameters.alignment_id == ALIGNMENT_GROR) {
        alignment_result = alignGror(src, tgt, correspondences, parameters);
    } else if (parameters.alignment_id == ALIGNMENT_TEASER) {
        alignment_result = alignTeaser(src, tgt, correspondences, parameters);
    } else {
        if (parameters.alignment_id != ALIGNMENT_DEFAULT) {
            PCL_WARN("[executeAlignmentStep] Transformation estimation method %s isn't supported,"
                     " default LRF will be used.\n", parameters.alignment_id.c_str());
        }
        alignment_result = alignRansac(src, tgt, correspondences, parameters);
    }
    alignment_result.time_cs = time_correspondence_search;
    alignment_result.time_ds_ne = time_downsampling_and_normals;
    if (parameters.ground_truth.has_value()) {
        saveTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                           constructName(parameters, "transformation_gt"), parameters.ground_truth.value());
    }
    saveTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                       constructName(parameters, "transformation"), alignment_result.transformation);
    return alignment_result;
}

AlignmentResult alignPointClouds(const PointNCloud::Ptr &src_fullsize,
                                 const PointNCloud::Ptr &tgt_fullsize,
                                 const AlignmentParameters &parameters) {
    pcl::ScopeTime t("Alignment");
    AlignmentResult final_result;
    if (parameters.coarse_to_fine) {
        PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
        downsamplePointCloud(src_fullsize, src, parameters);
        downsamplePointCloud(tgt_fullsize, tgt, parameters);
        float final_voxel_size = parameters.voxel_size;
        float initial_voxel_size = INITIAL_STEP_VOXEL_SIZE_COEF * final_voxel_size;

        // initial step
        AlignmentParameters initial_parameters(parameters);
        initial_parameters.voxel_size = initial_voxel_size;
        auto initial_result = executeAlignmentStep(src, tgt, initial_parameters);

        // final step
        AlignmentParameters final_parameters(parameters);
        final_parameters.voxel_size = final_voxel_size;
        final_parameters.match_search_radius = MATCH_SEARCH_RADIUS_COEF * initial_voxel_size;
        final_parameters.guess = std::optional<Eigen::Matrix4f>{initial_result.transformation};
        final_parameters.matching_id = MATCHING_LEFT_TO_RIGHT;
        final_parameters.metric_id = METRIC_CORRESPONDENCES;
        final_parameters.edge_thr_coef = FINAL_STEP_EDG_THR_COEF;
        final_result = executeAlignmentStep(src, tgt, final_parameters);

        saveIterationsInfo(fs::path(DATA_DEBUG_PATH) / fs::path(ITERATIONS_CSV),
                           constructName(parameters, "iterations"),
                           {initial_voxel_size, final_voxel_size},
                           {initial_parameters.matching_id, final_parameters.matching_id});
    } else {
        final_result = executeAlignmentStep(src_fullsize, tgt_fullsize, parameters);
        saveIterationsInfo(fs::path(DATA_DEBUG_PATH) / fs::path(ITERATIONS_CSV),
                           constructName(parameters, "iterations"),
                           {parameters.voxel_size}, {parameters.matching_id});
    }
    return final_result;
}