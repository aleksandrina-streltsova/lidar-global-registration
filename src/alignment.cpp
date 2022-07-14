#include <filesystem>
#include <pcl/common/time.h>

#include "alignment.h"
#include "sac_prerejective_omp.h"
#include "correspondence_search.h"
#include "downsample.h"

#define INITIAL_STEP_VOXEL_SIZE_COEF 8
#define FINAL_STEP_EDG_THR_COEF 0.99
#define MATCH_SEARCH_RADIUS_COEF 4

namespace fs = std::filesystem;

AlignmentResult executeAlignmentStep(const PointNCloud::Ptr &src_final,
                                     const PointNCloud::Ptr &tgt_final,
                                     const AlignmentParameters &parameters) {
    bool is_initial = !parameters.guess.has_value();
    pcl::console::print_highlight("Starting%s alignment step...\n", std::string(is_initial ? " initial" : "").c_str());

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    NormalCloud::Ptr normals_src(new NormalCloud), normals_tgt(new NormalCloud);

    float voxel_size = parameters.voxel_size;
    float normal_radius = parameters.normal_radius_coef * voxel_size;
    float feature_radius = parameters.feature_radius_coef * voxel_size;

    // Downsample
    pcl::console::print_highlight("Downsampling [voxel_size = %.5f]...\n", parameters.voxel_size);
    downsamplePointCloud(src_final, src, parameters);
    downsamplePointCloud(tgt_final, tgt, parameters);

    // Estimate normals
    if (parameters.use_normals) {
        pcl::console::print_highlight("Normals are from point clouds. Smoothing normals...\n");
        smoothNormals(normal_radius, voxel_size, src);
        smoothNormals(normal_radius, voxel_size, tgt);
        pcl::copyPointCloud(*src, *normals_src);
        pcl::copyPointCloud(*tgt, *normals_tgt);
    } else {
        pcl::console::print_highlight("Estimating normals...\n");
        estimateNormalsRadius(normal_radius, src, normals_src, parameters.normals_available);
        estimateNormalsRadius(normal_radius, tgt, normals_tgt, parameters.normals_available);
        pcl::concatenateFields(*src, *normals_src, *src);
        pcl::concatenateFields(*tgt, *normals_tgt, *tgt);
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
    }
    SampleConsensusPrerejectiveOMP ransac(src, tgt, correspondences, parameters);
    AlignmentResult alignment_result;
    {
        pcl::ScopeTime t(is_initial ? "Initial alignment step" : "Alignment step");
        alignment_result = ransac.align();
    }
    saveTransformation(fs::path(DATA_DEBUG_PATH) / fs::path(TRANSFORMATIONS_CSV),
                       constructName(parameters, "transformation"), alignment_result.transformation);
    return alignment_result;
}

AlignmentResult alignPointClouds(const PointNCloud::Ptr &src_fullsize,
                                 const PointNCloud::Ptr &tgt_fullsize,
                                 const AlignmentParameters &parameters) {
    AlignmentResult final_result;
    double time;
    {
        pcl::ScopeTime t("Alignment");
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
        time = t.getTimeSeconds();
    }
    final_result.time = time;
    return final_result;
}