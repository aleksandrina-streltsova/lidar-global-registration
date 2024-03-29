add_executable(flann_bf_matcher flann_bf_matcher.cpp ../src/config.cpp ../src/csv_parser.cpp ../src/alignment.cpp ../src/common.cpp ../src/analysis.cpp ../src/feature_analysis.cpp ../src/downsample.cpp ../src/utils.cpp ../src/metric.cpp ../src/weights.cpp ../src/gror/gror_pre.cpp ../src/transformation.cpp ../src/matching.cpp ../src/correspondence_search.cpp ../src/sac_prerejective_omp.cpp ../src/hypotheses.cpp ../src/pcl/iss_debug.cpp ../src/pcl/shot_debug.cpp ../src/quadric.cpp)
target_link_libraries(flann_bf_matcher  ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp) #teaserpp::teaser_registration)
target_include_directories(flann_bf_matcher PUBLIC ../include/)
add_test(NAME test_flann_bf_matcher COMMAND flann_bf_matcher data/tests/flann_bf_matcher.yaml WORKING_DIRECTORY ..)

add_executable(knn_result knn_result.cpp)
target_link_libraries(knn_result ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
target_include_directories(knn_result PUBLIC ../include/)
add_test(NAME test_knn_result COMMAND knn_result)

add_executable(keypoint_extraction keypoint_extraction.cpp ../src/config.cpp ../src/csv_parser.cpp ../src/alignment.cpp ../src/common.cpp ../src/analysis.cpp ../src/feature_analysis.cpp ../src/downsample.cpp ../src/utils.cpp ../src/metric.cpp ../src/weights.cpp ../src/gror/gror_pre.cpp ../src/transformation.cpp ../src/matching.cpp ../src/correspondence_search.cpp ../src/sac_prerejective_omp.cpp ../src/hypotheses.cpp ../src/pcl/iss_debug.cpp ../src/pcl/shot_debug.cpp ../src/quadric.cpp)
target_link_libraries(keypoint_extraction ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp) #teaserpp::teaser_registration)
target_include_directories(keypoint_extraction PUBLIC ../include/)
add_test(NAME test_keypoint_extraction COMMAND keypoint_extraction data/tests/keypoint_extraction.yaml WORKING_DIRECTORY ..)

add_executable(point2plane_distance point2plane_distance.cpp ../src/config.cpp ../src/csv_parser.cpp ../src/alignment.cpp ../src/common.cpp ../src/analysis.cpp ../src/feature_analysis.cpp ../src/downsample.cpp ../src/utils.cpp ../src/metric.cpp ../src/weights.cpp ../src/gror/gror_pre.cpp ../src/transformation.cpp ../src/matching.cpp ../src/correspondence_search.cpp ../src/sac_prerejective_omp.cpp ../src/hypotheses.cpp ../src/pcl/iss_debug.cpp ../src/pcl/shot_debug.cpp ../src/quadric.cpp)
target_link_libraries(point2plane_distance ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp) #teaserpp::teaser_registration)
target_include_directories(point2plane_distance PUBLIC ../include/)
add_test(NAME test_point2plane_distance COMMAND point2plane_distance WORKING_DIRECTORY ..)