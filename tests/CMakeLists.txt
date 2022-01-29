add_executable(flann_bf_matcher flann_bf_matcher.cpp ../src/config.cpp ../src/csv_parser.cpp ../src/align.cpp ../src/common.cpp ../src/filter.cpp ../src/analysis.cpp ../src/feature_analysis.cpp ../src/downsample.cpp)
target_link_libraries(flann_bf_matcher ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
target_include_directories(flann_bf_matcher PUBLIC ../include/)
add_test(NAME test_flann_bf_matcher COMMAND flann_bf_matcher data/tests/flann_bf_matcher.yaml
        WORKING_DIRECTORY ..)