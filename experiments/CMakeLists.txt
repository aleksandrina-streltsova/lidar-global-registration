add_executable(experiments experiments.cpp ../src/config.cpp ../src/csv_parser.cpp ../src/align.cpp ../src/common.cpp ../src/filter.cpp ../src/analysis.cpp ../src/feature_analysis.cpp ../src/downsample.cpp ../src/metric.cpp ../src/weights.cpp ../src/utils.cpp experiments.cpp keypoints.cpp)
target_link_libraries(experiments ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
target_include_directories(experiments PUBLIC ../include/)

