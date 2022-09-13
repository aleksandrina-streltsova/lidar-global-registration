#include <filesystem>

#include "flann_bf_matcher.h"
#include "io.h"

int main(int argc, char **argv) {
    YamlConfig config;
    std::cout << std::filesystem::current_path() << std::endl;
    config.init(argv[1]);

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    std::string testname;

    // Load src and tgt
    loadPointClouds(config, testname, src, tgt, fields_src, fields_tgt);

    for (auto &parameters: getParametersFromConfig(config, src, tgt, fields_src, fields_tgt)) {
        std::cout << "descriptor id: " << parameters.descriptor_id << std::endl;
        auto descriptor_id = parameters.descriptor_id;
        if (descriptor_id == "fpfh") {
            runTest<FPFH>(src, tgt, parameters);
        }  else if (descriptor_id == "rops") {
            runTest<RoPS135>(src, tgt, parameters);
        } else if (descriptor_id == "shot"){
            runTest<SHOT>(src, tgt, parameters);
        } else {
            pcl::console::print_error("Descriptor %s isn't supported!\n", descriptor_id.c_str());
        }
    }
}

