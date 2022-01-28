#include "flann_bf_matcher.h"

#include <pcl/io/ply_io.h>

int main(int argc, char **argv) {
    YamlConfig config;
    config.init(argv[1]);

    PointCloudT::Ptr src(new PointCloudT), tgt(new PointCloudT);

    std::vector<AlignmentParameters> parameters_container = getParametersFromConfig(config);

    // Load src and tgt
    pcl::console::print_highlight("Loading point clouds...\n");
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    if (pcl::io::loadPLYFile<PointT>(src_path, *src) < 0 ||
        pcl::io::loadPLYFile<PointT>(tgt_path, *tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }

    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);

    for (auto &parameters: parameters_container) {
        std::cout << "descriptor id: " << parameters.descriptor_id << std::endl;
        auto descriptor_id = parameters.descriptor_id;
        if (descriptor_id == "fpfh") {
            run_test<FPFH>(src, tgt, parameters);
        }  else if (descriptor_id == "rops") {
            run_test<RoPS135>(src, tgt, parameters);
        } else if (descriptor_id == "shot"){
            run_test<SHOT>(src, tgt, parameters);
        } else {
            pcl::console::print_error("Descriptor %s isn't supported!\n", descriptor_id.c_str());
        }
    }
}

