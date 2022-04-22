#include <pcl/io/ply_io.h>

#include "flann_bf_matcher.h"
#include "io.h"

int main(int argc, char **argv) {
    YamlConfig config;
    config.init(argv[1]);

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;

    // Load src and tgt
    pcl::console::print_highlight("Loading point clouds...\n");
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    if (loadPLYFile<PointN>(src_path, *src, fields_src) < 0 ||
        loadPLYFile<PointN>(tgt_path, *tgt, fields_tgt) < 0) {
        pcl::console::print_error("Error loading src/tgt file!\n");
        exit(1);
    }
    std::vector<AlignmentParameters> parameters_container = getParametersFromConfig(config, fields_src, fields_tgt);

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

