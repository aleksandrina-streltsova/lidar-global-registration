#include <filesystem>

#include "flann_bf_matcher.h"
#include "io.h"

int main(int argc, char **argv) {
    YamlConfig config;
    std::cout << std::filesystem::current_path() << std::endl;
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
    float src_density = calculatePointCloudDensity<PointN>(src);
    float tgt_density = calculatePointCloudDensity<PointN>(tgt);
    float min_voxel_size = std::max(src_density, tgt_density);

    std::string src_filename = src_path.substr(src_path.find_last_of("/\\") + 1);
    std::string tgt_filename = tgt_path.substr(tgt_path.find_last_of("/\\") + 1);

    for (auto &parameters: getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)) {
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

