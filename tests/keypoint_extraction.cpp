#include <filesystem>

#include "alignment.h"

#define TMP_DIR "tmp"
#define NR_DIMS_SHOT 352

namespace fs = std::filesystem;

typedef std::array<int, NR_DIMS_SHOT> Feature;
typedef std::vector<Feature> Features;

void readFeatures(Features &features, pcl::Indices &indices, const AlignmentParameters &parameters, bool is_source) {
    std::string filepath = constructPath(parameters,  std::string("histograms_") + (is_source ? "src" : "tgt"), "csv");
    bool file_exists = std::filesystem::exists(filepath);
    features.clear();
    indices.clear();
    if (file_exists) {
        std::ifstream fin(filepath);
        if (fin.is_open()) {
            std::string line;
            std::vector<std::string> tokens;
            Feature feature{};
            while (std::getline(fin, line)) {
                split(line, tokens, ",");
                indices.push_back(std::stoi(tokens[0]));
                for (int i = 0; i < NR_DIMS_SHOT; ++i) {
                    feature[i] = std::stof(tokens[i + 1]);
                }
                features.push_back(feature);
            }
        } else {
            perror(("error while opening file " + filepath).c_str());
        }
    } else {
        perror(("file " + filepath + " doesn't exist").c_str());
    }
}

void assertFeaturesEqual(const std::string &prefix, const Feature &expected, const Feature &actual) {
    for (int i = 0; i < NR_DIMS_SHOT; ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << prefix << " actual value at position " << i << " [" << actual[i] << "] differs from expected [" << expected[i] << "]" << std::endl;
            abort();
        }
    }
}

int main(int argc, char **argv) {
    YamlConfig config;
    config.init(argv[1]);

    PointNCloud::Ptr src(new PointNCloud), tgt(new PointNCloud);
    std::vector<::pcl::PCLPointField> fields_src, fields_tgt;
    std::string testname;
    float min_voxel_size;
    std::string src_path = config.get<std::string>("source").value();
    std::string tgt_path = config.get<std::string>("target").value();

    loadPointClouds(src_path, tgt_path, testname, src, tgt, fields_src, fields_tgt,
                    config.get<float>("density"), min_voxel_size);
    auto parameters = getParametersFromConfig(config, fields_src, fields_tgt, min_voxel_size)[0];
    parameters.testname = testname;
    parameters.dir_path = TMP_DIR;
    fs::create_directory(TMP_DIR);

    pcl::Indices key_point_indices_src, key_point_indices_tgt;
    Features features_src, features_tgt, key_point_features_src, key_point_features_tgt;

    parameters.keypoint_id = KEYPOINT_ANY;
    alignPointClouds(src, tgt, parameters);
    readFeatures(features_src, key_point_indices_src, parameters, true);
    readFeatures(features_tgt, key_point_indices_tgt, parameters, false);

    parameters.keypoint_id = KEYPOINT_ISS;
    alignPointClouds(src, tgt, parameters);
    readFeatures(key_point_features_src, key_point_indices_src, parameters, true);
    readFeatures(key_point_features_tgt, key_point_indices_tgt, parameters, false);

    fs::remove_all(TMP_DIR);

    for (int i = 0; i < key_point_indices_src.size(); ++i) {
        std::string prefix = "[" + std::to_string(i) + "/" + std::to_string(key_point_indices_src.size()) + "]";
        assertFeaturesEqual(prefix,features_src[key_point_indices_src[i]], key_point_features_src[i]);
    }
    for (int i = 0; i < key_point_indices_tgt.size(); ++i) {
        std::string prefix = "[" + std::to_string(i) + "/" + std::to_string(key_point_indices_tgt.size()) + "]";
        assertFeaturesEqual(prefix,features_tgt[key_point_indices_tgt[i]], key_point_features_tgt[i]);
    }
    return 0;
}
