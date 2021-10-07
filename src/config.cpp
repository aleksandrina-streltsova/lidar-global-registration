#include "../include/config.h"

void YamlConfig::init(const std::string &config_path) {
    config = YAML::LoadFile(config_path);
}
