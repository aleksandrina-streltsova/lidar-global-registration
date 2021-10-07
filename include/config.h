#pragma once

#include <optional>
#include <yaml-cpp/yaml.h>

struct YamlConfig {
    YAML::Node config;

    void init(const std::string& config_path);

    template <class T>
    std::optional<T> get(const std::string& option_name) const;

    template <class T>
    T get(const std::string& option_name, const T& default_value) const;
};

template<class T>
std::optional<T> YamlConfig::get(const std::string &option_name) const {
    if (config[option_name]) {
        return config[option_name].as<T>();
    }
    return {};
}

template<class T>
T YamlConfig::get(const std::string &option_name, const T &default_value) const {
    if (config[option_name]) {
        return config[option_name].as<T>();
    }
    return default_value;
}