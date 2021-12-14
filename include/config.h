#pragma once

#include <optional>
#include <yaml-cpp/yaml.h>

struct YamlConfig {
    YAML::Node config;

    void init(const std::string &config_path);

    template<class T>
    std::optional<T> get(const std::string &option_name) const;

    template<class T>
    T get(const std::string &option_name, const T &default_value) const;

    template<class T>
    void set(const std::string &option_name, const T &value);

    template<class T>
    std::optional<std::vector<T>> getVector(const std::string &option_name) const;

    template<class T>
    std::vector<T> getVector(const std::string &option_name, const T &default_value) const;
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

template<class T>
void YamlConfig::set(const std::string &option_name, const T &value) {
    config[option_name] = value;
}

template<class T>
std::optional<std::vector<T>> YamlConfig::getVector(const std::string &option_name) const {
    std::vector<T> v;
    if (config[option_name]) {
        try {
            v = config[option_name].as<std::vector<T>>();
        } catch (const YAML::TypedBadConversion<std::vector<T, std::allocator<T>>> &e) {
            v.push_back(config[option_name].as<T>());
        }
        return v;
    }
    return {};
}

template<class T>
std::vector<T> YamlConfig::getVector(const std::string &option_name, const T &default_value) const {
    std::optional<std::vector<T>> v = getVector<T>(option_name);
    if (v.has_value()) {
        return v.value();
    }
    return std::vector<T>(1, default_value);
}