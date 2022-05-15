#ifndef REGISTRATION_UTILS_H
#define REGISTRATION_UTILS_H

#include <iostream>
#include <random>
#include <functional>

class UniformRandIntGenerator {
public:
    UniformRandIntGenerator(const int min, const int max, std::mt19937::result_type seed = std::random_device{}())
            : distribution_(min, max), generator_(seed) {}

    int operator()() { return distribution_(generator_); }

protected:
    std::uniform_int_distribution<int> distribution_;
    std::mt19937 generator_;
};

template<typename T>
inline void combineHash(std::size_t& seed, const T& val) {
    std::hash<T> hasher;
    seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
T calculate_combination_or_max(T n, T k) {
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result *= n - i;
        result /= i + 1;
    }
    T max = std::numeric_limits<T>::max();
    return result > max ? max : (T)result;
}

template <typename T>
T quantile(double q, const std::vector<T> &values) {
    if (q < 0.0 || q > 1.0 || values.empty()) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (values.size() == 1) {
        return values[0];
    }
    std::size_t n = values.size();
    std::size_t i = std::floor(q * (double)(n - 1));
    std::size_t j = std::min(i + 1, n - 1);

    std::vector<T> v = values;
    std::nth_element(v.begin(), v.begin() + i, v.end());
    T ith = v[i];
    if (i < j) {
        std::nth_element(v.begin(), v.begin() + j, v.end());
        T jth = v[j];
        return ith * ((double)n * q - (double)i) + jth * ((double)j - (double)n * q);
    }
    return ith;
}

template<typename T>
T calculate_mean(const std::vector<T> &v) {
    if (v.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / (T) v.size();
}

template<typename T>
T calculate_standard_deviation(const std::vector<T> &v) {
    if (v.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    T mean = calculate_mean(v);
    T deviation = 0.0;
    for (T x: v) {
        deviation += (x - mean) * (x - mean);
    }
    deviation = std::sqrt(deviation / (T) v.size());
    return deviation;
}

void split(const std::string &str, std::vector<std::string> &tokens, const std::string &delimiter);

#endif
