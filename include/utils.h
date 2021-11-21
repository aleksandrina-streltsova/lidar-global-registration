#ifndef REGISTRATION_UTILS_H
#define REGISTRATION_UTILS_H

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

#endif
