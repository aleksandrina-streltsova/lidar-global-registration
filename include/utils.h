#ifndef REGISTRATION_UTILS_H
#define REGISTRATION_UTILS_H

#include <random>

class UniformRandIntGenerator {
public:
    UniformRandIntGenerator(const int min, const int max, std::mt19937::result_type seed = std::random_device{}())
            : distribution_(min, max), generator_(seed) {}

    int operator()() { return distribution_(generator_); }

protected:
    std::uniform_int_distribution<int> distribution_;
    std::mt19937 generator_;
};

#endif
