#include "matching.h"

void assertResultsEqual(const KNNResult<float> &knnResult,
                        const std::vector<int> &indices_expected,
                        const std::vector<float> &distances_expected) {
    std::vector<int> indices_actual = knnResult.getIndices();
    std::vector<float> distances_actual = knnResult.getDistances();
    if (indices_actual.size() != indices_expected.size()) {
        std::cerr << "actual size of indices [" << indices_actual.size() << "] differs from expected [" << indices_expected.size() << "]" << std::endl;
        abort();
    }
    if (distances_actual.size() != distances_expected.size()) {
        std::cerr << "actual size of distances [" << distances_actual.size() << "] differs from expected [" << distances_expected.size() << "]" << std::endl;
        abort();
    }
    for (int i = 0; i < indices_actual.size(); ++i) {
        if (indices_actual[i] != indices_expected[i]) {
            std::cerr << "actual index at position " << i << " [" << indices_actual[i] << "] differs from expected [" << indices_expected[i] << "]" << std::endl;
            abort();
        }
    }
    for (int i = 0; i < indices_actual.size(); ++i) {
        if (distances_actual[i] != distances_expected[i]) {
            std::cerr << "actual distance at position " << i << " [" << distances_actual[i] << "] differs from expected [" << distances_expected[i] << "]" << std::endl;
            abort();
        }
    }
}

void runTest() {
    std::vector<int> indices;
    std::vector<float> distances;
    KNNResult<float> knnResult(3);

    assertResultsEqual(knnResult, {}, {});

    knnResult.addPoint(3.f, 3);
    assertResultsEqual(knnResult, {3}, {3.f});

    knnResult.addPoint(2.f, 2);
    assertResultsEqual(knnResult, {2, 3}, {2.f, 3.f});

    knnResult.addPoint(4.f, 4);
    assertResultsEqual(knnResult, {2, 3, 4}, {2.f, 3.f, 4.f});

    knnResult.addPoint(1.f, 1);
    assertResultsEqual(knnResult, {1, 2, 3}, {1.f, 2.f, 3.f});

    knnResult.addPoint(1.f, 5);
    assertResultsEqual(knnResult, {1, 5, 2}, {1.f, 1.f, 2.f});
}

int main() {
    runTest();
    return 0;
}

