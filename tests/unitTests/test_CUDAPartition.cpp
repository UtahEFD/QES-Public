#include <iostream>
#include <random>
#include <ctime>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "CUDAPartitionKernel.h"

TEST_CASE("GPU Partitioning Tests")
{
    std::default_random_engine prng;
    std::uniform_real_distribution<float> distribution(-1000.0f, 1000.0f);
    prng.seed(std::time(nullptr));
 
    const int N = 10000;
    std::vector<float> data(N);

    float sum = 0.0;
    for (auto idx=0; idx<N; ++idx) {
        data[idx] = distribution(prng);
        sum += data[idx];
    }
    // pick the mean of the data as the pivot
    float pivotValue = sum/(float)data.size();

    partitionData(data, pivotValue);

    int lowerCount = 0, upperCount = 0;
    for (auto idx=0; idx<data.size(); ++idx) {
        if (data[idx] <= pivotValue)
            lowerCount++;
        else
            upperCount++;
    }

    // std::cout << "lowerCount: " << lowerCount << ", upperCount: " << upperCount << std::endl;

    for (auto idx=0; idx<lowerCount; ++idx) {
        REQUIRE( data[idx] <= pivotValue );
	// std::cout << data[idx] << ' ';
    }
    // std::cout << std::endl;
    // std::cout << "\n******* Pivot: " << pivotValue << " *********\n" << std::endl;
    for (auto idx=lowerCount; idx<lowerCount+upperCount; ++idx) {
        REQUIRE( data[idx] > pivotValue );
	// std::cout << data[idx] << ' ';
    }
    // std::cout << std::endl;
}
