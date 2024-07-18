#include <iostream>
#include <random>
#include <ctime>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "CUDAPartitionKernel.h"

/**
 *
 */
float generateRandomData(std::vector<float> &data)
{
  std::default_random_engine prng;
  std::uniform_real_distribution<float> distribution(-1000.0f, 1000.0f);
  prng.seed(std::time(nullptr));

  const int N = 10000;
  data.clear();
  data.resize(N);

  float sum = 0.0;
  for (auto idx = 0; idx < N; ++idx) {
    data[idx] = distribution(prng);
    sum += data[idx];
  }
  // pick the mean of the data as the pivot
  return sum / (float)data.size();
}

TEST_CASE("GPU Partitioning Tests")
{
  std::vector<float> data;
  float pivotValue = generateRandomData(data);

  partitionData(data, pivotValue);

  int lowerCount = 0, upperCount = 0;
  for (auto idx = 0; idx < data.size(); ++idx) {
    if (data[idx] <= pivotValue)
      lowerCount++;
    else
      upperCount++;
  }

  // if the partition worked, then all values in the data array, from
  // index 0 to lowerCount-1, should be less than the pivot.
  for (auto idx = 0; idx < lowerCount; ++idx) {
    REQUIRE(data[idx] <= pivotValue);
  }

  for (auto idx = lowerCount; idx < lowerCount + upperCount; ++idx) {
    REQUIRE(data[idx] > pivotValue);
  }

  BENCHMARK("Partition")
  {
    pivotValue = generateRandomData(data);
    return partitionData(data, pivotValue);
  };

}
