#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"

#include "CUDA_advect_partition_testkernel.h"

TEST_CASE("test")
{

  std::cout << "======================================\n"
            << "testing advection on GPU              \n"
            << "======================================\n"
            << std::endl;
  double avgTime = 0.0;
  auto startTime = std::chrono::high_resolution_clock::now();
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;

  std::cout << "TEST advection using data structure and paritioning" << std::endl;
  std::cout << "--------------------------------------" << std::endl;

  // for (auto lIdx = 0; lIdx < 3; ++lIdx) {

  startTime = std::chrono::high_resolution_clock::now();
  test_gpu(1E5, 5E2, 1E5);
  endTime = std::chrono::high_resolution_clock::now();

  elapsed = endTime - startTime;
  std::cout << "Total elapsed time:  " << elapsed.count() << " s\n";
  std::cout << "======================================" << std::endl;
}
