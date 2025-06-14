#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "CUDA_concentration_testkernel.h"

TEST_CASE("test")
{

  std::cout << "======================================\n"
            << "testing concentration on GPU          \n"
            << "======================================\n"
            << std::endl;
  double avgTime = 0.0;
  auto startTime = std::chrono::high_resolution_clock::now();
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;

  std::cout << "TEST CONCENTRATION" << std::endl;
  std::cout << "--------------------------------------" << std::endl;

  // for (auto lIdx = 0; lIdx < 3; ++lIdx) {

  startTime = std::chrono::high_resolution_clock::now();
  test_gpu(1E4, 1, 1E6);
  endTime = std::chrono::high_resolution_clock::now();

  elapsed = endTime - startTime;
  std::cout << "Total elapsed time:  " << elapsed.count() << " s\n";
  std::cout << "======================================" << std::endl;
}
