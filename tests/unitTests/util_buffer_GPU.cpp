#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"

#include "CUDA_buffer_testkernel.h"

TEST_CASE("test buffer")
{
  std::cout << "======================================\n"
            << "testing advection on GPU              \n"
            << "--------------------------------------\n"
            << std::endl;

  double avgTime = 0.0;
  auto startTime = std::chrono::high_resolution_clock::now();
  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;

  startTime = std::chrono::high_resolution_clock::now();
  test_gpu_buffer(2E5);
  endTime = std::chrono::high_resolution_clock::now();

  std::cout << "Total elapsed time: " << elapsed.count() << " s\n";
  std::cout << "======================================" << std::endl;
}
