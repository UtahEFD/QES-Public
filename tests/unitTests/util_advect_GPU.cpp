#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"

#include "CUDA_advect_testkernel.h"


TEST_CASE("vector math class test")
{

  std::cout << "======================================\n"
            << "testing advection on GPU              \n"
            << "--------------------------------------\n"
            << std::endl;

  auto startTime = std::chrono::high_resolution_clock::now();
  test_gpu_AOS(1E5);
  auto endTime = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = endTime - startTime;
  std::cout << "Total  elapsed time: " << elapsed.count() << " s\n";
  std::cout << "======================================" << std::endl;

  startTime = std::chrono::high_resolution_clock::now();
  test_gpu_SOA(1E5);
  endTime = std::chrono::high_resolution_clock::now();

  elapsed = endTime - startTime;
  std::cout << "Total  elapsed time: " << elapsed.count() << " s\n";
  std::cout << "======================================" << std::endl;
}
