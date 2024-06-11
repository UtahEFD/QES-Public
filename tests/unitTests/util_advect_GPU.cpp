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
            << "--------------------------------------" << std::endl;

  test_gpu(1E5);

  std::cout << "======================================" << std::endl;
}
