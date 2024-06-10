#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/Matrix3.h"
#include "util/VectorMath.h"
#include "CUDA_vector_testkernel.h"

TEST_CASE("vector math class test")
{

  std::cout << "======================================\n"
            << "testing vector math\n"
            << "--------------------------------------" << std::endl;

  test_gpu(1E8);
}
