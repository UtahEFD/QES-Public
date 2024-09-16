#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "CUDA_plume_testkernel.h"

TEST_CASE("PLUME")
{
  std::cout << "==============================================================\n"
            << "testing PLUME on GPU     \n"
            << "--------------------------------------\n"
            << std::endl;

  // for (auto lIdx = 0; lIdx < 3; ++lIdx) {
  Timer totTimer("test plume GPU");

  totTimer.start();
  test_gpu(1E4, 8E3, 5E5);
  // test_gpu(2100, 8E3, 5E5);
  // test_gpu(100000, 400, 5E5);
  //  test_gpu(2100, 400, 50000);
  totTimer.stop();

  totTimer.show();
  std::cout << "==============================================================" << std::endl;
}
