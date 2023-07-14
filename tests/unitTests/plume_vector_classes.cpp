#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util.h"
#include "testFunctions.h"
#include "test_WINDSGeneralData.h"
#include "test_TURBGeneralData.h"
#include "test_PlumeGeneralData.h"

TEST_CASE("vector math")
{

  printf("======================================\n");
  printf("testing vector math\n");
  printf("--------------------------------------\n");
  std::string results = TEST_PASS;

  int gridSize[3] = { 10, 10, 10 };
  float gridRes[3] = { 0.1, 0.1, 0.1 };

  auto *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  auto *TGD = new test_TURBGeneralData(WGD);
  auto *PGD = new test_PlumeGeneralData(WGD, TGD);

  printf("--------------------------------------\n");
  printf("starting PLUME vector math CPU...\n");
  PGD->testCPU(1000000);

  printf("--------------------------------------\n");

  REQUIRE(results == TEST_PASS);
#ifdef HAS_CUDA
  printf("--------------------------------------\n");
  printf("starting PLUME vector math CUDA...\n");
  // PGD->testGPU(100000);
  PGD->testGPU_struct(1000000);
#endif
}