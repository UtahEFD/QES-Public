#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "test_functions.h"
#include "test_WINDSGeneralData.h"
#include "test_TURBGeneralData.h"
#include "test_PlumeGeneralData.h"

TEST_CASE("vector math")
{

  printf("======================================\n");
  printf("testing vector math\n");
  printf("--------------------------------------\n");
  std::string results = "";

  int gridSize[3] = { 10, 10, 10 };
  float gridRes[3] = { 0.1, 0.1, 0.1 };

  auto *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  auto *TGD = new test_TURBGeneralData(WGD);
  PlumeParameters PP("", false, false);
  auto *PGD = new test_PlumeGeneralData(PP, WGD, TGD);

  printf("--------------------------------------\n");
  printf("starting PLUME vector math CPU...\n");
  // PGD->testCPU(1000000);

  printf("--------------------------------------\n");

  REQUIRE(results == "");
#ifdef HAS_CUDA
  printf("--------------------------------------\n");
  printf("starting PLUME vector math CUDA...\n");
  // PGD->testGPU(100000);
  // PGD->testGPU_struct(1000000);
#endif
  
  delete WGD;
  delete TGD;
  delete PGD;
}

#if 0
void test_PlumeGeneralData::testCPU(int length)
{

  mat3 tmp = { 1, 2, 3, 2, 1, 2, 3, 2, 1 };
  std::vector<mat3> A;
  A.resize(length, tmp);

  std::vector<vec3> b;
  b.resize(length, { 1.0, 1.0, 1.0 });

  std::vector<vec3> x;
  x.resize(length, { 0.0, 0.0, 0.0 });

  std::vector<mat3sym> tau;
  // tau.resize(length, { 1, 2, 3, 1, 2, 1 });
  tau.resize(length, { 1, 0, 3, 0, 0, 1 });
  std::vector<vec3> invar;
  invar.resize(length, { 0.0, 0.0, 0.0 });

  auto cpuStartTime = std::chrono::high_resolution_clock::now();
  for (auto it = 0; it < length; ++it) {
    bool tt = vectorMath::invert3(A[it]);
    vectorMath::matmult(A[it], b[it], x[it]);
  }

  for (auto it = 0; it < length; ++it) {
    vectorMath::makeRealizable(10e-4, tau[it]);
    vectorMath::calcInvariants(tau[it], invar[it]);
  }

  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpuElapsed = cpuEndTime - cpuStartTime;
  std::cout << "CPU  elapsed time: " << cpuElapsed.count() << " s\n";

  std::cout << A[0]._11 << " " << A[0]._12 << " " << A[0]._13 << std::endl;
  std::cout << A[0]._21 << " " << A[0]._22 << " " << A[0]._23 << std::endl;
  std::cout << A[0]._31 << " " << A[0]._32 << " " << A[0]._33 << std::endl;
  std::cout << x[0]._1 << " " << x[0]._2 << " " << x[0]._3 << std::endl;

  std::cout << std::endl;

  std::cout << tau[0]._11 << " " << tau[0]._12 << " " << tau[0]._13 << std::endl;
  std::cout << tau[0]._12 << " " << tau[0]._22 << " " << tau[0]._23 << std::endl;
  std::cout << tau[0]._13 << " " << tau[0]._23 << " " << tau[0]._33 << std::endl;
  std::cout << invar[0]._1 << " " << invar[0]._2 << " " << invar[0]._3 << std::endl;

  return;
}
#endif