#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util.h"
#include "test_WINDSGeneralData.h"
#include "test_TURBGeneralData.h"
#include "test_PlumeGeneralData.h"

std::string testInterpolation();
std::string testVectorMath();

void setTestVelocity(WINDSGeneralData *);
std::string check1DDerivative(std::vector<float> *,
                              std::vector<float> *,
                              std::vector<float> *,
                              std::vector<float> *,
                              std::vector<float> *,
                              std::vector<float> *,
                              std::vector<float> *,
                              std::vector<float> *,
                              std::vector<float> *,
                              WINDSGeneralData *,
                              test_TURBGeneralData *);
float compError1Dx(std::vector<float> *,
                   std::vector<float> *,
                   WINDSGeneralData *,
                   test_TURBGeneralData *);
float compError1Dy(std::vector<float> *,
                   std::vector<float> *,
                   WINDSGeneralData *,
                   test_TURBGeneralData *);
float compError1Dz(std::vector<float> *,
                   std::vector<float> *,
                   WINDSGeneralData *,
                   test_TURBGeneralData *);

int main()
{
  std::string results;

  /******************
   * TURBULENCE * 
   ******************/
  printf("======================================\n");
  printf("starting PLUME tests...\n");
  printf("======================================\n");
  printf("testing interpolation\n");
  printf("--------------------------------------\n");
  results = testInterpolation();
  printf("--------------------------------------\n");
  if (results == TEST_PASS) {
    printf("PLUME: Success!\n");
  } else {
    printf("PLUME: Failure\n%s\n", results.c_str());
    exit(EXIT_FAILURE);
  }
  printf("======================================\n");
  printf("testing vector math\n");
  printf("--------------------------------------\n");
  results = testVectorMath();
  printf("--------------------------------------\n");
  if (results == TEST_PASS) {
    printf("PLUME: Success!\n");
  } else {
    printf("PLUME: Failure\n%s\n", results.c_str());
    exit(EXIT_FAILURE);
  }


  printf("======================================\n");
  printf("All tests pass!\n");
  exit(EXIT_SUCCESS);

  return 0;
}

std::string testInterpolation()
{

  std::string results = TEST_PASS;

  int gridSize[3] = { 400, 400, 400 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };

  WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  TURBGeneralData *TGD = new test_TURBGeneralData(WGD);
  test_PlumeGeneralData *PGD = new test_PlumeGeneralData(WGD, TGD);

  PGD->setInterpMethod("triLinear", WGD, TGD);

  printf("--------------------------------------\n");
  printf("testing for accuracy\n");
  results = PGD->testInterp(WGD, TGD);
  if (results != TEST_PASS)
    return results;

  printf("--------------------------------------\n");
  printf("testing for time on CPU\n");
  results = PGD->timeInterpCPU(WGD, TGD);
  if (results != TEST_PASS)
    return results;

  return TEST_PASS;
}

std::string testVectorMath()
{

  std::string results = TEST_PASS;

  int gridSize[3] = { 10, 10, 10 };
  float gridRes[3] = { 0.1, 0.1, 0.1 };

  WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  TURBGeneralData *TGD = new test_TURBGeneralData(WGD);
  test_PlumeGeneralData *PGD = new test_PlumeGeneralData(WGD, TGD);

  printf("--------------------------------------\n");
  printf("starting PLUME vector math CPU...\n");
  PGD->testCPU(1000000);

#ifdef HAS_CUDA
  printf("--------------------------------------\n");
  printf("starting PLUME vector math CUDA...\n");
  //PGD->testGPU(100000);
  PGD->testGPU_struct(1000000);
#endif
  return results;
}
