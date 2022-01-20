#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util.h"
#include "test_WINDSGeneralData.h"
#include "test_TURBGeneralData.h"
#include "test_PlumeGeneralData.h"

std::string mainTest();
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
  printf("starting TURBULENCE tests...\n");
  results = mainTest();
  if (results == "") {
    printf("TURBULENCE: Success!\n");
  } else {
    printf("TURBULENCE: Failure\n%s\n", results.c_str());
    exit(EXIT_FAILURE);
  }

  printf("======================================\n");
  printf("All tests pass!\n");
  exit(EXIT_SUCCESS);

  return 0;
}

std::string mainTest()
{

  std::string results = TEST_PASS;

  int gridSize[3] = { 400, 400, 400 };
  float gridRes[3] = { 0.1, 0.1, 0.1 };

  WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  TURBGeneralData *TGD = new test_TURBGeneralData(WGD);
  test_PlumeGeneralData *PGD = new test_PlumeGeneralData(WGD, TGD);

  setTestVelocity(WGD);

  PGD->setInterpMethod("triLinear", WGD, TGD);

  PGD->testInterp(WGD, TGD);

  //PGD->testCPU(100000);
  //PGD->testGPU(100000);
  PGD->testGPU_struct(100000);

  return results;
}

void setTestVelocity(WINDSGeneralData *WGD)
{

  // a = 2 * 2pi/Lx
  float a = 2.0 * 2.0 * M_PI / (WGD->nx * WGD->dx);
  // b = 6 * 2pi/Ly
  float b = 6.0 * 2.0 * M_PI / (WGD->ny * WGD->dy);
  // c = 4 * 2pi/Lz
  float c = 4.0 * 2.0 * M_PI / ((WGD->nz - 1) * WGD->dz);

  // uv on vertical face -> k=0...nz-2
  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u[faceID] = cos(a * i * WGD->dx) + cos(b * WGD->y[j]) + sin(c * WGD->z[k]);
      }
    }
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->v[faceID] = cos(a * WGD->x[i]) + cos(b * j * WGD->dy) + sin(c * WGD->z[k]);
      }
    }
  }

  // w on horizontal face -> k=0...nz-1
  for (int k = 1; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->w[faceID] = cos(a * WGD->x[i]) + cos(b * WGD->y[j]) + sin(c * WGD->z_face[k]);
      }
    }
  }

  return;
}
