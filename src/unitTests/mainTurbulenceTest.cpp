#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util.h"
#include "test_WINDSGeneralData.h"
#include "test_TURBGeneralData.h"

std::string mainTest();
float compError1Dx(std::vector<float> *, std::vector<float> *, WINDSGeneralData *, test_TURBGeneralData *);
float compError1Dy(std::vector<float> *, std::vector<float> *, WINDSGeneralData *, test_TURBGeneralData *);
float compError1Dz(std::vector<float> *, std::vector<float> *, WINDSGeneralData *, test_TURBGeneralData *);

int main()
{
  std::string results;
  bool testsPassed = true;

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
    if (testsPassed) testsPassed = false;
  }

  printf("======================================\n");
  if (testsPassed) {
    printf("All tests pass!\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("At least one test failed!\n");
    exit(EXIT_FAILURE);
  }

  return 0;
}

std::string mainTest()
{

  int gridSize[3] = { 100, 100, 100 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };

  WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  test_TURBGeneralData *TGD = new test_TURBGeneralData(WGD);


  std::cout << "Check z-derivatives" << std::endl;
  float a(0.0);
  // a = 2pi/Lz
  a = 4.0 * 2.0 * M_PI / (gridSize[2] * gridRes[2]);
  std::vector<float> duv_dz, dw_dz;
  duv_dz.resize(WGD->nz - 1, 0.0);
  dw_dz.resize(WGD->nz - 1, 0.0);

  // dudz and dvdz at cell-center face -> k=1...nz-2
  for (int k = 1; k < WGD->nz - 1; k++) {
    duv_dz[k] = a * cos(a * WGD->z[k]);
  }
  // uv on vertical face -> k=0...nz-2
  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 2; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u[faceID] = sin(a * WGD->z[k]);
      }
    }
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 2; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->v[faceID] = sin(a * WGD->z[k]);
      }
    }
  }

  // dwdz at cell-center -> k=1...nz-2
  for (int k = 0; k < WGD->nz - 1; k++) {
    dw_dz[k] = a * cos(a * WGD->z[k + 1]);
  }
  // w on horizontal face -> k=0...nz-1
  for (int k = 0; k < WGD->nz; k++) {
    for (int j = 0; j < WGD->ny - 2; j++) {
      for (int i = 0; i < WGD->nx - 2; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->w[faceID] = sin(a * WGD->z_face[k]);
      }
    }
  }

  TGD->test_compDerivatives(WGD);

  float RMSE(0.0);
  float tol(1.0e-2);
  RMSE = compError1Dz(&duv_dz, &(TGD->Gxz), WGD, TGD);
  if (RMSE > tol) {
    return util_errorReport("compDerivatives",
                            "error in pure z-derivative of u-velocity\n RMSE = "
                              + std::to_string(RMSE));
  }
  RMSE = compError1Dz(&duv_dz, &(TGD->Gyz), WGD, TGD);
  if (RMSE > tol) {
    return util_errorReport("compDerivatives",
                            "error in pure z-derivative of v-velocity\n RMSE = "
                              + std::to_string(RMSE));
  }
  RMSE = compError1Dz(&dw_dz, &(TGD->Gzz), WGD, TGD);
  if (RMSE > tol) {
    return util_errorReport("compDerivatives",
                            "error in pure z-derivative of w-velocity\n RMS = "
                              + std::to_string(RMSE));
  }


  return TEST_PASS;
}

float compError1Dx(std::vector<float> *deriv,
                   std::vector<float> *var,
                   WINDSGeneralData *WGD,
                   test_TURBGeneralData *TGD)
{

  float error(0.0), numcell(0.0);

  for (std::vector<int>::iterator it = TGD->icellfluid.begin(); it != TGD->icellfluid.end(); ++it) {
    int cellID = *it;
    int k = (int)(cellID / ((WGD->nx - 1) * (WGD->ny - 1)));
    int j = (int)((cellID - k * (WGD->nx - 1) * (WGD->ny - 1)) / (WGD->nx - 1));
    int i = cellID - j * (WGD->nx - 1) - k * (WGD->nx - 1) * (WGD->ny - 1);

    error += pow(var->at(cellID) - deriv->at(i), 2.0);
    numcell++;
  }
  //std::cout << "\tEuclidian distance = " << error << std::endl;
  //std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}

float compError1Dy(std::vector<float> *deriv,
                   std::vector<float> *var,
                   WINDSGeneralData *WGD,
                   test_TURBGeneralData *TGD)
{

  float error(0.0), numcell(0.0);

  for (std::vector<int>::iterator it = TGD->icellfluid.begin(); it != TGD->icellfluid.end(); ++it) {
    int cellID = *it;
    int k = (int)(cellID / ((WGD->nx - 1) * (WGD->ny - 1)));
    int j = (int)((cellID - k * (WGD->nx - 1) * (WGD->ny - 1)) / (WGD->nx - 1));
    int i = cellID - j * (WGD->nx - 1) - k * (WGD->nx - 1) * (WGD->ny - 1);

    error += pow(var->at(cellID) - deriv->at(j), 2.0);
    numcell++;
  }
  //std::cout << "\tEuclidian distance = " << error << std::endl;
  //std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}

float compError1Dz(std::vector<float> *deriv,
                   std::vector<float> *var,
                   WINDSGeneralData *WGD,
                   test_TURBGeneralData *TGD)
{

  float error(0.0), numcell(0.0);

  for (std::vector<int>::iterator it = TGD->icellfluid.begin(); it != TGD->icellfluid.end(); ++it) {
    int cellID = *it;
    int k = (int)(cellID / ((WGD->nx - 1) * (WGD->ny - 1)));
    int j = (int)((cellID - k * (WGD->nx - 1) * (WGD->ny - 1)) / (WGD->nx - 1));
    int i = cellID - j * (WGD->nx - 1) - k * (WGD->nx - 1) * (WGD->ny - 1);

    error += pow(var->at(cellID) - deriv->at(k), 2.0);
    numcell++;
  }
  //std::cout << "\tEuclidian distance = " << error << std::endl;
  //std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}
