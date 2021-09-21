#include "test_Turbulence.h"

std::string test_Turbulence::mainTest()
{

  int gridSize[3] = { 100, 100, 100 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };

  WGD = new test_WINDSGeneralData(gridSize, gridRes);
  TGD = new test_TURBGeneralData(WGD);


  float a(0.0);
  // a = 2pi/Lz
  a = 2.0 * M_PI / (gridSize[2] * gridRes[2]);
  std::vector<float> deriv;
  deriv.resize(WGD->nz - 1, 0.0);

  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 2; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u[faceID] = sin(a * WGD->z[k]);
        deriv[k] = a * cos(a * WGD->z[k]);
      }
    }
  }

  TGD->test_compDerivatives(WGD);

  bool test = checkError1Dz(deriv, 1.0e-4);

  return TEST_PASS;
}

bool test_Turbulence::checkError1Dz(std::vector<float> deriv, float tol)
{

  float error(0.0), numcell(0.0);

  for (std::vector<int>::iterator it = TGD->icellfluid.begin(); it != TGD->icellfluid.end(); ++it) {
    int cellID = *it;
    int k = (int)(cellID / ((WGD->nx - 1) * (WGD->ny - 1)));
    int j = (int)((cellID - k * (WGD->nx - 1) * (WGD->ny - 1)) / (WGD->nx - 1));
    int i = cellID - j * (WGD->nx - 1) - k * (WGD->nx - 1) * (WGD->ny - 1);
    //std::cout << WGD->z[k] << " " << TGD->Gxz[cellID]
    //          << " " << a * cos(a * WGD->z[k]) << std::endl;

    error += pow(TGD->Gxz[cellID] - deriv[k], 2.0);
    numcell++;
  }
  std::cout << "Euclidian distance = " << error << std::endl;
  std::cout << "RMSE = " << sqrt(error / numcell) << std::endl;
  if (sqrt(error / numcell) < tol) {
    return true;
  } else {
    return false;
  }
}
