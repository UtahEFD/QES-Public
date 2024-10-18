#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "qes/Domain.h"

#include "test_TURBGeneralData.h"

std::string mainTest();
void set1DDerivative(std::vector<float> *,
                     std::vector<float> *,
                     std::vector<float> *,
                     std::vector<float> *,
                     std::vector<float> *,
                     std::vector<float> *,
                     std::vector<float> *,
                     std::vector<float> *,
                     std::vector<float> *,
                     WINDSGeneralData *);
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
                              TURBGeneralData *);
float compError1Dx(std::vector<float> *,
                   std::vector<float> *,
                   WINDSGeneralData *,
                   TURBGeneralData *);
float compError1Dy(std::vector<float> *,
                   std::vector<float> *,
                   WINDSGeneralData *,
                   TURBGeneralData *);
float compError1Dz(std::vector<float> *,
                   std::vector<float> *,
                   WINDSGeneralData *,
                   TURBGeneralData *);

TEST_CASE("Testing QES-Turb derivatives CPU")
{
  std::string results;

  int gridSize[3] = { 400, 400, 400 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };
  qes::Domain domain(gridSize[0], gridSize[1], gridSize[2], gridRes[0], gridRes[1], gridRes[2]);

  auto *WGD = new WINDSGeneralData(domain);
  auto *TGD = new TURBGeneralData(WGD);


  std::cout << "Checking derivatives" << std::endl;

  std::vector<float> dudx, dvdx, dwdx;
  dudx.resize(WGD->domain.nx() - 1, 0.0);
  dvdx.resize(WGD->domain.nx() - 1, 0.0);
  dwdx.resize(WGD->domain.nx() - 1, 0.0);

  std::vector<float> dudy, dvdy, dwdy;
  dudy.resize(WGD->domain.ny() - 1, 0.0);
  dvdy.resize(WGD->domain.ny() - 1, 0.0);
  dwdy.resize(WGD->domain.ny() - 1, 0.0);

  std::vector<float> dudz, dvdz, dwdz;
  dudz.resize(WGD->domain.nz() - 1, 0.0);
  dvdz.resize(WGD->domain.nz() - 1, 0.0);
  dwdz.resize(WGD->domain.nz() - 1, 0.0);

  set1DDerivative(&dudx, &dudy, &dudz, &dvdx, &dvdy, &dvdz, &dwdx, &dwdy, &dwdz, WGD);

  std::cout << "Calculation derivatives CPU" << std::endl;
  TGD->derivativeVelocity();

  std::cout << "Checking derivatives CPU" << std::endl;

  float tol(1.0e-3);

  REQUIRE(compError1Dx(&dudx, &(TGD->Gxx), WGD, TGD) < tol);
  REQUIRE(compError1Dx(&dvdx, &(TGD->Gyx), WGD, TGD) < tol);
  REQUIRE(compError1Dx(&dwdx, &(TGD->Gzx), WGD, TGD) < tol);

  REQUIRE(compError1Dy(&dudy, &(TGD->Gxy), WGD, TGD) < tol);
  REQUIRE(compError1Dy(&dvdy, &(TGD->Gyy), WGD, TGD) < tol);
  REQUIRE(compError1Dy(&dwdy, &(TGD->Gzy), WGD, TGD) < tol);

  REQUIRE(compError1Dz(&dudz, &(TGD->Gxz), WGD, TGD) < tol);
  REQUIRE(compError1Dz(&dvdz, &(TGD->Gyz), WGD, TGD) < tol);
  REQUIRE(compError1Dz(&dwdz, &(TGD->Gzz), WGD, TGD) < tol);
}

void set1DDerivative(std::vector<float> *dudx,
                     std::vector<float> *dudy,
                     std::vector<float> *dudz,
                     std::vector<float> *dvdx,
                     std::vector<float> *dvdy,
                     std::vector<float> *dvdz,
                     std::vector<float> *dwdx,
                     std::vector<float> *dwdy,
                     std::vector<float> *dwdz,
                     WINDSGeneralData *WGD)
{

  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  // a = 2 * 2pi/Lx
  float a = 2.0 * 2.0 * M_PI / (nx * dx);
  // b = 6 * 2pi/Ly
  float b = 6.0 * 2.0 * M_PI / (ny * dy);
  // c = 4 * 2pi/Lz
  float c = 4.0 * 2.0 * M_PI / ((nz - 1) * dz);

  // uv on vertical face -> k=0...nz-2
  for (int k = 0; k < nz - 1; k++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        // int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        int faceID = WGD->domain.face(i, j, k);
        WGD->u[faceID] = cos(a * i * dx) + cos(b * WGD->domain.y[j]) + sin(c * WGD->domain.z[k]);
      }
    }
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        int faceID = WGD->domain.face(i, j, k);
        WGD->v[faceID] = cos(a * WGD->domain.x[i]) + cos(b * j * dy) + sin(c * WGD->domain.z[k]);
      }
    }
  }
  // dudx and dvdx at cell-center face -> i=1...ny-2
  for (int i = 0; i < nx - 2; i++) {
    dudx->at(i) = -a * sin(a * WGD->domain.x[i]);
    dvdx->at(i) = -a * sin(a * WGD->domain.x[i]);
  }
  // dudy and dvdy at cell-center face -> j=0...ny-2
  for (int j = 0; j < ny - 2; j++) {
    dudy->at(j) = -b * sin(b * WGD->domain.y[j]);
    dvdy->at(j) = -b * sin(b * WGD->domain.y[j]);
  }
  // dudz and dvdz at cell-center face -> k=1...nz-2
  for (int k = 1; k < nz - 2; k++) {
    dudz->at(k) = c * cos(c * WGD->domain.z[k]);
    dvdz->at(k) = c * cos(c * WGD->domain.z[k]);
  }

  // w on horizontal face -> k=0...nz-1
  for (int k = 1; k < nz - 1; k++) {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        int faceID = WGD->domain.face(i, j, k);
        WGD->w[faceID] = cos(a * WGD->domain.x[i]) + cos(b * WGD->domain.y[j]) + sin(c * WGD->domain.z_face[k]);
      }
    }
  }
  // dwdx at cell-center -> i=1...nx-3
  for (int i = 0; i < nx - 2; i++) {
    dwdx->at(i) = -a * sin(a * WGD->domain.x[i]);
  }
  // dwdx at cell-center -> j=1...nz-3
  for (int j = 0; j < nz - 2; j++) {
    dwdy->at(j) = -b * sin(b * WGD->domain.y[j]);
  }
  // dwdx at cell-center -> k=1...nz-2
  for (int k = 1; k < nz - 2; k++) {
    dwdz->at(k) = c * cos(c * WGD->domain.z[k]);
  }

  return;
}

float compError1Dx(std::vector<float> *deriv,
                   std::vector<float> *var,
                   WINDSGeneralData *WGD,
                   TURBGeneralData *TGD)
{

  float error(0.0), numcell(0.0);

  for (std::vector<int>::iterator it = TGD->icellfluid.begin(); it != TGD->icellfluid.end(); ++it) {
    int cellID = *it;
    auto [i, j, k] = WGD->domain.getCellIdx(cellID);

    error += pow((var->at(cellID) - deriv->at(i)), 2.0);
    numcell++;
  }
  // std::cout << "\tEuclidian distance = " << error << std::endl;
  // std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}

float compError1Dy(std::vector<float> *deriv,
                   std::vector<float> *var,
                   WINDSGeneralData *WGD,
                   TURBGeneralData *TGD)
{

  float error(0.0), numcell(0.0);

  for (std::vector<int>::iterator it = TGD->icellfluid.begin(); it != TGD->icellfluid.end(); ++it) {
    int cellID = *it;
    auto [i, j, k] = WGD->domain.getCellIdx(cellID);

    error += pow((var->at(cellID) - deriv->at(j)), 2.0);
    numcell++;
  }
  // std::cout << "\tEuclidian distance = " << error << std::endl;
  // std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}

float compError1Dz(std::vector<float> *deriv,
                   std::vector<float> *var,
                   WINDSGeneralData *WGD,
                   TURBGeneralData *TGD)
{

  float error(0.0), numcell(0.0);

  for (std::vector<int>::iterator it = TGD->icellfluid.begin(); it != TGD->icellfluid.end(); ++it) {
    int cellID = *it;
    auto [i, j, k] = WGD->domain.getCellIdx(cellID);

    error += pow((var->at(cellID) - deriv->at(k)), 2.0);
    numcell++;
  }
  // std::cout << "\tEuclidian distance = " << error << std::endl;
  // std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}
