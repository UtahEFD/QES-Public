/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file turbulence_derivative_CPU.cpp
 * @brief This is a unit test of derivative calculations
 */

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "qes/Domain.h"

#include "test_TURBGeneralData.h"

void setVelocityDerivatives(std::vector<float> *,
                            std::vector<float> *,
                            std::vector<float> *,
                            std::vector<float> *,
                            std::vector<float> *,
                            std::vector<float> *,
                            std::vector<float> *,
                            std::vector<float> *,
                            std::vector<float> *,
                            WINDSGeneralData *);
void setStressDerivatives(std::vector<float> *,
                          std::vector<float> *,
                          std::vector<float> *,
                          TURBGeneralData *);
float compError1Dx(std::vector<float> *, std::vector<float> *, const qes::Domain &);
float compError1Dy(std::vector<float> *, std::vector<float> *, const qes::Domain &);
float compError1Dz(std::vector<float> *, std::vector<float> *, const qes::Domain &);

TEST_CASE("Testing QES-TURB derivatives GPU")
{
  std::cout << "--------------------------------------------------------------\n"
            << "Testing QES-TURB derivatives GPU \n"
            << "--------------------------------------------------------------" << std::endl;
  const float tol(1.0e-3);

  int gridSize[3] = { 400, 400, 400 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };
  qes::Domain domain(gridSize[0], gridSize[1], gridSize[2], gridRes[0], gridRes[1], gridRes[2]);

  auto *WGD = new WINDSGeneralData(domain);
  auto *TGD = new TURBGeneralData(WGD);

  std::cout << "checking velocity derivatives" << std::endl;

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

  setVelocityDerivatives(&dudx, &dudy, &dudz, &dvdx, &dvdy, &dvdz, &dwdx, &dwdy, &dwdz, WGD);
  TGD->derivativeVelocity();

  REQUIRE(compError1Dx(&dudx, &(TGD->Gxx), domain) < tol);
  REQUIRE(compError1Dx(&dvdx, &(TGD->Gyx), domain) < tol);
  REQUIRE(compError1Dx(&dwdx, &(TGD->Gzx), domain) < tol);

  REQUIRE(compError1Dy(&dudy, &(TGD->Gxy), domain) < tol);
  REQUIRE(compError1Dy(&dvdy, &(TGD->Gyy), domain) < tol);
  REQUIRE(compError1Dy(&dwdy, &(TGD->Gzy), domain) < tol);

  REQUIRE(compError1Dz(&dudz, &(TGD->Gxz), domain) < tol);
  REQUIRE(compError1Dz(&dvdz, &(TGD->Gyz), domain) < tol);
  REQUIRE(compError1Dz(&dwdz, &(TGD->Gzz), domain) < tol);

  std::cout << "checking stress tensor derivatives" << std::endl;

  std::vector<float> div_tau_x, div_tau_y, div_tau_z;
  div_tau_x.resize(WGD->domain.nx() - 1, 0.0);
  div_tau_y.resize(WGD->domain.ny() - 1, 0.0);
  div_tau_z.resize(WGD->domain.nz() - 1, 0.0);

  setStressDerivatives(&div_tau_x, &div_tau_y, &div_tau_z, TGD);
  TGD->divergenceStress();

  REQUIRE(compError1Dx(&div_tau_x, &(TGD->div_tau_x), domain) < tol);
  REQUIRE(compError1Dy(&div_tau_y, &(TGD->div_tau_y), domain) < tol);
  REQUIRE(compError1Dz(&div_tau_z, &(TGD->div_tau_z), domain) < tol);
}

void setVelocityDerivatives(std::vector<float> *dudx,
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
  // dudx and dvdx at cell-center face -> i=1...nx-2
  for (int i = 1; i < nx - 2; i++) {
    dudx->at(i) = -a * sin(a * WGD->domain.x[i]);
    dvdx->at(i) = -a * sin(a * WGD->domain.x[i]);
  }
  // dudy and dvdy at cell-center face -> j=0...ny-2
  for (int j = 1; j < ny - 2; j++) {
    dudy->at(j) = -b * sin(b * WGD->domain.y[j]);
    dvdy->at(j) = -b * sin(b * WGD->domain.y[j]);
  }
  // dudz and dvdz at cell-center face -> k=1...nz-2
  for (int k = 1; k < nz - 2; k++) {
    dudz->at(k) = c * cos(c * WGD->domain.z[k]);
    dvdz->at(k) = c * cos(c * WGD->domain.z[k]);
  }

  // w on horizontal face -> k=1...nz-1
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
}

void setStressDerivatives(std::vector<float> *div_tau_x,
                          std::vector<float> *div_tau_y,
                          std::vector<float> *div_tau_z,
                          TURBGeneralData *TGD)
{
  auto [nx, ny, nz] = TGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = TGD->domain.getDomainSize();

  // a = 2 * 2pi/Lx
  float a = 2.0 * 2.0 * M_PI / (nx * dx);
  // b = 6 * 2pi/Ly
  float b = 6.0 * 2.0 * M_PI / (ny * dy);
  // c = 4 * 2pi/Lz
  float c = 4.0 * 2.0 * M_PI / ((nz - 1) * dz);

  for (int k = 0; k < nz - 1; ++k) {
    for (int j = 0; j < ny - 1; ++j) {
      for (int i = 0; i < nx - 1; ++i) {
        TGD->txx[TGD->domain.cell(i, j, k)] = cos(a * TGD->domain.x[i]);
      }
    }
  }
  for (int i = 0; i < nx - 1; ++i) {
    div_tau_x->at(i) = -a * sin(a * TGD->domain.x[i]);
  }

  for (int k = 0; k < nz - 1; ++k) {
    for (int j = 0; j < ny - 1; ++j) {
      for (int i = 0; i < nx - 1; ++i) {
        TGD->tyy[TGD->domain.cell(i, j, k)] = cos(b * TGD->domain.y[j]);
      }
    }
  }
  for (int j = 0; j < ny - 1; ++j) {
    div_tau_y->at(j) = -b * sin(b * TGD->domain.y[j]);
  }

  for (int k = 0; k < nz - 1; ++k) {
    for (int j = 0; j < ny - 1; ++j) {
      for (int i = 0; i < nx - 1; ++i) {
        TGD->tzz[TGD->domain.cell(i, j, k)] = sin(c * TGD->domain.z[k]);
      }
    }
  }
  for (int k = 0; k < nz - 1; ++k) {
    div_tau_z->at(k) = c * cos(c * TGD->domain.z[k]);
  }
}

float compError1Dx(std::vector<float> *deriv, std::vector<float> *var, const qes::Domain &domain)
{

  float error(0.0), numcell(0.0);
  for (int k = 1; k < domain.nz() - 2; ++k) {
    for (int j = 1; j < domain.ny() - 2; ++j) {
      for (int i = 1; i < domain.nx() - 2; ++i) {
        error += pow((var->at(domain.cell(i, j, k)) - deriv->at(i)), 2.0);
        numcell++;
      }
    }
  }
  // std::cout << "\tEuclidian distance = " << error << std::endl;
  // std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}

float compError1Dy(std::vector<float> *deriv, std::vector<float> *var, const qes::Domain &domain)
{

  float error(0.0), numcell(0.0);

  for (int k = 1; k < domain.nz() - 2; ++k) {
    for (int j = 1; j < domain.ny() - 2; ++j) {
      for (int i = 1; i < domain.nx() - 2; ++i) {
        error += powf((var->at(domain.cell(i, j, k)) - deriv->at(j)), 2.0);
        numcell++;
      }
    }
  }

  // std::cout << "\tEuclidian distance = " << error << std::endl;
  // std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}
float compError1Dz(std::vector<float> *deriv, std::vector<float> *var, const qes::Domain &domain)
{

  float error(0.0), numcell(0.0);
  for (int k = 1; k < domain.nz() - 2; ++k) {
    for (int j = 1; j < domain.ny() - 2; ++j) {
      for (int i = 1; i < domain.nx() - 2; ++i) {
        error += pow((var->at(domain.cell(i, j, k)) - deriv->at(k)), 2.0);
        numcell++;
      }
    }
  }
  // std::cout << "\tEuclidian distance = " << error << std::endl;
  // std::cout << "\tRMSE = " << sqrt(error / numcell) << std::endl;
  return sqrt(error / numcell);
}
