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
 * @file plume_interpolation_CPU.cpp
 * @brief This is a unit test of tri-linear interpolation for CPU
 */

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "plume/InterpTriLinear.h"

#include "test_functions.h"

TEST_CASE("test tri-linear interpolation fine grid", "[Working]")
{

  int gridSize[3] = { 200, 200, 200 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };
  qes::Domain domain(gridSize[0], gridSize[1], gridSize[2], gridRes[0], gridRes[1], gridRes[2]);
  WINDSGeneralData *WGD = new WINDSGeneralData(domain);
  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  auto interp = new InterpTriLinear(WGD->domain, true);

  auto *tf = new test_functions(WGD, TGD, "trig");

  SECTION("testing for accuracy")
  {
    std::vector<float> xArray = { 10 + .01, 10 + .01, 10 + .51, 10 + .00, 125 };
    std::vector<float> yArray = { 10 + .51, 10 + .51, 10 + .51, 10 + .00, 136 };
    std::vector<float> zArray = { 10 + .51, 10 + .50, 10 + .01, 10 + .01, 190 };

    float tol(1.0e-2);

    for (size_t it = 0; it < xArray.size(); ++it) {
      vec3 pos = { xArray[it], yArray[it], zArray[it] };
      vec3 vel = { 0.0, 0.0, 0.0 };
      mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      vec3 flux_div = { 0.0, 0.0, 0.0 };
      float CoEps = 1e-6, nuT = 0.0;

      interp->interpWindsValues(WGD, pos, vel);
      interp->interpTurbValues(TGD, pos, tau, flux_div, nuT, CoEps);
      float err = 0.0;

      err = std::abs(tf->u_test_function->val(pos._1, pos._2, pos._3) - vel._1);
      REQUIRE(err < tol);

      err = std::abs(tf->v_test_function->val(pos._1, pos._2, pos._3) - vel._2);
      REQUIRE(err < tol);

      err = std::abs(tf->w_test_function->val(pos._1, pos._2, pos._3) - vel._3);
      REQUIRE(err < tol);

      err = std::abs(tf->c_test_function->val(pos._1, pos._2, pos._3) - tau._11);
      REQUIRE(err < tol);
    }
  }

  SECTION("testing random points")
  {
    int N = 1000000;

    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{ rnd_device() };// Generates random integers
    std::uniform_real_distribution<float> disX{ WGD->domain.x[0], WGD->domain.x.back() };
    std::uniform_real_distribution<float> disY{ WGD->domain.y[0], WGD->domain.y.back() };
    std::uniform_real_distribution<float> disZ{ 0, WGD->domain.z_face[WGD->domain.nz() - 3] };

    std::vector<float> xArray, yArray, zArray;

    for (int it = 0; it < N; ++it) {
      xArray.push_back(disX(mersenne_engine));
      yArray.push_back(disY(mersenne_engine));
      zArray.push_back(disZ(mersenne_engine));
    }

    float tol(1.0e-2);
    float errU = 0.0, errV = 0.0, errW = 0.0, errT = 0.0;

    for (size_t it = 0; it < xArray.size(); ++it) {
      vec3 pos = { xArray[it], yArray[it], zArray[it] };
      vec3 vel = { 0.0, 0.0, 0.0 };
      mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      vec3 flux_div = { 0.0, 0.0, 0.0 };
      float CoEps = 1e-6, nuT = 0.0;

      interp->interpWindsValues(WGD, pos, vel);
      interp->interpTurbValues(TGD, pos, tau, flux_div, nuT, CoEps);


      errU += std::abs(tf->u_test_function->val(pos._1, pos._2, pos._3) - vel._1);
      errV += std::abs(tf->v_test_function->val(pos._1, pos._2, pos._3) - vel._2);
      errW += std::abs(tf->w_test_function->val(pos._1, pos._2, pos._3) - vel._3);
      errT += std::abs(tf->c_test_function->val(pos._1, pos._2, pos._3) - tau._11);
    }

    errU = errU / float(N);
    REQUIRE(errU < tol);
    errV = errV / float(N);
    REQUIRE(errV < tol);
    errW = errW / float(N);
    REQUIRE(errW < tol);
    errT = errT / float(N);
    REQUIRE(errT < tol);
  }

  delete WGD;
  delete TGD;
  delete interp;
}

TEST_CASE("test tri-linear interpolation coarse grid", "[Working]")
{

  int gridSize[3] = { 200, 200, 200 };
  float gridRes[3] = { 2.0, 2.0, 2.0 };
  qes::Domain domain(gridSize[0], gridSize[1], gridSize[2], gridRes[0], gridRes[1], gridRes[2]);
  WINDSGeneralData *WGD = new WINDSGeneralData(domain);
  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  auto interp = new InterpTriLinear(WGD->domain, true);

  test_functions *tf = new test_functions(WGD, TGD, "trig");

  SECTION("testing for accuracy")
  {
    // results = PGD->testInterp(WGD, TGD);
    std::vector<float> xArray = { 10 + .01, 10 + .01, 10 + .51, 10 + .00, 125 };
    std::vector<float> yArray = { 10 + .51, 10 + .51, 10 + .51, 10 + .00, 136 };
    std::vector<float> zArray = { 10 + .51, 10 + .50, 10 + .01, 10 + .01, 190 };

    float tol(1.0e-2);

    for (size_t it = 0; it < xArray.size(); ++it) {
      vec3 pos = { xArray[it], yArray[it], zArray[it] };
      vec3 vel = { 0.0, 0.0, 0.0 };
      mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      vec3 flux_div = { 0.0, 0.0, 0.0 };
      float CoEps = 1e-6, nuT = 0.0;

      interp->interpWindsValues(WGD, pos, vel);
      interp->interpTurbValues(TGD, pos, tau, flux_div, nuT, CoEps);

      float err = 0.0;

      err = std::abs(tf->u_test_function->val(pos._1, pos._2, pos._3) - vel._1);
      REQUIRE(err < tol);

      err = std::abs(tf->v_test_function->val(pos._1, pos._2, pos._3) - vel._2);
      REQUIRE(err < tol);

      err = std::abs(tf->w_test_function->val(pos._1, pos._2, pos._3) - vel._3);
      REQUIRE(err < tol);

      err = std::abs(tf->c_test_function->val(pos._1, pos._2, pos._3) - tau._11);
      REQUIRE(err < tol);
    }
  }
  delete WGD;
  delete TGD;
  delete interp;
}

TEST_CASE("test tri-linear interpolation stretched grid", "[in progress]")
{

  int gridSize[3] = { 200, 200, 200 };
  float gridRes[3] = { 2.0, 2.0, 1.0 };
  float dz_array[200] = {};
  for (int k = 0; k < gridSize[2]; ++k) {
    dz_array[k] = (k + 1) * gridRes[2];
  }
  qes::Domain domain(gridSize[0], gridSize[1], gridSize[2], gridRes[0], gridRes[1], gridRes[2]);
  WINDSGeneralData *WGD = new WINDSGeneralData(domain);
  TURBGeneralData *TGD = new TURBGeneralData(WGD);

  auto interp = new InterpTriLinear(WGD->domain, true);

  test_functions *tf = new test_functions(WGD, TGD, "trig");

  SECTION("testing for accuracy")
  {
    // results = PGD->testInterp(WGD, TGD);
    std::vector<float> xArray = { 10 + .01, 10 + .01, 10 + .51, 10 + .00, 125 };
    std::vector<float> yArray = { 10 + .51, 10 + .51, 10 + .51, 10 + .00, 136 };
    std::vector<float> zArray = { 10 + .51, 10 + .50, 10 + .01, 10 + .01, 190 };

    float tol(1.0e-2);

    for (size_t it = 0; it < xArray.size(); ++it) {
      vec3 pos = { xArray[it], yArray[it], zArray[it] };
      vec3 vel = { 0.0, 0.0, 0.0 };
      mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      vec3 flux_div = { 0.0, 0.0, 0.0 };
      float CoEps = 1e-6, nuT = 0.0;

      interp->interpWindsValues(WGD, pos, vel);
      interp->interpTurbValues(TGD, pos, tau, flux_div, nuT, CoEps);

      float err = 0.0;

      err = std::abs(tf->u_test_function->val(pos._1, pos._2, pos._3) - vel._1);
      REQUIRE(err < tol);

      err = std::abs(tf->v_test_function->val(pos._1, pos._2, pos._3) - vel._2);
      REQUIRE(err < tol);

      err = std::abs(tf->w_test_function->val(pos._1, pos._2, pos._3) - vel._3);
      REQUIRE(err < tol);

      err = std::abs(tf->c_test_function->val(pos._1, pos._2, pos._3) - tau._11);
      REQUIRE(err < tol);
    }
  }

  SECTION("testing random points")
  {
    int N = 1000000;

    // First create an instance of an engine.
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{ rnd_device() };// Generates random integers
    std::uniform_real_distribution<float> disX{ WGD->domain.x[0], WGD->domain.x.back() };
    std::uniform_real_distribution<float> disY{ WGD->domain.y[0], WGD->domain.y.back() };
    std::uniform_real_distribution<float> disZ{ 0, WGD->domain.z_face[WGD->domain.nz() - 3] };

    std::vector<float> xArray, yArray, zArray;

    for (int it = 0; it < N; ++it) {
      xArray.push_back(disX(mersenne_engine));
      yArray.push_back(disY(mersenne_engine));
      zArray.push_back(disZ(mersenne_engine));
    }

    float tol(1.0e-2);
    float errU = 0.0, errV = 0.0, errW = 0.0, errT = 0.0;

    for (size_t it = 0; it < xArray.size(); ++it) {
      vec3 pos = { xArray[it], yArray[it], zArray[it] };
      vec3 vel = { 0.0, 0.0, 0.0 };
      mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      vec3 flux_div = { 0.0, 0.0, 0.0 };
      float CoEps = 1e-6, nuT = 0.0;

      interp->interpWindsValues(WGD, pos, vel);
      interp->interpTurbValues(TGD, pos, tau, flux_div, nuT, CoEps);


      errU += std::abs(tf->u_test_function->val(pos._1, pos._2, pos._3) - vel._1);
      errV += std::abs(tf->v_test_function->val(pos._1, pos._2, pos._3) - vel._2);
      errW += std::abs(tf->w_test_function->val(pos._1, pos._2, pos._3) - vel._3);
      errT += std::abs(tf->c_test_function->val(pos._1, pos._2, pos._3) - tau._11);
    }

    errU = errU / float(N);
    REQUIRE(errU < tol);
    errV = errV / float(N);
    REQUIRE(errV < tol);
    errW = errW / float(N);
    REQUIRE(errW < tol);
    errT = errT / float(N);
    REQUIRE(errT < tol);
  }

  delete WGD;
  delete TGD;
  delete interp;
}
