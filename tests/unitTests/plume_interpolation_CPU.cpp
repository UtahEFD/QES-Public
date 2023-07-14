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

TEST_CASE("interpolation fine grid", "[Working]")
{

  int gridSize[3] = { 200, 200, 200 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };

  test_WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  test_TURBGeneralData *TGD = new test_TURBGeneralData(WGD);
  test_PlumeGeneralData *PGD = new test_PlumeGeneralData(WGD, TGD);
  PGD->setInterpMethod("triLinear", WGD, TGD);

  testFunctions *tf = new testFunctions(WGD, TGD, "trig");

  SECTION("testing for accuracy")
  {
    // results = PGD->testInterp(WGD, TGD);
    std::vector<float> xArray = { 10 + .01, 10 + .01, 10 + .51, 10 + .00, 125 };
    std::vector<float> yArray = { 10 + .51, 10 + .51, 10 + .51, 10 + .00, 136 };
    std::vector<float> zArray = { 10 + .51, 10 + .50, 10 + .01, 10 + .01, 190 };

    float tol(1.0e-2);

    for (size_t it = 0; it < xArray.size(); ++it) {
      double xPos = xArray[it];
      double yPos = yArray[it];
      double zPos = zArray[it];

      double uMean = 0.0, vMean = 0.0, wMean = 0.0;
      double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
      double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;
      double CoEps = 1e-6;

      PGD->interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
      float err = 0.0;

      err = std::abs(tf->u_testFunction->val(xPos, yPos, zPos) - uMean);
      REQUIRE(err < tol);

      err = std::abs(tf->v_testFunction->val(xPos, yPos, zPos) - vMean);
      REQUIRE(err < tol);

      err = std::abs(tf->w_testFunction->val(xPos, yPos, zPos) - wMean);
      REQUIRE(err < tol);

      err = std::abs(tf->c_testFunction->val(xPos, yPos, zPos) - txx);
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
    std::uniform_real_distribution<float> disX{ WGD->x[0], WGD->x.back() };
    std::uniform_real_distribution<float> disY{ WGD->y[0], WGD->y.back() };
    std::uniform_real_distribution<float> disZ{ 0, WGD->z_face[WGD->nz - 3] };

    std::vector<float> xArray, yArray, zArray;

    for (int it = 0; it < N; ++it) {
      xArray.push_back(disX(mersenne_engine));
      yArray.push_back(disY(mersenne_engine));
      zArray.push_back(disZ(mersenne_engine));
    }

    float tol(1.0e-2);
    float errU = 0.0, errV = 0.0, errW = 0.0, errT = 0.0;

    for (size_t it = 0; it < xArray.size(); ++it) {
      double xPos = xArray[it];
      double yPos = yArray[it];
      double zPos = zArray[it];

      double uMean = 0.0, vMean = 0.0, wMean = 0.0;
      double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
      double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;
      double CoEps = 1e-6;

      PGD->interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);

      errU += std::abs(tf->u_testFunction->val(xPos, yPos, zPos) - uMean);
      errV += std::abs(tf->v_testFunction->val(xPos, yPos, zPos) - vMean);
      errW += std::abs(tf->w_testFunction->val(xPos, yPos, zPos) - wMean);
      errT += std::abs(tf->c_testFunction->val(xPos, yPos, zPos) - txx);
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
}

TEST_CASE("interpolation coarse grid", "[Working]")
{

  int gridSize[3] = { 200, 200, 200 };
  float gridRes[3] = { 2.0, 2.0, 2.0 };

  test_WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes);
  test_TURBGeneralData *TGD = new test_TURBGeneralData(WGD);
  test_PlumeGeneralData *PGD = new test_PlumeGeneralData(WGD, TGD);
  PGD->setInterpMethod("triLinear", WGD, TGD);

  testFunctions *tf = new testFunctions(WGD, TGD, "trig");

  SECTION("testing for accuracy")
  {
    // results = PGD->testInterp(WGD, TGD);
    std::vector<float> xArray = { 10 + .01, 10 + .01, 10 + .51, 10 + .00, 125 };
    std::vector<float> yArray = { 10 + .51, 10 + .51, 10 + .51, 10 + .00, 136 };
    std::vector<float> zArray = { 10 + .51, 10 + .50, 10 + .01, 10 + .01, 190 };

    float tol(1.0e-2);

    for (size_t it = 0; it < xArray.size(); ++it) {
      double xPos = xArray[it];
      double yPos = yArray[it];
      double zPos = zArray[it];

      double uMean = 0.0, vMean = 0.0, wMean = 0.0;
      double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
      double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;
      double CoEps = 1e-6;

      PGD->interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
      float err = 0.0;

      err = std::abs(tf->u_testFunction->val(xPos, yPos, zPos) - uMean);
      REQUIRE(err < tol);

      err = std::abs(tf->v_testFunction->val(xPos, yPos, zPos) - vMean);
      REQUIRE(err < tol);

      err = std::abs(tf->w_testFunction->val(xPos, yPos, zPos) - wMean);
      REQUIRE(err < tol);

      err = std::abs(tf->c_testFunction->val(xPos, yPos, zPos) - txx);
      REQUIRE(err < tol);
    }
  }
}

TEST_CASE("interpolation stretched grid", "[Working]")
{

  int gridSize[3] = { 200, 200, 200 };
  float gridRes[3] = { 2.0, 2.0, 0.1 };
  float dz_array[200] = {};
  for (int k = 0; k < gridSize[2]; ++k) {
    dz_array[k] = (k + 1) * gridRes[2];
  }

  test_WINDSGeneralData *WGD = new test_WINDSGeneralData(gridSize, gridRes, dz_array);

  test_TURBGeneralData *TGD = new test_TURBGeneralData(WGD);
  test_PlumeGeneralData *PGD = new test_PlumeGeneralData(WGD, TGD);
  PGD->setInterpMethod("triLinear", WGD, TGD);

  testFunctions *tf = new testFunctions(WGD, TGD, "trig");

  SECTION("testing for accuracy")
  {
    // results = PGD->testInterp(WGD, TGD);
    std::vector<float> xArray = { 10 + .01, 10 + .01, 10 + .51, 10 + .00, 125 };
    std::vector<float> yArray = { 10 + .51, 10 + .51, 10 + .51, 10 + .00, 136 };
    std::vector<float> zArray = { 10 + .51, 10 + .50, 10 + .01, 10 + .01, 190 };

    float tol(1.0e-2);

    for (size_t it = 0; it < xArray.size(); ++it) {
      double xPos = xArray[it];
      double yPos = yArray[it];
      double zPos = zArray[it];

      double uMean = 0.0, vMean = 0.0, wMean = 0.0;
      double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
      double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;
      double CoEps = 1e-6;

      PGD->interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
      float err = 0.0;

      err = std::abs(tf->u_testFunction->val(xPos, yPos, zPos) - uMean);
      REQUIRE(err < tol);

      err = std::abs(tf->v_testFunction->val(xPos, yPos, zPos) - vMean);
      REQUIRE(err < tol);

      err = std::abs(tf->w_testFunction->val(xPos, yPos, zPos) - wMean);
      REQUIRE(err < tol);

      err = std::abs(tf->c_testFunction->val(xPos, yPos, zPos) - txx);
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
    std::uniform_real_distribution<float> disX{ WGD->x[0], WGD->x.back() };
    std::uniform_real_distribution<float> disY{ WGD->y[0], WGD->y.back() };
    std::uniform_real_distribution<float> disZ{ 0, WGD->z_face[WGD->nz - 3] };

    std::vector<float> xArray, yArray, zArray;

    for (int it = 0; it < N; ++it) {
      xArray.push_back(disX(mersenne_engine));
      yArray.push_back(disY(mersenne_engine));
      zArray.push_back(disZ(mersenne_engine));
    }

    float tol(1.0e-2);
    float errU = 0.0, errV = 0.0, errW = 0.0, errT = 0.0;

    for (size_t it = 0; it < xArray.size(); ++it) {
      double xPos = xArray[it];
      double yPos = yArray[it];
      double zPos = zArray[it];

      double uMean = 0.0, vMean = 0.0, wMean = 0.0;
      double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
      double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;
      double CoEps = 1e-6;

      PGD->interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);

      errU += std::abs(tf->u_testFunction->val(xPos, yPos, zPos) - uMean);
      errV += std::abs(tf->v_testFunction->val(xPos, yPos, zPos) - vMean);
      errW += std::abs(tf->w_testFunction->val(xPos, yPos, zPos) - wMean);
      errT += std::abs(tf->c_testFunction->val(xPos, yPos, zPos) - txx);
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
}