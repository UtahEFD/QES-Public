#include "test_PlumeGeneralData.h"

void test_PlumeGeneralData::setInterpMethod(std::string interpMethod,
                                            WINDSGeneralData *WGD,
                                            TURBGeneralData *TGD)
{
  std::cout << "[Plume] \t Interpolation Method set to: " << interpMethod << std::endl;
  if (interpMethod == "analyticalPowerLaw") {
    interp = new InterpPowerLaw(WGD, TGD, debug);
  } else if (interpMethod == "nearestCell") {
    interp = new InterpNearestCell(WGD, TGD, debug);
  } else if (interpMethod == "triLinear") {
    interp = new InterpTriLinear(WGD, TGD, debug);
  } else {
    std::cerr << "[ERROR] unknown interpolation method" << std::endl;
    exit(EXIT_FAILURE);
  }
  return;
}

void test_PlumeGeneralData::setTestFunctions(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{

  // uv on vertical face -> k=0...nz-2
  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u[faceID] = (this->*u_testFunction)(WGD, i * WGD->dx, WGD->y[j], WGD->z[k]);
      }
    }
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        // WGD->v[faceID] = cos(a * WGD->x[i]) + cos(b * j * WGD->dy) + sin(c * WGD->z[k]);
        // WGD->v[faceID] = a * WGD->x[i] + b * j * WGD->dy + c * WGD->z[k];
        WGD->v[faceID] = (this->*v_testFunction)(WGD, WGD->x[i], j * WGD->dy, WGD->z[k]);
      }
    }
  }

  // w on horizontal face -> k=0...nz-1
  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->w[faceID] = (this->*w_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z_face[k]);
      }
    }
  }

  // w on horizontal face -> k=0...nz-1
  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int cellID = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        TGD->txx[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->txy[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->txz[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->tyy[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->tyz[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->tzz[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);

        TGD->div_tau_x[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->div_tau_y[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->div_tau_z[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);

        TGD->CoEps[cellID] = (this->*c_testFunction)(WGD, WGD->x[i], WGD->y[j], WGD->z[k]);
      }
    }
  }

  return;
}
float test_PlumeGeneralData::testFunction_linearX(WINDSGeneralData *WGD, float x, float y, float z)
{
  // a = 2 * 2pi/Lx
  float a = 2.0 * 2.0 * M_PI / (WGD->nx * WGD->dx);
  // b = 6 * 2pi/Ly
  // float b = 6.0 * 2.0 * M_PI / (WGD->ny * WGD->dy);
  // c = 4 * 2pi/Lz
  // float c = 4.0 * 2.0 * M_PI / ((WGD->nz - 1) * WGD->dz);

  return a * x;
}

float test_PlumeGeneralData::testFunction_linearY(WINDSGeneralData *WGD, float x, float y, float z)
{
  // a = 2 * 2pi/Lx
  // float a = 2.0 * 2.0 * M_PI / (WGD->nx * WGD->dx);
  // b = 6 * 2pi/Ly
  float b = 6.0 * 2.0 * M_PI / (WGD->ny * WGD->dy);
  // c = 4 * 2pi/Lz
  // float c = 4.0 * 2.0 * M_PI / ((WGD->nz - 1) * WGD->dz);

  return b * y;
}

float test_PlumeGeneralData::testFunction_linearZ(WINDSGeneralData *WGD, float x, float y, float z)
{
  // a = 2 * 2pi/Lx
  // float a = 2.0 * 2.0 * M_PI / (WGD->nx * WGD->dx);
  // b = 6 * 2pi/Ly
  // float b = 6.0 * 2.0 * M_PI / (WGD->ny * WGD->dy);
  // c = 4 * 2pi/Lz
  float c = 4.0 * 2.0 * M_PI / ((WGD->nz - 1) * WGD->dz);

  return c * z;
}

float test_PlumeGeneralData::testFunction_trig(WINDSGeneralData *WGD, float x, float y, float z)
{
  // a = 2 * 2pi/Lx
  float a = 2.0 * 2.0 * M_PI / (WGD->nx * WGD->dx);
  // b = 6 * 2pi/Ly
  float b = 6.0 * 2.0 * M_PI / (WGD->ny * WGD->dy);
  // c = 4 * 2pi/Lz
  float c = 4.0 * 2.0 * M_PI / ((WGD->nz - 1) * WGD->dz);

  return cos(a * x) + cos(b * y) + sin(c * z);
}

std::string test_PlumeGeneralData::testInterp(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  double xPos = 10;
  double yPos = 10;
  double zPos = 1;

  double uMean = 0.0;
  double vMean = 0.0;
  double wMean = 0.0;

  double txx = 0.0;
  double txy = 0.0;
  double txz = 0.0;
  double tyy = 0.0;
  double tyz = 0.0;
  double tzz = 0.0;

  double flux_div_x = 0.0;
  double flux_div_y = 0.0;
  double flux_div_z = 0.0;

  double CoEps = 1e-6;

  // u_testFunction = &test_PlumeGeneralData::testFunction_linearY;
  // v_testFunction = &test_PlumeGeneralData::testFunction_linearZ;
  // w_testFunction = &test_PlumeGeneralData::testFunction_linearX;
  // c_testFunction = &test_PlumeGeneralData::testFunction_linearZ;

  u_testFunction = &test_PlumeGeneralData::testFunction_trig;
  v_testFunction = &test_PlumeGeneralData::testFunction_trig;
  w_testFunction = &test_PlumeGeneralData::testFunction_trig;
  c_testFunction = &test_PlumeGeneralData::testFunction_trig;

  setTestFunctions(WGD, TGD);

  std::vector<float> xArray = { 10 + .01, 10 + .01, 10 + .51, 10 + .00, 325 };
  std::vector<float> yArray = { 10 + .51, 10 + .51, 10 + .51, 10 + .00, 136 };
  std::vector<float> zArray = { 10 + .51, 10 + .50, 10 + .01, 10 + .01, 290 };

  bool verbose = false;
  float tol(1.0e-2);

  for (size_t it = 0; it < xArray.size(); ++it) {
    xPos = xArray[it];
    yPos = yArray[it];
    zPos = zArray[it];
    interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
    if (verbose) {
      std::cout << (this->*u_testFunction)(WGD, xPos, yPos, zPos) << "->" << uMean << std::endl;
      std::cout << (this->*v_testFunction)(WGD, xPos, yPos, zPos) << "->" << vMean << std::endl;
      std::cout << (this->*w_testFunction)(WGD, xPos, yPos, zPos) << "->" << wMean << std::endl;
      std::cout << (this->*c_testFunction)(WGD, xPos, yPos, zPos) << "->" << txx << std::endl;
    }
    float err = 0.0;

    err = std::abs((this->*u_testFunction)(WGD, xPos, yPos, zPos) - uMean);
    if (err > tol)
      return util_errorReport("testInterpolation", "error in interpolation of u variable \n err = " + std::to_string(err));

    err = std::abs((this->*v_testFunction)(WGD, xPos, yPos, zPos) - vMean);
    if (err > tol)
      return util_errorReport("testInterpolation", "error in interpolation of v variable \n err = " + std::to_string(err));

    err = std::abs((this->*w_testFunction)(WGD, xPos, yPos, zPos) - wMean);
    if (err > tol)
      return util_errorReport("testInterpolation", "error in interpolation of w variable \n err = " + std::to_string(err));

    err = std::abs((this->*c_testFunction)(WGD, xPos, yPos, zPos) - txx);
    if (err > tol)
      return util_errorReport("testInterpolation", "error in interpolation of c variable \n err = " + std::to_string(err));
  }

  return TEST_PASS;
}

std::string test_PlumeGeneralData::timeInterpCPU(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{

  int N = 100000;

  double xPos = 10;
  double yPos = 10;
  double zPos = 1;

  double uMean = 0.0;
  double vMean = 0.0;
  double wMean = 0.0;

  double txx = 0.0;
  double txy = 0.0;
  double txz = 0.0;
  double tyy = 0.0;
  double tyz = 0.0;
  double tzz = 0.0;

  double flux_div_x = 0.0;
  double flux_div_y = 0.0;
  double flux_div_z = 0.0;

  double CoEps = 1e-6;

  // u_testFunction = &test_PlumeGeneralData::testFunction_linearY;
  // v_testFunction = &test_PlumeGeneralData::testFunction_linearZ;
  // w_testFunction = &test_PlumeGeneralData::testFunction_linearX;
  // c_testFunction = &test_PlumeGeneralData::testFunction_linearZ;

  u_testFunction = &test_PlumeGeneralData::testFunction_trig;
  v_testFunction = &test_PlumeGeneralData::testFunction_trig;
  w_testFunction = &test_PlumeGeneralData::testFunction_trig;
  c_testFunction = &test_PlumeGeneralData::testFunction_trig;

  setTestFunctions(WGD, TGD);

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

  // bool verbose = false;
  // float tol(1.0e-2);

  auto interpStartTime = std::chrono::high_resolution_clock::now();
  for (size_t it = 0; it < xArray.size(); ++it) {
    xPos = xArray[it];
    yPos = yArray[it];
    zPos = zArray[it];
    interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);
  }
  auto interpEndTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> interpElapsed = interpEndTime - interpStartTime;
  std::cout << "interpolation elapsed time: " << interpElapsed.count() << " s\n";

  return TEST_PASS;
}


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
