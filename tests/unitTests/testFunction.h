#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <math.h>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

class testFunction
{
public:
  testFunction(WINDSGeneralData *WGD)
  {
  nx = WGD->nx;
  ny = WGD->ny;
  nz = WGD->nz;

  dx = WGD->dx;
  dy = WGD->dy;
  dz = WGD->dz;
  }
  ~testFunction() {}

  virtual float val(float x, float y, float z) = 0;

protected:
  int nx{}, ny{}, nz{};
  float dx{}, dy{}, dz{};

private:
  testFunction() = default;
};

class testFunction_linearX : public testFunction
{
public:
  testFunction_linearX(WINDSGeneralData *WGD) : testFunction(WGD)
  {}
  float val(float x, float y, float z) override
  {
    // a = 2 * 2pi/Lx
    float a = 2.0 * 2.0 * M_PI / (nx * dx);
    // b = 6 * 2pi/Ly
    // float b = 6.0 * 2.0 * M_PI / (ny * dy);
    // c = 4 * 2pi/Lz
    // float c = 4.0 * 2.0 * M_PI / ((nz - 1) * dz);

    return a * x;
  }
};

class testFunction_linearY : public testFunction
{
public:
  testFunction_linearY(WINDSGeneralData *WGD) : testFunction(WGD)
  {}
  float val(float x, float y, float z) override
  {
    // float a = 2 * 2pi/Lx
    // float a = 2.0 * 2.0 * M_PI / (nx * dx);
    // float b = 6 * 2pi/Ly
    float b = 6.0 * 2.0 * M_PI / (ny * dy);
    // float c = 4 * 2pi/Lz
    // float c = 4.0 * 2.0 * M_PI / ((nz - 1) * dz);

    return b * y;
  }
};

class testFunction_linearZ : public testFunction
{
public:
  testFunction_linearZ(WINDSGeneralData *WGD) : testFunction(WGD)
  {}
  float val(float x, float y, float z) override
  {
    // float a = 2 * 2pi / Lx;
    // float a = 2.0 * 2.0 * M_PI / (nx * dx);
    // float b = 6 * 2pi/Ly
    // float b = 6.0 * 2.0 * M_PI / (ny * dy);
    // float c = 4 * 2pi/Lz
    float c = 4.0 * 2.0 * M_PI / ((nz - 1) * dz);

    return c * z;
  }
};

class testFunction_trig : public testFunction
{
public:
  testFunction_trig(WINDSGeneralData *WGD) : testFunction(WGD)
  {}
  float val(float x, float y, float z) override
  {
    // a = 2 * 2pi/Lx
    float a = 2.0 * 2.0 * M_PI / (nx * dx);
    // b = 6 * 2pi/Ly
    float b = 6.0 * 2.0 * M_PI / (ny * dy);
    // c = 4 * 2pi/Lz
    float c = 4.0 * 2.0 * M_PI / ((nz - 1) * dz);

    return cos(a * x) + cos(b * y) + sin(c * z);
  }
};