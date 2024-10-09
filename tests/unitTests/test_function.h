#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <math.h>

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

class test_function
{
public:
  test_function(WINDSGeneralData *WGD)
    : domain(WGD->domain)
  {
    lx = domain.nx() * domain.dx();
    ly = domain.ny() * domain.dy();
    lz = domain.z_face.back();
  }
  ~test_function() = default;

  virtual float val(float x, float y, float z) = 0;

protected:
  float lx{}, ly{}, lz{};
  qes::Domain domain;

private:
  test_function() : domain(0, 0, 0, 0, 0, 0) {}
};

class test_function_linearX : public test_function
{
public:
  test_function_linearX(WINDSGeneralData *WGD) : test_function(WGD)
  {}
  float val(float x, float y, float z) override
  {
    // a = 2 * 2pi/Lx
    float a = 2.0 * 2.0 * M_PI / (lx);
    // b = 6 * 2pi/Ly
    // float b = 6.0 * 2.0 * M_PI / (ny * dy);
    // c = 4 * 2pi/Lz
    // float c = 4.0 * 2.0 * M_PI / ((nz - 1) * dz);

    return a * x;
  }
};

class test_function_linearY : public test_function
{
public:
  test_function_linearY(WINDSGeneralData *WGD) : test_function(WGD)
  {}
  float val(float x, float y, float z) override
  {
    // float a = 2 * 2pi/Lx
    // float a = 2.0 * 2.0 * M_PI / (nx * dx);
    // float b = 6 * 2pi/Ly
    float b = 6.0 * 2.0 * M_PI / (ly);
    // float c = 4 * 2pi/Lz
    // float c = 4.0 * 2.0 * M_PI / ((nz - 1) * dz);

    return b * y;
  }
};

class test_function_linearZ : public test_function
{
public:
  test_function_linearZ(WINDSGeneralData *WGD) : test_function(WGD)
  {}
  float val(float x, float y, float z) override
  {
    // float a = 2 * 2pi / Lx;
    // float a = 2.0 * 2.0 * M_PI / (nx * dx);
    // float b = 6 * 2pi/Ly
    // float b = 6.0 * 2.0 * M_PI / (ny * dy);
    // float c = 4 * 2pi/Lz
    float c = 4.0 * 2.0 * M_PI / (lz);

    return c * z;
  }
};

class test_function_trig : public test_function
{
public:
  test_function_trig(WINDSGeneralData *WGD) : test_function(WGD)
  {}
  float val(float x, float y, float z) override
  {
    // a = 2 * 2pi/Lx
    float a = 2.0 * 2.0 * M_PI / (lx);
    // b = 6 * 2pi/Ly
    float b = 6.0 * 2.0 * M_PI / (ly);
    // c = 4 * 2pi/Lz
    float c = 4.0 * 2.0 * M_PI / (lz);

    return cos(a * x) + cos(b * y) + sin(c * z);
  }
};