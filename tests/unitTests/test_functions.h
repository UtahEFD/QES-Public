/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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
 * @file test_functions.h
 * @brief This generate test function for unit tests
 */

#pragma once

#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

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

class test_functions
{
public:
  test_functions(WINDSGeneralData *, TURBGeneralData *, const std::string &);
  void setTestValues(WINDSGeneralData *, TURBGeneralData *);

  test_function *u_test_function;
  test_function *v_test_function;
  test_function *w_test_function;
  test_function *c_test_function;

private:
  test_functions() : domain(0, 0, 0, 0, 0, 0) {}

  qes::Domain domain;
};