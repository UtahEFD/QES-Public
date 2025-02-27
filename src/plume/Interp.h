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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file Interp.h
 * @brief
 */

#pragma once

#include <iostream>
#include <ctime>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "qes/Domain.h"

#include "util/calcTime.h"
#include "util/VectorMath.h"
#include "util/Vector3Int.h"
#include "util/Vector3Float.h"
#include "util/Vector3Double.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Random.h"

class Interp
{

public:
  // constructor
  // copies the turb grid values for nx, ny, nz, nt, dx, dy, and dz to the QES grid,
  // then calculates the tau gradients which are then used to calculate the flux_div grid values.
  Interp(qes::Domain);
  virtual ~Interp() = default;

  void getDomainBounds(float &, float &, float &, float &, float &, float &);
  float getZstart() { return zStart; }

  virtual void interpWindsValues(const WINDSGeneralData *WGD,
                                 const vec3 &pos,
                                 vec3 &vel_out) = 0;

  virtual void interpTurbValues(const TURBGeneralData *TGD,
                                const vec3 &pos,
                                mat3sym &tau_out,
                                vec3 &flux_div_out,
                                float &nuT_out,
                                float &CoEps_out) = 0;

  virtual void interpTurbInitialValues(const TURBGeneralData *TGD,
                                       const vec3 &pos,
                                       mat3sym &tau_out,
                                       vec3 &sig_out) = 0;

  int getCellId(const float &, const float &, const float &);
  int getCellId(Vector3Float &);
  int getCellId(const vec3 &);
  int getCellId2d(const float &, const float &);
  std::tuple<int, int, int> getCellIndex(const long &);
  std::tuple<int, int, int> getCellIndex(const float &, const float &, const float &);
  std::tuple<int, int, int> getCellIndex(const vec3 &);

protected:
  // General QES Domain Data
  qes::Domain domain;

  // The Plume domain bounds.
  float xStart, xEnd;
  float yStart, yEnd;
  float zStart, zEnd;

  // the QES data held in this class is on the WINDS grid,
  // a copy of the WINDS grid information.
  int nx;
  int ny;
  int nz;
  // a copy of the grid resolution.
  float dx;
  float dy;
  float dz;

  // index of domain bounds
  int iStart, iEnd;
  int jStart, jEnd;
  int kStart, kEnd;

  // timer class useful for debugging and timing different operations
  // calcTime timers;

  // copies of debug related information from the input arguments
  // bool debug;

private:
  Interp() : domain(0, 0, 0, 0.0f, 0.0f, 0.0f) {}
};

inline int Interp::getCellId(const float &xPos, const float &yPos, const float &zPos)
{
  int i = floor((xPos - 0.0 * dx) / (dx + 1e-9));
  int j = floor((yPos - 0.0 * dy) / (dy + 1e-9));
  int k = floor((zPos + 1.0 * dz) / (dz + 1e-9));

  return domain.cell(i, j, k);
}

inline int Interp::getCellId(Vector3Float &X)
{
  // int i = floor((xPos - xStart + 0.5*dx)/(dx+1e-9));
  // int j = floor((yPos - yStart + 0.5*dy)/(dy+1e-9));
  // int k = floor((zPos - zStart + dz)/(dz+1e-9));

  int i = floor((X[0] - 0.0 * dx) / (dx + 1e-9));
  int j = floor((X[1] - 0.0 * dy) / (dy + 1e-9));
  int k = floor((X[2] + 1.0 * dz) / (dz + 1e-9));

  return domain.cell(i, j, k);
}

inline int Interp::getCellId(const vec3 &X)
{
  // int i = floor((xPos - xStart + 0.5*dx)/(dx+1e-9));
  // int j = floor((yPos - yStart + 0.5*dy)/(dy+1e-9));
  // int k = floor((zPos - zStart + dz)/(dz+1e-9));

  int i = floor((X._1 - 0.0 * dx) / (dx + 1e-9));
  int j = floor((X._2 - 0.0 * dy) / (dy + 1e-9));
  int k = floor((X._3 + 1.0 * dz) / (dz + 1e-9));

  return domain.cell(i, j, k);
}

inline int Interp::getCellId2d(const float &xPos, const float &yPos)
{
  int i = floor((xPos - 0.0 * dx) / (dx + 1e-9));
  int j = floor((yPos - 0.0 * dy) / (dy + 1e-9));

  return domain.cell2d(i, j);
}

inline std::tuple<int, int, int> Interp::getCellIndex(const long &cellId)
{
  return domain.getCellIdx(cellId);
}

inline std::tuple<int, int, int> Interp::getCellIndex(const float &xPos, const float &yPos, const float &zPos)
{
  return domain.getCellIdx(getCellId(xPos, yPos, zPos));
}

inline std::tuple<int, int, int> Interp::getCellIndex(const vec3 &X)
{
  return domain.getCellIdx(getCellId(X));
}
