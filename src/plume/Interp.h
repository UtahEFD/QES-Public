/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

#include "util/calcTime.h"
#include "Random.h"
#include "util/Vector3Int.h"
#include "util/Vector3.h"
#include "util/Vector3Double.h"

#include "PlumeInputData.hpp"
#include "src/winds/WINDSGeneralData.h"
#include "src/winds/TURBGeneralData.h"


class Interp
{

public:
  // constructor
  // copies the turb grid values for nx, ny, nz, nt, dx, dy, and dz to the QES grid,
  // then calculates the tau gradients which are then used to calculate the flux_div grid values.
  Interp()
  {}
  ~Interp()
  {}

  // the QES data held in this class is on the turb grid, so these are copies of the turb grid values
  int nx;// a copy of the turb grid information. This is the number of points in the x dimension
  int ny;// a copy of the turb grid information. This is the number of points in the y dimension
  int nz;// a copy of the turb grid information. This is the number of points in the z dimension

  double dx;// a copy of the turb grid information. This is the difference between points in the x dimension, eventually could become an array
  double dy;// a copy of the turb grid information. This is the difference between points in the y dimension, eventually could become an array
  double dz;// a copy of the TGD grid information. This is the difference between points in the z dimension, eventually could become an array

  // The QES grid information.
  double xStart, xEnd;
  double yStart, yEnd;
  double zStart, zEnd;

  // other input variable
  double C_0;// a copy of the TGD grid information. This is used to separate out CoEps into its separate parts when doing debug output

  double vel_threshold;

  virtual void interpValues(const double &xPos,
                            const double &yPos,
                            const double &zPos,
                            const WINDSGeneralData *WGD,
                            double &uMain_out,
                            double &vMean_out,
                            double &wMean_out,
                            const TURBGeneralData *TGD,
                            double &txx_out,
                            double &txy_out,
                            double &txz_out,
                            double &tyy_out,
                            double &tyz_out,
                            double &tzz_out,
                            double &flux_div_x_out,
                            double &flux_div_y_out,
                            double &flux_div_z_out,
                            double &CoEps_out) = 0;

  virtual void interpInitialValues(const double &xPos,
                                   const double &yPos,
                                   const double &zPos,
                                   const TURBGeneralData *TGD,
                                   double &sig_x_out,
                                   double &sig_y_out,
                                   double &sig_z_out,
                                   double &txx_out,
                                   double &txy_out,
                                   double &txz_out,
                                   double &tyy_out,
                                   double &tyz_out,
                                   double &tzz_out) = 0;

  int getCellId(const double &, const double &, const double &);
  int getCellId(Vector3Double &);
  Vector3Int getCellIndex(const int &);

protected:
  // index of domain bounds
  int iStart, iEnd;
  int jStart, jEnd;
  int kStart, kEnd;

  // timer class useful for debugging and timing different operations
  //calcTime timers;

  // copies of debug related information from the input arguments
  //bool debug;
};

inline int Interp::getCellId(const double &xPos, const double &yPos, const double &zPos)
{
  int i = floor((xPos - 0.0 * dx) / (dx + 1e-9));
  int j = floor((yPos - 0.0 * dy) / (dy + 1e-9));
  int k = floor((zPos + 1.0 * dz) / (dz + 1e-9));

  return i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
}

inline int Interp::getCellId(Vector3Double &X)
{
  //int i = floor((xPos - xStart + 0.5*dx)/(dx+1e-9));
  //int j = floor((yPos - yStart + 0.5*dy)/(dy+1e-9));
  //int k = floor((zPos - zStart + dz)/(dz+1e-9));

  int i = floor((X[0] - 0.0 * dx) / (dx + 1e-9));
  int j = floor((X[1] - 0.0 * dy) / (dy + 1e-9));
  int k = floor((X[2] + 1.0 * dz) / (dz + 1e-9));

  return i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
}

inline Vector3Int Interp::getCellIndex(const int &cellId)
{
  int k = (int)(cellId / ((nx - 1) * (ny - 1)));
  int j = (int)((cellId - k * (nx - 1) * (ny - 1)) / (nx - 1));
  int i = cellId - j * (nx - 1) - k * (nx - 1) * (ny - 1);

  return { i, j, k };
}
