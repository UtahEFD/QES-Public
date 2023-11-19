/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

/** @file InterpNearestCell.cpp */

#include "InterpNearestCell.h"


InterpNearestCell::InterpNearestCell(WINDSGeneralData *WGD, TURBGeneralData *TGD, const bool &debug_val)
  : Interp(WGD)
{
  //std::cout << "[InterpNearestCell] \t Setting InterpNearestCell fields " << std::endl;

  // copy debug information
  debug = debug_val;

  if (debug == true) {
    std::cout << "[InterpNearestCell] \t DEBUG - Domain boundary" << std::endl;
    std::cout << "\t\t xStart=" << xStart << " xEnd=" << xEnd << std::endl;
    std::cout << "\t\t yStart=" << yStart << " yEnd=" << yEnd << std::endl;
    std::cout << "\t\t zStart=" << zStart << " zEnd=" << zEnd << std::endl;
  }
}

void InterpNearestCell::interpInitialValues(const double &xPos,
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
                                            double &tzz_out)
{
  // this replaces the old indexing trick, set the indexing variables for the
  // interp3D for each particle, then get interpolated values from the
  // InterpTriLinear grid to the particle Lagrangian values for multiple datatypes


  Vector3Int cellIndex = getCellIndex(xPos, yPos, zPos);
  int cellId = cellIndex[0]
               + cellIndex[1] * (nx - 1)
               + cellIndex[2] * (nx - 1) * (ny - 1);

  // this is the current reynolds stress tensor
  txx_out = TGD->txx[cellId];
  txy_out = TGD->txy[cellId];
  txz_out = TGD->txz[cellId];
  tyy_out = TGD->tyy[cellId];
  tyz_out = TGD->tyz[cellId];
  tzz_out = TGD->tzz[cellId];

  sig_x_out = std::sqrt(std::abs(txx_out));
  if (sig_x_out == 0.0)
    sig_x_out = 1e-8;
  sig_y_out = std::sqrt(std::abs(tyy_out));
  if (sig_y_out == 0.0)
    sig_y_out = 1e-8;
  sig_z_out = std::sqrt(std::abs(tzz_out));
  if (sig_z_out == 0.0)
    sig_z_out = 1e-8;

  return;
}

void InterpNearestCell::interpValues(const double &xPos,
                                     const double &yPos,
                                     const double &zPos,
                                     const WINDSGeneralData *WGD,
                                     double &uMean_out,
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
                                     double &nuT_out,
                                     double &CoEps_out)
{

  Vector3Int cellIndex = getCellIndex(xPos, yPos, zPos);
  int cellId = cellIndex[0]
               + cellIndex[1] * (WGD->nx - 1)
               + cellIndex[2] * (WGD->nx - 1) * (WGD->ny - 1);
  int faceId = cellIndex[0]
               + cellIndex[1] * WGD->nx
               + cellIndex[2] * WGD->nx * WGD->ny;

  uMean_out = 0.5 * (WGD->u[faceId] + WGD->u[faceId + 1]);
  vMean_out = 0.5 * (WGD->v[faceId] + WGD->v[faceId + WGD->nx]);
  vMean_out = 0.5 * (WGD->w[faceId] + WGD->w[faceId + WGD->nx * WGD->ny]);

  CoEps_out = TGD->CoEps[cellId];
  // make sure CoEps is always bigger than zero
  if (CoEps_out <= 1e-6) {
    CoEps_out = 1e-6;
  }

  // this is the current reynolds stress tensor
  txx_out = TGD->txx[cellId];
  txy_out = TGD->txy[cellId];
  txz_out = TGD->txz[cellId];
  tyy_out = TGD->tyy[cellId];
  tyz_out = TGD->tyz[cellId];
  tzz_out = TGD->tzz[cellId];


  flux_div_x_out = TGD->div_tau_x[cellId];
  flux_div_y_out = TGD->div_tau_y[cellId];
  flux_div_z_out = TGD->div_tau_z[cellId];
  return;
}
