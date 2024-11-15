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

/** @file InterpNearestCell.cpp */

#include "InterpNearestCell.h"


InterpNearestCell::InterpNearestCell(qes::Domain domain_in, bool debug_val = false)
  : Interp(domain)
{
  // std::cout << "[InterpNearestCell] \t Setting InterpNearestCell fields " << std::endl;

  // copy debug information
  debug = debug_val;

  if (debug == true) {
    std::cout << "[InterpNearestCell] \t DEBUG - Domain boundary" << std::endl;
    std::cout << "\t\t xStart=" << xStart << " xEnd=" << xEnd << std::endl;
    std::cout << "\t\t yStart=" << yStart << " yEnd=" << yEnd << std::endl;
    std::cout << "\t\t zStart=" << zStart << " zEnd=" << zEnd << std::endl;
  }
}

void InterpNearestCell::interpWindsValues(const WINDSGeneralData *WGD,
                                          const vec3 &pos,
                                          vec3 &vel_out)
{
  auto [i, j, k] = getCellIndex(pos);
  long faceId = domain.face(i, j, k);

  vel_out._1 = 0.5 * (WGD->u[faceId] + WGD->u[WGD->domain.faceAdd(faceId, 1, 0, 0)]);
  vel_out._2 = 0.5 * (WGD->v[faceId] + WGD->v[WGD->domain.faceAdd(faceId, 0, 1, 0)]);
  vel_out._3 = 0.5 * (WGD->w[faceId] + WGD->w[WGD->domain.faceAdd(faceId, 0, 0, 1)]);
}

void InterpNearestCell::interpTurbValues(const TURBGeneralData *TGD,
                                         const vec3 &pos,
                                         mat3sym &tau_out,
                                         vec3 &flux_div_out,
                                         float &nuT_out,
                                         float &CoEps_out)
{
  auto [i, j, k] = getCellIndex(pos);
  long cellId = domain.cell(i, j, k);

  CoEps_out = TGD->CoEps[cellId];
  // make sure CoEps is always bigger than zero
  if (CoEps_out <= 1e-6) {
    CoEps_out = 1e-6;
  }
  nuT_out = TGD->nuT[cellId];

  // this is the current reynolds stress tensor
  tau_out._11 = TGD->txx[cellId];
  tau_out._12 = TGD->txy[cellId];
  tau_out._13 = TGD->txz[cellId];
  tau_out._22 = TGD->tyy[cellId];
  tau_out._23 = TGD->tyz[cellId];
  tau_out._33 = TGD->tzz[cellId];


  flux_div_out._1 = TGD->div_tau_x[cellId];
  flux_div_out._2 = TGD->div_tau_y[cellId];
  flux_div_out._3 = TGD->div_tau_z[cellId];
}


void InterpNearestCell::interpTurbInitialValues(const TURBGeneralData *TGD,
                                                const vec3 &pos,
                                                mat3sym &tau_out,
                                                vec3 &sig_out)
{
  // this replaces the old indexing trick, set the indexing variables for the
  // interp3D for each particle, then get interpolated values from the
  // InterpTriLinear grid to the particle Lagrangian values for multiple datatypes
  auto [i, j, k] = getCellIndex(pos);
  long cellId = domain.cell(i, j, k);

  // this is the current reynolds stress tensor
  tau_out._11 = TGD->txx[cellId];
  tau_out._12 = TGD->txy[cellId];
  tau_out._13 = TGD->txz[cellId];
  tau_out._22 = TGD->tyy[cellId];
  tau_out._23 = TGD->tyz[cellId];
  tau_out._33 = TGD->tzz[cellId];

  sig_out._1 = std::sqrt(std::abs(tau_out._11));
  if (sig_out._1 == 0.0)
    sig_out._1 = 1e-8;
  sig_out._2 = std::sqrt(std::abs(tau_out._22));
  if (sig_out._2 == 0.0)
    sig_out._2 = 1e-8;
  sig_out._3 = std::sqrt(std::abs(tau_out._33));
  if (sig_out._3 == 0.0)
    sig_out._3 = 1e-8;
}
