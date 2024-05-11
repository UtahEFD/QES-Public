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

/** @file Interp.cpp */

#include "Interp.h"


Interp::Interp(WINDSGeneralData *WGD)
{
  // std::cout << "[Interp] \t Setting Interp fields " << std::endl;

  // copy WGD grid information
  nz = WGD->nz;
  ny = WGD->ny;
  nx = WGD->nx;

  dz = WGD->dz;
  dy = WGD->dy;
  dx = WGD->dx;

  // domain beginning for interpolation in each direction
  // in x-direction (halo cell to account for TURB variables)
  iStart = 1;
  iEnd = nx - 3;
  // in y-direction (halo cell to account for TURB variables)
  jStart = 1;
  jEnd = ny - 3;
  // in z-direction (ghost cell at bottom and halo cell at top)
  kStart = 1;
  kEnd = nz - 2;

  // get the TGD domain start and end values, other WGD grid information
  // in x-direction (face)
  xStart = WGD->x[iStart] - 0.5 * dx;
  xEnd = WGD->x[iEnd] + 0.5 * dx;
  // in y-direction (face)
  yStart = WGD->y[jStart] - 0.5 * dy;
  yEnd = WGD->y[jEnd] + 0.5 * dy;
  // in z-direction (face)
  zStart = WGD->z_face[kStart];
  zEnd = WGD->z_face[kEnd];
}
