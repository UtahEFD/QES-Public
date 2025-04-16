/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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


Interp::Interp(qes::Domain domain_in)
  : domain(std::move(domain_in))
{

  // std::cout << "[Interp] \t Setting Interp fields " << std::endl;

  // copy WGD grid information
  std::tie(nx, ny, nz) = domain.getDomainCellNum();
  std::tie(dx, dy, dz) = domain.getDomainSize();

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
  xStart = domain.x[iStart] - 0.5 * dx;
  xEnd = domain.x[iEnd] + 0.5 * dx;
  // in y-direction (face)
  yStart = domain.y[jStart] - 0.5 * dy;
  yEnd = domain.y[jEnd] + 0.5 * dy;
  // in z-direction (face)
  zStart = domain.z_face[kStart];
  zEnd = domain.z_face[kEnd];
}

void Interp::getDomainBounds(float &domainXstart,
                             float &domainXend,
                             float &domainYstart,
                             float &domainYend,
                             float &domainZstart,
                             float &domainZend)
{
  domainXstart = xStart;
  domainXend = xEnd;
  domainYstart = yStart;
  domainYend = yEnd;
  domainZstart = zStart;
  domainZend = zEnd;
}
