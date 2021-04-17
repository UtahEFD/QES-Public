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

/** @file CanopyHomogeneous.cpp */

#include "CanopyHomogeneous.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

// set et attenuation coefficient
void CanopyHomogeneous::setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_id)
{
  // When THIS canopy calls this function, we need to do the
  // following:
  // readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
  // canopy_atten, canopy_top);

  // this function need to be called to defined the boundary of the canopy and the icellflags
  setCanopyGrid(WGD, building_id);

  // Resize the canopy-related vectors
  canopy_atten.resize(numcell_cent_3d, 0.0);

  for (auto j = 0; j < ny_canopy; j++) {
    for (auto i = 0; i < nx_canopy; i++) {
      int icell_2d = i + j * nx_canopy;
      for (auto k = canopy_bot_index[icell_2d]; k < canopy_top_index[icell_2d]; k++) {
        int icell_3d = i + j * nx_canopy + k * nx_canopy * ny_canopy;
        // initiate all attenuation coefficients to the canopy coefficient
        canopy_atten[icell_3d] = attenuationCoeff;
      }
    }
  }

  return;
}


void CanopyHomogeneous::canopyVegetation(WINDSGeneralData *WGD, int building_id)
{
  // Apply canopy parameterization
  canopyCioncoParam(WGD);

  return;
}

void CanopyHomogeneous::canopyWake(WINDSGeneralData *WGD, int building_id)
{
  return;
}
