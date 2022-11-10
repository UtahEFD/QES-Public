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
 * @file TURBWallBuilding.cpp
 * @brief :document this:
 */

#include "TURBWallBuilding.h"

TURBWallBuilding::TURBWallBuilding(const WINDSInputData *WID,
                                   WINDSGeneralData *WGD,
                                   TURBGeneralData *TGD)
{
  if (WID->simParams->meshTypeFlag == 1) {
    use_cutcell = true;
    // fill array with cellid  of cutcell cells
    get_cutcell_wall_id(WGD, icellflag_cutcell);
    // fill itrublfag with cutcell flag
    set_cutcell_wall_flag(TGD, iturbflag_cutcell);

    // [FM] temporary fix -> use stairstep within the cut-cell
    get_stairstep_wall_id(WGD, icellflag_building);
    set_stairstep_wall_flag(TGD, iturbflag_stairstep);
    comp_wall_velocity_deriv = &TURBWallBuilding::comp_velocity_deriv_finitediff_stairstep;
    comp_wall_stress_deriv = &TURBWallBuilding::comp_stress_deriv_finitediff_stairstep;
  } else {
    use_cutcell = false;
    // use stairstep
    get_stairstep_wall_id(WGD, icellflag_building);
    set_stairstep_wall_flag(TGD, iturbflag_stairstep);
    if (WID->turbParams->buildingWallFlag == 1) {
      comp_wall_velocity_deriv = &TURBWallBuilding::set_loglaw_stairstep;
    } else if (WID->turbParams->buildingWallFlag == 2) {
      comp_wall_velocity_deriv = &TURBWallBuilding::comp_velocity_deriv_finitediff_stairstep;
    } else {
      comp_wall_velocity_deriv = nullptr;
    }
    comp_wall_stress_deriv = &TURBWallBuilding::comp_stress_deriv_finitediff_stairstep;
  }

  return;
}


void TURBWallBuilding::setWallsVelocityDeriv(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  if (comp_wall_velocity_deriv) (this->*comp_wall_velocity_deriv)(WGD, TGD);
}

void TURBWallBuilding::setWallsStressDeriv(WINDSGeneralData *WGD,
                                           TURBGeneralData *TGD,
                                           const std::vector<float> &tox,
                                           const std::vector<float> &toy,
                                           const std::vector<float> &toz)
{
  if (comp_wall_stress_deriv) (this->*comp_wall_stress_deriv)(WGD, TGD, tox, toy, toz);
}

void TURBWallBuilding::set_loglaw_stairstep(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  int nx = WGD->nx;
  int ny = WGD->ny;

  // ## HORIZONTAL WALL
  // set BC for horizontal wall below the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_below_indices.begin();
       it != wall_below_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));

    // Gxz = dudz
    TGD->Gxz[it->cellID] = 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                           / (0.5 * WGD->dz_array[k] * log(0.5 * WGD->dz_array[k] / WGD->z0));
    // Gyz = dvdz
    TGD->Gyz[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * WGD->dz_array[k] * log(0.5 * WGD->dz_array[k] / WGD->z0));
  }
  // set BC for horizontal wall above the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_above_indices.begin();
       it != wall_above_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));

    // Gxz = dudz
    TGD->Gxz[it->cellID] = 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                           / (0.5 * WGD->dz_array[k] * log(0.5 * WGD->dz_array[k] / WGD->z0));
    // Gyz = dvdz
    TGD->Gyz[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + WGD->nx])
                           / (0.5 * WGD->dz_array[k] * log(0.5 * WGD->dz_array[k] / WGD->z0));
  }

  // ## VERTICAL WALL ALONG X (front/back)
  // set BC for vertical wall in back of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_back_indices.begin();
       it != wall_back_indices.end();
       ++it) {
    // Gyx = dvdx
    TGD->Gyx[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * WGD->dx * log(0.5 * WGD->dx / WGD->z0));
    // Gzx = dwdx
    TGD->Gzx[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * WGD->dx * log(0.5 * WGD->dx / WGD->z0));
  }
  // set BC for vertical wall in front of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_front_indices.begin();
       it != wall_front_indices.end();
       ++it) {
    // Gyx = dvdx
    TGD->Gyx[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * WGD->dx * log(0.5 * WGD->dx / WGD->z0));
    // Gzx = dwdx
    TGD->Gzx[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * WGD->dx * log(0.5 * WGD->dx / WGD->z0));
  }

  // ## VERTICAL WALL ALONG Y (right/left)
  // set BC for vertical wall right of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_right_indices.begin();
       it != wall_right_indices.end();
       ++it) {
    // Gxy = dudy
    TGD->Gxy[it->cellID] = 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                           / (0.5 * WGD->dy * log(0.5 * WGD->dy / WGD->z0));
    // Gzy = dwdy
    TGD->Gzy[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * WGD->dy * log(0.5 * WGD->dy / WGD->z0));
  }
  // set BC for vertical wall left of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_below_indices.begin();
       it != wall_below_indices.end();
       ++it) {
    // Gxy = dudy
    TGD->Gxy[it->cellID] = 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                           / (0.5 * WGD->dy * log(0.5 * WGD->dy / WGD->z0));
    // Gzy = dwdy
    TGD->Gzy[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * WGD->dy * log(0.5 * WGD->dy / WGD->z0));
  }

  return;
}
