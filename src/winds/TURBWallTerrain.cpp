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
 * @file TURBWallTerrain.cpp
 * @brief :document this:
 */

#include "TURBWallTerrain.h"

TURBWallTerrain::TURBWallTerrain(const WINDSInputData *WID,
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
    get_stairstep_wall_id(WGD, icellflag_terrain);
    set_stairstep_wall_flag(TGD, iturbflag_stairstep);
    comp_wall_velocity_deriv = &TURBWallTerrain::comp_velocity_deriv_finitediff_stairstep;
    comp_wall_stress_deriv = &TURBWallTerrain::comp_stress_deriv_finitediff_stairstep;
  } else {
    use_cutcell = false;
    // use stairstep
    get_stairstep_wall_id(WGD, icellflag_terrain);
    set_stairstep_wall_flag(TGD, iturbflag_stairstep);
    if (WID->turbParams->terrainWallFlag == 1) {
      comp_wall_velocity_deriv = &TURBWallTerrain::set_loglaw_stairstep;
    } else if (WID->turbParams->terrainWallFlag == 2) {
      comp_wall_velocity_deriv = &TURBWallTerrain::comp_velocity_deriv_finitediff_stairstep;
    } else {
      comp_wall_velocity_deriv = nullptr;
    }
    comp_wall_stress_deriv = &TURBWallTerrain::comp_stress_deriv_finitediff_stairstep;
  }


  return;
}


void TURBWallTerrain::setWallsVelocityDeriv(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  if (comp_wall_velocity_deriv) (this->*comp_wall_velocity_deriv)(WGD, TGD);
}

void TURBWallTerrain::setWallsStressDeriv(WINDSGeneralData *WGD,
                                          TURBGeneralData *TGD,
                                          const std::vector<float> &tox,
                                          const std::vector<float> &toy,
                                          const std::vector<float> &toz)
{
  if (comp_wall_stress_deriv) (this->*comp_wall_stress_deriv)(WGD, TGD, tox, toy, toz);
}

void TURBWallTerrain::set_loglaw_stairstep(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{

  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  // ## HORIZONTAL WALL
  // set BC for horizontal wall below the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_below_indices.begin();
       it != wall_below_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((it->cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = it->cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);

    float z0 = (WGD->z0_domain_u[i + j * nx] + WGD->z0_domain_u[i + 1 + j * nx]
                + WGD->z0_domain_v[i + j * nx] + WGD->z0_domain_v[i + (j + 1) * nx])
               / 4.0;

    // Gxz = dudz
    TGD->Gxz[it->cellID] = 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                           / (0.5 * WGD->domain.dz_array[k] * log(0.5 * WGD->domain.dz_array[k] / z0));
    // Gyz = dvdz
    TGD->Gyz[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * WGD->domain.dz_array[k] * log(0.5 * WGD->domain.dz_array[k] / z0));
  }
  // set BC for horizontal wall above the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_above_indices.begin();
       it != wall_above_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((it->cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = it->cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);

    float z0 = (WGD->z0_domain_u[i + j * nx] + WGD->z0_domain_u[i + 1 + j * nx]
                + WGD->z0_domain_v[i + j * nx] + WGD->z0_domain_v[i + (j + 1) * nx])
               / 4.0;


    // Gxz = dudz
    TGD->Gxz[it->cellID] = 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                           / (0.5 * WGD->domain.dz_array[k] * log(0.5 * WGD->domain.dz_array[k] / z0));
    // Gyz = dvdz
    TGD->Gyz[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * WGD->domain.dz_array[k] * log(0.5 * WGD->domain.dz_array[k] / z0));
  }

  // ## VERTICAL WALL ALONG X (front/back)
  // set BC for vertical wall in back of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_back_indices.begin();
       it != wall_back_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((it->cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = it->cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);

    float z0 = (WGD->z0_domain_u[i + j * nx] + WGD->z0_domain_u[i + 1 + j * nx]
                + WGD->z0_domain_v[i + j * nx] + WGD->z0_domain_v[i + (j + 1) * nx])
               / 4.0;

    // Gyx = dvdx
    TGD->Gyx[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * dx * log(0.5 * dx / z0));
    // Gzx = dwdx
    TGD->Gzx[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * dx * log(0.5 * dx / z0));
  }
  // set BC for vertical wall in front of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_front_indices.begin();
       it != wall_front_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((it->cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = it->cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);

    float z0 = (WGD->z0_domain_u[i + j * nx] + WGD->z0_domain_u[i + 1 + j * nx]
                + WGD->z0_domain_v[i + j * nx] + WGD->z0_domain_v[i + (j + 1) * nx])
               / 4.0;

    // Gyx = dvdx
    TGD->Gyx[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * dx * log(0.5 * dx / z0));
    // Gzx = dwdx
    TGD->Gzx[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * dx * log(0.5 * dx / z0));
  }

  // ## VERTICAL WALL ALONG Y (right/left)
  // set BC for vertical wall right of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_right_indices.begin();
       it != wall_right_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((it->cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = it->cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);

    float z0 = (WGD->z0_domain_u[i + j * nx] + WGD->z0_domain_u[i + 1 + j * nx]
                + WGD->z0_domain_v[i + j * nx] + WGD->z0_domain_v[i + (j + 1) * nx])
               / 4.0;

    // Gxy = dudy
    TGD->Gxy[it->cellID] = 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                           / (0.5 * dy * log(0.5 * dy / z0));
    // Gzy = dwdy
    TGD->Gzy[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * dy * log(0.5 * dy / z0));
  }
  // set BC for vertical wall left of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_below_indices.begin();
       it != wall_below_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((it->cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = it->cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);

    float z0 = (WGD->z0_domain_u[i + j * nx] + WGD->z0_domain_u[i + 1 + j * nx]
                + WGD->z0_domain_v[i + j * nx] + WGD->z0_domain_v[i + (j + 1) * nx])
               / 4.0;

    // Gxy = dudy
    TGD->Gxy[it->cellID] = 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                           / (0.5 * dy * log(0.5 * dy / z0));
    // Gzy = dwdy
    TGD->Gzy[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * dy * log(0.5 * dy / z0));
  }

  return;
}
