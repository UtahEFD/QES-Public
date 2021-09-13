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

/**
 * @file TURBWallTerrain.cpp
 * @brief :document this:
 */

#include "TURBWallTerrain.h"

void TURBWallTerrain::defineWalls(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  // fill array with cellid  of cutcell cells
  get_cutcell_wall_id(WGD, icellflag_cutcell);
  // fill itrublfag with cutcell flag
  set_cutcell_wall_flag(TGD, iturbflag_cutcell);

  // [FM] temporary fix -> use stairstep within the cut-cell
  get_stairstep_wall_id(WGD, icellflag_terrain);
  set_stairstep_wall_flag(TGD, iturbflag_stairstep);

  /*
    [FM] temporary fix -> when cut-cell treatement is implmented
    if(cutcell_wall_id.size()==0) {
    get_stairstep_wall_id(WGD,icellflag_terrain);
    set_stairstep_wall_flag(TGD,iturbflag_stairstep);
    } else {
    use_cutcell = true;
    }
  */

  return;
}


void TURBWallTerrain::setWallsBC(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{

  /*
    This function apply the loglow at the wall for terrain
    Note:
    - only stair-step is implemented
    - need to do: cut-cell for terrain
  */

  if (!use_cutcell) {
    set_loglaw_stairstep(WGD, TGD);
  } else {
    // [FM] temporary fix because the cut-cell are messing with the wall
    // at the terrain
    /* 
       for (size_t i = 0; i < cutcell_wall_id.size(); i++) {
       int id_cc = cutcell_wall_id[i];
       TGD->Sxx[id_cc] = 0.0;
       TGD->Sxy[id_cc] = 0.0;
       TGD->Sxz[id_cc] = 0.0;
       TGD->Syy[id_cc] = 0.0;
       TGD->Syz[id_cc] = 0.0;
       TGD->Szz[id_cc] = 0.0;
      TGD->Lm[id_cc] = 0.0;
      }
    */
  }
}


void TURBWallTerrain::set_loglaw_stairstep(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  int nx = WGD->nx;
  int ny = WGD->ny;

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
                           / (0.5 * WGD->dz_array[k] * log(0.5 * WGD->dz_array[k] / z0));
    // Gyz = dvdz
    TGD->Gyz[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * WGD->dz_array[k] * log(0.5 * WGD->dz_array[k] / z0));
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
                           / (0.5 * WGD->dz_array[k] * log(0.5 * WGD->dz_array[k] / z0));
    // Gyz = dvdz
    TGD->Gyz[it->cellID] = 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                           / (0.5 * WGD->dz_array[k] * log(0.5 * WGD->dz_array[k] / z0));
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
                           / (0.5 * WGD->dx * log(0.5 * WGD->dx / z0));
    // Gzx = dwdx
    TGD->Gzx[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * WGD->dx * log(0.5 * WGD->dx / z0));
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
                           / (0.5 * WGD->dx * log(0.5 * WGD->dx / z0));
    // Gzx = dwdx
    TGD->Gzx[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * WGD->dx * log(0.5 * WGD->dx / z0));
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
                           / (0.5 * WGD->dy * log(0.5 * WGD->dy / z0));
    // Gzy = dwdy
    TGD->Gzy[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * WGD->dy * log(0.5 * WGD->dy / z0));
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
                           / (0.5 * WGD->dy * log(0.5 * WGD->dy / z0));
    // Gzy = dwdy
    TGD->Gzy[it->cellID] = 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                           / (0.5 * WGD->dy * log(0.5 * WGD->dy / z0));
  }

  return;
}
