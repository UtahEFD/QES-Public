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
 * @file TURBWallBuilding.cpp
 * @brief :document this:
 */

#include "TURBWallBuilding.h"

void TURBWallBuilding::defineWalls(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  // fill array with cellid  of cutcell cells
  get_cutcell_wall_id(WGD, icellflag_cutcell);
  // fill itrublfag with cutcell flag
  set_cutcell_wall_flag(TGD, iturbflag_cutcell);

  // [FM] temporary fix -> use stairstep within the cut-cell
  get_stairstep_wall_id(WGD, icellflag_building);
  set_stairstep_wall_flag(TGD, iturbflag_stairstep);

  /*
    [FM] temporary fix -> when cut-cell treatement is implmented
    if(cutcell_wall_id.size()==0) {
    get_stairstep_wall_id(WGD,icellflag_building);
    set_stairstep_wall_flag(TGD,iturbflag_stairstep);
    }else{
    use_cutcell = true;
    }
  */

  return;
}


void TURBWallBuilding::setWallsBC(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{

  /*
    This function apply the loglow at the wall for building
    Note:
    - only stair-step is implemented
    - need to do: cut-cell for building
  */

  if (!use_cutcell) {
    //set_loglaw_stairstep(WGD, TGD);
    set_finitediff_stairstep(WGD, TGD);
  } else {
    //[FM] temporary fix because the cut-cell are messing with the wall
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


void TURBWallBuilding::set_finitediff_stairstep(WINDSGeneralData *WGD, TURBGeneralData *TGD)
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
    TGD->Gxz[it->cellID] = (0.5 * (WGD->u[it->faceID + nx * ny] + WGD->u[it->faceID + 1 + nx * ny])
                            - 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1]))
                           / (WGD->z[k + 1] - WGD->z[k]);

    // Gyz = dvdz
    TGD->Gyz[it->cellID] = (0.5 * (WGD->v[it->faceID + nx * ny] + WGD->v[it->faceID + nx + nx * ny])
                            - 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + nx]))
                           / (WGD->z[k + 1] - WGD->z[k]);
  }
  // set BC for horizontal wall above the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_above_indices.begin();
       it != wall_above_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));

    // Gxz = dudz
    TGD->Gxz[it->cellID] = (0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                            - 0.5 * (WGD->u[it->faceID - nx * ny] + WGD->u[it->faceID + 1 - nx * ny]))
                           / (WGD->z[k] - WGD->z[k - 1]);
    // Gyz = dvdz
    TGD->Gyz[it->cellID] = (0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                            - 0.5 * (WGD->u[it->faceID - nx * ny] + WGD->u[it->faceID + nx - nx * ny]))
                           / (WGD->z[k] - WGD->z[k - 1]);
  }

  // ## VERTICAL WALL ALONG X (front/back)
  // set BC for vertical wall in back of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_back_indices.begin();
       it != wall_back_indices.end();
       ++it) {
    // Gyx = dvdx
    TGD->Gyx[it->cellID] = (0.5 * (WGD->v[it->faceID + 1] + WGD->v[it->faceID + 1 + nx])
                            - 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx]))
                           / WGD->dx;
    // Gzx = dwdx
    TGD->Gzx[it->cellID] = (0.5 * (WGD->w[it->faceID + 1] + WGD->w[it->faceID + 1 + nx * ny])
                            - 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny]))
                           / WGD->dx;
  }
  // set BC for vertical wall in front of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_front_indices.begin();
       it != wall_front_indices.end();
       ++it) {
    // Gyx = dvdx
    TGD->Gyx[it->cellID] = (0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx])
                            - 0.5 * (WGD->v[it->faceID - 1] + WGD->v[it->faceID - 1 + nx]))
                           / WGD->dx;
    // Gzx = dwdx
    TGD->Gzx[it->cellID] = (0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                            - 0.5 * (WGD->w[it->faceID - 1] + WGD->w[it->faceID - 1 + nx * ny]))
                           / WGD->dx;
  }

  // ## VERTICAL WALL ALONG Y (right/left)
  // set BC for vertical wall right of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_right_indices.begin();
       it != wall_right_indices.end();
       ++it) {
    // Gxy = dudy
    TGD->Gxy[it->cellID] = (0.5 * (WGD->u[it->faceID + nx] + WGD->u[it->faceID + 1 + nx])
                            - 0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1]))
                           / WGD->dy;
    // Gzy = dwdy
    TGD->Gzy[it->cellID] = (0.5 * (WGD->w[it->faceID + nx] + WGD->w[it->faceID + nx + nx * ny])
                            - 0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny]))
                           / WGD->dy;
  }
  // set BC for vertical wall left of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_left_indices.begin();
       it != wall_left_indices.end();
       ++it) {
    // Gxy = dudy
    TGD->Gxy[it->cellID] = (0.5 * (WGD->u[it->faceID] + WGD->u[it->faceID + 1])
                            - 0.5 * (WGD->u[it->faceID - nx] + WGD->u[it->faceID + 1 - nx]))
                           / WGD->dy;
    // Gzy = dwdy
    TGD->Gzy[it->cellID] = (0.5 * (WGD->w[it->faceID] + WGD->w[it->faceID + nx * ny])
                            - 0.5 * (WGD->w[it->faceID - nx] + WGD->w[it->faceID - nx + nx * ny]))
                           / WGD->dy;
  }

  return;
}
