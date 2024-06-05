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
 * @file TURBWall.cpp
 * @brief :document this:
 */

#include "TURBWall.h"
#include "TURBGeneralData.h"

void TURBWall::get_stairstep_wall_id(WINDSGeneralData *WGD, int cellflag)
{
  int nx = WGD->nx;
  int ny = WGD->ny;
  int nz = WGD->nz;

  // container for cell above terrain (needed to remove dublicate for the wall law)
  // -> need to treat the wall all at once because of strain-rate tensor
  for (int i = 1; i < nx - 2; i++) {
    for (int j = 1; j < ny - 2; j++) {
      for (int k = 1; k < nz - 2; k++) {
        int cellID = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);

        if (WGD->icellflag[cellID] != 0 && WGD->icellflag[cellID] != 2) {
          /// Terrain below
          if (WGD->icellflag[cellID - (nx - 1) * (ny - 1)] == cellflag) {
            stairstep_wall_id.push_back(cellID);
          }
          /// Terrain in back
          if (WGD->icellflag[cellID - 1] == cellflag) {
            stairstep_wall_id.push_back(cellID);
          }
          /// Terrain in front
          if (WGD->icellflag[cellID + 1] == cellflag) {
            stairstep_wall_id.push_back(cellID);
          }
          /// Terrain on right
          if (WGD->icellflag[cellID - (nx - 1)] == cellflag) {
            stairstep_wall_id.push_back(cellID);
          }
          /// Terrain on left
          if (WGD->icellflag[cellID + (nx - 1)] == cellflag) {
            stairstep_wall_id.push_back(cellID);
          }
        }
      }
    }
  }

  // erase duplicates and sort above terrain indices.
  std::unordered_set<int> s;
  for (int i : stairstep_wall_id) {
    s.insert(i);
  }
  stairstep_wall_id.assign(s.begin(), s.end());
  sort(stairstep_wall_id.begin(), stairstep_wall_id.end());

  for (auto i = 1; i < nx - 1; i++) {
    for (auto j = 1; j < ny - 1; j++) {
      for (auto k = 1; k < nz - 2; k++) {
        int cellID = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        int faceID = i + j * nx + k * nx * ny;

        if (WGD->icellflag[cellID] != 0 && WGD->icellflag[cellID] != 2) {

          // Wall below
          if (WGD->icellflag[cellID - (nx - 1) * (ny - 1)] == cellflag) {
            wall_below_indices.push_back({ cellID, faceID });
          }
          // Wall above
          if (WGD->icellflag[cellID + (nx - 1) * (ny - 1)] == cellflag) {
            wall_above_indices.push_back({ cellID, faceID });
          }
          // Wall in back
          if (WGD->icellflag[cellID - 1] == cellflag) {
            wall_back_indices.push_back({ cellID, faceID });
          }
          // Wall in front
          if (WGD->icellflag[cellID + 1] == cellflag) {
            wall_front_indices.push_back({ cellID, faceID });
          }
          // Wall on right
          if (WGD->icellflag[cellID - (nx - 1)] == cellflag) {
            wall_right_indices.push_back({ cellID, faceID });
          }
          // Wall on left
          if (WGD->icellflag[cellID + (nx - 1)] == cellflag) {
            wall_left_indices.push_back({ cellID, faceID });
          }
        }
      }
    }
  }

  return;
}

void TURBWall::set_stairstep_wall_flag(TURBGeneralData *TGD, int cellflag)
{

  for (std::vector<pairCellFaceID>::iterator it = wall_below_indices.begin();
       it != wall_below_indices.end();
       ++it) {
    TGD->iturbflag.at(it->cellID) = cellflag;
  }
  for (std::vector<pairCellFaceID>::iterator it = wall_above_indices.begin();
       it != wall_above_indices.end();
       ++it) {
    TGD->iturbflag.at(it->cellID) = cellflag;
  }

  for (std::vector<pairCellFaceID>::iterator it = wall_back_indices.begin();
       it != wall_back_indices.end();
       ++it) {
    TGD->iturbflag.at(it->cellID) = cellflag;
  }
  for (std::vector<pairCellFaceID>::iterator it = wall_front_indices.begin();
       it != wall_front_indices.end();
       ++it) {
    TGD->iturbflag.at(it->cellID) = cellflag;
  }

  for (std::vector<pairCellFaceID>::iterator it = wall_right_indices.begin();
       it != wall_right_indices.end();
       ++it) {
    TGD->iturbflag.at(it->cellID) = cellflag;
  }
  for (std::vector<pairCellFaceID>::iterator it = wall_left_indices.begin();
       it != wall_left_indices.end();
       ++it) {
    TGD->iturbflag.at(it->cellID) = cellflag;
  }

  for (size_t id = 0; id < stairstep_wall_id.size(); ++id) {
    int idcell = stairstep_wall_id.at(id);
    TGD->iturbflag.at(idcell) = cellflag;
  }

  return;
}

void TURBWall::get_cutcell_wall_id(WINDSGeneralData *WGD, int cellflag)
{
  int nx = WGD->nx;
  int ny = WGD->ny;
  int nz = WGD->nz;

  for (int k = 0; k < nz - 2; k++) {
    for (int j = 1; j < ny - 2; j++) {
      for (int i = 1; i < nx - 2; i++) {
        int id = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (WGD->icellflag[id] == cellflag) {
          cutcell_wall_id.push_back(id);
        }
      }
    }
  }

  return;
}

void TURBWall::set_cutcell_wall_flag(TURBGeneralData *TGD, int cellflag)
{
  for (size_t id = 0; id < cutcell_wall_id.size(); ++id) {
    int idcell = cutcell_wall_id.at(id);
    TGD->iturbflag.at(idcell) = cellflag;
  }

  return;
}


void TURBWall::set_loglaw_stairstep_at_id_cc(WINDSGeneralData *WGD,
                                             TURBGeneralData *TGD,
                                             int id_cc,
                                             int flag2check,
                                             float z0)
{
  return;
}

void TURBWall::comp_velocity_deriv_finitediff_stairstep(WINDSGeneralData *WGD, TURBGeneralData *TGD)
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
                            - 0.5 * (WGD->v[it->faceID] + WGD->v[it->faceID + nx]))
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
                            - 0.5 * (WGD->v[it->faceID - nx * ny] + WGD->v[it->faceID + nx - nx * ny]))
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

void TURBWall::comp_stress_deriv_finitediff_stairstep(WINDSGeneralData *WGD,
                                                      TURBGeneralData *TGD,
                                                      const std::vector<float> &tox,
                                                      const std::vector<float> &toy,
                                                      const std::vector<float> &toz)
{
  int nx = WGD->nx;
  int ny = WGD->ny;

  // ## HORIZONTAL WALL
  // set BC for horizontal wall below the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_below_indices.begin();
       it != wall_below_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));
    // dtozdz
    TGD->tmp_dtozdz[it->cellID] = (toz[it->cellID + (nx - 1) * (ny - 1)] - toz[it->cellID])
                                  / (WGD->z[k + 1] - WGD->z[k]);
  }
  // set BC for horizontal wall above the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_above_indices.begin();
       it != wall_above_indices.end();
       ++it) {
    int k = (int)(it->cellID / ((nx - 1) * (ny - 1)));
    // dtozdz
    TGD->tmp_dtozdz[it->cellID] = (toz[it->cellID] - toz[it->cellID - (nx - 1) * (ny - 1)])
                                  / (WGD->z[k + 1] - WGD->z[k]);
  }

  // ## VERTICAL WALL ALONG X (front/back)
  // set BC for vertical wall in back of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_back_indices.begin();
       it != wall_back_indices.end();
       ++it) {
    // dtoxdx
    TGD->tmp_dtoxdx[it->cellID] = (tox[it->cellID + 1] - tox[it->cellID]) / WGD->dx;
  }
  // set BC for vertical wall in front of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_front_indices.begin();
       it != wall_front_indices.end();
       ++it) {
    // dtoxdx
    TGD->tmp_dtoxdx[it->cellID] = (toy[it->cellID] - tox[it->cellID - 1]) / WGD->dx;
  }

  // ## VERTICAL WALL ALONG Y (right/left)
  // set BC for vertical wall right of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_right_indices.begin();
       it != wall_right_indices.end();
       ++it) {
    // dtoydy
    TGD->tmp_dtoydy[it->cellID] = (toy[it->cellID + (nx - 1)] - toy[it->cellID]) / WGD->dy;
  }
  // set BC for vertical wall left of the cell
  for (std::vector<pairCellFaceID>::iterator it = wall_left_indices.begin();
       it != wall_left_indices.end();
       ++it) {
    // dtoydy
    TGD->tmp_dtoydy[it->cellID] = (toy[it->cellID] - toy[it->cellID - (nx - 1)]) / WGD->dy;
  }

  return;
}
