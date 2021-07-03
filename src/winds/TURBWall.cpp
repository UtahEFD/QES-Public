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
        int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);

        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
          /// Terrain below
          if (WGD->icellflag[icell_cent - (nx - 1) * (ny - 1)] == cellflag) {
            stairstep_wall_id.push_back(icell_cent);
          }
          /// Terrain in back
          if (WGD->icellflag[icell_cent - 1] == cellflag) {
            stairstep_wall_id.push_back(icell_cent);
          }
          /// Terrain in front
          if (WGD->icellflag[icell_cent + 1] == cellflag) {
            stairstep_wall_id.push_back(icell_cent);
          }
          /// Terrain on right
          if (WGD->icellflag[icell_cent - (nx - 1)] == cellflag) {
            stairstep_wall_id.push_back(icell_cent);
          }
          /// Terrain on left
          if (WGD->icellflag[icell_cent + (nx - 1)] == cellflag) {
            stairstep_wall_id.push_back(icell_cent);
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

  return;
}

void TURBWall::set_stairstep_wall_flag(TURBGeneralData *TGD, int cellflag)
{
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

void TURBWall::set_loglaw_stairstep_at_id_cc(WINDSGeneralData *WGD, TURBGeneralData *TGD, int id_cc, int flag2check, float z0)
{
  int nx = TGD->nx;
  int ny = TGD->ny;

  int k = (int)(id_cc / ((nx - 1) * (ny - 1)));
  int j = (int)((id_cc - k * (nx - 1) * (ny - 1)) / (nx - 1));
  int i = id_cc - j * (nx - 1) - k * (nx - 1) * (ny - 1);
  int id_fc = i + j * nx + k * nx * ny;

  // index of neighbour cells
  int idp, idm;
  // interpolated velocity field at neighbour cell center
  float up, um, vp, vm, wp, wm;

  // wind gradients in x-direction
  float dudx = 0.0;
  float dvdx = 0.0;
  float dwdx = 0.0;

  // dudx =[u(i+1,j,k)-u(i,j,k)]/(Delta x)
  dudx = (WGD->u[id_fc + 1] - WGD->u[id_fc]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);

  // Three cases: 1) wall behind (i-1), 2) wall in front (i+1) 3) no wall in x-direction
  if (WGD->icellflag[id_cc - 1] == flag2check) {
    // dvdx=v_hat/(0.5dx*log(0.5dx/z0))
    idp = id_fc + nx;// i,j+1,k
    idm = id_fc;// i,j,k
    vp = ((TGD->y_cc[j] - TGD->y_fc[j]) * WGD->v[idp] + (TGD->y_fc[j + 1] - TGD->y_cc[j]) * WGD->v[idm]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);
    dvdx = vp / ((TGD->x_cc[i] - TGD->x_fc[i]) * log((TGD->x_cc[i] - TGD->x_fc[i]) / z0));

    // dwdx=w_hat/(0.5dx*log(0.5dx/z0))
    idp = id_fc + nx * ny;// i,j,k+1
    idm = id_fc;// i,j,k
    wp = ((TGD->z_cc[k] - TGD->z_fc[k]) * WGD->w[idp] + (TGD->z_fc[k + 1] - TGD->z_cc[k]) * WGD->w[idm]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);
    dwdx = wp / ((TGD->x_cc[i] - TGD->x_fc[i]) * log((TGD->x_cc[i] - TGD->x_fc[i]) / z0));

  } else if (WGD->icellflag[id_cc + 1] == flag2check) {
    // dvdx=-v_hat/(0.5dx*log(0.5dx/z0))
    idp = id_fc + nx;// i,j+1,k
    idm = id_fc;// i,j,k
    vp = ((TGD->y_cc[j] - TGD->y_fc[j]) * WGD->v[idp] + (TGD->y_fc[j + 1] - TGD->y_cc[j]) * WGD->v[idm]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);
    dvdx = -vp / ((TGD->x_fc[i + 1] - TGD->x_cc[i]) * log((TGD->x_fc[i + 1] - TGD->x_cc[i]) / z0));

    // dwdx=-w_hat/(0.5dx*log(0.5dx/z0))
    idp = id_fc + nx * ny;// i,j,k+1
    idm = id_fc;// i,j,k
    wp = ((TGD->z_cc[k] - TGD->z_fc[k]) * WGD->w[idp] + (TGD->z_fc[k + 1] - TGD->z_cc[k]) * WGD->w[idm]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);
    dwdx = -wp / ((TGD->x_fc[i + 1] - TGD->x_cc[i]) * log((TGD->x_fc[i + 1] - TGD->x_cc[i]) / z0));

  } else {
    // dvdx
    // v_hat+
    idp = id_fc + 1 + nx;// i+1,j+1,k
    idm = id_fc + 1;// i+1,j,k
    vp = ((TGD->y_cc[j] - TGD->y_fc[j]) * WGD->v[idp] + (TGD->y_fc[j + 1] - TGD->y_cc[j]) * WGD->v[idm]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);
    // v_hat-
    idp = id_fc - 1 + nx;// i-1,j+1,k
    idm = id_fc - 1;// i-1,j,k
    vm = ((TGD->y_cc[j] - TGD->y_fc[j]) * WGD->v[idp] + (TGD->y_fc[j + 1] - TGD->y_cc[j]) * WGD->v[idm]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);
    // ==>
    dvdx = (vp - vm) / (TGD->x_cc[i + 1] - TGD->x_cc[i - 1]);

    // dwdx
    // w_hat+
    idp = id_fc + 1 + nx * ny;// i+1,j,k+1
    idm = id_fc + 1;// i+1,j,k
    wp = ((TGD->z_cc[k] - TGD->z_fc[k]) * WGD->w[idp] + (TGD->z_fc[k + 1] - TGD->z_cc[k]) * WGD->w[idm]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);
    // w_hat-
    idp = id_fc - 1 + nx * ny;// i-1,j,k+1
    idm = id_fc - 1;// i-1,j,k
    wm = ((TGD->z_cc[k] - TGD->z_fc[k]) * WGD->w[idp] + (TGD->z_fc[k + 1] - TGD->z_cc[k]) * WGD->w[idm]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);
    // ==>
    dwdx = (wp - wm) / (TGD->x_cc[i + 1] - TGD->x_cc[i - 1]);
  }

  // wind gradients in Y-direction
  float dudy = 0.0;
  float dvdy = 0.0;
  float dwdy = 0.0;

  // dvdy = [v(i,j+1,k)-v(i,j,k)]/(Delta y)
  dvdy = (WGD->v[id_fc + nx] - WGD->v[id_fc]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);

  // Three cases: 1) wall right (j-1), 2) wall left (j+1) 3) no wall in y-direction
  if (WGD->icellflag[id_cc - (nx - 1)] == flag2check) {
    // dudy=u_hat/(0.5dy*log(0.5dy/z0))
    idp = id_fc + 1;// i+1,j,k
    idm = id_fc;// i,j,k
    up = ((TGD->x_cc[i] - TGD->x_fc[i]) * WGD->u[idp] + (TGD->x_fc[i + 1] - TGD->x_cc[i]) * WGD->u[idm]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);
    dudy = up / ((TGD->y_cc[j] - TGD->y_fc[j]) * log((TGD->y_cc[j] - TGD->y_fc[j]) / z0));

    // dwdx=w_hat/(0.5dx*log(0.5dx/z0))
    idp = id_fc + nx * ny;// i,j,k+1
    idm = id_fc;// i,j,k
    wp = ((TGD->z_cc[k] - TGD->z_fc[k]) * WGD->w[idp] + (TGD->z_fc[k + 1] - TGD->z_cc[k]) * WGD->w[idm]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);
    dwdy = wp / ((TGD->y_cc[j] - TGD->y_fc[j]) * log((TGD->y_cc[j] - TGD->y_fc[j]) / z0));

  } else if (WGD->icellflag[id_cc + (nx - 1)] == flag2check) {
    // dudy=-u_hat/(0.5dy*log(0.5dy/z0))
    idp = id_fc + 1;// i+1,j,k
    idm = id_fc;// i,j,k
    up = ((TGD->x_cc[i] - TGD->x_fc[i]) * WGD->u[idp] + (TGD->x_fc[i + 1] - TGD->x_cc[i]) * WGD->u[idm]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);
    dudy = -up / ((TGD->y_fc[j + 1] - TGD->y_cc[j]) * log((TGD->y_fc[j + 1] - TGD->y_cc[j]) / z0));

    // dwdy=-w_hat/(0.5dy*log(0.5dy/z0))
    idp = id_fc + nx * ny;// i,j,k+1
    idm = id_fc;// i,j,k
    wp = ((TGD->z_cc[k] - TGD->z_fc[k]) * WGD->w[idp] + (TGD->z_fc[k + 1] - TGD->z_cc[k]) * WGD->w[idm]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);
    dwdy = -wp / ((TGD->y_fc[i + 1] - TGD->y_cc[i]) * log((TGD->y_fc[i + 1] - TGD->y_cc[i]) / z0));

  } else {
    // u_hat+
    idp = id_fc + 1 + nx;// i+1,j+1,k
    idm = id_fc + nx;// i,j+1,k
    up = ((TGD->x_cc[i] - TGD->x_fc[i]) * WGD->u[idp] + (TGD->x_fc[i + 1] - TGD->x_cc[i]) * WGD->u[idm]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);
    // u_hat-
    idp = id_fc + 1 - nx;// i+1,j-1,k
    idm = id_fc - nx;// i,j-1,k
    um = ((TGD->x_cc[i] - TGD->x_fc[i]) * WGD->u[idp] + (TGD->x_fc[i + 1] - TGD->x_cc[i]) * WGD->u[idm]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);
    // ==>
    dudy = (up - um) / (TGD->y_cc[j + 1] - TGD->y_cc[j - 1]);

    // w_hat+
    idp = id_fc + nx + nx * ny;// i,j+1,k+1
    idm = id_fc + nx;// i,j,+1,k
    wp = ((TGD->z_cc[k - 1] - TGD->z_fc[k]) * WGD->w[idp] + (TGD->z_fc[k + 1] - TGD->z_cc[k]) * WGD->w[idm]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);
    // w_hat-
    idp = id_fc - nx + nx * ny;// i,j-1,k+1
    idm = id_fc - nx;// i,j-1,k
    wp = ((TGD->z_cc[k] - TGD->z_fc[k]) * WGD->w[idp] + (TGD->z_fc[k + 1] - TGD->z_cc[k]) * WGD->w[idm]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);
    // ==>
    dwdy = (wp - wm) / (TGD->y_cc[j + 1] - TGD->y_cc[j - 1]);
  }

  // wind gradients in Z-direction
  float dudz = 0.0;
  float dvdz = 0.0;
  float dwdz = 0.0;

  // dwdz = [w(i,j,k+1)-w(i,j,k)]/(Delta z)
  dwdz = (WGD->w[id_fc + nx * ny] - WGD->w[id_fc]) / (TGD->z_fc[k + 1] - TGD->z_fc[k]);

  // Three cases: 1) wall below (k-1), 2) wall above (k+1) 3) no wall in y-direction
  if (WGD->icellflag[id_cc - (nx - 1) * (ny - 1)] == flag2check) {
    // dudz=u_hat/(0.5dz*log(0.5dz/z0))
    idp = id_fc + 1;// i+1,j,k
    idm = id_fc;// i,j,k
    up = ((TGD->x_cc[i] - TGD->x_fc[i]) * WGD->u[idp] + (TGD->x_fc[i + 1] - TGD->x_cc[i]) * WGD->u[idm]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);
    dudz = up / ((TGD->z_cc[k] - TGD->z_fc[k]) * log((TGD->z_cc[k] - TGD->z_fc[k]) / z0));

    // dvdz=v_hat/(0.5dz*log(0.5dz/z0))
    idp = id_fc + nx;// i,j+1,k
    idm = id_fc;// i,j,k
    vp = ((TGD->y_cc[j] - TGD->y_fc[j]) * WGD->v[idp] + (TGD->y_fc[j + 1] - TGD->y_cc[j]) * WGD->v[idm]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);
    dvdz = vp / ((TGD->z_cc[k] - TGD->z_fc[k]) * log((TGD->z_cc[k] - TGD->z_fc[k]) / z0));

  } else if (WGD->icellflag[id_cc + (nx - 1) * (ny - 1)] == flag2check) {
    // dudz=-u_hat/(0.5dz*log(0.5dz/z0))
    idp = id_fc + 1;// i+1,j,k
    idm = id_fc;// i,j,k
    up = ((TGD->x_cc[i] - TGD->x_fc[i]) * WGD->u[idp] + (TGD->x_fc[i + 1] - TGD->x_cc[i]) * WGD->u[idm]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);
    dudz = -up / ((TGD->z_fc[k + 1] - TGD->z_cc[k]) * log((TGD->z_fc[k + 1] - TGD->z_cc[k]) / z0));

    // dvdz=-v_hat/(0.5dz*log(0.5dz/z0))
    idp = id_fc + nx;// i,j+1,k
    idm = id_fc;// i,j,k
    vp = ((TGD->y_cc[j] - TGD->y_fc[j]) * WGD->v[idp] + (TGD->y_fc[j + 1] - TGD->y_cc[j]) * WGD->v[idm]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);
    dvdz = -vp / ((TGD->z_fc[k + 1] - TGD->z_cc[k]) * log((TGD->z_fc[k + 1] - TGD->z_cc[k]) / z0));

  } else {
    // u_hat+
    idp = id_fc + 1 + nx * ny;// i+1,j,k+1
    idm = id_fc + nx * ny;// i,j,k+1
    up = ((TGD->x_cc[i] - TGD->x_fc[i]) * WGD->u[idp] + (TGD->x_fc[i + 1] - TGD->x_cc[i]) * WGD->u[idm]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);
    // u_hat-
    idp = id_fc + 1 - nx * ny;// i+1,j,k-1
    idm = id_fc - nx * ny;// i,j,k-1
    um = ((TGD->x_cc[i] - TGD->x_fc[i]) * WGD->u[idp] + (TGD->x_fc[i + 1] - TGD->x_cc[i]) * WGD->u[idm]) / (TGD->x_fc[i + 1] - TGD->x_fc[i]);
    // ==>
    dudz = (up - um) / (TGD->z_cc[k + 1] - TGD->z_cc[k - 1]);

    // v_hat+
    idp = id_fc + nx + nx * ny;// i,j+1,k+1
    idm = id_fc + nx * ny;// i,j,k+1
    vp = ((TGD->y_cc[j] - TGD->y_fc[j]) * WGD->v[idp] + (TGD->y_fc[j + 1] - TGD->y_cc[j]) * WGD->v[idm]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);
    // v_hat-
    idp = id_fc + nx - nx * ny;// i,j+1,k-1
    idm = id_fc - nx * ny;// i,j,k-1
    vm = ((TGD->y_cc[j] - TGD->y_fc[j]) * WGD->v[idp] + (TGD->y_fc[j + 1] - TGD->y_cc[j]) * WGD->v[idm]) / (TGD->y_fc[j + 1] - TGD->y_fc[j]);
    // ==>
    dvdz = (vp - vm) / (TGD->z_cc[k + 1] - TGD->z_cc[k - 1]);
  }

  // Strain-rate tensor
  // diagonal terms
  TGD->Sxx[id_cc] = dudx;
  TGD->Syy[id_cc] = dvdy;
  TGD->Szz[id_cc] = dwdz;
  // off-diagonal terms
  TGD->Sxy[id_cc] = 0.5 * (dudy + dvdx);
  TGD->Syz[id_cc] = 0.5 * (dvdz + dwdy);
  TGD->Sxz[id_cc] = 0.5 * (dudz + dwdx);

  return;
}
