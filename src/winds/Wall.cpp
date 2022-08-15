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
 * @file Wall.cpp
 * @brief :brief here:
 *
 * :long desc here if necessary:
 */
#include "Wall.h"

#include "WINDSGeneralData.h"
#include "WINDSInputData.h"


void Wall::defineWalls(WINDSGeneralData *WGD)
{

  //float dx = WGD->dx;
  //float dy = WGD->dy;
  //float dz = WGD->dz;
  int nx = WGD->nx;
  int ny = WGD->ny;
  int nz = WGD->nz;


  for (auto k = 1; k < nz - 1; k++) {
    for (auto i = 0; i < nx - 1; i++) {
      for (auto j = 0; j < ny - 1; j++) {
        int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (WGD->e[icell_cent] < 0.05) {
          WGD->e[icell_cent] = 0.0;
        }
        if (WGD->f[icell_cent] < 0.05) {
          WGD->f[icell_cent] = 0.0;
        }
        if (WGD->g[icell_cent] < 0.05) {
          WGD->g[icell_cent] = 0.0;
        }
        if (WGD->h[icell_cent] < 0.05) {
          WGD->h[icell_cent] = 0.0;
        }
        if (WGD->m[icell_cent] < 0.05) {
          WGD->m[icell_cent] = 0.0;
        }
        if (WGD->n[icell_cent] < 0.05) {
          WGD->n[icell_cent] = 0.0;
        }

        if (WGD->e[icell_cent] > 1.0) {
          WGD->e[icell_cent] = 1.0;
        }
        if (WGD->f[icell_cent] > 1.0) {
          WGD->f[icell_cent] = 1.0;
        }
        if (WGD->g[icell_cent] > 1.0) {
          WGD->g[icell_cent] = 1.0;
        }
        if (WGD->h[icell_cent] > 1.0) {
          WGD->h[icell_cent] = 1.0;
        }
        if (WGD->m[icell_cent] > 1.0) {
          WGD->m[icell_cent] = 1.0;
        }
        if (WGD->n[icell_cent] > 1.0) {
          WGD->n[icell_cent] = 1.0;
        }
      }
    }
  }

  for (auto k = 1; k < nz - 1; k++) {
    for (auto i = 0; i < nx - 1; i++) {
      for (auto j = 0; j < ny - 1; j++) {
        int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if ((WGD->icellflag[icell_cent] == 7 || WGD->icellflag[icell_cent] == 0) && WGD->building_volume_frac[icell_cent] <= 0.1) {
          WGD->icellflag[icell_cent] = 0;
          WGD->e[icell_cent] = 1.0;
          WGD->f[icell_cent] = 1.0;
          WGD->g[icell_cent] = 1.0;
          WGD->h[icell_cent] = 1.0;
          WGD->m[icell_cent] = 1.0;
          WGD->n[icell_cent] = 1.0;
        }
      }
    }
  }

  for (auto k = 1; k < nz - 1; k++) {
    for (auto i = 0; i < nx - 1; i++) {
      for (auto j = 0; j < ny - 1; j++) {
        int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (WGD->e[icell_cent] == 0.0 && WGD->f[icell_cent] == 0.0 && WGD->g[icell_cent] == 0.0
            && WGD->h[icell_cent] == 0.0 && WGD->m[icell_cent] == 0.0 && WGD->n[icell_cent] == 0.0 && WGD->icellflag[icell_cent] == 7) {
          WGD->icellflag[icell_cent] = 0;
          WGD->e[icell_cent] = 1.0;
          WGD->f[icell_cent] = 1.0;
          WGD->g[icell_cent] = 1.0;
          WGD->h[icell_cent] = 1.0;
          WGD->m[icell_cent] = 1.0;
          WGD->n[icell_cent] = 1.0;
        }
      }
    }
  }

  for (auto k = 1; k < nz - 2; k++) {
    for (auto i = 0; i < nx - 1; i++) {
      for (auto j = 0; j < ny - 1; j++) {
        int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        //int icell_face = i + j * nx + k * nx * ny;

        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

          // Wall below
          if (WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 0 || WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
            WGD->n[icell_cent] = 0.0;
          }
          // Wall above
          if (WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 0 || WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
            WGD->m[icell_cent] = 0.0;
          }
          // Wall in back
          if (WGD->icellflag[icell_cent - 1] == 0 || WGD->icellflag[icell_cent - 1] == 2) {
            if (i > 0) {
              WGD->f[icell_cent] = 0.0;
            }
          }
          // Wall in front
          if (WGD->icellflag[icell_cent + 1] == 0 || WGD->icellflag[icell_cent + 1] == 2) {
            WGD->e[icell_cent] = 0.0;
          }
          // Wall on right
          if (WGD->icellflag[icell_cent - (WGD->nx - 1)] == 0 || WGD->icellflag[icell_cent - (WGD->nx - 1)] == 2) {
            if (j > 0) {
              WGD->h[icell_cent] = 0.0;
            }
          }
          // Wall on left
          if (WGD->icellflag[icell_cent + (WGD->nx - 1)] == 0 || WGD->icellflag[icell_cent + (WGD->nx - 1)] == 2) {
            WGD->g[icell_cent] = 0.0;
          }
        }

        if (WGD->icellflag[icell_cent] == 7) {
          if (WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 7 || WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 8) {
            WGD->m[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] = WGD->n[icell_cent];
          }

          if (WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 7 || WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 8) {
            WGD->n[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] = WGD->m[icell_cent];
          }

          if (WGD->icellflag[icell_cent - 1] == 7 || WGD->icellflag[icell_cent - 1] == 8) {
            if (i > 0) {
              WGD->f[icell_cent] = WGD->e[icell_cent - 1];
            }
          }

          if (WGD->icellflag[icell_cent + 1] == 7 || WGD->icellflag[icell_cent + 1] == 8) {
            WGD->e[icell_cent] = WGD->f[icell_cent + 1];
          }

          if (WGD->icellflag[icell_cent - (WGD->nx - 1)] == 7 || WGD->icellflag[icell_cent - (WGD->nx - 1)] == 8) {
            if (j > 0) {
              WGD->h[icell_cent] = WGD->g[icell_cent - (WGD->nx - 1)];
            }
          }

          if (WGD->icellflag[icell_cent + (WGD->nx - 1)] == 7 || WGD->icellflag[icell_cent + (WGD->nx - 1)] == 8) {
            WGD->g[icell_cent] = WGD->h[icell_cent + (WGD->nx - 1)];
          }
        }

        if (WGD->icellflag[icell_cent] == 8) {
          if (WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 7 || WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 8) {
            WGD->m[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] = WGD->n[icell_cent];
          }

          if (WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 7 || WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 8) {
            WGD->n[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] = WGD->m[icell_cent];
          }

          if (WGD->icellflag[icell_cent - 1] == 7 || WGD->icellflag[icell_cent - 1] == 8) {
            if (i > 0) {
              WGD->f[icell_cent] = WGD->e[icell_cent - 1];
            }
          }

          if (WGD->icellflag[icell_cent + 1] == 7 || WGD->icellflag[icell_cent + 1] == 8) {
            WGD->e[icell_cent] = WGD->f[icell_cent + 1];
          }

          if (WGD->icellflag[icell_cent - (WGD->nx - 1)] == 7 || WGD->icellflag[icell_cent - (WGD->nx - 1)] == 8) {
            if (j > 0) {
              WGD->h[icell_cent] = WGD->g[icell_cent - (WGD->nx - 1)];
            }
          }

          if (WGD->icellflag[icell_cent + (WGD->nx - 1)] == 7 || WGD->icellflag[icell_cent + (WGD->nx - 1)] == 8) {
            WGD->g[icell_cent] = WGD->h[icell_cent + (WGD->nx - 1)];
          }
        }

        if (WGD->icellflag[icell_cent] == 1) {
          if (WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 7 || WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 8) {
            WGD->n[icell_cent] = WGD->m[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)];
          }

          if (WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 7 || WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 8) {
            WGD->m[icell_cent] = WGD->n[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)];
          }

          if (WGD->icellflag[icell_cent - 1] == 7 || WGD->icellflag[icell_cent - 1] == 8) {
            if (i > 0) {
              WGD->f[icell_cent] = WGD->e[icell_cent - 1];
            }
          }

          if (WGD->icellflag[icell_cent + 1] == 7 || WGD->icellflag[icell_cent + 1] == 8) {
            WGD->e[icell_cent] = WGD->f[icell_cent + 1];
          }

          if (WGD->icellflag[icell_cent - (WGD->nx - 1)] == 7 || WGD->icellflag[icell_cent - (WGD->nx - 1)] == 8) {
            if (j > 0) {
              WGD->h[icell_cent] = WGD->g[icell_cent - (WGD->nx - 1)];
            }
          }

          if (WGD->icellflag[icell_cent + (WGD->nx - 1)] == 7 || WGD->icellflag[icell_cent + (WGD->nx - 1)] == 8) {
            WGD->g[icell_cent] = WGD->h[icell_cent + (WGD->nx - 1)];
          }
        }
      }
    }
  }

  for (auto k = 1; k < nz - 1; k++) {
    for (auto i = 0; i < nx - 1; i++) {
      for (auto j = 0; j < ny - 1; j++) {
        int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        if (WGD->e[icell_cent] == 0.0 && WGD->f[icell_cent] == 0.0 && WGD->g[icell_cent] == 0.0
            && WGD->h[icell_cent] == 0.0 && WGD->m[icell_cent] == 0.0 && WGD->n[icell_cent] == 0.0
            && (WGD->icellflag[icell_cent] == 7 || WGD->icellflag[icell_cent] == 0)) {
          WGD->icellflag[icell_cent] = 0;
          WGD->e[icell_cent] = 1.0;
          WGD->f[icell_cent] = 1.0;
          WGD->g[icell_cent] = 1.0;
          WGD->h[icell_cent] = 1.0;
          WGD->m[icell_cent] = 1.0;
          WGD->n[icell_cent] = 1.0;
        }


        if (WGD->e[icell_cent] == 0.0 && WGD->f[icell_cent] == 0.0 && WGD->g[icell_cent] == 0.0
            && WGD->h[icell_cent] == 0.0 && WGD->m[icell_cent] == 0.0 && WGD->n[icell_cent] == 0.0
            && (WGD->icellflag[icell_cent] == 8 || WGD->icellflag[icell_cent] == 2)) {
          WGD->icellflag[icell_cent] = 2;
          WGD->e[icell_cent] = 1.0;
          WGD->f[icell_cent] = 1.0;
          WGD->g[icell_cent] = 1.0;
          WGD->h[icell_cent] = 1.0;
          WGD->m[icell_cent] = 1.0;
          WGD->n[icell_cent] = 1.0;
        }
      }
    }
  }

  for (auto k = 1; k < nz - 2; k++) {
    for (auto i = 0; i < nx - 1; i++) {
      for (auto j = 0; j < ny - 1; j++) {
        int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
        int icell_face = i + j * nx + k * nx * ny;
        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

          // Wall below
          if (WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 0 || WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
            if (WGD->icellflag[icell_cent] != 8) {
              WGD->wall_below_indices.push_back(icell_face);
            }
            WGD->n[icell_cent] = 0.0;
          }
          // Wall above
          if (WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 0 || WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
            if (WGD->icellflag[icell_cent] != 8) {
              WGD->wall_above_indices.push_back(icell_face);
            }
            WGD->m[icell_cent] = 0.0;
          }
          // Wall in back
          if (WGD->icellflag[icell_cent - 1] == 0 || WGD->icellflag[icell_cent - 1] == 2) {
            if (i > 0) {
              if (WGD->icellflag[icell_cent] != 8) {
                WGD->wall_back_indices.push_back(icell_face);
              }
              WGD->f[icell_cent] = 0.0;
            }
          }
          // Wall in front
          if (WGD->icellflag[icell_cent + 1] == 0 || WGD->icellflag[icell_cent + 1] == 2) {
            if (WGD->icellflag[icell_cent] != 8) {
              WGD->wall_front_indices.push_back(icell_face);
            }
            WGD->e[icell_cent] = 0.0;
          }
          // Wall on right
          if (WGD->icellflag[icell_cent - (WGD->nx - 1)] == 0 || WGD->icellflag[icell_cent - (WGD->nx - 1)] == 2) {
            if (j > 0) {
              if (WGD->icellflag[icell_cent] != 8) {
                WGD->wall_right_indices.push_back(icell_face);
              }
              WGD->h[icell_cent] = 0.0;
            }
          }
          // Wall on left
          if (WGD->icellflag[icell_cent + (WGD->nx - 1)] == 0 || WGD->icellflag[icell_cent + (WGD->nx - 1)] == 2) {
            if (WGD->icellflag[icell_cent] != 8) {
              WGD->wall_left_indices.push_back(icell_face);
            }
            WGD->g[icell_cent] = 0.0;
          }
        }

        if (WGD->icellflag[icell_cent] == 7 || WGD->icellflag[icell_cent] == 8) {
          WGD->wall_indices.push_back(icell_cent);
        }

        /*if (WGD->icellflag[icell_cent] == 1) {
          if (WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 0 || WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
            WGD->wall_indices.push_back(icell_cent);
          }
          // Wall above
          if (WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 0 || WGD->icellflag[icell_cent + (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
            WGD->wall_indices.push_back(icell_cent);
          }
          // Wall in back
          if (WGD->icellflag[icell_cent - 1] == 0 || WGD->icellflag[icell_cent - 1] == 2) {
            if (i > 0) {
              WGD->wall_indices.push_back(icell_cent);
            }
          }
          // Wall in front
          if (WGD->icellflag[icell_cent + 1] == 0 || WGD->icellflag[icell_cent + 1] == 2) {
            WGD->wall_indices.push_back(icell_cent);
          }
          // Wall on right
          if (WGD->icellflag[icell_cent - (WGD->nx - 1)] == 0 || WGD->icellflag[icell_cent - (WGD->nx - 1)] == 2) {
            if (j > 0) {
              WGD->wall_indices.push_back(icell_cent);
            }
          }
          // Wall on left
          if (WGD->icellflag[icell_cent + (WGD->nx - 1)] == 0 || WGD->icellflag[icell_cent + (WGD->nx - 1)] == 2) {
            WGD->wall_indices.push_back(icell_cent);
          }
        }*/
      }
    }
  }
}


void Wall::wallLogBC(WINDSGeneralData *WGD, bool isInitial)
{
  auto wallLog_start = std::chrono::high_resolution_clock::now();

  std::cout << "Applying log law parameterization..." << std::flush;

  float dx = WGD->dx;
  float dy = WGD->dy;
  float dz = WGD->dz;
  int nx = WGD->nx;
  int ny = WGD->ny;
  //int nz = WGD->nz;
  const float z0 = WGD->z0;
  std::vector<float> &u0 = WGD->u0;
  std::vector<float> &v0 = WGD->v0;
  std::vector<float> &w0 = WGD->w0;

  float ustar_wall;// velocity gradient at the wall
  float new_ustar;// new ustar value calculated
  float vel_mag1;// velocity magnitude at the nearest cell to wall in perpendicular direction
  float vel_mag2;// velocity magnitude at the second cell near wall in perpendicular direction
  float dist1;// distance of the center of the nearest cell in perpendicular direction from wall
  float dist2;// distance of the center of second near cell in perpendicular direction from wall
  float wind_dir;// wind direction in parallel planes to wall
  int icell_cent;
  float s_behind, s_front, s_right, s_left, s_below, s_above, s_cut;// Area of each face and the cut-cell filled by solid
  int first_i, first_j, first_k;
  int second_i, second_j, second_k;

  // Loop through all the cells
  for (auto i = 0; i < WGD->wall_indices.size(); i++) {
    if (WGD->ni[WGD->wall_indices[i]] == 0.0 && WGD->nj[WGD->wall_indices[i]] == 0.0 && WGD->nk[WGD->wall_indices[i]] == 0.0) {
      int k = WGD->wall_indices[i] / ((WGD->nx - 1) * (WGD->ny - 1));
      s_behind = WGD->f[WGD->wall_indices[i]] * (WGD->dy * WGD->dz_array[k]) * (WGD->dx * WGD->dx);
      s_front = WGD->e[WGD->wall_indices[i]] * (WGD->dy * WGD->dz_array[k]) * (WGD->dx * WGD->dx);
      s_right = WGD->h[WGD->wall_indices[i]] * (WGD->dx * WGD->dz_array[k]) * (WGD->dy * WGD->dy);
      s_left = WGD->g[WGD->wall_indices[i]] * (WGD->dx * WGD->dz_array[k]) * (WGD->dy * WGD->dy);
      s_below = WGD->n[WGD->wall_indices[i]] * (WGD->dx * WGD->dy) * (WGD->dz_array[k] * 0.5 * (WGD->dz_array[k] + WGD->dz_array[k + 1]));
      s_above = WGD->m[WGD->wall_indices[i]] * (WGD->dx * WGD->dy) * (WGD->dz_array[k] * 0.5 * (WGD->dz_array[k] + WGD->dz_array[k - 1]));

      s_cut = sqrt(pow(s_behind - s_front, 2.0) + pow(s_right - s_left, 2.0)
                   + pow(s_below - s_above, 2.0));
      // Calculate normal to the cut surface
      if (s_cut != 0.0) {
        WGD->ni[WGD->wall_indices[i]] = (s_front - s_behind) / s_cut;
        WGD->nj[WGD->wall_indices[i]] = (s_left - s_right) / s_cut;
        WGD->nk[WGD->wall_indices[i]] = (s_above - s_below) / s_cut;
      }
    }
  }

  float z_buffer;
  float dot_product;
  float ut, vt, wt;// Velocity components in tangential direction
  float un, vn, wn;
  int first_id, second_id, cell_id;
  float vel_tan_mag;
  float coeff;
  int count;
  float max_dist;
  // Loop through all the cut-cells
  for (auto id = 0; id < WGD->wall_indices.size(); id++) {
    int k = WGD->wall_indices[id] / ((WGD->nx - 1) * (WGD->ny - 1));
    int j = (WGD->wall_indices[id] - k * (WGD->nx - 1) * (WGD->ny - 1)) / (WGD->nx - 1);
    int i = WGD->wall_indices[id] - k * (WGD->nx - 1) * (WGD->ny - 1) - j * (WGD->nx - 1);
    coeff = 1.0;
    count = 0;
    max_dist = sqrt(pow(WGD->dx, 2.0) + pow(WGD->dy, 2.0) + pow(WGD->dz_array[k], 2.0));

    if (abs(WGD->ni[WGD->wall_indices[id]]) < 0.05) {
      WGD->ni[WGD->wall_indices[id]] = 0;
    }
    if (abs(WGD->nj[WGD->wall_indices[id]]) < 0.05) {
      WGD->nj[WGD->wall_indices[id]] = 0;
    }
    if (abs(WGD->nk[WGD->wall_indices[id]]) < 0.05) {
      WGD->nk[WGD->wall_indices[id]] = 0;
    }

    if (WGD->center_id[WGD->wall_indices[id]] == 1) {
      if (WGD->wall_distance[WGD->wall_indices[id]] <= WGD->z0) {
        count += 1;
      } else {
        if ((log(1 + max_dist / WGD->wall_distance[WGD->wall_indices[id]]) / log(WGD->wall_distance[WGD->wall_indices[id]] / WGD->z0)) > 1.0) {
          count += 1;
        }
      }
    }

    // If the center of the cut-cell is inside air
    if (WGD->center_id[WGD->wall_indices[id]] == 1 && count == 0) {

      // Finding indices for the first node location in normal to surface direction
      first_i = i;
      first_j = j;
      first_k = k;


      if (WGD->ni[WGD->wall_indices[id]] >= 0.0) {
        // Finding indices for the second node location in normal to surface direction
        second_i = std::round((WGD->x[i] - 0.001 + WGD->ni[WGD->wall_indices[id]] * coeff * WGD->dx) / WGD->dx);
      } else {
        // Finding indices for the second node location in normal to surface direction
        second_i = std::round((WGD->x[i] - 0.001 + WGD->ni[WGD->wall_indices[id]] * coeff * WGD->dx) / WGD->dx) - 1;
      }

      if (WGD->nj[WGD->wall_indices[id]] >= 0.0) {
        // Finding indices for the second node location in normal to surface direction
        second_j = std::round((WGD->y[j] - 0.001 + WGD->nj[WGD->wall_indices[id]] * coeff * WGD->dy) / WGD->dy);
      } else {
        // Finding indices for the second node location in normal to surface direction
        second_j = std::round((WGD->y[j] - 0.001 + WGD->nj[WGD->wall_indices[id]] * coeff * WGD->dy) / WGD->dy) - 1;
      }


      if (WGD->nk[WGD->wall_indices[id]] >= 0.0) {
        z_buffer = WGD->z[k] + WGD->nk[WGD->wall_indices[id]] * coeff * WGD->dz_array[k];
        for (auto kk = 0; kk < WGD->z.size(); kk++) {
          second_k = kk + 1;
          if (z_buffer <= WGD->z_face[kk + 1]) {
            break;
          }
        }
      } else {
        z_buffer = WGD->z[k] + WGD->nk[WGD->wall_indices[id]] * coeff * WGD->dz_array[k];
        for (auto kk = 0; kk < WGD->z.size(); kk++) {
          second_k = kk + 1;
          if (z_buffer <= WGD->z_face[kk + 1]) {
            break;
          }
        }
      }

      // Distance of the first cell center from the cut surface
      dist1 = sqrt(pow((WGD->x[first_i] - WGD->x[i]), 2.0) + pow((WGD->y[first_j] - WGD->y[j]), 2.0)
                   + pow((WGD->z[first_k] - WGD->z[k]), 2.0))
              + WGD->wall_distance[WGD->wall_indices[id]];
      // Distance of the second cell center from the cut surface
      dist2 = sqrt(pow((WGD->x[second_i] - WGD->x[i]), 2.0) + pow((WGD->y[second_j] - WGD->y[j]), 2.0)
                   + pow((WGD->z[second_k] - WGD->z[k]), 2.0))
              + WGD->wall_distance[WGD->wall_indices[id]];
    } else {
      bool condition = true;
      while (condition) {
        if (WGD->center_id[WGD->wall_indices[id]] == 1) {
          coeff = count;
        } else {
          coeff = 1.0 + count;
        }

        if (WGD->ni[WGD->wall_indices[id]] >= 0.0) {
          // Finding index for the first node location in normal to surface direction
          first_i = std::round((WGD->x[i] - 0.001 + WGD->ni[WGD->wall_indices[id]] * coeff * WGD->dx) / WGD->dx);
          // Finding index for the second node location in normal to surface direction
          second_i = std::round((WGD->x[i] - 0.001 + WGD->ni[WGD->wall_indices[id]] * (coeff + 1) * WGD->dx) / WGD->dx);
        } else {
          // Finding index for the first node location in normal to surface direction
          first_i = std::round((WGD->x[i] - 0.001 + WGD->ni[WGD->wall_indices[id]] * coeff * WGD->dx) / WGD->dx) - 1;
          // Finding index for the second node location in normal to surface direction
          second_i = std::round((WGD->x[i] - 0.001 + WGD->ni[WGD->wall_indices[id]] * (coeff + 1) * WGD->dx) / WGD->dx) - 1;
        }

        if (WGD->nj[WGD->wall_indices[id]] >= 0.0) {
          // Finding index for the first node location in normal to surface direction
          first_j = std::round((WGD->y[j] - 0.001 + WGD->nj[WGD->wall_indices[id]] * coeff * WGD->dy) / WGD->dy);
          // Finding index for the second node location in normal to surface direction
          second_j = std::round((WGD->y[j] - 0.001 + WGD->nj[WGD->wall_indices[id]] * (coeff + 1) * WGD->dy) / WGD->dy);
        } else {
          // Finding index for the first node location in normal to surface direction
          first_j = std::round((WGD->y[j] - 0.001 + WGD->nj[WGD->wall_indices[id]] * coeff * WGD->dy) / WGD->dy) - 1;
          // Finding index for the second node location in normal to surface direction
          second_j = std::round((WGD->y[j] - 0.001 + WGD->nj[WGD->wall_indices[id]] * (coeff + 1) * WGD->dy) / WGD->dy) - 1;
        }

        if (WGD->nk[WGD->wall_indices[id]] >= 0.0) {
          z_buffer = WGD->z[k] + WGD->nk[WGD->wall_indices[id]] * coeff * WGD->dz_array[k];
          for (auto kk = 0; kk < WGD->z.size(); kk++) {
            first_k = kk + 1;
            if (z_buffer <= WGD->z[kk + 1]) {
              break;
            }
          }
          z_buffer = WGD->z[k] + WGD->nk[WGD->wall_indices[id]] * (coeff + 1) * WGD->dz_array[k];
          for (auto kk = 0; kk < WGD->z.size(); kk++) {
            second_k = kk + 1;
            if (z_buffer <= WGD->z_face[kk + 1]) {
              break;
            }
          }
        } else {
          z_buffer = WGD->z[k] + WGD->nk[WGD->wall_indices[id]] * coeff * WGD->dz_array[k];
          for (auto kk = 0; kk < WGD->z.size(); kk++) {
            first_k = kk + 1;
            if (z_buffer <= WGD->z_face[kk + 1]) {
              break;
            }
          }
          z_buffer = WGD->z[k] + WGD->nk[WGD->wall_indices[id]] * (coeff + 1) * WGD->dz_array[k];
          for (auto kk = 0; kk < WGD->z.size(); kk++) {
            second_k = kk - 1;
            if (z_buffer <= WGD->z[kk]) {
              break;
            }
          }
        }
        if (first_i > WGD->nx - 2 || first_i < 0 || first_j > WGD->ny - 2 || first_j < 0 || first_k > WGD->nz - 3 || first_k < 1 || second_i > WGD->nx - 2 || second_i < 0 || second_j > WGD->ny - 2 || second_j < 0 || second_k > WGD->nz - 3 || second_k < 1) {
          break;
        }

        // Distance of the first cell center from the cut surface
        dist1 = sqrt(pow((WGD->x[first_i] - WGD->x[i]), 2.0) + pow((WGD->y[first_j] - WGD->y[j]), 2.0)
                     + pow((WGD->z[first_k] - WGD->z[k]), 2.0))
                + WGD->wall_distance[WGD->wall_indices[id]];
        // Distance of the second cell center from the cut surface
        dist2 = sqrt(pow((WGD->x[second_i] - WGD->x[i]), 2.0) + pow((WGD->y[second_j] - WGD->y[j]), 2.0)
                     + pow((WGD->z[second_k] - WGD->z[k]), 2.0))
                + WGD->wall_distance[WGD->wall_indices[id]];
        // If the cell distance from surface is in a way that creates negative velocity
        /*if (i != first_i || j != first_j || k != first_k) {
          int icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
          WGD->u0[icell_face] = 0.0;
          WGD->v0[icell_face] = 0.0;
          WGD->w0[icell_face] = 0.0;
        }*/

        if (((log(dist2 / dist1) / log(dist1 / WGD->z0)) <= 1.0) && dist1 > WGD->z0) {
          condition = false;
        } else {
          /*int icell_face = first_i + first_j * WGD->nx + first_k * WGD->nx * WGD->ny;
          WGD->u0[icell_face] = 0.0;
          WGD->v0[icell_face] = 0.0;
          WGD->w0[icell_face] = 0.0;*/
          count += 1;
        }
      }
    }


    if (first_i > WGD->nx - 2 || first_i < 0 || first_j > WGD->ny - 2 || first_j < 0 || first_k > WGD->nz - 3 || first_k < 1 || second_i > WGD->nx - 2 || second_i < 0 || second_j > WGD->ny - 2 || second_j < 0 || second_k > WGD->nz - 3 || second_k < 1) {
      continue;
    }

    // Id of the first cell for velocity components
    first_id = first_i + first_j * WGD->nx + first_k * WGD->nx * WGD->ny;

    if (isInitial == true) {
      // Velocity magnitude in normal direction (U.N = u0*ni+v0*nj+w0*nk)
      dot_product = WGD->u0[first_id] * WGD->ni[WGD->wall_indices[id]] + WGD->v0[first_id] * WGD->nj[WGD->wall_indices[id]]
                    + WGD->w0[first_id] * WGD->nk[WGD->wall_indices[id]];
    } else {
      // Velocity magnitude in normal direction (U.N = u0*ni+v0*nj+w0*nk)
      dot_product = WGD->u[first_id] * WGD->ni[WGD->wall_indices[id]] + WGD->v[first_id] * WGD->nj[WGD->wall_indices[id]]
                    + WGD->w[first_id] * WGD->nk[WGD->wall_indices[id]];
    }

    // Velocity components in normal direction
    un = dot_product * WGD->ni[WGD->wall_indices[id]];
    vn = dot_product * WGD->nj[WGD->wall_indices[id]];
    wn = dot_product * WGD->nk[WGD->wall_indices[id]];

    if (isInitial == true) {
      // Velocity components in tangential direction (Ut = U-Un)
      ut = WGD->u0[first_id] - un;
      vt = WGD->v0[first_id] - vn;
      wt = WGD->w0[first_id] - wn;
    } else {
      // Velocity components in tangential direction (Ut = U-Un)
      ut = WGD->u[first_id] - un;
      vt = WGD->v[first_id] - vn;
      wt = WGD->w[first_id] - wn;
    }

    // Velocity magnitude in tangential direction
    vel_tan_mag = sqrt(pow(ut, 2.0) + pow(vt, 2.0) + pow(wt, 2.0));
    // Calculating tangential unit vectors
    if (vel_tan_mag != 0.0) {
      WGD->ti[WGD->wall_indices[id]] = ut / vel_tan_mag;
      WGD->tj[WGD->wall_indices[id]] = vt / vel_tan_mag;
      WGD->tk[WGD->wall_indices[id]] = wt / vel_tan_mag;
    }


    // Id of the second cell for velocity components
    second_id = second_i + second_j * WGD->nx + second_k * WGD->nx * WGD->ny;
    if (isInitial == true) {
      // Velocity magnitude in tangential direction (U.T = u0*ti+v0*tj+w0*tk)
      vel_mag2 = abs(WGD->u0[second_id] * WGD->ti[WGD->wall_indices[id]] + WGD->v0[second_id] * WGD->tj[WGD->wall_indices[id]]
                     + WGD->w0[second_id] * WGD->tk[WGD->wall_indices[id]]);
    } else {
      // Velocity magnitude in tangential direction (U.T = u0*ti+v0*tj+w0*tk)
      vel_mag2 = abs(WGD->u[second_id] * WGD->ti[WGD->wall_indices[id]] + WGD->v[second_id] * WGD->tj[WGD->wall_indices[id]]
                     + WGD->w[second_id] * WGD->tk[WGD->wall_indices[id]]);
    }

    if (vel_mag2 == 0.0) {
      continue;
    }

    ustar_wall = 0.1;// reset default value for velocity gradient
    for (auto iter = 0; iter < 20; iter++) {
      vel_mag1 = vel_mag2 - (ustar_wall / WGD->vk) * log(dist2 / dist1);
      new_ustar = WGD->vk * vel_mag1 / log(dist1 / WGD->z0);
      ustar_wall = new_ustar;
    }

    /*if (first_i == 169 && first_j == 111 && first_k == 13) {
      std::cout << "dist1:  " << dist1 << std::endl;
      std::cout << "dist2:  " << dist2 << std::endl;
      std::cout << "vel_mag1:  " << vel_mag1 << std::endl;
      std::cout << "vel_mag2:  " << vel_mag2 << std::endl;
      std::cout << "i:  " << i << std::endl;
      std::cout << "j:  " << j << std::endl;
      std::cout << "k:  " << k << std::endl;
      std::cout << "second_i:  " << second_i << std::endl;
      std::cout << "second_j:  " << second_j << std::endl;
      std::cout << "second_k:  " << second_k << std::endl;
      std::cout << "WGD->ti[WGD->wall_indices[id]]:  " << WGD->ti[WGD->wall_indices[id]] << std::endl;
      std::cout << "WGD->tj[WGD->wall_indices[id]]:  " << WGD->tj[WGD->wall_indices[id]] << std::endl;
      std::cout << "WGD->tk[WGD->wall_indices[id]]:  " << WGD->tk[WGD->wall_indices[id]] << std::endl;
    }*/


    // Turn the velocity magnitude in the tangential direction to Cartesian grid (U0 = ut + un (un = 0))
    WGD->u0[first_id] = vel_mag1 * WGD->ti[WGD->wall_indices[id]];
    WGD->v0[first_id] = vel_mag1 * WGD->tj[WGD->wall_indices[id]];
    WGD->w0[first_id] = vel_mag1 * WGD->tk[WGD->wall_indices[id]];
  }


  // Total size of wall indices
  int wall_size = WGD->wall_right_indices.size() + WGD->wall_left_indices.size()
                  + WGD->wall_above_indices.size() + WGD->wall_below_indices.size()
                  + WGD->wall_front_indices.size() + WGD->wall_back_indices.size();

  std::vector<float> ustar;
  ustar.resize(wall_size, 0.0);
  std::vector<int> index;
  index.resize(wall_size, 0.0);
  int j;

  ustar_wall = 0.1;
  wind_dir = 0.0;
  vel_mag1 = 0.0;
  vel_mag2 = 0.0;

  dist1 = 0.5 * WGD->dz;
  dist2 = 1.5 * WGD->dz;

  // apply log law fix to the cells with wall below
  for (size_t i = 0; i < WGD->wall_below_indices.size(); i++) {
    ustar_wall = 0.1;// reset default value for velocity gradient
    for (auto iter = 0; iter < 20; iter++) {
      wind_dir = atan2(WGD->v0[WGD->wall_below_indices[i] + WGD->nx * WGD->ny], WGD->u0[WGD->wall_below_indices[i] + WGD->nx * WGD->ny]);
      vel_mag2 = sqrt(pow(WGD->u0[WGD->wall_below_indices[i] + WGD->nx * WGD->ny], 2.0) + pow(WGD->v0[WGD->wall_below_indices[i] + WGD->nx * WGD->ny], 2.0));
      vel_mag1 = vel_mag2 - (ustar_wall / WGD->vk) * log(dist2 / dist1);
      WGD->w0[WGD->wall_below_indices[i]] = 0;// normal component of velocity set to zero
      // parallel components of velocity to wall
      WGD->u0[WGD->wall_below_indices[i]] = vel_mag1 * cos(wind_dir);
      WGD->v0[WGD->wall_below_indices[i]] = vel_mag1 * sin(wind_dir);
      new_ustar = WGD->vk * vel_mag1 / log(dist1 / WGD->z0);
      ustar_wall = new_ustar;
    }
    index[i] = WGD->wall_below_indices[i];
    ustar[i] = ustar_wall;
  }

  // apply log law fix to the cells with wall above
  for (size_t i = 0; i < WGD->wall_above_indices.size(); i++) {
    ustar_wall = 0.1;// reset default value for velocity gradient
    for (auto iter = 0; iter < 20; iter++) {
      wind_dir = atan2(WGD->v0[WGD->wall_above_indices[i] - WGD->nx * WGD->ny], WGD->u0[WGD->wall_above_indices[i] - WGD->nx * WGD->ny]);
      vel_mag2 = sqrt(pow(WGD->u0[WGD->wall_above_indices[i] - WGD->nx * WGD->ny], 2.0) + pow(WGD->v0[WGD->wall_above_indices[i] - WGD->nx * WGD->ny], 2.0));
      vel_mag1 = vel_mag2 - (ustar_wall / WGD->vk) * log(dist2 / dist1);
      WGD->w0[WGD->wall_above_indices[i]] = 0;// normal component of velocity set to zero
      // parallel components of velocity to wall
      WGD->u0[WGD->wall_above_indices[i]] = vel_mag1 * cos(wind_dir);
      WGD->v0[WGD->wall_above_indices[i]] = vel_mag1 * sin(wind_dir);
      new_ustar = WGD->vk * vel_mag1 / log(dist1 / WGD->z0);
      ustar_wall = new_ustar;
    }
    j = i + WGD->wall_below_indices.size();
    index[j] = WGD->wall_above_indices[i];
    ustar[j] = ustar_wall;
  }

  dist1 = 0.5 * WGD->dx;
  dist2 = 1.5 * WGD->dx;

  // apply log law fix to the cells with wall in back
  for (size_t i = 0; i < WGD->wall_back_indices.size(); i++) {
    ustar_wall = 0.1;
    for (auto iter = 0; iter < 20; iter++) {
      wind_dir = atan2(WGD->w0[WGD->wall_back_indices[i] + 1], WGD->v0[WGD->wall_back_indices[i] + 1]);
      vel_mag2 = sqrt(pow(WGD->v0[WGD->wall_back_indices[i] + 1], 2.0) + pow(WGD->w0[WGD->wall_back_indices[i] + 1], 2.0));
      vel_mag1 = vel_mag2 - (ustar_wall / WGD->vk) * log(dist2 / dist1);
      WGD->u0[WGD->wall_back_indices[i]] = 0;// normal component of velocity set to zero
      // parallel components of velocity to wall
      WGD->v0[WGD->wall_back_indices[i]] = vel_mag1 * cos(wind_dir);
      WGD->w0[WGD->wall_back_indices[i]] = vel_mag1 * sin(wind_dir);
      new_ustar = WGD->vk * vel_mag1 / log(dist1 / WGD->z0);
      ustar_wall = new_ustar;
    }
    j = i + WGD->wall_below_indices.size() + WGD->wall_above_indices.size();
    index[j] = WGD->wall_back_indices[i];
    ustar[j] = ustar_wall;
  }


  // apply log law fix to the cells with wall in front
  for (size_t i = 0; i < WGD->wall_front_indices.size(); i++) {
    ustar_wall = 0.1;// reset default value for velocity gradient
    for (auto iter = 0; iter < 20; iter++) {
      wind_dir = atan2(WGD->w0[WGD->wall_front_indices[i] - 1], WGD->v0[WGD->wall_front_indices[i] - 1]);
      vel_mag2 = sqrt(pow(WGD->v0[WGD->wall_front_indices[i] - 1], 2.0) + pow(WGD->w0[WGD->wall_front_indices[i] - 1], 2.0));
      vel_mag1 = vel_mag2 - (ustar_wall / WGD->vk) * log(dist2 / dist1);
      WGD->u0[WGD->wall_front_indices[i]] = 0;// normal component of velocity set to zero
      // parallel components of velocity to wall
      WGD->v0[WGD->wall_front_indices[i]] = vel_mag1 * cos(wind_dir);
      WGD->w0[WGD->wall_front_indices[i]] = vel_mag1 * sin(wind_dir);
      new_ustar = WGD->vk * vel_mag1 / log(dist1 / WGD->z0);
      ustar_wall = new_ustar;
    }
    j = i + WGD->wall_below_indices.size() + WGD->wall_above_indices.size() + WGD->wall_back_indices.size();
    index[j] = WGD->wall_front_indices[i];
    ustar[j] = ustar_wall;
  }


  dist1 = 0.5 * WGD->dy;
  dist2 = 1.5 * WGD->dy;

  // apply log law fix to the cells with wall to right
  for (size_t i = 0; i < WGD->wall_right_indices.size(); i++) {
    ustar_wall = 0.1;// reset default value for velocity gradient
    for (auto iter = 0; iter < 20; iter++) {
      wind_dir = atan2(WGD->w0[WGD->wall_right_indices[i] + WGD->nx], WGD->u0[WGD->wall_right_indices[i] + WGD->nx]);
      vel_mag2 = sqrt(pow(WGD->u0[WGD->wall_right_indices[i] + WGD->nx], 2.0) + pow(WGD->w0[WGD->wall_right_indices[i] + WGD->nx], 2.0));
      vel_mag1 = vel_mag2 - (ustar_wall / WGD->vk) * log(dist2 / dist1);
      WGD->v0[WGD->wall_right_indices[i]] = 0;// normal component of velocity set to zero
      // parallel components of velocity to wall
      WGD->u0[WGD->wall_right_indices[i]] = vel_mag1 * cos(wind_dir);
      WGD->w0[WGD->wall_right_indices[i]] = vel_mag1 * sin(wind_dir);
      new_ustar = WGD->vk * vel_mag1 / log(dist1 / WGD->z0);
      ustar_wall = new_ustar;
    }
    j = i + WGD->wall_below_indices.size() + WGD->wall_above_indices.size() + WGD->wall_back_indices.size() + WGD->wall_front_indices.size();
    index[j] = WGD->wall_right_indices[i];
    ustar[j] = ustar_wall;
  }

  // apply log law fix to the cells with wall to left
  for (size_t i = 0; i < WGD->wall_left_indices.size(); i++) {
    ustar_wall = 0.1;// reset default value for velocity gradient
    for (auto iter = 0; iter < 20; iter++) {
      wind_dir = atan2(WGD->w0[WGD->wall_left_indices[i] - WGD->nx], WGD->u0[WGD->wall_left_indices[i] - WGD->nx]);
      vel_mag2 = sqrt(pow(WGD->u0[WGD->wall_left_indices[i] - WGD->nx], 2.0) + pow(WGD->w0[WGD->wall_left_indices[i] - WGD->nx], 2.0));
      vel_mag1 = vel_mag2 - (ustar_wall / WGD->vk) * log(dist2 / dist1);
      WGD->v0[WGD->wall_left_indices[i]] = 0;// normal component of velocity set to zero
      // parallel components of velocity to wall
      WGD->u0[WGD->wall_left_indices[i]] = vel_mag1 * cos(wind_dir);
      WGD->w0[WGD->wall_left_indices[i]] = vel_mag1 * sin(wind_dir);
      new_ustar = WGD->vk * vel_mag1 / log(dist1 / WGD->z0);
      ustar_wall = new_ustar;
    }
    j = i + WGD->wall_below_indices.size() + WGD->wall_above_indices.size() + WGD->wall_back_indices.size() + WGD->wall_front_indices.size() + WGD->wall_right_indices.size();
    index[j] = WGD->wall_left_indices[i];
    ustar[j] = ustar_wall;
  }

  std::cout << "[done]" << std::endl;

  auto wallLog_finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  std::chrono::duration<float> elapsed_wallLog = wallLog_finish - wallLog_start;
  std::cout << "Elapsed time for applying log law: " << elapsed_wallLog.count() << " s\n";
}

void Wall::setVelocityZero(WINDSGeneralData *WGD)
{
  for (auto k = 0; k < WGD->nz - 1; k++) {
    for (auto j = 1; j < WGD->ny - 1; j++) {
      for (auto i = 1; i < WGD->nx - 1; i++) {
        int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        int icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2) {
          WGD->u0[icell_face] = 0.0;// Set velocity inside the building to zero
          WGD->u0[icell_face + 1] = 0.0;
          WGD->v0[icell_face] = 0.0;// Set velocity inside the building to zero
          WGD->v0[icell_face + WGD->nx] = 0.0;
          WGD->w0[icell_face] = 0.0;// Set velocity inside the building to zero
          WGD->w0[icell_face + WGD->nx * WGD->ny] = 0.0;
        }
      }
    }
  }
}


void Wall::solverCoefficients(WINDSGeneralData *WGD)
{
  // New boundary condition implementation
  // This needs to be done only once
  for (auto k = 1; k < WGD->nz - 2; k++) {
    for (auto j = 0; j < WGD->ny - 1; j++) {
      for (auto i = 0; i < WGD->nx - 1; i++) {
        int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        WGD->e[icell_cent] = WGD->e[icell_cent] / (WGD->dx * WGD->dx);
        WGD->f[icell_cent] = WGD->f[icell_cent] / (WGD->dx * WGD->dx);
        WGD->g[icell_cent] = WGD->g[icell_cent] / (WGD->dy * WGD->dy);
        WGD->h[icell_cent] = WGD->h[icell_cent] / (WGD->dy * WGD->dy);
        WGD->m[icell_cent] = WGD->m[icell_cent] / (WGD->dz_array[k] * 0.5 * (WGD->dz_array[k] + WGD->dz_array[k + 1]));
        WGD->n[icell_cent] = WGD->n[icell_cent] / (WGD->dz_array[k] * 0.5 * (WGD->dz_array[k] + WGD->dz_array[k - 1]));
      }
    }
  }
}
