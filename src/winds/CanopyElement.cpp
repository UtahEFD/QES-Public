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

#include "CanopyElement.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

void CanopyElement::setPolyBuilding(WINDSGeneralData *WGD)
{

  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  // Calculate the centroid coordinates of the building (average of all nodes coordinates)
  building_cent_x = 0.0;// x-coordinate of the centroid of the building
  building_cent_y = 0.0;// y-coordinate of the centroid of the building

  for (size_t i = 0; i < polygonVertices.size() - 1; i++) {
    building_cent_x += polygonVertices[i].x_poly;
    building_cent_y += polygonVertices[i].y_poly;
  }
  building_cent_x /= polygonVertices.size() - 1;
  building_cent_y /= polygonVertices.size() - 1;

  i_building_cent = std::round(building_cent_x / dx) - 1;// Index of building centroid in x-direction
  j_building_cent = std::round(building_cent_y / dy) - 1;// Index of building centroid in y-direction

  // Loop to calculate maximum and minimum of x and y values of the building
  x_min = x_max = polygonVertices[0].x_poly;
  y_min = y_max = polygonVertices[0].y_poly;
  for (size_t id = 1; id < polygonVertices.size(); id++) {
    if (polygonVertices[id].x_poly > x_max) {
      x_max = polygonVertices[id].x_poly;
    }
    if (polygonVertices[id].x_poly < x_min) {
      x_min = polygonVertices[id].x_poly;
    }
    if (polygonVertices[id].y_poly > y_max) {
      y_max = polygonVertices[id].y_poly;
    }
    if (polygonVertices[id].y_poly < y_min) {
      y_min = polygonVertices[id].y_poly;
    }
  }

  // i_start and i_end are faces and not cells
  i_start = x_min / dx;// Index of canopy start location in x-direction
  i_end = x_max / dx + 1;// Index of canopy end location in x-direction
  // j_start and j_end are faces and not cells
  j_start = y_min / dy;// Index of canopy end location in y-direction
  j_end = y_max / dy + 1;// Index of canopy start location in y-direction

  if (i_start < 0 || i_end >= nx - 1 || j_start < 0 || j_end >= ny - 1) {
    std::cerr << "==============================================================" << std::endl;
    std::cerr << "[ERROR] canopy element out of bound" << std::endl;
    std::cout << "x value (min,man) (indeces) (" << x_min << "," << x_max << ") (" << i_start << "," << i_end << ")" << std::endl;
    std::cout << "y value (min,man) (indeces) (" << y_min << "," << y_max << ") (" << j_start << "," << j_end << ")" << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}

void CanopyElement::setCanopyGrid(WINDSGeneralData *WGD, int building_number)
{
  float ray_intersect;
  unsigned int num_crossing, vert_id, start_poly;

#if 1
  std::cerr << "THIS FUNCTION IS OBSOLETE" << std::endl;
  exit(EXIT_FAILURE);
#else

  // Loop to calculate maximum and minimum of x and y values of the building
  x_min = x_max = polygonVertices[0].x_poly;
  y_min = y_max = polygonVertices[0].y_poly;
  for (size_t id = 1; id < polygonVertices.size(); id++) {
    if (polygonVertices[id].x_poly > x_max) {
      x_max = polygonVertices[id].x_poly;
    }
    if (polygonVertices[id].x_poly < x_min) {
      x_min = polygonVertices[id].x_poly;
    }
    if (polygonVertices[id].y_poly > y_max) {
      y_max = polygonVertices[id].y_poly;
    }
    if (polygonVertices[id].y_poly < y_min) {
      y_min = polygonVertices[id].y_poly;
    }
  }

  // i_start and i_end are faces and not cells
  i_start = x_min / WGD->dx;// Index of canopy start location in x-direction
  i_end = x_max / WGD->dx + 1;// Index of canopy end location in x-direction
  // j_start and j_end are faces and not cells
  j_start = y_min / WGD->dy;// Index of canopy end location in y-direction
  j_end = y_max / WGD->dy + 1;// Index of canopy start location in y-direction

  // size of the canopy array -> with ghost cell before and after (hence +2)
  nx_canopy = (i_end - i_start - 1) + 2;
  ny_canopy = (j_end - j_start - 1) + 2;


  // number of cell cell-center elements (2D)
  numcell_cent_2d = nx_canopy * ny_canopy;

  // Resize the canopy-related vectors
  canopy_bot.resize(numcell_cent_2d, 0.0);
  canopy_top.resize(numcell_cent_2d, 0.0);

  canopy_bot_index.resize(numcell_cent_2d, 0);
  canopy_top_index.resize(numcell_cent_2d, -1);

  canopy_base.resize(numcell_cent_2d, 0.0);
  canopy_height.resize(numcell_cent_2d, 0.0);

  canopy_z0.resize(numcell_cent_2d, 0.0);
  canopy_ustar.resize(numcell_cent_2d, 0.0);
  canopy_d.resize(numcell_cent_2d, 0.0);

  canopy_cellMap3D.clear();
  canopy_cellMap2D.clear();

  // Find out which cells are going to be inside the polygone
  // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
  // Check the center of each cell, if it's inside, set that cell to building
  for (auto j = j_start; j < j_end; j++) {
    // Center of cell y coordinate
    float y_cent = (j + 0.5) * WGD->dy;
    for (auto i = i_start; i < i_end; i++) {
      float x_cent = (i + 0.5) * WGD->dx;
      // Node index
      vert_id = 0;
      start_poly = vert_id;
      num_crossing = 0;
      while (vert_id < polygonVertices.size() - 1) {
        if ((polygonVertices[vert_id].y_poly <= y_cent && polygonVertices[vert_id + 1].y_poly > y_cent)
            || (polygonVertices[vert_id].y_poly > y_cent && polygonVertices[vert_id + 1].y_poly <= y_cent)) {
          ray_intersect = (y_cent - polygonVertices[vert_id].y_poly) / (polygonVertices[vert_id + 1].y_poly - polygonVertices[vert_id].y_poly);
          if (x_cent < (polygonVertices[vert_id].x_poly + ray_intersect * (polygonVertices[vert_id + 1].x_poly - polygonVertices[vert_id].x_poly))) {
            num_crossing += 1;
          }
        }
        vert_id += 1;
        if (polygonVertices[vert_id].x_poly == polygonVertices[start_poly].x_poly
            && polygonVertices[vert_id].y_poly == polygonVertices[start_poly].y_poly) {
          vert_id += 1;
          start_poly = vert_id;
        }
      }

      // if num_crossing is odd = cell is oustside of the polygon
      // if num_crossing is even = cell is inside of the polygon
      if ((num_crossing % 2) != 0) {
        int icell_cent_2d = i + j * (WGD->nx - 1);
        int icell_canopy_2d = (i + 1 - i_start) + (j + 1 - j_start) * nx_canopy;

        for (size_t k = 0; k < WGD->z.size() - 1; k++) {
          int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
          int icell_canopy_3d = i + j * nx_canopy + k * nx_canopy * ny_canopy;

          if (WGD->icellflag[icell_cent] == 2) {
            continue;
          } else if (WGD->icellflag[icell_cent] == 0) {
            break;
          }

          // Define start index of the canopy in z-direction
          if (WGD->terrain[icell_cent_2d] + base_height > WGD->z_face[k]
              && WGD->terrain[icell_cent_2d] + base_height <= WGD->z_face[k + 1]) {
            canopy_bot_index[icell_canopy_2d] = k;
            canopy_bot[icell_canopy_2d] = WGD->terrain[icell_cent_2d] + base_height;
            canopy_base[icell_canopy_2d] = WGD->z_face[k - 1];
            // break;
          }

          // Define end index of the canopy in z-direction
          if (WGD->terrain[icell_cent_2d] + H >= WGD->z_face[k]
              && WGD->terrain[icell_cent_2d] + H < WGD->z_face[k + 1]) {
            canopy_top_index[icell_canopy_2d] = k + 1;
            canopy_top[icell_canopy_2d] = WGD->terrain[icell_cent_2d] + H;
            // break;
          }


          // Canopy cell
          WGD->icellflag[icell_cent] = getCellFlagCanopy();
          WGD->ibuilding_flag[icell_cent] = building_number;

          canopy_cellMap2D[icell_cent_2d] = icell_canopy_2d;
          canopy_cellMap3D[icell_cent] = icell_canopy_3d;
        }


        // Define hieght of the canopy base in z-direction
        canopy_height[icell_canopy_2d] = canopy_top[icell_canopy_2d] - canopy_bot[icell_canopy_2d];

        // define icellflag @ (x,y) for all z(k) in [k_start...k_end]
        /*for (auto k=canopy_bot_index[icell_canopy_2d]; k<canopy_top_index[icell_canopy_2d]; k++) {
            int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
            int icell_canopy_3d = i + j*nx_canopy + k*nx_canopy*ny_canopy;
            if( WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                // Canopy cell
                WGD->icellflag[icell_cent] = getCellFlagCanopy();
                WGD->ibuilding_flag[icell_cent] = building_number;

                canopy_cellMap2D[icell_cent_2d] = icell_canopy_2d;
                canopy_cellMap3D[icell_cent] = icell_canopy_3d;
            }
        }
        */
      }// end define icellflag!
    }
  }


  float linH_tot = 0;
  for (auto k = 0u; k < canopy_height.size(); k++) {
    linH_tot += canopy_height[k];
  }

  if (linH_tot > 0) {
    // Define start/end index of the canopy in z-direction
    k_start = WGD->nz;
    k_end = 0;
    for (auto j = 0; j < ny_canopy; j++) {
      for (auto i = 0; i < nx_canopy; i++) {
        int icell_canopy_2d = i + j * nx_canopy;
        if (canopy_bot_index[icell_canopy_2d] < k_start)
          k_start = canopy_bot_index[icell_canopy_2d];
        if (canopy_top_index[icell_canopy_2d] > k_end)
          k_end = canopy_top_index[icell_canopy_2d];
      }
    }

    // define base heigh and effective height
    base_height = 100000;
    height_eff = 0;
    for (auto j = 0; j < ny_canopy; j++) {
      for (auto i = 0; i < nx_canopy; i++) {
        int icell_canopy_2d = i + j * nx_canopy;
        if (canopy_top_index[icell_canopy_2d] >= 0) {
          if (canopy_base[icell_canopy_2d] < base_height)
            base_height = canopy_base[icell_canopy_2d];
          if (canopy_top[icell_canopy_2d] > height_eff)
            height_eff = canopy_top[icell_canopy_2d];
        }
      }
    }

    // number of point in z -> + 2 (1 ghost below, 1 ghost above)
    // the canopy subarray start at the ground (with bottom ghost cell below the ground)
    nz_canopy = k_end + 2;
    // k_end to match definition used by all building (top face and not top cell)
    k_end++;

  } else {
    k_start = 0;
    k_end = 0;
    base_height = 0;
    height_eff = 0;

    nx_canopy = 0;
    ny_canopy = 0;
    nz_canopy = 0;
  }

  // number of cell-center elements (3D)
  numcell_cent_3d = nx_canopy * ny_canopy * nz_canopy;
#endif

  return;
}


// Function to apply the urban canopy parameterization
// Based on the version contain Lucas Ulmerlmer's modifications
void CanopyElement::canopyCioncoParam(WINDSGeneralData *WGD)
{

#if 1
  std::cerr << "THIS FUNCTION IS OBSOLETE" << std::endl;
  exit(EXIT_FAILURE);
#else

  float avg_atten; /**< average attenuation of the canopy */
  float veg_vel_frac; /**< vegetation velocity fraction */
  int num_atten;

  // Call regression to define ustar and surface roughness of the canopy
  // canopyRegression(WGD);

  for (auto j = 0; j < ny_canopy; j++) {
    for (auto i = 0; i < nx_canopy; i++) {
      int icell_2d = i + j * nx_canopy;

      if (canopy_top[icell_2d] > 0) {
        int icell_3d = i + j * nx_canopy + (canopy_top_index[icell_2d] - 1) * nx_canopy * ny_canopy;

        // Call the bisection method to find the root
        canopy_d[icell_2d] = canopyBisection(canopy_ustar[icell_2d],
                                             canopy_z0[icell_2d],
                                             canopy_height[icell_2d],
                                             canopy_atten[icell_3d],
                                             WGD->vk,
                                             0.0);
        // std::cout << "WGD->vk:" << WGD->vk << "\n";
        // std::cout << "WGD->canopy_atten[icell_cent]:" << WGD->canopy_atten[icell_cent] << "\n";
        if (canopy_d[icell_2d] == 10000) {
          std::cout << "bisection failed to converge"
                    << "\n";
          canopy_d[icell_2d] = canopySlopeMatch(canopy_z0[icell_2d],
                                                canopy_height[icell_2d],
                                                canopy_atten[icell_3d]);
        }

        /**< velocity at the height of the canopy */
        // Local variable - not being used by anything... so
        // commented out for now.
        //
        // float u_H = (WGD->canopy_ustar[id]/WGD->vk)*
        //  log((WGD->canopy_top[id]-WGD->canopy_d[id])/WGD->canopy_z0[id]);

        for (auto k = 1; k < WGD->nz - 1; k++) {
          int icell_face = (i - 1 + i_start) + (j - 1 + j_start) * WGD->nx + k * WGD->nx * WGD->ny;
          float z_rel = WGD->z[k] - canopy_base[icell_2d];

          if (WGD->z[k] < canopy_base[icell_2d]) {
            // below the terrain or building
          } else if (WGD->z[k] < canopy_top[icell_2d]) {
            if (canopy_atten[icell_3d] > 0) {
              icell_3d = i + j * nx_canopy + k * nx_canopy * ny_canopy;
              avg_atten = canopy_atten[icell_3d];


              if (canopy_atten[icell_3d + nx_canopy * ny_canopy] != canopy_atten[icell_3d]
                  || canopy_atten[icell_3d - nx_canopy * ny_canopy] != canopy_atten[icell_3d]) {
                num_atten = 1;
                if (canopy_atten[icell_3d + nx_canopy * ny_canopy] > 0) {
                  avg_atten += canopy_atten[icell_3d + nx_canopy * ny_canopy];
                  num_atten += 1;
                }
                if (canopy_atten[icell_3d - nx_canopy * ny_canopy] > 0) {
                  avg_atten += canopy_atten[icell_3d - nx_canopy * ny_canopy];
                  num_atten += 1;
                }
                avg_atten /= num_atten;
              }

              /*
              veg_vel_frac = log((canopy_top[icell_2d] - canopy_d[icell_2d])/
                                 canopy_z0[icell_2d])*exp(avg_atten*((WGD->z[k]/canopy_top[icell_2d])-1))/
                  log(WGD->z[k]/canopy_z0[icell_2d]);
              */

              // correction on the velocity within the canopy
              veg_vel_frac = log((canopy_height[icell_2d] - canopy_d[icell_2d]) / canopy_z0[icell_2d])
                             * exp(avg_atten * ((z_rel / canopy_height[icell_2d]) - 1)) / log(z_rel / canopy_z0[icell_2d]);
              // check if correction is bound and well defined
              if (veg_vel_frac > 1 || veg_vel_frac < 0) {
                veg_vel_frac = 1;
              }

              WGD->u0[icell_face] *= veg_vel_frac;
              WGD->v0[icell_face] *= veg_vel_frac;

              // at the edge of the canopy need to adjust velocity at the next face
              // use canopy_top to detect the edge (worke with level changes)
              if (j - 1 + j_start < WGD->ny - 2) {
                if (canopy_top[icell_2d + nx_canopy] == 0.0) {
                  WGD->v0[icell_face + WGD->nx] *= veg_vel_frac;
                }
              }
              if (i - 1 + i_start < WGD->nx - 2) {
                if (canopy_top[icell_2d + 1] == 0.0) {
                  WGD->u0[icell_face + 1] *= veg_vel_frac;
                }
              }
            }
          } else {
            // correction on the velocity above the canopy
            veg_vel_frac = log((z_rel - canopy_d[icell_2d]) / canopy_z0[icell_2d]) / log(z_rel / canopy_z0[icell_2d]);
            // check if correction is bound and well defined
            if (veg_vel_frac > 1 || veg_vel_frac < 0) {
              veg_vel_frac = 1;
            }

            WGD->u0[icell_face] *= veg_vel_frac;
            WGD->v0[icell_face] *= veg_vel_frac;

            // at the edge of the canopy need to adjust velocity at the next face
            // use canopy_top to detect the edge (worke with level changes)
            if (j - 1 + j_start < WGD->ny - 2) {
              icell_3d = i + j * nx_canopy + canopy_bot_index[icell_2d] * nx_canopy * ny_canopy;
              if (canopy_top[icell_2d + nx_canopy] == 0.0) {
                WGD->v0[icell_face + WGD->nx] *= veg_vel_frac;
              }
            }
            if (i - 1 + i_start < WGD->nx - 2) {
              icell_3d = i + j * nx_canopy + canopy_bot_index[icell_2d] * nx_canopy * ny_canopy;
              if (canopy_top[icell_2d + 1] == 0.0) {
                WGD->u0[icell_face + 1] *= veg_vel_frac;
              }
            }
          }
        }// end of for(auto k=1; k < WGD->nz-1; k++)
      }
    }
  }
#endif

  return;
}

float CanopyElement::canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m)
{
  int iter;
  float uhc, d, d1, d2;
  float tol, fnew, fi;

  tol = z0 / 100;
  fnew = tol * 10;

  d1 = z0;
  d2 = canopy_top;
  d = (d1 + d2) / 2;

  uhc = (ustar / vk) * (log((canopy_top - d1) / z0) + psi_m);
  fi = ((canopy_atten * uhc * vk) / ustar) - canopy_top / (canopy_top - d1);

  if (canopy_atten > 0) {
    iter = 0;
    while (iter < 200 && abs(fnew) > tol && d < canopy_top && d > z0) {
      iter += 1;
      d = (d1 + d2) / 2;
      uhc = (ustar / vk) * (log((canopy_top - d) / z0) + psi_m);
      fnew = ((canopy_atten * uhc * vk) / ustar) - canopy_top / (canopy_top - d);
      if (fnew * fi > 0) {
        d1 = d;
      } else if (fnew * fi < 0) {
        d2 = d;
      }
    }
    if (d > canopy_top) {
      d = 10000;
    }

  } else {
    d = 0.99 * canopy_top;
  }

  return d;
}

float CanopyElement::canopySlopeMatch(float z0, float canopy_top, float canopy_atten)
{

  int iter;
  float tol, d, d1, d2, f;

  tol = z0 / 100;
  // f is the root of the equation (to find d)
  // log[(H-d)/z0] = H/[a(H-d)]
  f = tol * 10;

  // initial bound for bisection method (d1,d2)
  // d1 min displacement possible
  // d2 max displacement possible - canopy top
  d1 = z0;
  if (z0 <= canopy_top) {
    d1 = z0;
  } else if (z0 > canopy_top) {
    d1 = 0.1;
  }
  d2 = canopy_top;
  d = (d1 + d2) / 2;

  if (canopy_atten > 0) {
    iter = 0;
    // bisection method to find the displacement height
    while (iter < 200 && abs(f) > tol && d < canopy_top && d > z0) {
      iter += 1;
      d = (d1 + d2) / 2;
      f = log((canopy_top - d) / z0) - (canopy_top / (canopy_atten * (canopy_top - d)));
      if (f > 0) {
        d1 = d;
      } else if (f < 0) {
        d2 = d;
      }
    }
    // if displacement found higher that canopy top => shifted down
    if (d > canopy_top) {
      d = 0.7 * canopy_top;
    }
  } else {
    // return this if attenuation coeff is 0.
    d = 10000;
  }

  // return displacement height
  return d;
}
