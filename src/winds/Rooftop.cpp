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

/** @file Rooftop.cpp */

#include "PolyBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

/**
 *
 * This function applies the rooftop parameterization to the qualified space on top of buildings defined as polygons.
 * This function reads in building features like nodes, building height and base height and uses
 * features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
 * cells qualified on top of buildings and applies the approperiate parameterization to them.
 * More information:
 *
 * @param WID :document this:
 * @param WGD :document this:
 */
void PolyBuilding::rooftop(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  float tol = 30 * M_PI / 180.0;// Rooftop criteria for vortex parameterization
  //unused-variable:float wing_tol = 70 * M_PI / 180.0;// Rooftop criteria for delta wing parameterization

  int rooftop_method = 0;

  int k_ref(0);
  float R_scale;
  float R_cx;
  float vd;
  float z_ref(0.0);
  int k_end_v(0);
  int u_flag, v_flag, w_flag;
  float x_u, y_u;
  float x_v, y_v;
  //unused-variable:float x_w, y_w;
  float h_x, h_y, hd_u, hd_v;
  //unused-variable:float h_xu, h_yu, h_xv, h_yv, h_xw, h_yw;
  //unused-variable:float hd_wx, hd_wy;
  float z_ref_u, z_ref_v;
  int k_shell_u, k_shell_v;//k_shell_w;
  float denom_u, denom_v;
  //unused-variable:float velocity_mag;
  float velocity_mag_u, velocity_mag_v;
  float z_roof;
  float shell_heightu_part, shell_heightv_part;
  float shell_height_u, shell_height_v;
  std::vector<int> perpendicular_flag;

  //unused-variable:float roof_angle;
  float u0_roof(0.0), v0_roof(0.0);

  // check which rooftop method to use
  if (WID->buildingsParams->rooftopFlag == 1) {
    // everything uses log-law
    rooftop_method = 1;
  } else if (WID->buildingsParams->rooftopFlag == 2) {
    // rectangulare building can use rooftop vortex
    if (rectangular_flag && rooftop_flag == 1) {
      rooftop_method = 2;
    } else {
      rooftop_method = 1;
    }
  } else {
    //
    rooftop_method = 0;
    return;
  }

  upwind_rel_dir.resize(polygonVertices.size(), 0.0);// Upwind reletive direction for each face
  perpendicular_flag.resize(polygonVertices.size(), 0);


  int index_building_face = i_building_cent + j_building_cent * WGD->nx + (k_end)*WGD->nx * WGD->ny;
  u0_h = WGD->u0[index_building_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[index_building_face];// v velocity at the height of building at the centroid

  // Wind direction of initial velocity at the height of building at the centroid
  upwind_dir = atan2(v0_h, u0_h);

  xi.resize(polygonVertices.size(), 0.0);// Difference of x values of the centroid and each node
  yi.resize(polygonVertices.size(), 0.0);// Difference of y values of the centroid and each node

  float x_front = 0.0;
  float y_front = 0.0;
  //unused-variable:int ns_flag = 0;

  // Loop to calculate x and y values of each polygon point in rotated coordinates
  for (size_t id = 0; id < polygonVertices.size(); id++) {
    xi[id] = (polygonVertices[id].x_poly - building_cent_x) * cos(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * sin(upwind_dir);
    yi[id] = -(polygonVertices[id].x_poly - building_cent_x) * sin(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * cos(upwind_dir);
    if (xi[id] < x_front) {
      x_front = xi[id];
    }
    if (yi[id] < y_front) {
      y_front = yi[id];
    }
  }

  // Finding index of 1.5 height of the building
  for (auto k = k_end; k < WGD->nz - 1; k++) {
    k_ref = k;
    if (1.5 * (H + base_height) < WGD->z_face[k]) {
      break;
    }
  }

  if (k_ref < WGD->nz - 1) {
    // Smaller of H and the effective cross-wind width (Weff)
    small_dimension = MIN_S(width_eff, H);
    // Larger of H and the effective cross-wind width (Weff)
    long_dimension = MAX_S(width_eff, H);
    R_scale = pow(small_dimension, (2.0 / 3.0)) * pow(long_dimension, (1.0 / 3.0));// Scaling length
    //R_scale = 2.0 / 3.0 * small_dimension + 1.0 / 3.0 * long_dimension;// Scaling length
    //R_cx = 0.9 * R_scale;// Normalized cavity length
    //vd = 0.5 * 0.22 * R_scale;// Cavity height
    R_cx = 0.99 * R_scale;// Normalized cavity length
    vd = 0.22 * R_scale;// Cavity height
    z_ref = (vd / sqrt(0.5 * R_cx));
    // Finding index related to height of building plus rooftop height
    for (auto k = k_end; k <= k_ref; k++) {
      k_end_v = k;
      if (H + base_height + vd < WGD->z[k]) {
        break;
      }
    }

    // Log parameterization
    if (rooftop_method == 1) {
      for (auto j = j_start; j < j_end - 1; j++) {
        for (auto i = i_start; i < i_end - 1; i++) {
          // Defining which velocity component is applicable for the parameterization
          u_flag = 0;
          v_flag = 0;
          w_flag = 0;
          icell_cent = i + j * (WGD->nx - 1) + (k_end - 1) * (WGD->nx - 1) * (WGD->ny - 1);
          if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 7)// Cell below is building
          {
            u_flag = 1;
            v_flag = 1;
            w_flag = 1;
          } else// No building in cell below
          {
            icell_cent = (i - 1) + j * (WGD->nx - 1) + (k_end - 1) * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 7)// Cell behind is building
            {
              u_flag = 1;
            }
            icell_cent = i + (j - 1) * (WGD->nx - 1) + (k_end - 1) * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 7)// Cell on right is building
            {
              v_flag = 1;
            }
          }
          icell_cent = i + j * (WGD->nx - 1) + k_end * (WGD->nx - 1) * (WGD->ny - 1);
          // If cell at building height is street canyon or wake behind building
          if (WGD->icellflag[icell_cent] == 4 || WGD->icellflag[icell_cent] == 6) {
            u_flag = 0;
            v_flag = 0;
            w_flag = 0;
          }
          // No solid cell and at least one velocity component is applicable for the parameterization
          if ((u_flag + v_flag + w_flag) > 0 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
            // x location of u component in local coordinates
            x_u = (i * WGD->dx - building_cent_x) * cos(upwind_dir) + ((j + 0.5) * WGD->dy - building_cent_y) * sin(upwind_dir);
            // y location of u component in local coordinates
            y_u = -(i * WGD->dx - building_cent_x) * sin(upwind_dir) + ((j + 0.5) * WGD->dy - building_cent_y) * cos(upwind_dir);
            // x location of v component in local coordinates
            x_v = ((i + 0.5) * WGD->dx - building_cent_x) * cos(upwind_dir) + (j * WGD->dy - building_cent_y) * sin(upwind_dir);
            // y location of v component in local coordinates
            y_v = -((i + 0.5) * WGD->dx - building_cent_x) * sin(upwind_dir) + (j * WGD->dy - building_cent_y) * cos(upwind_dir);
            // Distance from front face of the building in x direction for u component
            h_x = abs(x_u - x_front);
            // Distance from front face of the building in y direction for u component
            h_y = abs(y_u - y_front);
            // Minimum distance from front face of the building for u component
            hd_u = MIN_S(h_x, h_y);
            // Distance from front face of the building in x direction for v component
            h_x = abs(x_v - x_front);
            // Distance from front face of the building in y direction for v component
            h_y = abs(y_v - y_front);
            // Minimum distance from front face of the building for v component
            hd_v = MIN_S(h_x, h_y);
            z_ref_u = z_ref * sqrt(hd_u);// z location of the rooftop ellipse for u component
            z_ref_v = z_ref * sqrt(hd_v);// z location of the rooftop ellipse for v component
            k_shell_u = 0;
            k_shell_v = 0;
            // Looping through to find indices of the cell located at the top of the rooftop ellipse
            for (auto k = k_end - 1; k < WGD->nz - 2; k++) {
              if ((z_ref_u + H + base_height) < WGD->z[k + 1] && k_shell_u < 1) {
                k_shell_u = k;
              }
              if ((z_ref_v + H + base_height) < WGD->z[k + 1] && k_shell_v < 1) {
                k_shell_v = k;
              }
              if (k_shell_u > 0 && k_shell_v > 0) {
                break;
              }
            }
            if (k_shell_u <= k_end - 1) {
              u_flag = 0;
              denom_u = 1.0;
            } else {
              denom_u = 1.0 / log((WGD->z[k_shell_u] - (H + base_height)) / WGD->z0);
            }
            if (k_shell_v <= k_end - 1) {
              v_flag = 0;
              denom_v = 1.0;
            } else {
              denom_v = 1.0 / log((WGD->z[k_shell_v] - (H + base_height)) / WGD->z0);
            }
            icell_face = i + j * WGD->nx + k_shell_u * WGD->nx * WGD->ny;
            velocity_mag_u = WGD->u0[icell_face];
            icell_face = i + j * WGD->nx + k_shell_v * WGD->nx * WGD->ny;
            velocity_mag_v = WGD->v0[icell_face];
            for (auto k = k_end; k <= k_ref; k++) {
              icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
              if (WGD->icellflag[icell_cent] == 0) {
                break;
              } else {
                k_shell_u = 0;
                k_shell_v = 0;
                z_roof = WGD->z[k] - (H + base_height);
                if (u_flag > 0) {
                  if (z_roof <= z_ref_u) {
                    icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
                    WGD->u0[icell_face] = velocity_mag_u * log(z_roof / WGD->z0) * denom_u;
                    k_shell_u = 1;
                    if (w_flag == 1 && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                      icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                      WGD->icellflag[icell_cent] = 10;
                    }
                  }
                }
                if (v_flag > 0) {
                  if (z_roof <= z_ref_v) {
                    icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
                    WGD->v0[icell_face] = velocity_mag_v * log(z_roof / WGD->z0) * denom_v;
                    k_shell_v = 1;
                    if (w_flag == 1 && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                      icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                      WGD->icellflag[icell_cent] = 10;
                    }
                  }
                }
              }
              if ((k_shell_u + k_shell_v) < 1) {
                break;
              }
            }
          }
        }
      }
    }

    // Vortex parameterization
    if (rooftop_method == 2) {
      for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
        // Calculate upwind reletive direction for each face
        upwind_rel_dir[id] = atan2(yi[id + 1] - yi[id], xi[id + 1] - xi[id]) + 0.5 * M_PI;

        if (upwind_rel_dir[id] > M_PI) {
          upwind_rel_dir[id] -= 2 * M_PI;
        }

        if (abs(upwind_rel_dir[id]) >= M_PI - tol) {
          for (auto j = j_start; j < j_end - 1; j++) {
            for (auto i = i_start; i < i_end - 1; i++) {
              // Defining which velocity component is applicable for the parameterization
              u_flag = 0;
              v_flag = 0;
              w_flag = 0;
              icell_cent = i + j * (WGD->nx - 1) + (k_end - 1) * (WGD->nx - 1) * (WGD->ny - 1);
              if (WGD->icellflag[icell_cent] == 0)// Cell below is building
              {
                u_flag = 1;
                v_flag = 1;
                w_flag = 1;
              } else// Cell below is not building
              {
                icell_cent = (i - 1) + j * (WGD->nx - 1) + (k_end - 1) * (WGD->nx - 1) * (WGD->ny - 1);
                if (WGD->icellflag[icell_cent] == 0)// Cell behind is building
                {
                  u_flag = 1;
                }
                icell_cent = i + (j - 1) * (WGD->nx - 1) + (k_end - 1) * (WGD->nx - 1) * (WGD->ny - 1);
                if (WGD->icellflag[icell_cent] == 0)// Cell on right is building
                {
                  v_flag = 1;
                }
              }
              icell_cent = i + j * (WGD->nx - 1) + k_end * (WGD->nx - 1) * (WGD->ny - 1);
              // If cell at building height is street canyon or wake behind building
              if (WGD->icellflag[icell_cent] == 4 || WGD->icellflag[icell_cent] == 6) {
                u_flag = 0;
                v_flag = 0;
                w_flag = 0;
              }
              // No solid cell and at least one velocity component is applicable for the parameterization
              if ((u_flag + v_flag + w_flag) > 0 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                // x location of u component in local coordinates
                x_u = (i * WGD->dx - building_cent_x) * cos(upwind_dir) + ((j + 0.5) * WGD->dy - building_cent_y) * sin(upwind_dir);
                // y location of u component in local coordinates
                y_u = -(i * WGD->dx - building_cent_x) * sin(upwind_dir) + ((j + 0.5) * WGD->dy - building_cent_y) * cos(upwind_dir);
                // x location of v component in local coordinates
                x_v = ((i + 0.5) * WGD->dx - building_cent_x) * cos(upwind_dir) + (j * WGD->dy - building_cent_y) * sin(upwind_dir);
                // y location of v component in local coordinates
                y_v = -((i + 0.5) * WGD->dx - building_cent_x) * sin(upwind_dir) + (j * WGD->dy - building_cent_y) * cos(upwind_dir);
                // Distance from front face of the building in x direction for u component
                h_x = abs(x_u - x_front);
                // Distance from front face of the building in y direction for u component
                h_y = abs(y_u - y_front);
                // Minimum distance from front face of the building for u component
                hd_u = h_x;
                // Distance from front face of the building in x direction for v component
                h_x = abs(x_v - x_front);
                // Distance from front face of the building in y direction for v component
                h_y = abs(y_v - y_front);
                // Minimum distance from front face of the building for v component
                hd_v = h_x;
                z_ref_u = z_ref * sqrt(hd_u);// z location of the rooftop ellipse for u component
                z_ref_v = z_ref * sqrt(hd_v);// z location of the rooftop ellipse for v component
                k_shell_u = 0;
                k_shell_v = 0;
                // Looping through to find indices of the cell located at the top of the rooftop ellipse
                for (auto k = k_end - 1; k < WGD->nz; k++) {
                  if ((z_ref_u + H + base_height) < WGD->z[k + 1] && k_shell_u < 1) {
                    k_shell_u = k;
                  }
                  if ((z_ref_v + H + base_height) < WGD->z[k + 1] && k_shell_v < 1) {
                    k_shell_v = k;
                  }
                  if (k_shell_u > 0 && k_shell_v > 0) {
                    break;
                  }
                }
                shell_heightu_part = 1.0 - pow((0.5 * R_cx - hd_u) / (0.5 * R_cx), 2.0);
                shell_heightv_part = 1.0 - pow((0.5 * R_cx - hd_v) / (0.5 * R_cx), 2.0);
                if (shell_heightu_part > 0.0) {
                  shell_height_u = vd * sqrt(shell_heightu_part);
                } else {
                  shell_height_u = 0.0;
                }
                if (shell_heightv_part > 0.0) {
                  shell_height_v = vd * sqrt(shell_heightv_part);
                } else {
                  shell_height_v = 0.0;
                }
                if (k_shell_u <= k_end - 1) {
                  u_flag = 0;
                  denom_u = 1.0;
                } else {
                  denom_u = 1.0 / log((WGD->z[k_shell_u] - (H + base_height)) / WGD->z0);
                }
                if (k_shell_v <= k_end - 1) {
                  v_flag = 0;
                  denom_v = 1.0;
                } else {
                  denom_v = 1.0 / log((WGD->z[k_shell_v] - (H + base_height)) / WGD->z0);
                }
                icell_face = i + j * WGD->nx + k_shell_u * WGD->nx * WGD->ny;
                velocity_mag_u = WGD->u0[icell_face];
                icell_face = i + j * WGD->nx + k_shell_v * WGD->nx * WGD->ny;
                velocity_mag_v = WGD->v0[icell_face];
                for (auto k = k_end; k <= k_end_v; k++) {
                  k_shell_u = 0;
                  k_shell_v = 0;
                  z_roof = WGD->z[k] - (H + base_height);
                  icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                  if (WGD->icellflag[icell_cent] == 0) {
                    break;
                  } else {
                    if (u_flag == 1) {
                      if (z_roof <= z_ref_u) {
                        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
                        u0_roof = WGD->u0[icell_face];
                        WGD->u0[icell_face] = velocity_mag_u * log(z_roof / WGD->z0) * denom_u;
                        k_shell_u = 1;
                        if (w_flag == 1 && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                          icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                          WGD->icellflag[icell_cent] = 10;
                        }
                      }
                      if (hd_u < R_cx && z_roof <= shell_height_u) {
                        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
                        WGD->u0[icell_face] = -u0_roof * abs((shell_height_u - z_roof) / vd);
                        k_shell_u = 1;
                        if (w_flag == 1 && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                          icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                          WGD->icellflag[icell_cent] = 10;
                        }
                      }
                    }
                    if (v_flag == 1) {
                      if (z_roof <= z_ref_v) {
                        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
                        v0_roof = WGD->v0[icell_face];
                        WGD->v0[icell_face] = velocity_mag_v * log(z_roof / WGD->z0) * denom_v;
                        k_shell_v = 1;
                        if (w_flag == 1 && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                          icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                          WGD->icellflag[icell_cent] = 10;
                        }
                      }
                      if (hd_v < R_cx && z_roof <= shell_height_v) {
                        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
                        WGD->v0[icell_face] = -v0_roof * abs((shell_height_v - z_roof) / vd);
                        k_shell_v = 1;
                        if (w_flag == 1 && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                          icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                          WGD->icellflag[icell_cent] = 10;
                        }
                      }
                    }
                  }
                  if ((k_shell_u + k_shell_v) < 1) {
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
