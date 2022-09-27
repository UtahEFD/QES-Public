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
 * @file PolygonWake.cpp
 * @brief :document this:
 */

#include "PolyBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


/**
 *
 * This function applies wake behind the building parameterization to buildings defined as polygons.
 * The parameterization has two parts: near wake and far wake. This function reads in building features
 * like nodes, building height and base height and uses features of the building defined in the class
 * constructor ans setCellsFlag function. It defines cells in each wake area and applies the approperiate
 * parameterization to them.
 *
 * @param WID :document this:
 * @param WGD :document this:
 * @param building_id :document this:
 */
void PolyBuilding::polygonWake(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_id)
{

  std::vector<float> Lr_face, Lr_node;
  std::vector<int> perpendicular_flag;
  Lr_face.resize(polygonVertices.size(), -1.0);// Length of wake for each face
  Lr_node.resize(polygonVertices.size(), 0.0);// Length of wake for each node
  perpendicular_flag.resize(polygonVertices.size(), 0);
  upwind_rel_dir.resize(polygonVertices.size(), 0.0);// Upwind reletive direction for each face
  float z_build;// z value of each building point from its base height
  float yc, xc;
  float Lr_local, Lr_local_u, Lr_local_v, Lr_local_w;// Local length of the wake for each velocity component
  float x_wall, x_wall_u, x_wall_v, x_wall_w;
  float y_norm, canyon_factor;
  int x_id_min;

  float Lr_ave;// Average length of Lr
  float total_seg_length;// Length of each edge
  int index_previous, index_next;// Indices of previous and next nodes
  int stop_id = 0;
  int kk(1);
  const float tol = 0.01 * M_PI / 180.0;
  float farwake_exp = 1.5;
  float farwake_factor = 3.0;
  float epsilon = 10e-10;
  int u_wake_flag, v_wake_flag, w_wake_flag;
  int i_u, j_u, i_v, j_v, i_w, j_w;// i and j indices for x, y and z directions
  float xp, yp;
  float xu, yu, xv, yv, xw, yw;
  float dn_u, dn_v, dn_w;// Length of cavity zone
  float farwake_vel;
  std::vector<double> u_temp, v_temp;
  //u_temp.resize(WGD->nx * WGD->ny, 0.0);
  //v_temp.resize(WGD->nx * WGD->ny, 0.0);
  std::vector<double> u0_modified, v0_modified;
  std::vector<int> u0_mod_id, v0_mod_id;
  float R_scale, R_cx, vd, hd, shell_height;
  int k_bottom, k_top;
  int tall_flag = 0;
  float z_s;// Saddle point height
  float x_sep;// In-canyon separation-point location
  int k_s;// Index of the saddle point in vetical dir


  int index_building_face = i_building_cent + j_building_cent * WGD->nx + (k_end)*WGD->nx * WGD->ny;
  u0_h = WGD->u0[index_building_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[index_building_face];// v velocity at the height of building at the centroid

  // Wind direction of initial velocity at the height of building at the centroid
  upwind_dir = atan2(v0_h, u0_h);

  x1 = x2 = y1 = y2 = 0.0;
  xi.resize(polygonVertices.size(), 0.0);// Difference of x values of the centroid and each node
  yi.resize(polygonVertices.size(), 0.0);// Difference of y values of the centroid and each node
  polygon_area = 0.0;

  for (size_t id = 0; id < polygonVertices.size(); id++) {
    xi[id] = (polygonVertices[id].x_poly - building_cent_x) * cos(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * sin(upwind_dir);
    yi[id] = -(polygonVertices[id].x_poly - building_cent_x) * sin(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * cos(upwind_dir);
  }

  // Loop to calculate polygon area, projections of x and y values of each point wrt upwind wind
  for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
    polygon_area += 0.5 * (polygonVertices[id].x_poly * polygonVertices[id + 1].y_poly - polygonVertices[id].y_poly * polygonVertices[id + 1].x_poly);
    // Find maximum and minimum x and y values in rotated coordinates
    if (xi[id] < x1) {
      x1 = xi[id];// Minimum x
    }
    if (xi[id] > x2) {
      x2 = xi[id];// Maximum x
    }
    if (yi[id] < y1) {
      y1 = yi[id];// Minimum y
    }
    if (yi[id] > y2) {
      y2 = yi[id];// Maximum y
    }
  }

  polygon_area = abs(polygon_area);
  width_eff = polygon_area / (x2 - x1);// Effective width of the building
  length_eff = polygon_area / (y2 - y1);// Effective length of the building
  float wake_height = height_eff;// effective top of the wake

  // Loop through points to find the height added to the effective height because of rooftop parameterization
  for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
    // Calculate upwind reletive direction for each face
    upwind_rel_dir[id] = atan2(yi[id + 1] - yi[id], xi[id + 1] - xi[id]) + 0.5 * M_PI;
    if (upwind_rel_dir[id] > M_PI + 0.0001) {
      upwind_rel_dir[id] -= 2 * M_PI;
    }
    if (abs(upwind_rel_dir[id]) < tol) {
      perpendicular_flag[id] = 1;
    }
  }

  L_over_H = length_eff / wake_height;// Length over height
  W_over_H = width_eff / wake_height;// Width over height
  H_over_L = wake_height / length_eff;

  if (H_over_L >= 2.0) {
    tall_flag = 1;
  }

  // Checking bounds of length over height and width over height
  if (L_over_H > 3.0) {
    L_over_H = 3.0;
  }
  if (L_over_H < 0.3) {
    L_over_H = 0.3;
  }
  if (W_over_H > 10.0) {
    W_over_H = 10.0;
  }

  // If the building is not tall enough
  if (tall_flag == 0) {
    // Calculating length of the downwind wake based on Fackrell (1984) formulation
    Lr = 1.8 * wake_height * W_over_H / (pow(L_over_H, 0.3) * (1 + 0.24 * W_over_H));
  } else {// New tall building parameterization
    Lr = 2.34 * wake_height * W_over_H / (pow(L_over_H, -1.37) * (-0.27 + 1.86 * W_over_H));
  }


  // if rectangular building and rooftop vortex, recalculate the top of the wake
  if (rectangular_flag && WID->buildingsParams->rooftopFlag == 2) {
    int id_valid = -1;
    int bldg_upwind = 0;

    // Rooftop applies if upcoming wind angle is in -/+30 degrees of perpendicular direction of the face
    const float tolRT = 30.0 * M_PI / 180.0;

    // search for valid face (can only have one here)
    for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
      if (abs(upwind_rel_dir[id]) >= M_PI - tolRT) {
        id_valid = id;
      }
    }

    // if valid face found
    if (id_valid >= 0) {

      // search for building upstream
      for (auto y_id = 0; y_id <= 2 * ceil(abs(yi[id_valid] - yi[id_valid + 1]) / WGD->dxy); y_id++) {
        yc = MIN_S(yi[id_valid], yi[id_valid + 1]) + 0.5 * y_id * WGD->dxy;

        // Checking to see whether the face is perpendicular to the wind direction
        if (perpendicular_flag[id_valid] == 1) {
          x_wall = xi[id_valid];
        } else {
          x_wall = ((xi[id_valid + 1] - xi[id_valid]) / (yi[id_valid + 1] - yi[id_valid])) * (yc - yi[id_valid]) + xi[id_valid];
        }

        for (auto x_id = ceil(Lr / WGD->dxy) + 1; x_id >= 1; x_id--) {
          xc = -x_id * WGD->dxy;
          int i = ceil(((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x)) / WGD->dx - 1;
          int j = ceil(((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y)) / WGD->dy - 1;
          if (i < WGD->nx - 2 && i > 0 && j < WGD->ny - 2 && j > 0) {
            int icell_cent = i + j * (WGD->nx - 1) + (k_end - 1) * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->icellflag[icell_cent] == 0) {
              // found building
              bldg_upwind++;
              break;
            }
          }
        }
      }

      // set rooftop_flag based on presence of building upstream
      if (bldg_upwind >= ceil(abs(yi[id_valid] - yi[id_valid + 1]) / WGD->dxy)) {
        rooftop_flag = 0;
      } else {
        rooftop_flag = 1;
      }

      if (rooftop_flag == 1) {
        // Smaller of the effective height (height_eff) and the effective cross-wind width (Weff)
        small_dimension = MIN_S(width_eff, height_eff);
        // Larger of the effective height (height_eff) and the effective cross-wind width (Weff)
        long_dimension = MAX_S(width_eff, height_eff);
        R_scale = pow(small_dimension, (2.0 / 3.0)) * pow(long_dimension, (1.0 / 3.0));// Scaling length
        R_cx = 0.9 * R_scale;// Normalized cavity length
        vd = 0.5 * 0.22 * R_scale;// Cavity height
        // Smaller of the effective length (length_eff) and the effective cross-wind width (Weff)
        hd = MIN_S(width_eff, length_eff);
        if (hd < R_cx) {
          shell_height = vd * sqrt(1.0 - pow((0.5 * R_cx - hd) / (0.5 * R_cx), 2.0));// Additional height because of the rooftop
          if (shell_height > 0.0) {
            wake_height += shell_height;
          }
        }

        // recaluclate based one new height
        L_over_H = length_eff / wake_height;// Length over height
        W_over_H = width_eff / wake_height;// Width over height
        H_over_L = wake_height / length_eff;

        if (H_over_L >= 2.0) {
          tall_flag = 1;
        }

        // Checking bounds of length over height and width over height
        if (L_over_H = 3.0) {
          L_over_H = 3.0;
        }
        if (L_over_H < 0.3) {
          L_over_H = 0.3;
        }
        if (W_over_H > 10.0) {
          W_over_H = 10.0;
        }

        // If the building is not tall enough
        if (tall_flag == 0) {
          // Calculating length of the downwind wake based on Fackrell (1984) formulation
          Lr = 1.8 * wake_height * W_over_H / (pow(L_over_H, 0.3) * (1 + 0.24 * W_over_H));
        } else {// New tall building parameterization
          Lr = 2.34 * wake_height * W_over_H / (pow(L_over_H, -1.37) * (-0.27 + 1.86 * W_over_H));
        }
      }
    }
  }

  if (tall_flag == 1) {
    // Caluculate the saddle point height and the in-canyon separation-point Location
    z_s = (-1.47 * L_over_H - 0.69 * W_over_H + 1.25) * wake_height;
    x_sep = 4.34 * wake_height * W_over_H / (pow(L_over_H, -0.16) * (2.99 - 0.81 * W_over_H));
    wake_height = (1.4077 * L_over_H - 1.5705 * W_over_H + 0.9297) * wake_height;
    for (auto k = k_start; k <= WGD->nz - 2; k++) {
      k_s = k;
      if (z_s <= WGD->z[k]) {
        break;
      }
    }
  }

  for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
    // Finding faces that are eligible for applying the far-wake parameterizations
    // angle between two points should be in -180 to 0 degree
    if (abs(upwind_rel_dir[id]) < 0.5 * M_PI) {
      // Calculate length of the far wake zone for each face
      Lr_face[id] = Lr * cos(upwind_rel_dir[id]);
    }
  }

  Lr_ave = total_seg_length = 0.0;
  // This loop interpolates the value of Lr for eligible faces to nodes of those faces
  for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
    // If the face is eligible for parameterization
    if (Lr_face[id] > 0.0) {
      index_previous = (id + polygonVertices.size() - 2) % (polygonVertices.size() - 1);// Index of previous face
      index_next = (id + 1) % (polygonVertices.size() - 1);// Index of next face
      if (Lr_face[index_previous] < 0.0 && Lr_face[index_next] < 0.0) {
        Lr_node[id] = Lr_face[id];
        Lr_node[id + 1] = Lr_face[id];
      } else if (Lr_face[index_previous] < 0.0) {
        Lr_node[id] = Lr_face[id];
        Lr_node[id + 1] = ((yi[index_next] - yi[index_next + 1]) * Lr_face[index_next] + (yi[id] - yi[index_next]) * Lr_face[id])
                          / (yi[id] - yi[index_next + 1]);
      } else if (Lr_face[index_next] < 0.0) {
        Lr_node[id] = ((yi[id] - yi[index_next]) * Lr_face[id] + (yi[index_previous] - yi[id]) * Lr_face[index_previous])
                      / (yi[index_previous] - yi[index_next]);
        Lr_node[id + 1] = Lr_face[id];
      } else {
        Lr_node[id] = ((yi[id] - yi[index_next]) * Lr_face[id] + (yi[index_previous] - yi[id]) * Lr_face[index_previous])
                      / (yi[index_previous] - yi[index_next]);
        Lr_node[id + 1] = ((yi[index_next] - yi[index_next + 1]) * Lr_face[index_next] + (yi[id] - yi[index_next]) * Lr_face[id])
                          / (yi[id] - yi[index_next + 1]);
      }
      Lr_ave += Lr_face[id] * (yi[id] - yi[index_next]);
      total_seg_length += (yi[id] - yi[index_next]);
    }

    if ((polygonVertices[id + 1].x_poly > polygonVertices[0].x_poly - 0.1) && (polygonVertices[id + 1].x_poly < polygonVertices[0].x_poly + 0.1)
        && (polygonVertices[id + 1].y_poly > polygonVertices[0].y_poly - 0.1) && (polygonVertices[id + 1].y_poly < polygonVertices[0].y_poly + 0.1)) {
      stop_id = id;
      break;
    }
  }

  Lr = Lr_ave / total_seg_length;

  for (auto k = 1; k <= k_start; k++) {
    k_bottom = k;
    if (base_height <= WGD->z[k]) {
      break;
    }
  }

  for (auto k = k_start; k < WGD->nz - 2; k++) {
    k_top = k;
    if (wake_height < WGD->z[k + 1]) {
      break;
    }
  }

  for (auto k = k_start; k < k_end; k++) {
    kk = k;
    if (0.75 * H + base_height <= WGD->z[k]) {
      break;
    }
  }

  for (auto k = k_top; k >= k_bottom; k--) {
    z_build = WGD->z[k] - base_height;
    for (auto id = 0; id <= stop_id; id++) {
      if (abs(upwind_rel_dir[id]) < 0.5 * M_PI) {
        if (perpendicular_flag[id] == 1) {
          x_wall = xi[id];
        }
        for (auto y_id = 0; y_id <= 2 * ceil(abs(yi[id] - yi[id + 1]) / WGD->dxy); y_id++) {
          yc = yi[id] - 0.5 * y_id * WGD->dxy;
          Lr_local = Lr_node[id] + (yc - yi[id]) * (Lr_node[id + 1] - Lr_node[id]) / (yi[id + 1] - yi[id]);
          // Checking to see whether the face is perpendicular to the wind direction
          if (perpendicular_flag[id] == 0) {
            x_wall = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (yc - yi[id]) + xi[id];
          } else {
            x_wall = xi[id];
          }
          if (yc >= 0.0) {
            y_norm = y2;
          } else {
            y_norm = y1;
          }
          canyon_factor = 1.0;
          x_id_min = -1;
          for (auto x_id = 1; x_id <= ceil(Lr_local / WGD->dxy); x_id++) {
            xc = x_id * WGD->dxy;
            int i = ((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x) / WGD->dx;
            int j = ((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y) / WGD->dy;
            if (i >= WGD->nx - 2 || i <= 0 || j >= WGD->ny - 2 || j <= 0) {
              break;
            }
            int icell_cent = i + j * (WGD->nx - 1) + kk * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && x_id_min < 0) {
              x_id_min = x_id;
            }
            if ((WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2) && x_id_min > 0) {
              canyon_factor = xc / Lr;
              break;
            }
          }
          x_id_min = -1;
          for (auto x_id = 1; x_id <= 2 * ceil(farwake_factor * Lr_local / WGD->dxy); x_id++) {
            u_wake_flag = 1;
            v_wake_flag = 1;
            w_wake_flag = 1;
            xc = 0.5 * x_id * WGD->dxy;
            int i = ((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x) / WGD->dx;
            int j = ((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y) / WGD->dy;
            if (i >= WGD->nx - 2 || i <= 0 || j >= WGD->ny - 2 || j <= 0) {
              break;
            }
            icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && x_id_min < 0) {
              x_id_min = x_id;
            }
            if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2) {
              if (x_id_min >= 0) {
                if (WGD->ibuilding_flag[icell_cent] == building_id) {
                  x_id_min = -1;
                } else if (canyon_factor < 1.0) {
                  break;
                } else if (WGD->icellflag[i + j * (WGD->nx - 1) + kk * (WGD->nx - 1) * (WGD->ny - 1)] == 0
                           || WGD->icellflag[i + j * (WGD->nx - 1) + kk * (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
                  break;
                }
              }
            }

            if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
              i_u = std::round(((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x) / WGD->dx);
              j_u = ((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y) / WGD->dy;
              if (i_u < WGD->nx - 1 && i_u > 0 && j_u < WGD->ny - 1 && j_u > 0) {
                xp = i_u * WGD->dx - building_cent_x;
                yp = (j_u + 0.5) * WGD->dy - building_cent_y;
                xu = xp * cos(upwind_dir) + yp * sin(upwind_dir);
                yu = -xp * sin(upwind_dir) + yp * cos(upwind_dir);
                Lr_local_u = Lr_node[id] + (yu - yi[id]) * (Lr_node[id + 1] - Lr_node[id]) / (yi[id + 1] - yi[id]);
                if (perpendicular_flag[id] > 0) {
                  x_wall_u = xi[id];

                } else {
                  x_wall_u = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (yu - yi[id]) + xi[id];
                }

                xu -= x_wall_u;
                if (abs(yu) < abs(y_norm) && abs(y_norm) > epsilon && z_build < wake_height && wake_height > epsilon) {
                  if (tall_flag == 0) {
                    dn_u = sqrt((1.0 - pow((yu / y_norm), 2.0)) * (1.0 - pow((z_build / wake_height), 2.0)) * pow((canyon_factor * Lr_local_u), 2.0));
                  } else if (tall_flag == 1 && z_build > z_s) {
                    dn_u = sqrt((1.0 - pow((yu / y_norm), 2.0)) * (1.0 - pow(((z_build - z_s) / (wake_height - z_s)), 2.0)) * pow((canyon_factor * Lr_local_u), 2.0));
                  } else if (tall_flag == 1 && z_build <= z_s) {
                    dn_u = sqrt((1.0 - pow((yu / y_norm), 2.0))) * ((sqrt(1.0 - pow(((z_build - z_s) / z_s), 2.0)) * (canyon_factor * (Lr_local_u - x_sep))) + x_sep);
                  }
                } else {
                  dn_u = 0.0;
                }
                if (xu > farwake_factor * dn_u) {
                  u_wake_flag = 0;
                }
                icell_cent = i_u + j_u * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                icell_face = i_u + j_u * WGD->nx + k * WGD->nx * WGD->ny;
                if (dn_u > 0.0 && u_wake_flag == 1 && yu <= yi[id] && yu >= yi[id + 1]
                    && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                  // Far wake zone
                  if (xu > dn_u) {
                    farwake_vel = WGD->u0[icell_face] * (1.0 - pow((dn_u / (xu + WGD->wake_factor * dn_u)), farwake_exp));
                    if (canyon_factor == 1.0) {
                      u0_modified.push_back(farwake_vel);
                      u0_mod_id.push_back(icell_face);
                      WGD->w0[icell_face] = 0.0;
                    }
                  }
                  // Cavity zone
                  else {
                    WGD->u0[icell_face] = -u0_h * MIN_S(pow((1.0 - xu / (WGD->cavity_factor * dn_u)), 2.0), 1.0)
                                          * MIN_S(sqrt(1.0 - abs(yu / y_norm)), 1.0);
                    if (tall_flag == 1 && xu <= (dn_u / 2) && z_build >= z_s) {
                      WGD->w0[icell_face] = -1.0 * (abs(z_build - z_s) / wake_height) * WGD->u0[icell_face];
                    } else if (tall_flag == 1 && xu > (dn_u / 2) && z_build >= z_s) {
                      WGD->w0[icell_face] = 1.0 * (abs(z_build - z_s) / wake_height) * WGD->u0[icell_face];
                    } else if (tall_flag == 1 && z_build < z_s) {
                      WGD->w0[icell_face] = -2.0 * (abs(z_build - z_s) / wake_height) * WGD->u0[icell_face];
                    } else {
                      WGD->w0[icell_face] = 0.0;
                    }
                  }
                }
              }

              i_v = ((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x) / WGD->dx;
              j_v = std::round(((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y) / WGD->dy);
              if (i_v < WGD->nx - 1 && i_v > 0 && j_v < WGD->ny - 1 && j_v > 0) {
                xp = (i_v + 0.5) * WGD->dx - building_cent_x;
                yp = j_v * WGD->dy - building_cent_y;
                xv = xp * cos(upwind_dir) + yp * sin(upwind_dir);
                yv = -xp * sin(upwind_dir) + yp * cos(upwind_dir);
                Lr_local_v = Lr_node[id] + (yv - yi[id]) * (Lr_node[id + 1] - Lr_node[id]) / (yi[id + 1] - yi[id]);
                if (perpendicular_flag[id] > 0) {
                  x_wall_v = xi[id];
                } else {
                  x_wall_v = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (yv - yi[id]) + xi[id];
                }
                xv -= x_wall_v;

                if (abs(yv) < abs(y_norm) && abs(y_norm) > epsilon && z_build < wake_height && wake_height > epsilon) {
                  if (tall_flag == 0) {
                    dn_v = sqrt((1.0 - pow((yv / y_norm), 2.0)) * (1.0 - pow((z_build / wake_height), 2.0)) * pow((canyon_factor * Lr_local_v), 2.0));
                  } else if (tall_flag == 1 && z_build >= z_s) {
                    dn_v = sqrt((1.0 - pow((yv / y_norm), 2.0)) * (1.0 - pow(((z_build - z_s) / (wake_height - z_s)), 2.0)) * pow((canyon_factor * Lr_local_v), 2.0));
                  } else if (tall_flag == 1 && z_build < z_s) {
                    dn_v = sqrt((1.0 - pow((yv / y_norm), 2.0))) * ((sqrt(1.0 - pow(((z_build - z_s) / z_s), 2.0)) * (canyon_factor * (Lr_local_v - x_sep))) + x_sep);
                  }
                } else {
                  dn_v = 0.0;
                }
                if (xv > farwake_factor * dn_v) {
                  v_wake_flag = 0;
                }
                icell_cent = i_v + j_v * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                icell_face = i_v + j_v * WGD->nx + k * WGD->nx * WGD->ny;
                if (dn_v > 0.0 && v_wake_flag == 1 && yv <= yi[id] && yv >= yi[id + 1]
                    && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                  // Far wake zone
                  if (xv > dn_v) {
                    farwake_vel = WGD->v0[icell_face] * (1.0 - pow((dn_v / (xv + WGD->wake_factor * dn_v)), farwake_exp));
                    if (canyon_factor == 1) {
                      v0_modified.push_back(farwake_vel);
                      v0_mod_id.push_back(icell_face);

                      WGD->w0[icell_face] = 0.0;
                    }
                  }
                  // Cavity zone
                  else {
                    WGD->v0[icell_face] = -v0_h * MIN_S(pow((1.0 - xv / (WGD->cavity_factor * dn_v)), 2.0), 1.0)
                                          * MIN_S(sqrt(1.0 - abs(yv / y_norm)), 1.0);
                    if (tall_flag == 1 && xv <= (dn_v / 2) && z_build >= z_s) {
                      WGD->w0[icell_face] = MAX_S(-1.0 * (abs(z_build - z_s) / wake_height) * WGD->v0[icell_face], WGD->w0[icell_face]);
                    } else if (tall_flag == 1 && xv > (dn_v / 2) && z_build >= z_s) {
                      WGD->w0[icell_face] = MIN_S(1.0 * (abs(z_build - z_s) / wake_height) * WGD->v0[icell_face], WGD->w0[icell_face]);
                    } else if (tall_flag == 1 && z_build < z_s) {
                      WGD->w0[icell_face] = MAX_S(-2.0 * (abs(z_build - z_s) / wake_height) * WGD->v0[icell_face], WGD->w0[icell_face]);
                    } else {
                      WGD->w0[icell_face] = 0.0;
                    }
                  }
                }
              }

              i_w = ceil(((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x) / WGD->dx) - 1;
              j_w = ceil(((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y) / WGD->dy) - 1;
              if (i_w < WGD->nx - 2 && i_w > 0 && j_w < WGD->ny - 2 && j_w > 0) {
                xp = (i_w + 0.5) * WGD->dx - building_cent_x;
                yp = (j_w + 0.5) * WGD->dy - building_cent_y;
                xw = xp * cos(upwind_dir) + yp * sin(upwind_dir);
                yw = -xp * sin(upwind_dir) + yp * cos(upwind_dir);
                Lr_local_w = Lr_node[id] + (yw - yi[id]) * (Lr_node[id + 1] - Lr_node[id]) / (yi[id + 1] - yi[id]);
                if (perpendicular_flag[id] > 0) {
                  x_wall_w = xi[id];
                } else {
                  x_wall_w = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (yw - yi[id]) + xi[id];
                }
                xw -= x_wall_w;
                if (abs(yw) < abs(y_norm) && abs(y_norm) > epsilon && z_build < wake_height && wake_height > epsilon) {
                  if (tall_flag == 0) {
                    dn_w = sqrt((1.0 - pow(yw / y_norm, 2.0)) * (1.0 - pow(z_build / wake_height, 2.0)) * pow(canyon_factor * Lr_local_w, 2.0));
                  } else if (tall_flag == 1 && k >= k_s) {
                    dn_w = sqrt((1.0 - pow((yw / y_norm), 2.0)) * (1.0 - pow(((z_build - z_s) / (wake_height - z_s)), 2.0)) * pow((canyon_factor * Lr_local_w), 2.0));
                  } else if (tall_flag == 1 && k < k_s) {
                    dn_w = sqrt((1.0 - pow((yw / y_norm), 2.0)) * (1.0 - pow(((z_build - z_s) / z_s), 2.0)) * pow((canyon_factor * (Lr_local_w - x_sep)), 2.0)) + x_sep;
                  }
                } else {
                  dn_w = 0.0;
                }

                if (xw > farwake_factor * dn_w) {
                  w_wake_flag = 0;
                }
                icell_cent = i_w + j_w * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                icell_face = i_w + j_w * WGD->nx + k * WGD->nx * WGD->ny;
                if (dn_w > 0.0 && w_wake_flag == 1 && yw <= yi[id] && yw >= yi[id + 1]
                    && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                  if (xw > dn_w) {
                    if (canyon_factor == 1) {
                      if ((WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                        WGD->icellflag[icell_cent] = 5;
                      }
                    }
                  } else {
                    if ((WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                      WGD->icellflag[icell_cent] = 4;
                    }
                  }
                }
                if (u_wake_flag == 0 && v_wake_flag == 0 && w_wake_flag == 0) {
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  for (size_t x_id = 0; x_id < u0_mod_id.size(); x_id++) {
    WGD->u0[u0_mod_id[x_id]] = u0_modified[x_id];
  }

  for (size_t y_id = 0; y_id < v0_mod_id.size(); y_id++) {
    WGD->v0[v0_mod_id[y_id]] = v0_modified[y_id];
  }

  u0_mod_id.clear();
  v0_mod_id.clear();
  u0_modified.clear();
  v0_modified.clear();
}
