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

/** Sidewall.cpp */

#include "PolyBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


/**
 *
 * This function applies the sidewall parameterization to the qualified space on the side of buildings defined as polygons.
 * This function reads in building features like nodes, building height and base height and uses
 * features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
 * cells qualified on the side of buildings and applies the approperiate parameterization to them.
 * More information: "Comprehensive Evaluation of Fast-Response, Reynolds-Averaged Navier–Stokes, and Large-Eddy Simulation
 * Methods Against High-Spatial-Resolution Wind-Tunnel Data in Step-Down Street Canyons, A. N. Hayati et al. (2017)"
 *
 * @param WID :document this:
 * @param WGD :document this:
 */
void PolyBuilding::sideWall(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  float tol = 10 * M_PI / 180.0;// Sidewall is applied if outward normal of the face is in +/-10 degree perpendicular
    // to the local wind
  int side_wall_flag = 0;// If 1, indicates that there are faces that are nominally parallel with the wind
  std::vector<float> face_rel_dir; /**< Face relative angle to the perpendicular direction of the local wind */
  face_rel_dir.resize(polygonVertices.size(), 0.0);
  float R_scale_side; /**< Vortex size scaling factor */
  float R_cx_side; /**< Downwind length of the half-ellipse that defines the vortex recirculation region */
  float vd; /**< Half of lateral width of the elliptical recirculation region */
  float y_pref;
  int right_flag, left_flag;// 1, dependent face eligible for parameterization; 0, not eligible
  int index_previous, index_next;// Previous or next vertex
  float x_start_left, x_end_left, x_start_right, x_end_right;// Start and end point of each left/right faces in x-direction
  float y_start_left, y_end_left, y_start_right, y_end_right;// Start and end point of each left/right faces in y-direction
  float face_length;// Length of the face
  float face_dir;// Direction of the face
  int i_start_right, j_start_right;// i and j indices of the starting point for right face
  int i_start_left, j_start_left;// i and j indices of the starting point for left face
  float u0_right, v0_right;// u0 and v0 values for the right face
  float u0_left, v0_left;// u0 and v0 values for the left face
  float x_p, y_p;
  float shell_width, shell_width_calc;
  float x, y;
  float x_u, x_v, y_u, y_v;
  float xp_u, xp_v, xp_c, yp_u, yp_v, yp_c;
  float internal_BL_width;
  int x_id_max, y_id_max;

  int index_building_face = i_building_cent + j_building_cent * WGD->nx + (k_end)*WGD->nx * WGD->ny;
  u0_h = WGD->u0[index_building_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[index_building_face];// v velocity at the height of building at the centroid

  // Wind direction of initial velocity at the height of building at the centroid
  upwind_dir = atan2(v0_h, u0_h);

  xi.resize(polygonVertices.size(), 0.0);// Difference of x values of the centroid and each node
  yi.resize(polygonVertices.size(), 0.0);// Difference of y values of the centroid and each node

  // Loop to calculate x and y values of each polygon point in rotated coordinates
  for (size_t id = 0; id < polygonVertices.size(); id++) {
    xi[id] = (polygonVertices[id].x_poly - building_cent_x) * cos(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * sin(upwind_dir);
    yi[id] = -(polygonVertices[id].x_poly - building_cent_x) * sin(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * cos(upwind_dir);
  }

  for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
    // Face relative angle to the perpendicular angle to the local wind
    face_rel_dir[id] = atan2(yi[id + 1] - yi[id], xi[id + 1] - xi[id]) + 0.5 * M_PI;
    if (face_rel_dir[id] > M_PI) {
      face_rel_dir[id] -= 2 * M_PI;
    }
    if (abs(face_rel_dir[id]) >= 0.5 * M_PI - tol && abs(face_rel_dir[id]) <= 0.5 * M_PI + tol) {
      side_wall_flag = 1;// Indicates that there are faces that are nominally parallel with the wind
    }
  }

  if (side_wall_flag == 1) {
    // Smaller of the building height (H) and the effective cross-wind width (Weff)
    small_dimension = MIN_S(width_eff, H);
    // Larger of the building height (H) and the effective cross-wind width (Weff)
    long_dimension = MAX_S(width_eff, H);
    // Scaling length
    R_scale_side = pow(small_dimension, (2.0 / 3.0)) * pow(long_dimension, (1.0 / 3.0));
    R_cx_side = 0.9 * R_scale_side;// Normalized shell length
    vd = 0.5 * 0.22 * R_scale_side;// Shell width
    y_pref = vd / sqrt(0.5 * R_cx_side);

    for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
      // +/-10 degree perpendicular to the local wind
      if (abs(face_rel_dir[id]) >= 0.5 * M_PI - tol && abs(face_rel_dir[id]) <= 0.5 * M_PI + tol) {
        right_flag = 0;
        left_flag = 0;
        if (face_rel_dir[id] > 0.0) {
          index_previous = (id + polygonVertices.size() - 2) % (polygonVertices.size() - 1);
          // Finding the left face eligible for the parameterization
          if (abs(face_rel_dir[index_previous]) >= M_PI - tol) {
            left_flag = 1;
            x_start_left = polygonVertices[id].x_poly;
            y_start_left = polygonVertices[id].y_poly;
            x_end_left = polygonVertices[id + 1].x_poly;
            y_end_left = polygonVertices[id + 1].y_poly;
            face_length = sqrt(pow(x_start_left - x_end_left, 2.0) + pow(y_start_left - y_end_left, 2.0));
            face_dir = atan2(y_end_left - y_start_left, x_end_left - x_start_left);
          }
        } else {
          index_next = (id + 1) % (polygonVertices.size() - 1);
          // Finding the right face eligible for the parameterization
          if (abs(face_rel_dir[index_next]) >= M_PI - tol) {
            right_flag = 1;
            x_start_right = polygonVertices[id + 1].x_poly;
            y_start_right = polygonVertices[id + 1].y_poly;
            x_end_right = polygonVertices[id].x_poly;
            y_end_right = polygonVertices[id].y_poly;
            face_length = sqrt(pow(x_start_right - x_end_right, 2.0) + pow(y_start_right - y_end_right, 2.0));
            face_dir = atan2(y_end_right - y_start_right, x_end_right - x_start_right);
          }
        }
        // Loop through all points might be eligible for the parameterization
        for (auto k = k_end - 1; k >= k_start; k--) {
          if (right_flag == 1)// If the right face is eligible for the parameterization
          {
            // i and j indices for start of the right section of building
            i_start_right = ceil(x_start_right / WGD->dx) - 1;
            j_start_right = ceil(y_start_right / WGD->dy) - 1;
            for (auto j = MAX_S(0, j_start_right - 1); j <= MIN_S(WGD->ny - 2, j_start_right + 1); j++) {
              for (auto i = MAX_S(0, i_start_right - 1); i <= MIN_S(WGD->nx - 2, i_start_right + 1); i++) {
                icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                icell_face = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);
                // If the cell is building
                if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 7) {
                  u0_right = WGD->u0[icell_face];
                  v0_right = WGD->v0[icell_face];
                }
                // If the cell is air, upwind or canopy vegetation
                else if (WGD->icellflag[icell_cent] == 1 || WGD->icellflag[icell_cent] == 3 || WGD->icellflag[icell_cent] == 11) {
                }
                // If the cell is anything else (not eligible for the sidewall)
                else {
                  right_flag = 0;
                }
              }
            }
            if (right_flag == 1)// If the right face is eligible for the parameterization
            {
              // Finding id of the last cell eligible for sidewall from the upwind face of the buildin in x direction
              x_id_max = ceil(MAX_S(face_length, R_cx_side) / (0.5 * WGD->dxy));
              for (auto x_id = 1; x_id <= x_id_max; x_id++) {
                x_p = 0.5 * x_id * WGD->dxy;// x location of the point beng examined for parameterization in local coordinates
                // Width of the shell shape area on the sidewall of the building
                shell_width = y_pref * sqrt(x_p);
                y_id_max = (ceil(shell_width / (0.5 * WGD->dxy)) + 2);
                for (auto y_id = 1; y_id <= y_id_max; y_id++) {
                  y_p = -0.5 * y_id * WGD->dxy;// y location of the point beng examined for parameterization in local coordinates
                  x = x_start_right + x_p * cos(face_dir) - y_p * sin(face_dir);// x location in QUIC domain
                  y = y_start_right + x_p * sin(face_dir) + y_p * cos(face_dir);// y location in QUIC domain
                  // i and j indices of the point
                  int i = ceil(x / WGD->dx) - 1;
                  int j = ceil(y / WGD->dy) - 1;
                  icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                  icell_face = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);
                  if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)// No solid cell
                  {
                    x_u = i * WGD->dx;// x location of u component
                    y_u = (j + 0.5) * WGD->dy;// y location of u component
                    // x location of u component in local coordinates
                    xp_u = (x_u - x_start_right) * cos(face_dir) + (y_u - y_start_right) * sin(face_dir);
                    // y location of u component in local coordinates
                    yp_u = -(x_u - x_start_right) * sin(face_dir) + (y_u - y_start_right) * cos(face_dir);
                    // Shell width calculated for local x location of u component
                    shell_width_calc = 1 - pow((0.5 * R_cx_side - xp_u) / (0.5 * R_cx_side), 2.0);
                    if (shell_width_calc > 0.0) {
                      shell_width = vd * sqrt(shell_width_calc);
                    } else {
                      shell_width = 0.0;
                    }
                    internal_BL_width = y_pref * sqrt(xp_u);
                    if (abs(yp_u) <= shell_width) {
                      WGD->u0[icell_face] = -u0_right * abs((shell_width - abs(yp_u)) / vd);
                    } else if (abs(yp_u) <= internal_BL_width) {
                      WGD->u0[icell_face] = u0_right * log((abs(yp_u) + WGD->z0) / WGD->z0) / log((internal_BL_width + WGD->z0) / WGD->z0);
                    }

                    x_v = (i + 0.5) * WGD->dx;// x location of v component
                    y_v = j * WGD->dy;// y location of v component
                    // x location of v component in local coordinates
                    xp_v = (x_v - x_start_right) * cos(face_dir) + (y_v - y_start_right) * sin(face_dir);
                    // y location of v component in local coordinates
                    yp_v = -(x_v - x_start_right) * sin(face_dir) + (y_v - y_start_right) * cos(face_dir);
                    // Shell width calculated for local x location of v component
                    shell_width_calc = 1 - pow((0.5 * R_cx_side - xp_v) / (0.5 * R_cx_side), 2.0);
                    if (shell_width_calc > 0.0) {
                      shell_width = vd * sqrt(shell_width_calc);
                    } else {
                      shell_width = 0.0;
                    }
                    internal_BL_width = y_pref * sqrt(xp_v);
                    if (abs(yp_v) <= shell_width) {
                      WGD->v0[icell_face] = -v0_right * abs((shell_width - abs(yp_v)) / vd);
                    } else if (abs(yp_v) <= internal_BL_width) {
                      WGD->v0[icell_face] = v0_right * log((abs(yp_v) + WGD->z0) / WGD->z0) / log((internal_BL_width + WGD->z0) / WGD->z0);
                    }
                    // x location of cell center in local coordinates
                    xp_c = (x_v - x_start_right) * cos(face_dir) + (y_u - y_start_right) * sin(face_dir);
                    // y location of cell center in local coordinates
                    yp_c = -(x_v - x_start_right) * sin(face_dir) + (y_u - y_start_right) * cos(face_dir);
                    // Shell width calculated for local x location of cell center
                    shell_width_calc = 1 - pow((0.5 * R_cx_side - xp_c) / (0.5 * R_cx_side), 2.0);
                    internal_BL_width = y_pref * sqrt(xp_c);
                    if ((abs(yp_c) <= shell_width || abs(yp_c) <= internal_BL_width) && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                      WGD->icellflag[icell_cent] = 9;// Cell marked as sidewall cell
                    }
                  }
                }
              }
            }
          }
          if (left_flag == 1)// If the left face is eligible for the parameterization
          {
            // i and j indices for start of the left section of building
            i_start_left = ceil(x_start_left / WGD->dx) - 1;
            j_start_left = ceil(y_start_left / WGD->dy) - 1;
            for (auto j = MAX_S(0, j_start_left - 1); j <= MIN_S(WGD->ny - 2, j_start_left + 1); j++) {
              for (auto i = MAX_S(0, i_start_left - 1); i <= MIN_S(WGD->nx - 2, i_start_left + 1); i++) {
                icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                icell_face = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);
                // If the cell is solid (building or terrain)
                if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 7) {
                  u0_left = WGD->u0[icell_face];
                  v0_left = WGD->v0[icell_face];
                }
                // If the cell is air, upwind or canopy vegetation
                else if (WGD->icellflag[icell_cent] == 1 || WGD->icellflag[icell_cent] == 3 || WGD->icellflag[icell_cent] == 11) {
                }
                // If the cell anything else (not eligible for the parameterization)
                else {
                  left_flag = 0;
                }
              }
            }
            if (left_flag == 1)// If the left face is eligible for the parameterization
            {
              // Finding id of the last cell eligible for sidewall from the upwind face of the buildin in x direction
              x_id_max = ceil(MAX_S(face_length, R_cx_side) / (0.5 * WGD->dxy));
              for (auto x_id = 1; x_id <= x_id_max; x_id++) {
                x_p = 0.5 * x_id * WGD->dxy;// x location of the point beng examined for parameterization in local coordinates
                // Width of the shell shape area on the sidewall of the building
                shell_width = y_pref * sqrt(x_p);
                y_id_max = (ceil(shell_width / (0.5 * WGD->dxy)) + 2);
                for (auto y_id = 1; y_id <= y_id_max; y_id++) {
                  y_p = 0.5 * y_id * WGD->dxy;// y location of the point beng examined for parameterization in local coordinates
                  x = x_start_left + x_p * cos(face_dir) - y_p * sin(face_dir);// x location in QUIC domain
                  y = y_start_left + x_p * sin(face_dir) + y_p * cos(face_dir);// y location in QUIC domain
                  // i and j indices of the point
                  int i = ceil(x / WGD->dx) - 1;
                  int j = ceil(y / WGD->dy) - 1;
                  icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                  icell_face = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);
                  if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)// No solid cell
                  {
                    x_u = i * WGD->dx;// x location of u component
                    y_u = (j + 0.5) * WGD->dy;// y location of u component
                    // x location of u component in local coordinates
                    xp_u = (x_u - x_start_left) * cos(face_dir) + (y_u - y_start_left) * sin(face_dir);
                    // y location of u component in local coordinates
                    yp_u = -(x_u - x_start_left) * sin(face_dir) + (y_u - y_start_left) * cos(face_dir);
                    // Shell width calculated for local x location of u component
                    shell_width_calc = 1 - pow((0.5 * R_cx_side - xp_u) / (0.5 * R_cx_side), 2.0);
                    if (shell_width_calc > 0.0) {
                      shell_width = vd * sqrt(shell_width_calc);
                    } else {
                      shell_width = 0.0;
                    }
                    internal_BL_width = y_pref * sqrt(xp_u);
                    if (abs(yp_u) <= shell_width) {
                      WGD->u0[icell_face] = -u0_left * abs((shell_width - abs(yp_u)) / vd);
                    } else if (abs(yp_u) <= internal_BL_width) {
                      WGD->u0[icell_face] = u0_left * log((abs(yp_u) + WGD->z0) / WGD->z0) / log((internal_BL_width + WGD->z0) / WGD->z0);
                    }

                    x_v = (i + 0.5) * WGD->dx;// x location of v component
                    y_v = j * WGD->dy;// y location of v component
                    // x location of v component in local coordinates
                    xp_v = (x_v - x_start_left) * cos(face_dir) + (y_v - y_start_left) * sin(face_dir);
                    // y location of v component in local coordinates
                    yp_v = -(x_v - x_start_left) * sin(face_dir) + (y_v - y_start_left) * cos(face_dir);
                    // Shell width calculated for local x location of v component
                    shell_width_calc = 1 - pow((0.5 * R_cx_side - xp_v) / (0.5 * R_cx_side), 2.0);
                    if (shell_width_calc > 0.0) {
                      shell_width = vd * sqrt(shell_width_calc);
                    } else {
                      shell_width = 0.0;
                    }
                    internal_BL_width = y_pref * sqrt(xp_v);
                    if (abs(yp_v) <= shell_width) {
                      WGD->v0[icell_face] = -v0_left * abs((shell_width - abs(yp_v)) / vd);
                    } else if (abs(yp_v) <= internal_BL_width) {
                      WGD->v0[icell_face] = v0_left * log((abs(yp_v) + WGD->z0) / WGD->z0) / log((internal_BL_width + WGD->z0) / WGD->z0);
                    }
                    // x location of cell center in local coordinates
                    xp_c = (x_v - x_start_left) * cos(face_dir) + (y_u - y_start_left) * sin(face_dir);
                    // y location of cell center in local coordinates
                    yp_c = -(x_v - x_start_left) * sin(face_dir) + (y_u - y_start_left) * cos(face_dir);
                    // Shell width calculated for local x location of cell center
                    shell_width_calc = 1 - pow((0.5 * R_cx_side - xp_c) / (0.5 * R_cx_side), 2.0);
                    internal_BL_width = y_pref * sqrt(xp_c);
                    if ((abs(yp_c) <= shell_width || abs(yp_c) <= internal_BL_width) && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                      WGD->icellflag[icell_cent] = 9;// Cell marked as sidewall cell
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
}
