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

/** StreetCanyon.cpp */

#include "PolyBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

/**
 *
 * This function applies the street canyon parameterization to the qualified space between buildings defined as polygons.
 * This function reads in building features like nodes, building height and base height and uses
 * features of the building defined in the class constructor and setCellsFlag function. It defines
 * cells qualified in the space between buildings and applies the approperiate parameterization to them.
 * More information: "Improvements to a fast-response WINDSan wind model, M. Nelson et al. (2008)"
 *
 * @param WGD :document this:
 */
void PolyBuilding::streetCanyon(WINDSGeneralData *WGD)
{
  float tol = 0.01 * M_PI / 180.0;
  float angle_tol = 3.0 * M_PI / 4.0;
  float x_wall, x_wall_u, x_wall_v, x_wall_w;
  float xc, yc;
  int top_flag, canyon_flag;
  int k_ref;
  int reverse_flag;
  int x_id_min, x_id_max;
  int number_u, number_v;
  float u_component, v_component;
  float s;// Distance between two buildings
  float velocity_mag;
  float canyon_dir;
  int d_build;// Downwind building number
  int i_u, j_v;
  float x_u, y_u, x_v, y_v, x_w, y_w;
  float x_pos;
  float x_p, y_p;
  float x_ave, y_ave;
  float x_down, y_down;
  float segment_length;// Face length
  float downwind_rel_dir, along_dir, cross_dir, facenormal_dir;
  float cross_vel_mag, along_vel_mag;
  std::vector<int> perpendicular_flag;
  std::vector<float> perpendicular_dir;

  xi.resize(polygonVertices.size(), 0.0);// Difference of x values of the centroid and each node
  yi.resize(polygonVertices.size(), 0.0);// Difference of y values of the centroid and each node
  upwind_rel_dir.resize(polygonVertices.size(), 0.0);// Upwind reletive direction for each face
  perpendicular_flag.resize(polygonVertices.size(), 0);
  perpendicular_dir.resize(polygonVertices.size(), 0.0);

  int index_building_face = i_building_cent + j_building_cent * WGD->nx + (k_end)*WGD->nx * WGD->ny;
  u0_h = WGD->u0[index_building_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[index_building_face];// v velocity at the height of building at the centroid

  // Wind direction of initial velocity at the height of building at the centroid
  upwind_dir = atan2(v0_h, u0_h);
  // Loop through to calculate projected location for each polygon node in rotated coordinates
  for (size_t id = 0; id < polygonVertices.size(); id++) {
    xi[id] = (polygonVertices[id].x_poly - building_cent_x) * cos(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * sin(upwind_dir);
    yi[id] = -(polygonVertices[id].x_poly - building_cent_x) * sin(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * cos(upwind_dir);
  }

  // Loop over face of current building
  for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
    // Calculate upwind reletive direction for each face (out-facing normal)
    upwind_rel_dir[id] = atan2(yi[id + 1] - yi[id], xi[id + 1] - xi[id]) + 0.5 * M_PI;

    // Force the angle to be between (-pi;pi]
    if (upwind_rel_dir[id] > M_PI) {
      upwind_rel_dir[id] -= 2.0 * M_PI;
    }

    // valid only if relative angle between back face of building and wind angle is (-0.5*pi;0.5*pi)
    if (abs(upwind_rel_dir[id]) < 0.5 * M_PI - 0.0001) {
      // Checking to see whether the face is perpendicular to the wind direction
      if (abs(upwind_rel_dir[id]) > M_PI - tol || abs(upwind_rel_dir[id]) < tol) {
        perpendicular_flag[id] = 1;
        x_wall = xi[id];
      }
      // Calculating perpendicula direction to each face (out-facing nomral)
      perpendicular_dir[id] = atan2(polygonVertices[id + 1].y_poly - polygonVertices[id].y_poly,
                                    polygonVertices[id + 1].x_poly - polygonVertices[id].x_poly)
                              + 0.5 * M_PI;

      // Force the angle to be between (-pi;pi]
      if (perpendicular_dir[id] > M_PI) {
        perpendicular_dir[id] -= 2.0 * M_PI;
      }
      // Loop through y locations along each face in rotated coordinates
      for (auto y_id = 0; y_id <= 2 * ceil(abs(yi[id] - yi[id + 1]) / WGD->dxy); y_id++) {
        // y locations along each face in rotated coordinates
        yc = MIN_S(yi[id], yi[id + 1]) + 0.5 * y_id * WGD->dxy;
        // reset flag for every y-location along the face (if inside the canyon)
        top_flag = 0;

        if (perpendicular_flag[id] == 0) {
          // x location of each yc point parallel to the face
          x_wall = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (yc - yi[id]) + xi[id];
        }

        for (auto k = k_end - 1; k >= k_start; k--) {

          // checking that the cell before the wall is defined as building (loop if not)
          int i1 = ceil(((x_wall - WGD->dxy) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x) / WGD->dx) - 1;
          int j1 = ceil(((x_wall - WGD->dxy) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y) / WGD->dy) - 1;
          icell_cent = i1 + j1 * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
          if (WGD->icellflag[icell_cent] != 0)
            continue;

          // resting canyon parameters
          canyon_flag = 0;
          s = 0.0;
          reverse_flag = 0;
          x_id_min = -1;

          // (LoopX1) Loop through x locations along perpendicular direction of each face
          for (auto x_id = 1; x_id <= 2 * ceil(Lr / WGD->dxy); x_id++) {
            // x locations along perpendicular direction of each face
            xc = 0.5 * x_id * WGD->dxy;
            // Finding i and j indices of the cell (xc, yc) located in
            int i = ceil(((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x) / WGD->dx) - 1;
            int j = ceil(((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y) / WGD->dy) - 1;
            icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            // Making sure i and j are inside the domain
            if (i >= WGD->nx - 2 && i <= 0 && j >= WGD->ny - 2 && j <= 0) {
              break;// exit LoopX1
            }
            icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            // Finding id of the first cell in perpendicular direction of the face that is outside of the building
            if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && x_id_min < 0) {
              x_id_min = x_id;
            }
            // Finding id of the last cell in perpendicular direction of the face that is outside of the building
            if ((WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2) && x_id_min >= 0) {
              canyon_flag = 1;
              // x_id_max is last x_id before hitting downstream building
              x_id_max = x_id - 1;
              // Distance between two buildings
              s = 0.5 * (x_id_max - x_id_min) * WGD->dxy;

              // If inside the street canyon
              if (top_flag == 0) {
                // k_ref is one cell above top of street canyon zone
                k_ref = k + 1;
                int ic = ceil(((0.5 * x_id_max * WGD->dxy + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x - 0.001) / WGD->dx) - 1;
                int jc = ceil(((0.5 * x_id_max * WGD->dxy + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y - 0.001) / WGD->dy) - 1;
                // reference cell (x_id_max,y,k_ref)
                icell_cent = ic + jc * (WGD->nx - 1) + k_ref * (WGD->nx - 1) * (WGD->ny - 1);
                int icell_face = ic + jc * WGD->nx + k_ref * WGD->nx * WGD->ny;
                if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                  // compute average velocity at reference cell.
                  number_u = 0;
                  number_v = 0;
                  u_component = 0.0;
                  v_component = 0.0;
                  if (WGD->icellflag[icell_cent - 1] != 0 && WGD->icellflag[icell_cent - 1] != 2) {
                    number_u += 1;
                    u_component += WGD->u0[icell_face];
                  }
                  if (WGD->icellflag[icell_cent + 1] != 0 && WGD->icellflag[icell_cent + 1] != 2) {
                    number_u += 1;
                    u_component += WGD->u0[icell_face + 1];
                  }
                  if (WGD->icellflag[icell_cent - (WGD->nx - 1)] != 0 && WGD->icellflag[icell_cent - (WGD->nx - 1)] != 2) {
                    number_v += 1;
                    v_component += WGD->v0[icell_face];
                  }
                  if (WGD->icellflag[icell_cent + (WGD->nx - 1)] != 0 && WGD->icellflag[icell_cent + (WGD->nx - 1)] != 2) {
                    number_v += 1;
                    v_component += WGD->v0[icell_face + WGD->nx];
                  }

                  // compute average velocity at reference cell.
                  if (u_component != 0.0 && number_u > 0) {
                    u_component /= number_u;
                  } else {
                    u_component = 0.0;
                  }
                  if (v_component != 0.0 && number_v > 0) {
                    v_component /= number_v;
                  } else {
                    v_component = 0.0;
                  }

                  // compute velocity in polar coord (u_mag,u_dir)
                  if (number_u == 0 && number_v == 0) {
                    // break (LoopX1) if no valid velocit at reference cell
                    canyon_flag = 0;
                    top_flag = 0;
                    s = 0.0;
                    break;// exit LoopX1
                  } else if (number_u > 0 && number_v > 0) {
                    // 2D velocity
                    velocity_mag = sqrt(pow(u_component, 2.0) + pow(v_component, 2.0));
                    canyon_dir = atan2(v_component, u_component);
                  } else if (number_u > 0) {
                    // mean velocity along +/-x
                    velocity_mag = abs(u_component);
                    if (u_component > 0.0) {
                      canyon_dir = 0.0;
                    } else {
                      canyon_dir = M_PI;
                    }
                  } else {
                    // mean velocity along +/- y
                    velocity_mag = abs(v_component);
                    if (v_component > 0.0) {
                      canyon_dir = 0.5 * M_PI;
                    } else {
                      canyon_dir = -0.5 * M_PI;
                    }
                  }

                  // flag=1 -> top of canyon set, velocity and direction set
                  // this if-block will not be executed unless flag is reset
                  top_flag = 1;

                  // icell_cent set to cell inside the downstream building
                  icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);

                  if (abs(s) > 0.0) {
                    if ((WGD->ibuilding_flag[icell_cent] >= 0)
                        && (WGD->allBuildingsV[WGD->ibuilding_flag[icell_cent]]->height_eff < height_eff)
                        && (WGD->z_face[k] / s < 0.65)) {
                      // break if downstream building sorter than current building and H/S < 0.65 (WILL NOT WORK ABOVE TERRAIN)
                      canyon_flag = 0;
                      top_flag = 0;
                      s = 0.0;
                      break;// exit LoopX1
                    }
                  }
                } else {
                  // break if reference cell is building or terrain (x_id_max==0,2)
                  canyon_flag = 0;
                  top_flag = 0;
                  s = 0.0;
                  break;// exit LoopX1
                }
                if (velocity_mag > WGD->max_velmag) {
                  // break if velocity above threshold
                  canyon_flag = 0;
                  top_flag = 0;
                  s = 0.0;
                  break;// exit LoopX1
                }
              }

              // icell_cent set to cell (xc,yc,zc) (inside the downstream building if present)
              icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
              if (WGD->ibuilding_flag[icell_cent] >= 0) {
                d_build = WGD->ibuilding_flag[icell_cent];
                int i = ceil(((xc - 0.5 * WGD->dxy + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x - 0.001) / WGD->dx) - 1;
                int j = ceil(((xc - 0.5 * WGD->dxy + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y - 0.001) / WGD->dy) - 1;

                //if (WGD->ibuilding_flag[i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1)] >= 0) {
                //  std::cout << "error" << std::endl;
                //}

                // (LoopDB) Loop through each polygon node of the downstream buildings
                for (size_t j_id = 0; j_id < WGD->allBuildingsV[d_build]->polygonVertices.size() - 1; j_id++) {
                  // normal direction (out-facing normal) of face of downstream building (ref as db-face for that loop)
                  facenormal_dir = atan2(WGD->allBuildingsV[d_build]->polygonVertices[j_id + 1].y_poly
                                           - WGD->allBuildingsV[d_build]->polygonVertices[j_id].y_poly,
                                         WGD->allBuildingsV[d_build]->polygonVertices[j_id + 1].x_poly
                                           - WGD->allBuildingsV[d_build]->polygonVertices[j_id].x_poly)
                                   + 0.5 * M_PI;
                  // forcing facenormal_dir to be in (-pi,pi]
                  if (facenormal_dir > M_PI + 0.001) {
                    facenormal_dir -= 2.0 * M_PI;
                  }
                  if (facenormal_dir <= -M_PI) {
                    facenormal_dir += 2.0 * M_PI;
                  }

                  // (x,y) mid point of db-face
                  x_ave = 0.5 * (WGD->allBuildingsV[d_build]->polygonVertices[j_id + 1].x_poly + WGD->allBuildingsV[d_build]->polygonVertices[j_id].x_poly);
                  y_ave = 0.5 * (WGD->allBuildingsV[d_build]->polygonVertices[j_id + 1].y_poly + WGD->allBuildingsV[d_build]->polygonVertices[j_id].y_poly);

                  // x-/y-componants of segment between current location (center of cell) and mid-point of bd-face
                  // -> relative to the normal of current face  (facenormal_dir)
                  x_down = ((i + 0.5) * WGD->dx - x_ave) * cos(facenormal_dir) + ((j + 0.5) * WGD->dy - y_ave) * sin(facenormal_dir);
                  y_down = -((i + 0.5) * WGD->dx - x_ave) * sin(facenormal_dir) + ((j + 0.5) * WGD->dy - y_ave) * cos(facenormal_dir);

                  /* flow reverse means that the flow at the reference is reveresed compared to the upwind direction
                     - reverse flow = flow go down along front face - up along back face
                     - otherwise    = flow go up along front face - down along back face
                     this block check of flow conditions: 
                     1) check if location of current cell against db-face
                     |  x-dir (relative to center of face): location within one cell
                     |  y-dir (relative to center of face): less that 1/2 the length of the face
                     2) check relative angle between wind and bd-face :
                     |  if smaller that +/- 0.5pi -> flow reverse
                     |  else -> no flow reverse
		     3) define wind projection angle within the canyon (absolute angles)
                     |  cross_dir = angle of the projection of the canyon_dir wind on the perpendicular dir to the db-face (facenormal_dir)
		     |  along_dir = angle of the projection of the canyon_dir wind on the parallel dir to the db-face (facenormal_dir)
                     -> exit loop on db faces at first valid face (condition 1)
                  */

                  // checking distance to face of down-wind building
                  if (std::abs(x_down) < 1.0 * WGD->dxy) {
                    // length of current face
                    segment_length = sqrt(pow(WGD->allBuildingsV[d_build]->polygonVertices[j_id + 1].x_poly
                                                - WGD->allBuildingsV[d_build]->polygonVertices[j_id].x_poly,
                                              2.0)
                                          + pow(WGD->allBuildingsV[d_build]->polygonVertices[j_id + 1].y_poly
                                                  - WGD->allBuildingsV[d_build]->polygonVertices[j_id].y_poly,
                                                2.0));
                    if (std::abs(y_down) <= 0.5 * segment_length) {
                      downwind_rel_dir = canyon_dir - facenormal_dir;
                      // forcing downwind_rel_dir to be in [-pi,pi] and check for round-off error
                      if (downwind_rel_dir > M_PI + 0.001) {
                        downwind_rel_dir -= 2.0 * M_PI;
                        if (abs(downwind_rel_dir) < 0.001) {
                          downwind_rel_dir = 0.0;
                        }
                      }
                      // forcing downwind_rel_dir to be in [-pi,pi] and check for round-off error
                      if (downwind_rel_dir <= -M_PI) {
                        downwind_rel_dir += 2.0 * M_PI;
                        if (abs(downwind_rel_dir) < 0.001) {
                          downwind_rel_dir = 0.0;
                        }
                      }
                      // checking relative wind direction (define reverse flow)
                      if (abs(downwind_rel_dir) < 0.5 * M_PI) {
                        // out-facing normal and wind at reference cell 'same' direction
                        reverse_flag = 1;
                        cross_dir = facenormal_dir + M_PI;
                      } else {
                        // out-facing normal and wind at reference cell 'opposite' direction
                        reverse_flag = 0;
                        cross_dir = facenormal_dir;
                      }
                      // define along direction
                      if (downwind_rel_dir >= 0.0) {
                        along_dir = facenormal_dir - 0.5 * M_PI;
                      } else {
                        along_dir = facenormal_dir + 0.5 * M_PI;
                      }

                      // continue if wall found not valid (avoid false detection of concave corner)
                      if (cos(facenormal_dir - perpendicular_dir[id]) > cos(angle_tol)) {
                        continue;
                      } else {
                        break;// exit LoopDB
                      }
                    }
                  }
                }

                // check angle between the face of current and downstream buildings.
                // angle need to be (-pi;-0.75*pi) or (0.75*pi;pi)
                if (cos(facenormal_dir - perpendicular_dir[id]) > cos(angle_tol)) {
                  canyon_flag = 0;
                  s = 0;
                  top_flag = 0;
                }

                /* FM old version where cross_dir depend on direction of the flow
		  if (reverse_flag == 1) {
                  // angle need to be (-0.25*pi;0.25*pi)
                  //if (cos(cross_dir - perpendicular_dir[id]) < -cos(angle_tol)) {
                  if (cos(cross_dir - perpendicular_dir[id]) > cos(angle_tol)) {
                    canyon_flag = 0;
                    s = 0;
                    top_flag = 0;
                  }
		  } else {
                  // angle need to be (-pi;-0.75*pi) or (0.75*pi;pi)
                  if (cos(cross_dir - perpendicular_dir[id]) > cos(angle_tol)) {
                    canyon_flag = 0;
                    s = 0;
                    top_flag = 0;
                  }
                } */
                break;// exit LoopX1
              }
            }
          }

          // forcing along_dir to be in [-pi,pi]
          if (cross_dir > M_PI + 0.001) {
            cross_dir -= 2.0 * M_PI;
          }
          if (cross_dir <= -M_PI) {
            cross_dir += 2.0 * M_PI;
          }

          // forcing along_dir to be in [-pi,pi]
          if (along_dir > M_PI + 0.001) {
            along_dir -= 2.0 * M_PI;
          }
          if (along_dir <= -M_PI) {
            along_dir += 2.0 * M_PI;
          }

          // std::cout << "along_dir:   " << along_dir << std::endl;
          if (canyon_flag == 1 && s > 0.9 * WGD->dxy) {
            // along velocity adjusted for height (assuming log profile) (WILL NOT WORK OVER TERRAIN)
            along_vel_mag = abs(velocity_mag * cos(canyon_dir - along_dir)) * log(WGD->z[k] / WGD->z0) / log(WGD->z[k_ref] / WGD->z0);
            cross_vel_mag = abs(velocity_mag * cos(canyon_dir - cross_dir));
            for (auto x_id = x_id_min; x_id <= x_id_max + 2; x_id++) {
              xc = 0.5 * x_id * WGD->dxy;

              int i = ceil(((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x - 0.001) / WGD->dx) - 1;
              int j = ceil(((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y - 0.001) / WGD->dy) - 1;

              //icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
              //if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

              /* u-velocity parameterization (face)
		i_u x-index of u-face (j does not need correction cell)
		x_p,y_p non-rotated relative to building center (u-face)
		x_u,y_u rotated relative to building center (u-face)
	      */
              i_u = std::round(((xc + x_wall) * cos(upwind_dir) - yc * sin(upwind_dir) + building_cent_x) / WGD->dx);
              x_p = i_u * WGD->dx - building_cent_x;
              y_p = (j + 0.5) * WGD->dy - building_cent_y;
              x_u = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
              y_u = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

              if (perpendicular_flag[id] == 0) {
                x_wall_u = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (y_u - yi[id]) + xi[id];
              } else {
                x_wall_u = xi[id];
              }
              x_pos = x_u - x_wall_u;
              //if (x_pos <= s + 0.001 && x_pos > -0.5 * WGD->dxy) {
              if (x_pos > -0.5 * WGD->dxy && x_pos <= s + 0.5 * WGD->dxy) {
                icell_face = i_u + j * WGD->nx + k * WGD->nx * WGD->ny;
                WGD->u0[icell_face] = along_vel_mag * cos(along_dir) + cross_vel_mag * (2 * x_pos / s) * 2 * (1 - x_pos / s) * cos(cross_dir);
              }
              // end of u-velocity parameterization

              /* v-velocity parameterization (face)
		j_v y-index of v-face (i does not need correction cell)
		x_p,y_p non-rotated relative to building center (v-face)
		x_u,y_u rotated relative to building center (u-face)
	      */
              j_v = std::round(((xc + x_wall) * sin(upwind_dir) + yc * cos(upwind_dir) + building_cent_y) / WGD->dy);
              x_p = (i + 0.5) * WGD->dx - building_cent_x;
              y_p = j_v * WGD->dy - building_cent_y;
              x_v = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
              y_v = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

              if (perpendicular_flag[id] == 0) {
                x_wall_v = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (y_v - yi[id]) + xi[id];
              } else {
                x_wall_v = xi[id];
              }
              x_pos = x_v - x_wall_v;
              //if (x_pos <= s + 0.001 && x_pos > -0.5 * WGD->dxy) {
              if (x_pos > -0.5 * WGD->dxy && x_pos <= s + 0.5 * WGD->dxy) {
                icell_face = i + j_v * WGD->nx + k * WGD->nx * WGD->ny;
                WGD->v0[icell_face] = along_vel_mag * sin(along_dir) + cross_vel_mag * (2 * x_pos / s) * 2 * (1 - x_pos / s) * sin(cross_dir);
              }
              // end of v-velocity parameterization

              /* w-velocity parameterization (face) and cellflag (cell)
		(i,j do not need correction cell)
		x_p,y_p non-rotated relative to building center (w-face/cell)
		x_w,y_w rotated relative to building center (w-face/cell)
	      */
              x_p = (i + 0.5) * WGD->dx - building_cent_x;
              y_p = (j + 0.5) * WGD->dy - building_cent_y;
              x_w = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
              y_w = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

              if (perpendicular_flag[id] == 0) {
                x_wall_w = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (y_w - yi[id]) + xi[id];
              } else {
                x_wall_w = xi[id];
              }
              x_pos = x_w - x_wall_w;

              //if (x_pos <= s + 0.001 && x_pos > -0.5 * WGD->dxy) {
              if (x_pos > -0.5 * WGD->dxy && x_pos <= s + 0.5 * WGD->dxy) {
                icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
                if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2
                    && WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] != 0
                    && WGD->icellflag[icell_cent - (WGD->nx - 1) * (WGD->ny - 1)] != 2) {
                  icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
                  if (reverse_flag == 0) {
                    // flow go up along front face - down along back face
                    WGD->w0[icell_face] = -abs(0.5 * cross_vel_mag * (1 - 2 * x_pos / s)) * (1 - 2 * (s - x_pos) / s);
                  } else {
                    // flow go down along front face - up along back face
                    WGD->w0[icell_face] = abs(0.5 * cross_vel_mag * (1 - 2 * x_pos / s)) * (1 - 2 * (s - x_pos) / s);
                  }
                }
                if ((WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                  WGD->icellflag[icell_cent] = 6;
                }
              }
              // end of w-velocity parameterization
            }
          }
        }
      }
    }
  }
}
