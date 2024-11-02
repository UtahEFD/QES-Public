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

/** @file UpwindCavity.cpp */

#include "PolyBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


/**
 *
 * This function applies the upwind cavity in front of the building to buildings defined as polygons.
 * This function reads in building features like nodes, building height and base height and uses
 * features of the building defined in the class constructor and setCellsFlag function. It defines
 * cells in the upwind area and applies the approperiate parameterization to them.
 * More information: "Improvements to a fast-response urban wind model, M. Nelson et al. (2008)"
 *
 * @param WID :document this:
 * @param WGD :document this:
 */
void PolyBuilding::upwindCavity(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  float tol = 10.0 * M_PI / 180.0;// Upwind cavity is applied if the angle
                                  // is in 10 degree of the perpendicular direction

  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();
  float dxy = WGD->domain.dxy();

  float retarding_factor = 0.4;// In the outer region, velocities are reduced by 40% (Bagal et al. (2004))
  float height_factor = 0.6;// Height of both elipsoids (inner and outer) extends to 0.6H
  float length_factor = 0.4;
  int k_top(1);// Index of top of the upwind area
  float uh_rotation, vh_rotation;// Velocity components at the height of the building after rotation
  float vortex_height;// Height of the vortex
  float retarding_height;// Height of the obstacle causes retardation in flow
  float x_average, y_average;// Average x and y locations of face
  int upwind_i_start, upwind_i_end, upwind_j_start, upwind_j_end;// Start and end indices of upwind area
  float z_front;// Vertical location inside upwind area
  float x_u, y_u, x_v, y_v, x_w, y_w;// x and y locations of parameterized velocity components
  float x_intersect_u, x_ellipse_u, xrz_u;
  float x_intersect_v, x_ellipse_v, xrz_v;
  float x_intersect_w, x_ellipse_w;
  float rz_end;
  float u_rotation, v_rotation;// Velocity components in rotated coordinates
  long icell_face, icell_cent;
  std::vector<float> perpendicular_dir;// Perpendicular angle to each face
  std::vector<float> effective_gamma;// Effective angle of each face in rotated coordinates
  std::vector<float> face_length;// Length of each face
  std::vector<float> Lf_face;// Length of upwind cavity for each face

  upwind_rel_dir.resize(polygonVertices.size(), 0.0);// Upwind reletive direction for each face
  xf1.resize(polygonVertices.size(), 0.0);
  xf2.resize(polygonVertices.size(), 0.0);
  yf1.resize(polygonVertices.size(), 0.0);
  yf2.resize(polygonVertices.size(), 0.0);
  perpendicular_dir.resize(polygonVertices.size(), 0.0);
  effective_gamma.resize(polygonVertices.size(), 0.0);
  int counter = 0;

  long index_building_face = WGD->domain.face(i_building_cent, j_building_cent, k_end);
  u0_h = WGD->u0[index_building_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[index_building_face];// v velocity at the height of building at the centroid
  // Wind direction of initial velocity at the height of building at the centroid
  upwind_dir = atan2(v0_h, u0_h);
  for (auto id = 0u; id < polygonVertices.size() - 1; id++) {
    xf1[id] = 0.5 * (polygonVertices[id].x_poly - polygonVertices[id + 1].x_poly) * cos(upwind_dir)
              + 0.5 * (polygonVertices[id].y_poly - polygonVertices[id + 1].y_poly) * sin(upwind_dir);
    yf1[id] = -0.5 * (polygonVertices[id].x_poly - polygonVertices[id + 1].x_poly) * sin(upwind_dir)
              + 0.5 * (polygonVertices[id].y_poly - polygonVertices[id + 1].y_poly) * cos(upwind_dir);
    xf2[id] = 0.5 * (polygonVertices[id + 1].x_poly - polygonVertices[id].x_poly) * cos(upwind_dir)
              + 0.5 * (polygonVertices[id + 1].y_poly - polygonVertices[id].y_poly) * sin(upwind_dir);
    yf2[id] = -0.5 * (polygonVertices[id + 1].x_poly - polygonVertices[id].x_poly) * sin(upwind_dir)
              + 0.5 * (polygonVertices[id + 1].y_poly - polygonVertices[id].y_poly) * cos(upwind_dir);

    // Calculate upwind reletive direction for each face
    upwind_rel_dir[id] = atan2(yf2[id] - yf1[id], xf2[id] - xf1[id]) + 0.5 * M_PI;
    if (upwind_rel_dir[id] > M_PI + 0.0001) {
      upwind_rel_dir[id] -= 2 * M_PI;
    }

    if (abs(upwind_rel_dir[id]) > M_PI - tol) {
      perpendicular_dir[id] = atan2(polygonVertices[id + 1].y_poly - polygonVertices[id].y_poly,
                                    polygonVertices[id + 1].x_poly - polygonVertices[id].x_poly)
                              + 0.5 * M_PI;
      if (perpendicular_dir[id] <= -M_PI) {
        perpendicular_dir[id] += 2 * M_PI;
      }

      if (perpendicular_dir[id] >= 0.75 * M_PI) {
        effective_gamma[id] = perpendicular_dir[id] - M_PI;
      } else if (perpendicular_dir[id] >= 0.25 * M_PI) {
        effective_gamma[id] = perpendicular_dir[id] - 0.5 * M_PI;
      } else if (perpendicular_dir[id] < -0.75 * M_PI) {
        effective_gamma[id] = perpendicular_dir[id] + M_PI;
      } else if (perpendicular_dir[id] < -0.25 * M_PI) {
        effective_gamma[id] = perpendicular_dir[id] + 0.5 * M_PI;
      }
      uh_rotation = u0_h * cos(effective_gamma[id]) + v0_h * sin(effective_gamma[id]);
      vh_rotation = -u0_h * sin(effective_gamma[id]) + v0_h * cos(effective_gamma[id]);
      face_length.push_back(sqrt(pow(xf2[id] - xf1[id], 2.0) + pow(yf2[id] - yf1[id], 2.0)));
      Lf_face.push_back(abs(WGD->lengthf_coeff * face_length[counter] * cos(upwind_rel_dir[id]) / (1 + 0.8 * face_length[counter] / H)));

      // High-rise Modified Vortex Parameterization (HMVP) (Bagal et al. (2004))
      if (WID->buildingsParams->upwindCavityFlag == 3) {
        vortex_height = MIN_S(face_length[counter], H);
        retarding_height = H;
      } else {
        vortex_height = H;
        retarding_height = H;
      }

      // Defining index related to the height that upwind cavity is being applied
      for (size_t k = k_start; k < WGD->domain.z.size(); k++) {
        k_top = k + 1;
        if (height_factor * retarding_height + base_height <= WGD->domain.z[k]) {
          break;
        }
      }
      // Defining limits of the upwind cavity in x and y directions
      upwind_i_start = MAX_S(std::round(MIN_S(polygonVertices[id].x_poly, polygonVertices[id + 1].x_poly) / dx)
                               - std::round(1.5 * Lf_face[counter] / dx) - 1,
                             1);
      upwind_i_end = MIN_S(std::round(MAX_S(polygonVertices[id].x_poly, polygonVertices[id + 1].x_poly) / dx)
                             + std::round(1.5 * Lf_face[counter] / dx),
                           nx - 2);
      upwind_j_start = MAX_S(std::round(MIN_S(polygonVertices[id].y_poly, polygonVertices[id + 1].y_poly) / dy)
                               - std::round(1.5 * Lf_face[counter] / dy) - 1,
                             1);
      upwind_j_end = MIN_S(std::round(MAX_S(polygonVertices[id].y_poly, polygonVertices[id + 1].y_poly) / dy)
                             + std::round(1.5 * Lf_face[counter] / dy),
                           ny - 2);

      x_average = 0.5 * (polygonVertices[id].x_poly + polygonVertices[id + 1].x_poly);// x-location of middle of the face
      y_average = 0.5 * (polygonVertices[id].y_poly + polygonVertices[id + 1].y_poly);// y-location of middle of the face

      // Apply the upwind parameterization
      for (auto k = k_start; k < k_top; k++) {
        z_front = WGD->domain.z[k] - base_height;// Height from the base of the building
        for (auto j = upwind_j_start; j < upwind_j_end; j++) {
          for (auto i = upwind_i_start; i < upwind_i_end; i++) {
            x_u = (i * dx - x_average) * cos(upwind_dir) + ((j + 0.5) * dy - y_average) * sin(upwind_dir);// x-location of u velocity
            y_u = -(i * dx - x_average) * sin(upwind_dir) + ((j + 0.5) * dy - y_average) * cos(upwind_dir);// y-location of u velocity
            x_v = ((i + 0.5) * dx - x_average) * cos(upwind_dir) + (j * dy - y_average) * sin(upwind_dir);// x-location of v velocity
            y_v = -((i + 0.5) * dx - x_average) * sin(upwind_dir) + (j * dy - y_average) * cos(upwind_dir);// y-location of v velocity
            x_w = ((i + 0.5) * dx - x_average) * cos(upwind_dir) + ((j + 0.5) * dy - y_average) * sin(upwind_dir);// x-location of w velocity
            y_w = -((i + 0.5) * dx - x_average) * sin(upwind_dir) + ((j + 0.5) * dy - y_average) * cos(upwind_dir);// y-location of w velocity

            if ((abs(y_u) <= abs(yf2[id])) && (height_factor * vortex_height > z_front)) {
              // Intersection of a parallel line to the line goes through (xf1,yf1) and (xf2,yf2) and the y=y_u
              x_intersect_u = ((xf2[id] - xf1[id]) / (yf2[id] - yf1[id])) * (y_u - yf1[id]) + xf1[id];
              x_ellipse_u = -Lf_face[counter] * sqrt((1 - pow(y_u / abs(yf2[id]), 2.0)) * (1 - pow(z_front / (height_factor * vortex_height), 2.0)));
              xrz_u = -Lf_face[counter] * sqrt((1 - pow(y_u / abs(yf2[id]), 2.0)) * (1 - pow(z_front / (height_factor * retarding_height), 2.0)));
              rz_end = length_factor * x_ellipse_u;
              icell_cent = WGD->domain.cell(i, j, k);
              icell_face = WGD->domain.face(i, j, k);
              if (WID->buildingsParams->upwindCavityFlag == 1)// Rockle parameterization
              {
                if ((x_u - x_intersect_u >= x_ellipse_u) && (x_u - x_intersect_u <= 0.1 * dxy)
                    && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  WGD->u0[icell_face] = 0.0;
                }
              } else {
                if ((x_u - x_intersect_u >= xrz_u) && (x_u - x_intersect_u < rz_end)
                    && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  if (WID->buildingsParams->upwindCavityFlag == 3)// High-rise Modified Vortex Parameterization (HMVP) (Bagal et al. (2004))
                  {
                    WGD->u0[icell_face] *= ((x_u - x_intersect_u - xrz_u) * (retarding_factor - 1.0) / (rz_end - xrz_u) + 1.0);
                  } else// Modified Vortex Parameterization (MVP)
                  {
                    WGD->u0[icell_face] *= retarding_factor;
                  }
                }
                if ((x_u - x_intersect_u >= length_factor * x_ellipse_u) && (x_u - x_intersect_u <= 0.1 * dxy)
                    && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  u_rotation = WGD->u0[icell_face] * cos(effective_gamma[id]);
                  v_rotation = WGD->u0[icell_face] * sin(effective_gamma[id]);
                  if (abs(perpendicular_dir[id]) >= 0.25 * M_PI && abs(perpendicular_dir[id]) <= 0.75 * M_PI) {
                    v_rotation = -vh_rotation * (-height_factor * cos(M_PI * z_front / (0.5 * vortex_height)) + 0.05)
                                 * (-height_factor * sin(M_PI * abs(x_u - x_intersect_u) / (length_factor * Lf_face[counter])));
                  } else {
                    u_rotation = -uh_rotation * (-height_factor * cos(M_PI * z_front / (0.5 * vortex_height)) + 0.05)
                                 * (-height_factor * sin(M_PI * abs(x_u - x_intersect_u) / (length_factor * Lf_face[counter])));
                  }
                  WGD->u0[icell_face] = u_rotation * cos(-effective_gamma[id]) + v_rotation * sin(-effective_gamma[id]);
                }
              }
            }
            // v velocity
            if ((abs(y_v) <= abs(yf2[id])) && (height_factor * vortex_height > z_front)) {
              x_intersect_v = ((xf2[id] - xf1[id]) / (yf2[id] - yf1[id])) * (y_v - yf1[id]) + xf1[id];
              x_ellipse_v = -Lf_face[counter] * sqrt((1 - pow(y_v / abs(yf2[id]), 2.0)) * (1 - pow(z_front / (height_factor * vortex_height), 2.0)));
              xrz_v = -Lf_face[counter] * sqrt((1 - pow(y_v / abs(yf2[id]), 2.0)) * (1 - pow(z_front / (height_factor * retarding_height), 2.0)));
              rz_end = length_factor * x_ellipse_v;
              icell_cent = WGD->domain.cell(i, j, k);
              icell_face = WGD->domain.face(i, j, k);
              if (WID->buildingsParams->upwindCavityFlag == 1)// Rockle parameterization
              {
                if ((x_v - x_intersect_v >= x_ellipse_v) && (x_v - x_intersect_v <= 0.1 * dxy)
                    && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  WGD->v0[icell_face] = 0.0;
                }
              } else {
                if ((x_v - x_intersect_v >= xrz_v) && (x_v - x_intersect_v < rz_end)
                    && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  if (WID->buildingsParams->upwindCavityFlag == 3)// High-rise Modified Vortex Parameterization (HMVP) (Bagal et al. (2004))
                  {
                    WGD->v0[icell_face] *= ((x_v - x_intersect_v - xrz_v) * (retarding_factor - 1.0) / (rz_end - xrz_v) + 1.0);
                  } else// Modified Vortex Parameterization (MVP)
                  {
                    WGD->v0[icell_face] *= retarding_factor;
                  }
                }
                if ((x_v - x_intersect_v >= length_factor * x_ellipse_v) && (x_v - x_intersect_v <= 0.1 * dxy)
                    && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  u_rotation = WGD->v0[icell_face] * sin(effective_gamma[id]);
                  v_rotation = WGD->v0[icell_face] * cos(effective_gamma[id]);
                  if (abs(perpendicular_dir[id]) >= 0.25 * M_PI && abs(perpendicular_dir[id]) <= 0.75 * M_PI) {
                    v_rotation = -vh_rotation * (-height_factor * cos(M_PI * z_front / (0.5 * vortex_height)) + 0.05)
                                 * (-height_factor * sin(M_PI * abs(x_v - x_intersect_v) / (length_factor * Lf_face[counter])));
                  } else {
                    u_rotation = -uh_rotation * (-height_factor * cos(M_PI * z_front / (0.5 * vortex_height)) + 0.05)
                                 * (-height_factor * sin(M_PI * abs(x_v - x_intersect_v) / (length_factor * Lf_face[counter])));
                  }
                  WGD->v0[icell_face] = -u_rotation * sin(-effective_gamma[id]) + v_rotation * cos(-effective_gamma[id]);
                }
              }
            }

            // w velocity
            if ((abs(y_w) <= abs(yf2[id])) && (height_factor * vortex_height > z_front)) {
              x_intersect_w = ((xf2[id] - xf1[id]) / (yf2[id] - yf1[id])) * (y_w - yf1[id]) + xf1[id];
              x_ellipse_w = -Lf_face[counter] * sqrt((1 - pow(y_w / abs(yf2[id]), 2.0)) * (1 - pow(z_front / (height_factor * vortex_height), 2.0)));
              icell_cent = WGD->domain.cell(i, j, k);
              icell_face = WGD->domain.face(i, j, k);
              if (WID->buildingsParams->upwindCavityFlag == 1)// Rockle parameterization
              {
                if ((x_w - x_intersect_w >= x_ellipse_w) && (x_w - x_intersect_w <= 0.1 * dxy)
                    && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  WGD->w0[icell_face] = 0.0;
                  if (i < nx - 1 && j < ny - 1 && k < nz - 2
                      && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                    WGD->icellflag[icell_cent] = 3;
                  }
                }
              } else {
                if ((x_w - x_intersect_w >= x_ellipse_w) && (x_w - x_intersect_w < length_factor * x_ellipse_w)
                    && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  WGD->w0[icell_face] *= retarding_factor;
                  if (i < nx - 1 && j < ny - 1 && k < nz - 2
                      && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                    WGD->icellflag[icell_cent] = 3;
                  }
                }
                if ((x_w - x_intersect_w >= length_factor * x_ellipse_w)
                    && (x_w - x_intersect_w <= 0.1 * dxy) && (WGD->icellflag[icell_cent] != 0) && (WGD->icellflag[icell_cent] != 2)) {
                  WGD->w0[icell_face] = -sqrt(pow(u0_h, 2.0) + pow(v0_h, 2.0))
                                        * (0.1 * cos(M_PI * abs(x_w - x_intersect_w) / (length_factor * Lf_face[counter])) - 0.05);
                  if (i < nx - 1 && j < ny - 1 && k < nz - 2
                      && (WGD->icellflag[icell_cent] != 7) && (WGD->icellflag[icell_cent] != 8)) {
                    WGD->icellflag[icell_cent] = 3;
                  }
                }
              }
            }
          }
        }
      }
      counter += 1;
    }
  }
}
