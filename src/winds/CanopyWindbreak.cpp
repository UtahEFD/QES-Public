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

/** @file CanopyWindbreak.cpp */

#include "CanopyWindbreak.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

// set et attenuation coefficient
void CanopyWindbreak::setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int canopy_id)
{
  // Aerodyamic Porosity Models
  if (wbModel == 1) {
    // Guan 2D model porosity profile
    a_obf = beta;
  } else {
    // Guan real windbreak aerodynamic porosity profile
    a_obf = pow(beta, 0.4);
  }

  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  // this function need to be called to defined the boundary of the canopy and the icellflags
  float ray_intersect;
  unsigned int num_crossing, vert_id, start_poly;

  // Find out which cells are going to be inside the polygone
  // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
  // Check the center of each cell, if it's inside, set that cell to building
  for (auto j = j_start; j < j_end; j++) {
    // Center of cell y coordinate
    float y_cent = (j + 0.5) * dy;
    for (auto i = i_start; i < i_end; i++) {
      float x_cent = (i + 0.5) * dx;
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
        int icell_2d = WGD->domain.cell2d(i, j);

        if (WGD->icellflag_footprint[icell_2d] == 0) {
          // a  building exist here -> skip
        } else {
          // save the (x,y) location of the canopy
          canopy_cell2D.push_back(icell_2d);
          // set the footprint array for canopy
          WGD->icellflag_footprint[icell_2d] = getCellFlagCanopy();

          // Define start index of the canopy in z-direction
          for (size_t k = 1u; k < WGD->domain.z.size(); k++) {
            if (WGD->terrain[icell_2d] + base_height <= WGD->domain.z[k]) {
              WGD->canopy->canopy_bot_index[icell_2d] = k;
              WGD->canopy->canopy_bot[icell_2d] = WGD->terrain[icell_2d] + base_height;
              WGD->canopy->canopy_base[icell_2d] = WGD->domain.z_face[k];
              break;
            }
          }

          // Define end index of the canopy in z-direction
          for (size_t k = 0u; k < WGD->domain.z.size(); k++) {
            if (WGD->terrain[icell_2d] + H < WGD->domain.z[k + 1]) {
              WGD->canopy->canopy_top_index[icell_2d] = k + 1;
              WGD->canopy->canopy_top[icell_2d] = WGD->terrain[icell_2d] + H;
              break;
            }
          }

          // Define the height of the canopy
          WGD->canopy->canopy_height[icell_2d] = WGD->canopy->canopy_top[icell_2d] - WGD->canopy->canopy_bot[icell_2d];

          // define icellflag @ (x,y) for all z(k) in [k_start...k_end]
          for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < WGD->canopy->canopy_top_index[icell_2d]; k++) {
            long icell_3d = WGD->domain.cell(i, j, k);
            if (WGD->icellflag[icell_3d] != 0 && WGD->icellflag[icell_3d] != 2) {
              // Canopy cell
              WGD->icellflag[icell_3d] = getCellFlagCanopy();
              // WGD->canopy->canopy_atten_coeff[icell_3d] = attenuationCoeff;
              WGD->canopy->icanopy_flag[icell_3d] = canopy_id;
              canopy_cell3D.push_back(icell_3d);
            }
          }
        }// end define icellflag!
      }
    }
  }

  // check if the canopy is well defined
  if (canopy_cell2D.empty()) {
    k_start = 0;
    k_end = 0;
  } else {
    k_start = nz - 1;
    k_end = 0;
    for (size_t k = 0u; k < canopy_cell2D.size(); k++) {
      if (WGD->canopy->canopy_bot_index[canopy_cell2D[k]] < k_start)
        k_start = WGD->canopy->canopy_bot_index[canopy_cell2D[k]];
      if (WGD->canopy->canopy_top_index[canopy_cell2D[k]] > k_end)
        k_end = WGD->canopy->canopy_top_index[canopy_cell2D[k]];
    }
  }

  // check of illegal definition.
  if (k_start > k_end) {
    std::cerr << "ERROR in tree definition (k_start > k end)" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (ceil(1.5 * k_end) > nz - 1) {
    std::cerr << "ERROR domain too short for tree method" << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}


void CanopyWindbreak::canopyVegetation(WINDSGeneralData *WGD, int building_id)
{

  std::map<int, float> u0_modified, v0_modified;
  long icell_face = WGD->domain.face(i_building_cent, j_building_cent, k_end);
  u0_h = WGD->u0[icell_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[icell_face];// v velocity at the height of building at the centroid

  upwind_dir = atan2(v0_h, u0_h);

  // apply canopy parameterization
  for (auto icell_2d : canopy_cell2D) {
    auto [i, j, k] = WGD->domain.getCellIdx(icell_2d);

    // base of the canopy
    float z_b = WGD->domain.z[WGD->terrain_id[icell_2d]];
    for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < WGD->canopy->canopy_top_index[icell_2d]; ++k) {
      icell_face = WGD->domain.face(i, j, k);

      if ((WGD->domain.z_face[k] - z_b >= understory_height) && (WGD->domain.z_face[k] - z_b <= H)) {
        // adding modified velocity to the list of node to modifiy
        /*
            this method avoid double appication of the param.
            especially important when the param multiply the velocity at the
            current location!
            example: u0[]*=(1-p)
          */
        // all face of the cell i=icell_face & i+1 = icell_face+1
        // u0_mod_id.push_back(icell_face);
        u0_modified[icell_face] = a_obf;
        u0_modified[WGD->domain.faceAdd(icell_face, 1, 0, 0)] = a_obf;

        // all face of the cell j=icell_face & j+1 = icell_face+nx
        v0_modified[icell_face] = a_obf;
        v0_modified[WGD->domain.faceAdd(icell_face, 0, 1, 0)] = a_obf;
      }
    }
  }

  // apply the parameterization (only once per cell/face!)
  for (auto const &m : u0_modified)
    WGD->u0[m.first] *= m.second;

  for (auto const &m : v0_modified)
    WGD->v0[m.first] *= m.second;

  // clear memory
  u0_modified.clear();
  v0_modified.clear();

  return;
}

void CanopyWindbreak::canopyWake(WINDSGeneralData *WGD, int building_id)
{
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();
  float dxy = WGD->domain.dxy();

  const float tol = 0.01 * M_PI / 180.0;
  const float epsilon = 10e-10;

  const float wake_shear_coef = 7.5;
  const float wake_recov_coef = 4;
  const float wake_stream_coef = wake_shear_coef + wake_recov_coef;

  //*** spreading parameters ***!
  float udelt = (1 - a_obf);// udelt parameters
  float uave = 0.5 * (1 + a_obf);// uave  parameters
  float spreadclassicmix = 0.14 * udelt / uave;// classic mixing layer
  float spreadupstream = 0.0;// upstream spread rate
  float spreadrate = 0.0;// sum of two spreading models

  /* FM - OBSOLETE (using tanh to blend recovery zone)
     Perrera 1981 contains a typo "K=(2.0*k^2)/(ln(H-d)/zo)" where k is von Karman's constant.
     The correct form and the one used in this line is found in Counihan et al.
     "Wakes behind two-dimensional surface obstacles in turbulent boundary layers"
     J. Fluid Mech. (1974) vol 64, part 3, Eq. 2.19b on page 536
  */
  // const float n = 0.1429;
  // float zo = 0.1;
  // float K = (2.0 * 0.4 * 0.4) / log((H - d) / zo);
  // float xbar, eta, recovery_factor;

  // float z_b;
  float x_c, y_c, z_c, x1(0), x2(0), y1(0), y2(0), y_norm;
  float x_p, y_p, x_u, y_u, x_v, y_v, x_w, y_w;
  float x_wall, x_wall_u, x_wall_v, x_wall_w, dn_u, dn_v, dn_w;
  int u_wake_flag(0), v_wake_flag(0), w_wake_flag(0);

  // velocity at top of windbreak
  float u0_wh, v0_wh, w0_wh, umag0_wh;
  // shear zone orig. and spread
  float zwmo, d_shearzone, ds;

  // float u_defect,u_c,r_center,theta,delta,B_h;
  // float ustar_wake(0),ustar_us(0),mag_us(0);


  // int k_bottom(1), k_top(WGD->nz - 2);
  // int icell_canopy_2d;

  std::vector<float> u0_modified, v0_modified;
  std::vector<int> u0_mod_id, v0_mod_id;

  std::vector<float> xi, yi, upwind_rel_dir;
  std::vector<bool> wake_flag, perpendicular_flag;
  xi.resize(polygonVertices.size(), 0.0);// Difference of x values of the centroid and each node
  yi.resize(polygonVertices.size(), 0.0);// Difference of y values of the centroid and each node
  upwind_rel_dir.resize(polygonVertices.size(), 0.0);
  wake_flag.resize(polygonVertices.size(), false);
  perpendicular_flag.resize(polygonVertices.size(), false);

  // float Lt=0.5*W;
  Lr = H;

  long icell_face = WGD->domain.face(i_building_cent, j_building_cent, k_end);
  u0_h = WGD->u0[icell_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[icell_face];// v velocity at the height of building at the centroid

  upwind_dir = atan2(v0_h, u0_h);
  // float upwind_mag=sqrt(u0_h*u0_h + v0_h*v0_h);

  for (auto id = 0u; id < polygonVertices.size(); id++) {
    xi[id] = (polygonVertices[id].x_poly - building_cent_x) * cos(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * sin(upwind_dir);
    yi[id] = -(polygonVertices[id].x_poly - building_cent_x) * sin(upwind_dir) + (polygonVertices[id].y_poly - building_cent_y) * cos(upwind_dir);
  }
  int stop_id = 0;
  for (auto id = 0u; id < polygonVertices.size() - 1; id++) {
    // Calculate upwind reletive direction for each face
    upwind_rel_dir[id] = atan2(yi[id + 1] - yi[id], xi[id + 1] - xi[id]) + 0.5 * M_PI;
    if (upwind_rel_dir[id] > M_PI + 0.0001)
      upwind_rel_dir[id] -= 2 * M_PI;

    if (abs(upwind_rel_dir[id]) < tol)
      perpendicular_flag[id] = true;

    // Calculate length of the far wake zone for each face
    if ((abs(upwind_rel_dir[id]) < 0.5 * M_PI - tol) && (id % 2 == 0))
      wake_flag[id] = true;

    if (xi[id] < x1)
      x1 = xi[id];// Minimum x
    if (xi[id] > x2)
      x2 = xi[id];// Maximum x

    if (yi[id] < y1)
      y1 = yi[id];// Minimum y
    if (yi[id] > y2)
      y2 = yi[id];// Maximum y

    // calculate last id, ie, come back to itself
    if ((polygonVertices[id + 1].x_poly > polygonVertices[0].x_poly - 0.1)
        && (polygonVertices[id + 1].x_poly < polygonVertices[0].x_poly + 0.1)
        && (polygonVertices[id + 1].y_poly > polygonVertices[0].y_poly - 0.1)
        && (polygonVertices[id + 1].y_poly < polygonVertices[0].y_poly + 0.1)) {
      stop_id = id;
      break;
    }
  }

  for (auto id = 0; id <= stop_id; id++) {
    if (wake_flag[id]) {
      for (auto y_id = 0; y_id <= 2 * ceil(abs(yi[id] - yi[id + 1]) / dxy); y_id++) {

        // y-coord relative to center of the windbreak (left to right)
        y_c = yi[id] - 0.5 * y_id * dxy;

        if (perpendicular_flag[id]) {
          x_wall = xi[id];
        } else {
          x_wall = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (y_c - yi[id]) + xi[id];
        }
        if (y_c >= 0.0) {
          y_norm = y2;
        } else {
          y_norm = y1;
        }

        // ij index of cell just inside the rear wall
        int i = ((x_wall - dxy) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / dx;
        int j = ((x_wall - dxy) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / dy;

        if (i >= nx - 2 && i <= 0 && j >= ny - 2 && j <= 0)
          break;

        for (auto z_id = 5.0 * H / dz; z_id > 0; z_id--) {

          int x_id_min = -1;
          for (auto x_id = 1; x_id <= 2 * ceil(wake_stream_coef * Lr / dxy); x_id++) {
            // reset stop flag
            u_wake_flag = 1;
            v_wake_flag = 1;
            w_wake_flag = 1;

            // x-coord relative to center of the windbreak (downstream)
            x_c = 0.5 * x_id * dxy;

            int i = ceil(((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / dx) - 1;
            int j = ceil(((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / dy) - 1;

            int k = z_id + WGD->terrain_id[WGD->domain.cell2d(i, j)] - 1;
            int k_top = H / dz + WGD->terrain_id[WGD->domain.cell2d(i, j)];

            // z-coord relative to the base of the building.
            z_c = WGD->domain.z[k] - WGD->domain.z_face[WGD->terrain_id[WGD->domain.cell2d(i, j)]];

            if (i >= nx - 2 && i <= 0 && j >= ny - 2 && j <= 0)
              break;

            // adjustement of center of the shear zone
            icell_face = WGD->domain.face(i, j, k_top);

            u0_wh = WGD->u0[icell_face];
            v0_wh = WGD->v0[icell_face];
            w0_wh = WGD->w0[icell_face];
            umag0_wh = sqrt(u0_wh * u0_wh + v0_wh * v0_wh);

            if (umag0_wh != 0.0)
              zwmo = H + w0_wh / umag0_wh * dx;
            else
              zwmo = H;

            // upstream spread rate
            spreadupstream = 2.0 * stdw / umag0_wh;
            // sum of two spreading models
            spreadrate = sqrt(spreadclassicmix * spreadclassicmix + spreadupstream * spreadupstream);
            // size of the shear zone
            d_shearzone = spreadrate * x_c;

            // linearized indexes
            icell_cent = WGD->domain.cell(i, j, k);

            // check if not in canopy/building set start (was x_idx_min < 0) to x_idx_min > 0
            if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && x_id_min < 0)
              x_id_min = x_id;

            if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2 || WGD->icellflag[icell_cent] == getCellFlagCanopy()) {
              // check for canopy/building/terrain that will disrupt the wake
              if (x_id_min >= 0) {
                if (WGD->ibuilding_flag[icell_cent] == building_id) {
                  x_id_min = -1;
                } else if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2) {
                  // break if run into building or terrain
                  break;
                }
              }
            }

            if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2
                && WGD->icellflag[icell_cent] != getCellFlagCanopy()) {
              // START OF WAKE VELOCITY PARAMETRIZATION

              // wake u-values
              // ij coord of u-face
              int i_u = std::round(((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / dx);
              int j_u = ((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / dy;
              if (i_u < nx - 1 && i_u > 0 && j_u < ny - 1 && j_u > 0) {
                // not rotated relative coordinate of u-face
                x_p = i_u * dx - building_cent_x;
                y_p = (j_u + 0.5) * dy - building_cent_y;
                // rotated relative coordinate of u-face
                x_u = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
                y_u = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

                // if(std::abs(y_u) > std::abs(y_norm))
                // break;

                if (perpendicular_flag[id]) {
                  x_wall_u = xi[id];
                } else {
                  x_wall_u = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (y_u - yi[id]) + xi[id];
                }

                // adjusted downstream value
                x_u -= x_wall_u;

                if (std::abs(y_u) < std::abs(y_norm) && std::abs(y_norm) > epsilon && H > epsilon) {
                  dn_u = H;
                } else {
                  dn_u = 0.0;
                }

                if (x_u > wake_stream_coef * dn_u)
                  u_wake_flag = 0;

                // linearized indexes
                icell_cent = WGD->domain.cell(i_u, j_u, k);
                icell_face = WGD->domain.face(i_u, j_u, k);

                if (dn_u > 0.0 && u_wake_flag == 1 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

                  // x_u,y_u,z_c
                  float aeropor = (0.5 * (1.0 - a_obf)) * tanh(1.5 * (z_c - zwmo) / d_shearzone) + 0.5 * (1.0 + a_obf);
                  float recovery = 0.5 - 0.5 * tanh((x_u - wake_shear_coef * Lr - 0.5 * wake_recov_coef * Lr) / (0.25 * wake_recov_coef * Lr));
                  WGD->canopy->wake_u_defect[icell_face] = (1.0 - aeropor) * recovery;

                  // latteral shear zone
                  ds = 0.5 * spreadclassicmix * x_u;
                  WGD->canopy->wake_u_defect[icell_face] *= (0.5 - 0.5 * tanh(1.5 * (y_u - (y2 - ds)) / ds));
                  WGD->canopy->wake_u_defect[icell_face] *= (0.5 - 0.5 * tanh(1.5 * ((y1 + ds) - y_u) / ds));
                }
              }

              // wake v-values
              // ij coord of v-face
              int i_v = ((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / dx;
              int j_v = std::round(((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / dy);
              if (i_v < nx - 1 && i_v > 0 && j_v < ny - 1 && j_v > 0) {
                // not rotated relative coordinate of v-face
                x_p = (i_v + 0.5) * dx - building_cent_x;
                y_p = j_v * dy - building_cent_y;
                // rotated relative coordinate of u-face
                x_v = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
                y_v = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

                if (perpendicular_flag[id]) {
                  x_wall_v = xi[id];
                } else {
                  x_wall_v = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (y_v - yi[id]) + xi[id];
                }

                // adjusted downstream value
                x_v -= x_wall_v;

                if (std::abs(y_v) < std::abs(y_norm) && std::abs(y_norm) > epsilon && H > epsilon) {
                  dn_v = H;
                } else {
                  dn_v = 0.0;
                }

                if (x_v > wake_stream_coef * dn_v)
                  v_wake_flag = 0;

                // linearized indexes
                icell_cent = WGD->domain.cell(i_v, j_v, k);
                icell_face = WGD->domain.face(i_v, j_v, k);

                if (dn_v > 0.0 && v_wake_flag == 1 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

                  // x_v,y_v,z_c
                  float aeropor = (0.5 * (1.0 - a_obf)) * tanh(1.5 / (d_shearzone) * (z_c)-1.5 / (d_shearzone)*zwmo) + 0.5 * (1.0 + a_obf);
                  float recovery = 0.5 - 0.5 * tanh((x_v - wake_shear_coef * Lr - 0.5 * wake_recov_coef * Lr) / (0.25 * wake_recov_coef * Lr));

                  WGD->canopy->wake_v_defect[icell_face] = (1.0 - aeropor) * recovery;

                  // latteral shear zone
                  ds = 0.5 * spreadclassicmix * x_v;
                  WGD->canopy->wake_v_defect[icell_face] *= (-0.5 * tanh(1.5 * (y_v - (y2 - ds)) / ds) + 0.5);
                  WGD->canopy->wake_v_defect[icell_face] *= (-0.5 * tanh(1.5 * ((y1 + ds) - y_v) / ds) + 0.5);
                }
              }


              // wake celltype w-values
              // ij coord of cell-center
              int i_w = ceil(((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / dx) - 1;
              int j_w = ceil(((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / dy) - 1;

              if (i_w < nx - 1 && i_w > 0 && j_w < ny - 1 && j_w > 0) {
                // not rotated relative coordinate of cell-center
                x_p = (i_w + 0.5) * dx - building_cent_x;
                y_p = (j_w + 0.5) * dy - building_cent_y;
                // rotated relative coordinate of cell-center
                x_w = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
                y_w = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

                // if(std::abs(y_w) > std::abs(y_norm))
                //     break;

                if (perpendicular_flag[id]) {
                  x_wall_w = xi[id];
                } else {
                  x_wall_w = ((xi[id + 1] - xi[id]) / (yi[id + 1] - yi[id])) * (y_w - yi[id]) + xi[id];
                }

                // adjusted downstream value
                x_w -= x_wall_w;

                if (std::abs(y_w) < std::abs(y_norm) && std::abs(y_norm) > epsilon && H > epsilon) {
                  dn_w = H;
                } else {
                  dn_w = 0.0;
                }

                if (x_w > wake_stream_coef * dn_w)
                  w_wake_flag = 0;

                // linearized indexes
                icell_cent = WGD->domain.cell(i_w, j_w, k);

                if (dn_w > 0.0 && w_wake_flag == 1 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

                  WGD->icellflag[icell_cent] = getCellFlagWake();
                }
              }
              // if u,v, and w are done -> exit x-loop
              if (u_wake_flag == 0 && v_wake_flag == 0 && w_wake_flag == 0)
                break;
              // END OF WAKE VELOCITY PARAMETRIZATION
            }
          }
        }
      }
    }
  }
}
