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

/**
 * @file CutBuilding.cpp
 * @brief Designed for applying the cut-cell method to the general building shape (polygons).
 *
 * In this class, the cut-cell method to the polygon buildings.
 *
 */

#include "CutBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "PolyBuilding.h"


/**
 *
 * This function applies the cut-cell method to buildings.
 *
 */
void CutBuilding::setCutCellFlags(WINDSGeneralData *WGD, const PolyBuilding *PolyB, int building_number)
{

  float ray_intersect;
  int num_crossing, start_poly;
  unsigned int vert_id;
  std::vector<int> i_face_start, i_face_end;
  std::vector<int> j_face_start, j_face_end;
  std::vector<float> x_min_face, x_max_face;
  std::vector<float> y_min_face, y_max_face;
  std::vector<float> slope;
  std::vector<int> i_face_first, j_face_first;
  std::vector<int> i_face_second, j_face_second;
  float x1i, x2i, x1j, x2j;
  float y1i, y2i, y1j, y2j;
  std::vector<cutCell> cut_points;
  std::vector<int> cut_cell_id;
  std::vector<int>::iterator it;
  unsigned int counter;
  cut_points.clear();
  cut_cell_id.clear();
  unsigned int index_next;
  float x1i_intersect, x2i_intersect;
  float y1i_intersect, y2i_intersect;
  float x1j_intersect, x2j_intersect;
  float y1j_intersect, y2j_intersect;
  std::vector<cutVert> face_intersect;
  int condition;
  int height_flag;
  float S_cut;
  float solid_V_frac;
  std::vector<unsigned int> skip_id;
  skip_id.clear();
  float y_face, x_face;
  long icell_cent, icell_face;
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  i_face_start.resize(PolyB->polygonVertices.size(), 1);
  i_face_end.resize(PolyB->polygonVertices.size(), 1);
  j_face_start.resize(PolyB->polygonVertices.size(), 1);
  j_face_end.resize(PolyB->polygonVertices.size(), 1);
  x_min_face.resize(PolyB->polygonVertices.size(), 0.0);
  x_max_face.resize(PolyB->polygonVertices.size(), 0.0);
  y_min_face.resize(PolyB->polygonVertices.size(), 0.0);
  y_max_face.resize(PolyB->polygonVertices.size(), 0.0);
  slope.resize(PolyB->polygonVertices.size(), 0.0);
  i_face_first.resize(PolyB->polygonVertices.size(), 1);
  j_face_first.resize(PolyB->polygonVertices.size(), 1);
  i_face_second.resize(PolyB->polygonVertices.size(), 1);
  j_face_second.resize(PolyB->polygonVertices.size(), 1);

  // Loop to find the maximum and minimum in x and y directions and calculate slope of the line between each two points of polygon
  for (auto id = 0u; id < PolyB->polygonVertices.size() - 1; id++) {
    // Find the maximum and minimum in x direction of each line of the polygon
    if (PolyB->polygonVertices[id].x_poly < PolyB->polygonVertices[id + 1].x_poly) {
      x_min_face[id] = PolyB->polygonVertices[id].x_poly;
      x_max_face[id] = PolyB->polygonVertices[id + 1].x_poly;
    } else {
      x_min_face[id] = PolyB->polygonVertices[id + 1].x_poly;
      x_max_face[id] = PolyB->polygonVertices[id].x_poly;
    }

    // Calculate the start and end indices for each line of polygon in x direction
    i_face_start[id] = (x_min_face[id] / dx) - 1;
    i_face_end[id] = std::floor(x_max_face[id] / dx);

    // Find the maximum and minimum in y direction of each line of the polygon
    if (PolyB->polygonVertices[id].y_poly < PolyB->polygonVertices[id + 1].y_poly) {
      y_min_face[id] = PolyB->polygonVertices[id].y_poly;
      y_max_face[id] = PolyB->polygonVertices[id + 1].y_poly;
    } else {
      y_min_face[id] = PolyB->polygonVertices[id + 1].y_poly;
      y_max_face[id] = PolyB->polygonVertices[id].y_poly;
    }

    // Calculate the start and end indices for each line of polygon in y direction
    j_face_start[id] = (y_min_face[id] / dy) - 1;
    j_face_end[id] = std::floor(y_max_face[id] / dy);

    // Calculate the slope for each line of polygon in x direction
    if (PolyB->polygonVertices[id + 1].x_poly != PolyB->polygonVertices[id].x_poly) {
      slope[id] = (PolyB->polygonVertices[id + 1].y_poly - PolyB->polygonVertices[id].y_poly) / (PolyB->polygonVertices[id + 1].x_poly - PolyB->polygonVertices[id].x_poly);
    } else {
      slope[id] = (PolyB->polygonVertices[id + 1].y_poly - PolyB->polygonVertices[id].y_poly) / (PolyB->polygonVertices[id + 1].x_poly - PolyB->polygonVertices[id].x_poly + 0.000001);
    }
  }

  unsigned int number = 0;

  // The main loop going through all points in polygon and processing cut-cells
  for (auto id = 0u; id < PolyB->polygonVertices.size() - 1; id++) {
    if (skip_id.size() != 0) {
      if (id == skip_id[number]) {
        if (number < skip_id.size()) {
          number += 1;
        }
        continue;
      }
    }

    // Loops to go through all the cells that encompass the line
    for (auto i = i_face_start[id]; i <= i_face_end[id]; i++) {
      for (auto j = j_face_start[id]; j <= j_face_end[id]; j++) {
        // x and y values of intersection points with 1 and 2 lines of the cell in x-direction
        x1i = x2i = 0.0;
        y1i = y2i = 0.0;
        // x and y values of intersection points with 1 and 2 lines of the cell in y-direction
        x1j = x2j = 0.0;
        y1j = y2j = 0.0;
        face_intersect.clear();
        condition = 0;
        // x values of intersection points with 1 and 2 lines of the cell in x-direction
        x1i = i * dx;
        x2i = (i + 1) * dx;
        // y values of intersection points with 1 and 2 lines of the cell in y-direction
        y1j = j * dy;
        y2j = (j + 1) * dy;
        i_face_first[id] = PolyB->polygonVertices[id].x_poly / dx;
        j_face_first[id] = PolyB->polygonVertices[id].y_poly / dy;
        i_face_second[id] = PolyB->polygonVertices[id + 1].x_poly / dx;
        j_face_second[id] = PolyB->polygonVertices[id + 1].y_poly / dy;

        // If the first and second points of line are not in the same cell, ignore processing the first point cells
        // Since it has been processed as the second point of previous line
        if ((i == i_face_first[id] && j == j_face_first[id]) && (i != i_face_second[id] || j != j_face_second[id])) {
          continue;
        }

        // If the second point is in bounds of the current cell
        if (x2i >= PolyB->polygonVertices[id + 1].x_poly && x1i <= PolyB->polygonVertices[id + 1].x_poly && y2j >= PolyB->polygonVertices[id + 1].y_poly && y1j <= PolyB->polygonVertices[id + 1].y_poly) {
          x1i_intersect = x2i_intersect = 0.0;
          y1i_intersect = y2i_intersect = 0.0;
          x1j_intersect = x2j_intersect = 0.0;
          y1j_intersect = y2j_intersect = 0.0;
          // Add the second point to solid points in the cell
          face_intersect.push_back(cutVert(PolyB->polygonVertices[id + 1].x_poly, PolyB->polygonVertices[id + 1].y_poly, 0.0));
          // If the first line of the cell in x-direction is in range of the line
          if (x1i >= x_min_face[id] && x1i <= x_max_face[id] && x1i != PolyB->polygonVertices[id + 1].x_poly && x1i != PolyB->polygonVertices[id].x_poly) {
            // Calculate intersection of two lines in y-direction
            y1i_intersect = slope[id] * (x1i - PolyB->polygonVertices[id].x_poly) + PolyB->polygonVertices[id].y_poly;
            // If the intersection is outside of the cell bounds
            if (y1i_intersect > y2j || y1i_intersect < y1j) {
              y1i_intersect = 0.0;
            }
            x1i_intersect = x1i;
          }
          // If the second line of the cell in x-direction is in range of the line
          if (x2i >= x_min_face[id] && x2i <= x_max_face[id] && x2i != PolyB->polygonVertices[id + 1].x_poly && x2i != PolyB->polygonVertices[id].x_poly) {
            // Calculate intersection of two lines in y-direction
            y1i_intersect = slope[id] * (x2i - PolyB->polygonVertices[id].x_poly) + PolyB->polygonVertices[id].y_poly;
            // If the intersection is outside of the cell bounds
            if (y1i_intersect > y2j || y1i_intersect < y1j) {
              y1i_intersect = 0.0;
            }
            x1i_intersect = x2i;
          }
          // If the first line of the cell in y-direction is in range of the line
          if (y1j >= y_min_face[id] && y1j <= y_max_face[id] && y1j != PolyB->polygonVertices[id + 1].y_poly && y1j != PolyB->polygonVertices[id].y_poly) {
            // Calculate intersection of two lines in x-direction
            x1j_intersect = ((y1j - PolyB->polygonVertices[id].y_poly) / slope[id]) + PolyB->polygonVertices[id].x_poly;
            // If the intersection is outside of the cell bounds
            if (x1j_intersect > x2i || x1j_intersect < x1i) {
              x1j_intersect = 0.0;
            }
            y1j_intersect = y1j;
          }
          // If the second line of the cell in y-direction is in range of the line
          if (y2j >= y_min_face[id] && y2j <= y_max_face[id] && y2j != PolyB->polygonVertices[id + 1].y_poly && y2j != PolyB->polygonVertices[id].y_poly) {
            // Calculate intersection of two lines in x-direction
            x1j_intersect = ((y2j - PolyB->polygonVertices[id].y_poly) / slope[id]) + PolyB->polygonVertices[id].x_poly;
            // If the intersection is outside of the cell bounds
            if (x1j_intersect > x2i || x1j_intersect < x1i) {
              x1j_intersect = 0.0;
            }
            y1j_intersect = y2j;
          }

          // Index of the next line of polygon
          index_next = (id + 1) % (PolyB->polygonVertices.size() - 1);

          // If the first line of the cell in x-direction is in range of the next line
          if (x1i >= x_min_face[index_next] && x1i <= x_max_face[index_next] && x1i != PolyB->polygonVertices[index_next].x_poly) {
            // Calculate intersection of two lines in y-direction
            y2i_intersect = slope[index_next] * (x1i - PolyB->polygonVertices[index_next].x_poly) + PolyB->polygonVertices[index_next].y_poly;
            // If the intersection is outside of the cell bounds
            if (y2i_intersect > y2j || y2i_intersect < y1j) {
              y2i_intersect = 0.0;
            }
            x2i_intersect = x1i;
          }
          // If the second line of the cell in x-direction is in range of the next line
          if (x2i >= x_min_face[index_next] && x2i <= x_max_face[index_next] && x2i != PolyB->polygonVertices[index_next].x_poly) {
            // Calculate intersection of two lines in y-direction
            y2i_intersect = slope[index_next] * (x2i - PolyB->polygonVertices[index_next].x_poly) + PolyB->polygonVertices[index_next].y_poly;
            // If the intersection is outside of the cell bounds
            if (y2i_intersect > y2j || y2i_intersect < y1j) {
              y2i_intersect = 0.0;
            }
            x2i_intersect = x2i;
          }
          // If the first line of the cell in y-direction is in range of the next line
          if (y1j >= y_min_face[index_next] && y1j <= y_max_face[index_next] && y1j != PolyB->polygonVertices[index_next].y_poly) {
            // Calculate intersection of two lines in x-direction
            x2j_intersect = ((y1j - PolyB->polygonVertices[index_next].y_poly) / slope[index_next]) + PolyB->polygonVertices[index_next].x_poly;
            // If the intersection is outside of the cell bounds
            if (x2j_intersect > x2i || x2j_intersect < x1i) {
              x2j_intersect = 0.0;
            }
            y2j_intersect = y1j;
          }
          // If the second line of the cell in y-direction is in range of the next line
          if (y2j >= y_min_face[index_next] && y2j <= y_max_face[index_next] && y2j != PolyB->polygonVertices[index_next].y_poly) {
            // Calculate intersection of two lines in x-direction
            x2j_intersect = ((y2j - PolyB->polygonVertices[index_next].y_poly) / slope[index_next]) + PolyB->polygonVertices[index_next].x_poly;
            // If the intersection is outside of the cell bounds
            if (x2j_intersect > x2i || x2j_intersect < x1i) {
              x2j_intersect = 0.0;
            }
            y2j_intersect = y2j;
          }

          unsigned int count = 0;
          // Loop to calculate all the intersection points in case the next line is in the current cell
          // The loop is complete once the second point of a line is outside of the cell
          while (x2i >= PolyB->polygonVertices[index_next + 1].x_poly && x1i <= PolyB->polygonVertices[index_next + 1].x_poly
                 && count != PolyB->polygonVertices.size() - 2
                 && y2j >= PolyB->polygonVertices[index_next + 1].y_poly && y1j <= PolyB->polygonVertices[index_next + 1].y_poly
                 && index_next != PolyB->polygonVertices.size() - 1) {
            // Add the second point to solid points in the cell
            face_intersect.push_back(cutVert(PolyB->polygonVertices[index_next + 1].x_poly, PolyB->polygonVertices[index_next + 1].y_poly, 0.0));
            // Add the ID to the list to skip it in the process
            skip_id.push_back(index_next);

            // Index of the next line of polygon
            index_next = (index_next + 1) % (PolyB->polygonVertices.size() - 1);
            //////////////////////////////////////////////////////////////////
            ////////    Finding intersection points for the next line  ///////
            //////////////////////////////////////////////////////////////////
            if (x1i >= x_min_face[index_next] && x1i <= x_max_face[index_next] && x1i != PolyB->polygonVertices[index_next].x_poly) {
              y2i_intersect = slope[index_next] * (x1i - PolyB->polygonVertices[index_next].x_poly) + PolyB->polygonVertices[index_next].y_poly;
              if (y2i_intersect > y2j || y2i_intersect < y1j) {
                y2i_intersect = 0.0;
              }
              x2i_intersect = x1i;
            }
            if (x2i >= x_min_face[index_next] && x2i <= x_max_face[index_next] && x2i != PolyB->polygonVertices[index_next].x_poly) {
              y2i_intersect = slope[index_next] * (x2i - PolyB->polygonVertices[index_next].x_poly) + PolyB->polygonVertices[index_next].y_poly;
              if (y2i_intersect > y2j || y2i_intersect < y1j) {
                y2i_intersect = 0.0;
              }
              x2i_intersect = x2i;
            }
            if (y1j >= y_min_face[index_next] && y1j <= y_max_face[index_next] && y1j != PolyB->polygonVertices[index_next].y_poly) {
              x2j_intersect = ((y1j - PolyB->polygonVertices[index_next].y_poly) / slope[index_next]) + PolyB->polygonVertices[index_next].x_poly;
              if (x2j_intersect > x2i || x2j_intersect < x1i) {
                x2j_intersect = 0.0;
              }
              y2j_intersect = y1j;
            }
            if (y2j >= y_min_face[index_next] && y2j <= y_max_face[index_next] && y2j != PolyB->polygonVertices[index_next].y_poly) {
              x2j_intersect = ((y2j - PolyB->polygonVertices[index_next].y_poly) / slope[index_next]) + PolyB->polygonVertices[index_next].x_poly;
              if (x2j_intersect > x2i || x2j_intersect < x1i) {
                x2j_intersect = 0.0;
              }
              y2j_intersect = y2j;
            }
            count += 1;
          }

          // If all the intersection points are outside of cell bounds, skip to the next cell
          if (x1j_intersect == 0.0 && x2j_intersect == 0.0) {
            if ((y1i_intersect > y2j && y2i_intersect > y2j) || (y1i_intersect < y1j && y2i_intersect < y1j)) {
              continue;
            }
          }
          // If all the intersection points are outside of cell bounds, skip to the next cell
          if (y1i_intersect == 0.0 && y2i_intersect == 0.0) {
            if ((x1j_intersect > x2i && x2j_intersect > x2i) || (x1j_intersect < x1i && x2j_intersect < x1i)) {
              continue;
            }
          }

          if (y1j == PolyB->polygonVertices[id + 1].y_poly && x1i == PolyB->polygonVertices[id + 1].x_poly) {
            if ((x1i_intersect == x2i && y1i_intersect == y1j) || (x1j_intersect == x2i && y1j_intersect == y1j) || (x2i_intersect == x2i && y2i_intersect == y1j) || (x2j_intersect == x2i && y2j_intersect == y1j)) {
              continue;
            }
            if ((x1i_intersect == x1i && y1i_intersect == y2j) || (x1j_intersect == x1i && y1j_intersect == y2j) || (x2i_intersect == x1i && y2i_intersect == y2j) || (x2j_intersect == x1i && y2j_intersect == y2j)) {
              continue;
            }
          }

          if (y2j == PolyB->polygonVertices[id + 1].y_poly && x1i == PolyB->polygonVertices[id + 1].x_poly) {
            if ((x1i_intersect == x2i && y1i_intersect == y2j) || (x1j_intersect == x2i && y1j_intersect == y2j) || (x2i_intersect == x2i && y2i_intersect == y2j) || (x2j_intersect == x2i && y2j_intersect == y2j)) {

              continue;
            }
            if ((x1i_intersect == x1i && y1i_intersect == y1j) || (x1j_intersect == x1i && y1j_intersect == y1j) || (x2i_intersect == x1i && y2i_intersect == y1j) || (x2j_intersect == x1i && y2j_intersect == y1j)) {
              continue;
            }
          }

          if (y1j == PolyB->polygonVertices[id + 1].y_poly && x2i == PolyB->polygonVertices[id + 1].x_poly) {
            if ((x1i_intersect == x2i && y1i_intersect == y2j) || (x1j_intersect == x2i && y1j_intersect == y2j) || (x2i_intersect == x2i && y2i_intersect == y2j) || (x2j_intersect == x2i && y2j_intersect == y2j)) {

              continue;
            }
            if ((x1i_intersect == x1i && y1i_intersect == y1j) || (x1j_intersect == x1i && y1j_intersect == y1j) || (x2i_intersect == x1i && y2i_intersect == y1j) || (x2j_intersect == x1i && y2j_intersect == y1j)) {
              continue;
            }
          }

          if (y2j == PolyB->polygonVertices[id + 1].y_poly && x2i == PolyB->polygonVertices[id + 1].x_poly) {
            if ((x1i_intersect == x2i && y1i_intersect == y1j) || (x1j_intersect == x2i && y1j_intersect == y1j) || (x2i_intersect == x2i && y2i_intersect == y1j) || (x2j_intersect == x2i && y2j_intersect == y1j)) {
              continue;
            }
            if ((x1i_intersect == x1i && y1i_intersect == y2j) || (x1j_intersect == x1i && y1j_intersect == y2j) || (x2i_intersect == x1i && y2i_intersect == y2j) || (x2j_intersect == x1i && y2j_intersect == y2j)) {
              continue;
            }
          }
          // If all the intersection points are outside of cell bounds, skip to the next cell
          if (x1i_intersect == 0.0 && x2i_intersect == 0.0 && x1j_intersect == 0.0 && x2j_intersect == 0.0 && y1i_intersect == 0.0 && y2i_intersect == 0.0 && y1j_intersect == 0.0 && y2j_intersect == 0.0) {
            continue;
          }


          for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
            icell_cent = WGD->domain.cell(i, j, k);
            // If the cell is not in the terrain
            if (WGD->icellflag[icell_cent] != 2) {
              // If the cell is not marked as the cut-cell before, create one
              if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                WGD->icellflag[icell_cent] = 7;
                cut_points.push_back(cutCell(icell_cent));
                cut_cell_id.push_back(icell_cent);
                counter = cut_cell_id.size() - 1;
              } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                counter = std::distance(cut_cell_id.begin(), it);
                if (counter == cut_cell_id.size()) {
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                }
              } else {
                continue;
              }
            }

            // If it is a cut-cell, find the solid points and push them to the list
            if (WGD->icellflag[icell_cent] == 7) {
              height_flag = 1;
              cut_points[counter].z_solid = WGD->domain.dz_array[k];
              if (k == PolyB->k_cut_end) {
                if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                  height_flag = 0;
                  cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                }
              }
              for (auto ii = 0u; ii < face_intersect.size(); ii++) {
                cut_points[counter].intersect.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, 0.0));
                cut_points[counter].face_below.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, 0.0));
                if (height_flag == 1) {
                  cut_points[counter].face_above.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, cut_points[counter].z_solid));
                }
                if (face_intersect[ii].x_cut == x1i) {
                  cut_points[counter].face_behind.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, 0.0));
                  cut_points[counter].face_behind.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, cut_points[counter].z_solid));
                }
                if (face_intersect[ii].x_cut == x2i) {
                  cut_points[counter].face_front.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, 0.0));
                  cut_points[counter].face_front.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, cut_points[counter].z_solid));
                }
                if (face_intersect[ii].y_cut == y1j) {
                  cut_points[counter].face_right.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, 0.0));
                  cut_points[counter].face_right.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, cut_points[counter].z_solid));
                }
                if (face_intersect[ii].y_cut == y2j) {
                  cut_points[counter].face_left.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, 0.0));
                  cut_points[counter].face_left.push_back(cutVert(face_intersect[ii].x_cut - i * dx, face_intersect[ii].y_cut - j * dy, cut_points[counter].z_solid));
                }
              }

              if (x1i_intersect == x1i && x2i_intersect == x1i && y1i_intersect != y2i_intersect && y1i_intersect >= y1j && y1i_intersect <= y2j && y2i_intersect >= y1j && y2i_intersect <= y2j) {
                cut_points[counter].face_below.push_back(cutVert(x1i_intersect - i * dx, y1i_intersect - j * dy, 0.0));
                cut_points[counter].face_below.push_back(cutVert(x2i_intersect - i * dx, y2i_intersect - j * dy, 0.0));
                if (height_flag == 1) {
                  cut_points[counter].face_above.push_back(cutVert(x1i_intersect - i * dx, y1i_intersect - j * dy, cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert(x2i_intersect - i * dx, y2i_intersect - j * dy, cut_points[counter].z_solid));
                }
                cut_points[counter].face_behind.push_back(cutVert(x1i_intersect - i * dx, y1i_intersect - j * dy, 0.0));
                cut_points[counter].face_behind.push_back(cutVert(x2i_intersect - i * dx, y2i_intersect - j * dy, 0.0));
                cut_points[counter].face_behind.push_back(cutVert(x1i_intersect - i * dx, y1i_intersect - j * dy, cut_points[counter].z_solid));
                cut_points[counter].face_behind.push_back(cutVert(x2i_intersect - i * dx, y2i_intersect - j * dy, cut_points[counter].z_solid));
                condition = 1;
              }

              if (x1i_intersect == x2i && x2i_intersect == x2i && y1i_intersect != y2i_intersect && y1i_intersect >= y1j && y1i_intersect <= y2j && y2i_intersect >= y1j && y2i_intersect <= y2j) {
                cut_points[counter].face_below.push_back(cutVert(x1i_intersect - i * dx, y1i_intersect - j * dy, 0.0));
                cut_points[counter].face_below.push_back(cutVert(x2i_intersect - i * dx, y2i_intersect - j * dy, 0.0));
                if (height_flag == 1) {
                  cut_points[counter].face_above.push_back(cutVert(x1i_intersect - i * dx, y1i_intersect - j * dy, cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert(x2i_intersect - i * dx, y2i_intersect - j * dy, cut_points[counter].z_solid));
                }
                cut_points[counter].face_front.push_back(cutVert(x1i_intersect - i * dx, y1i_intersect - j * dy, 0.0));
                cut_points[counter].face_front.push_back(cutVert(x2i_intersect - i * dx, y2i_intersect - j * dy, 0.0));
                cut_points[counter].face_front.push_back(cutVert(x1i_intersect - i * dx, y1i_intersect - j * dy, cut_points[counter].z_solid));
                cut_points[counter].face_front.push_back(cutVert(x2i_intersect - i * dx, y2i_intersect - j * dy, cut_points[counter].z_solid));
                condition = 1;
              }

              if (y1j_intersect == y1j && y2j_intersect == y1j && x1j_intersect != x2j_intersect && x1j_intersect >= x1i && x1j_intersect <= x2i && x2j_intersect >= x1i && x2j_intersect <= x2i) {
                cut_points[counter].face_below.push_back(cutVert(x1j_intersect - i * dx, y1j_intersect - j * dy, 0.0));
                cut_points[counter].face_below.push_back(cutVert(x2j_intersect - i * dx, y2j_intersect - j * dy, 0.0));
                if (height_flag == 1) {
                  cut_points[counter].face_above.push_back(cutVert(x1j_intersect - i * dx, y1j_intersect - j * dy, cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert(x2j_intersect - i * dx, y2j_intersect - j * dy, cut_points[counter].z_solid));
                }
                cut_points[counter].face_right.push_back(cutVert(x1j_intersect - i * dx, y1j_intersect - j * dy, 0.0));
                cut_points[counter].face_right.push_back(cutVert(x2j_intersect - i * dx, y2j_intersect - j * dy, 0.0));
                cut_points[counter].face_right.push_back(cutVert(x1j_intersect - i * dx, y1j_intersect - j * dy, cut_points[counter].z_solid));
                cut_points[counter].face_right.push_back(cutVert(x2j_intersect - i * dx, y2j_intersect - j * dy, cut_points[counter].z_solid));
                condition = 1;
              }

              if (y1j_intersect == y2j && y2j_intersect == y2j && x1j_intersect != x2j_intersect && x1j_intersect >= x1i && x1j_intersect <= x2i && x2j_intersect >= x1i && x2j_intersect <= x2i) {
                cut_points[counter].face_below.push_back(cutVert(x1j_intersect - i * dx, y1j_intersect - j * dy, 0.0));
                cut_points[counter].face_below.push_back(cutVert(x2j_intersect - i * dx, y2j_intersect - j * dy, 0.0));
                if (height_flag == 1) {
                  cut_points[counter].face_above.push_back(cutVert(x1j_intersect - i * dx, y1j_intersect - j * dy, cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert(x2j_intersect - i * dx, y2j_intersect - j * dy, cut_points[counter].z_solid));
                }
                cut_points[counter].face_left.push_back(cutVert(x1j_intersect - i * dx, y1j_intersect - j * dy, 0.0));
                cut_points[counter].face_left.push_back(cutVert(x2j_intersect - i * dx, y2j_intersect - j * dy, 0.0));
                cut_points[counter].face_left.push_back(cutVert(x1j_intersect - i * dx, y1j_intersect - j * dy, cut_points[counter].z_solid));
                cut_points[counter].face_left.push_back(cutVert(x2j_intersect - i * dx, y2j_intersect - j * dy, cut_points[counter].z_solid));
                condition = 1;
              }
            }
          }

          if (condition == 1) {
            continue;
          }

          if (y1i_intersect != 0.0) {
            if (x1i_intersect == x1i) {
              y1i = y1i_intersect;
              if (y2i_intersect != 0.0 && x2i_intersect == x2i) {
                y2i = y2i_intersect;
              }
              if (x2j_intersect != 0.0) {
                if (y2j_intersect == y1j) {
                  x1j = x2j_intersect;
                  y2i = (j - 1) * dy;
                }
                if (y2j_intersect == y2j) {
                  x2j = x2j_intersect;
                  y2i = (j + 2) * dy;
                }
              }
              if (x1j_intersect != 0.0) {
                if (y1j_intersect == y1j) {
                  x1j = x1j_intersect;
                }
                if (y1j_intersect == y2j) {
                  x2j = x1j_intersect;
                }
              }
            }

            if (x1i_intersect == x2i) {
              y2i = y1i_intersect;
              if (y2i_intersect != 0.0 && x2i_intersect == x1i) {
                y1i = y2i_intersect;
              }
              if (x2j_intersect != 0.0) {
                if (y2j_intersect == y1j) {
                  x1j = x2j_intersect;
                  y1i = (j - 1) * dy;
                }
                if (y2j_intersect == y2j) {
                  x2j = x2j_intersect;
                  y1i = (j + 2) * dy;
                }
              }
              if (x1j_intersect != 0.0) {
                if (y1j_intersect == y1j) {
                  x1j = x1j_intersect;
                }
                if (y1j_intersect == y2j) {
                  x2j = x1j_intersect;
                }
              }
            }
          }

          else if (y2i_intersect != 0.0) {
            if (x2i_intersect == x1i) {
              y1i = y2i_intersect;
              if (x1j_intersect != 0.0) {
                if (y1j_intersect == y1j) {
                  x1j = x1j_intersect;
                  y2i = (j - 1) * dy;
                }
                if (y1j_intersect == y2j) {
                  x2j = x1j_intersect;
                  y2i = (j + 2) * dy;
                }
              }
            }

            if (x2i_intersect == x2i) {
              y2i = y2i_intersect;
              if (x1j_intersect != 0.0) {
                if (y1j_intersect == y1j) {
                  x1j = x1j_intersect;
                  y1i = (j - 1) * dy;
                }
                if (y1j_intersect == y2j) {
                  x2j = x1j_intersect;
                  y1i = (j + 2) * dy;
                }
              }
            }
          }

          else {
            if (y1j_intersect == y1j) {
              x1j = x1j_intersect;
              x2j = x2j_intersect;
              y1i = (j - 1) * dy;
              y2i = (j + 2) * dy;
            }
            if (y1j_intersect == y2j) {
              x1j = x2j_intersect;
              x2j = x1j_intersect;
              y2i = (j - 1) * dy;
              y1i = (j + 2) * dy;
            }
          }

        }
        // If the second point is not in bounds of the current cell
        else {
          if ((x1i < x_min_face[id] && x2i < x_min_face[id]) || (y1j < y_min_face[id] && y2j < y_min_face[id])) {
            continue;
          }
          // Calculate intersection of two lines in x and y directions
          if (x1i >= x_min_face[id] && x1i <= x_max_face[id]) {
            y1i = slope[id] * (x1i - PolyB->polygonVertices[id].x_poly) + PolyB->polygonVertices[id].y_poly;
          }
          if (x2i >= x_min_face[id] && x2i <= x_max_face[id]) {
            y2i = slope[id] * (x2i - PolyB->polygonVertices[id].x_poly) + PolyB->polygonVertices[id].y_poly;
          }
          if (y1j >= y_min_face[id] && y1j <= y_max_face[id]) {
            x1j = ((y1j - PolyB->polygonVertices[id].y_poly) / slope[id]) + PolyB->polygonVertices[id].x_poly;
          }
          if (y2j >= y_min_face[id] && y2j <= y_max_face[id]) {
            x2j = ((y2j - PolyB->polygonVertices[id].y_poly) / slope[id]) + PolyB->polygonVertices[id].x_poly;
          }
          // If intersection points are outside of cell bound, skip the cell
          if (x1j == 0.0 && x2j == 0.0 && y1i == 0.0 && y2i == 0.0) {
            continue;
          }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        ///////   Processing different combinations of the building cuts through cell lies     //////
        /////////////////////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////////////////////
        //////   Case 1: intersection point at the first line of cell in x-direction is in  ////////
        //////                          y bounds of cell.                                   ////////
        ////////////////////////////////////////////////////////////////////////////////////////////

        if (y1i > j * dy && y1i < (j + 1) * dy) {
          ////////////////////////////////////////////////////////////////////////////////////////////
          //////   Case 1A: intersection point at the second line of cell in x-direction      ////////
          //////                          is in y bounds of cell.                             ////////
          ////////////////////////////////////////////////////////////////////////////////////////////
          if (y2i >= j * dy && y2i <= (j + 1) * dy) {

            for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
              icell_cent = WGD->domain.cell(i, j, k);
              if (WGD->icellflag[icell_cent] != 2) {// If the cell is not terrain
                // Check to see if the cell is already a cut-cell
                if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                  WGD->icellflag[icell_cent] = 7;
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                  it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                  counter = std::distance(cut_cell_id.begin(), it);
                  if (counter == cut_cell_id.size()) {
                    cut_points.push_back(cutCell(icell_cent));
                    cut_cell_id.push_back(icell_cent);
                    counter = cut_cell_id.size() - 1;
                  }
                } else {
                  continue;
                }
              }

              if (WGD->icellflag[icell_cent] == 7) {
                // Add to the intersect points
                cut_points[counter].intersect.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                cut_points[counter].intersect.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));
                cut_points[counter].z_solid = WGD->domain.dz_array[k];
                if (k == PolyB->k_cut_end) {
                  if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                    // Define corner points
                    cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
                    // Define solid height of the cell
                    cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                  }
                }
                // Define solid points on each face
                // Below face points
                cut_points[counter].face_below.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                cut_points[counter].face_below.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));
                if (cut_points[counter].corner_id[4] != 0 || cut_points[counter].corner_id[5] != 0 || cut_points[counter].corner_id[6] != 0 || cut_points[counter].corner_id[7] != 0) {
                  // Above face points
                  cut_points[counter].face_above.push_back(cutVert((x1i - i * dx), (y1i - j * dy), cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert((x2i - i * dx), (y2i - j * dy), cut_points[counter].z_solid));
                }

                // Behind face points
                cut_points[counter].face_behind.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                cut_points[counter].face_behind.push_back(cutVert((x1i - i * dx), (y1i - j * dy), cut_points[counter].z_solid));
                // Front face points
                cut_points[counter].face_front.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));
                cut_points[counter].face_front.push_back(cutVert((x2i - i * dx), (y2i - j * dy), cut_points[counter].z_solid));
              }
            }
          }
          ////////////////////////////////////////////////////////////////////////////////////////////
          //////   Case 1B: intersection point at the second line of cell in x-direction      ////////
          //////                          is lower than cell's y bound                        ////////
          ////////////////////////////////////////////////////////////////////////////////////////////
          else if (y2i < j * dy) {
            for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
              icell_cent = WGD->domain.cell(i, j, k);
              if (WGD->icellflag[icell_cent] != 2) {// If the cell is not terrain
                // Check to see if the cell is already a cut-cell
                if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                  WGD->icellflag[icell_cent] = 7;
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                  it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                  counter = std::distance(cut_cell_id.begin(), it);
                  if (counter == cut_cell_id.size()) {
                    cut_points.push_back(cutCell(icell_cent));
                    cut_cell_id.push_back(icell_cent);
                    counter = cut_cell_id.size() - 1;
                  }
                } else {
                  continue;
                }
              }
              if (WGD->icellflag[icell_cent] == 7) {
                // Add to the intersect points
                cut_points[counter].intersect.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                cut_points[counter].z_solid = WGD->domain.dz_array[k];

                if (k == PolyB->k_cut_end) {
                  if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                    // Define corner point
                    cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
                    // Define solid height of the cell
                    cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                  }
                }

                // Define solid points on each face
                // Below face points
                cut_points[counter].face_below.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                cut_points[counter].face_below.push_back(cutVert((x1j - i * dx), (y1j - j * dy), 0.0));
                if (cut_points[counter].corner_id[4] != 0 || cut_points[counter].corner_id[5] != 0 || cut_points[counter].corner_id[6] != 0 || cut_points[counter].corner_id[7] != 0) {
                  // Above face points
                  cut_points[counter].face_above.push_back(cutVert((x1i - i * dx), (y1i - j * dy), cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert((x1j - i * dx), (y1j - j * dy), cut_points[counter].z_solid));
                }
                // Right face points
                cut_points[counter].face_right.push_back(cutVert((x1j - i * dx), (y1j - j * dy), cut_points[counter].z_solid));
                cut_points[counter].face_right.push_back(cutVert((x1j - i * dx), (y1j - j * dy), 0.0));

                // Behind face points
                cut_points[counter].face_behind.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                cut_points[counter].face_behind.push_back(cutVert((x1i - i * dx), (y1i - j * dy), cut_points[counter].z_solid));
              }
            }
          }
          ////////////////////////////////////////////////////////////////////////////////////////////
          //////   Case 1C: intersection point at the second line of cell in x-direction      ////////
          //////                          is greater than cell's y bound                        ////////
          ////////////////////////////////////////////////////////////////////////////////////////////
          else if (y2i > (j + 1) * dy) {
            for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
              icell_cent = WGD->domain.cell(i, j, k);
              if (WGD->icellflag[icell_cent] != 2) {// If the cell is not terrain
                // Check to see if the cell is already a cut-cell
                if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                  WGD->icellflag[icell_cent] = 7;
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                  it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                  counter = std::distance(cut_cell_id.begin(), it);
                  if (counter == cut_cell_id.size()) {
                    cut_points.push_back(cutCell(icell_cent));
                    cut_cell_id.push_back(icell_cent);
                    counter = cut_cell_id.size() - 1;
                  }
                } else {
                  continue;
                }
              }

              if (WGD->icellflag[icell_cent] == 7) {
                // Add to the intersect points
                cut_points[counter].intersect.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));

                cut_points[counter].z_solid = WGD->domain.dz_array[k];
                if (k == PolyB->k_cut_end) {
                  if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                    // Define corner points
                    cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
                    // Define solid height of the cell
                    cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                  }
                }
                // Define solid points on each face
                // Below face points
                cut_points[counter].face_below.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                cut_points[counter].face_below.push_back(cutVert((x2j - i * dx), (y2j - j * dy), 0.0));
                if (cut_points[counter].corner_id[4] != 0 || cut_points[counter].corner_id[5] != 0 || cut_points[counter].corner_id[6] != 0 || cut_points[counter].corner_id[7] != 0) {
                  // Above face points
                  cut_points[counter].face_above.push_back(cutVert((x1i - i * dx), (y1i - j * dy), cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert((x2j - i * dx), (y2j - j * dy), cut_points[counter].z_solid));
                }
                // Left face points
                cut_points[counter].face_left.push_back(cutVert((x2j - i * dx), (y2j - j * dy), 0.0));
                cut_points[counter].face_left.push_back(cutVert((x2j - i * dx), (y2j - j * dy), cut_points[counter].z_solid));

                // Behind face points
                cut_points[counter].face_behind.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                cut_points[counter].face_behind.push_back(cutVert((x1i - i * dx), (y1i - j * dy), cut_points[counter].z_solid));
              }
            }
          }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        //////   Case 2: intersection point at the second line of cell in x-direction is    ////////
        //////                          in y bounds of cell.                                ////////
        ////////////////////////////////////////////////////////////////////////////////////////////
        if (y2i > j * dy && y2i < (j + 1) * dy) {
          ////////////////////////////////////////////////////////////////////////////////////////////
          //////   Case 2A: intersection point at the first line of cell in x-direction       ////////
          //////                        is lower than cell's y bound.                         ////////
          ////////////////////////////////////////////////////////////////////////////////////////////
          if (y1i <= j * dy) {
            for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
              icell_cent = WGD->domain.cell(i, j, k);
              if (WGD->icellflag[icell_cent] != 2) {// If the cell is not terrain
                // Check to see if the cell is already a cut-cell
                if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                  WGD->icellflag[icell_cent] = 7;
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                  it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                  counter = std::distance(cut_cell_id.begin(), it);
                  if (counter == cut_cell_id.size()) {
                    cut_points.push_back(cutCell(icell_cent));
                    cut_cell_id.push_back(icell_cent);
                    counter = cut_cell_id.size() - 1;
                  }
                } else {
                  continue;
                }
              }

              if (WGD->icellflag[icell_cent] == 7) {
                // Add to the intersect points
                cut_points[counter].intersect.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));

                cut_points[counter].z_solid = WGD->domain.dz_array[k];
                if (k == PolyB->k_cut_end) {
                  if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                    // Define corner points
                    cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
                    // Define solid height of the cell
                    cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                  }
                }

                // Define solid points on each face
                // Below face points
                cut_points[counter].face_below.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));
                cut_points[counter].face_below.push_back(cutVert((x1j - i * dx), (y1j - j * dy), 0.0));
                if (cut_points[counter].corner_id[4] != 0 || cut_points[counter].corner_id[5] != 0 || cut_points[counter].corner_id[6] != 0 || cut_points[counter].corner_id[7] != 0) {
                  // Above face points
                  cut_points[counter].face_above.push_back(cutVert((x2i - i * dx), (y2i - j * dy), cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert((x1j - i * dx), (y1j - j * dy), cut_points[counter].z_solid));
                }

                // Right face points
                cut_points[counter].face_right.push_back(cutVert((x1j - i * dx), (y1j - j * dy), 0.0));
                cut_points[counter].face_right.push_back(cutVert((x1j - i * dx), (y1j - j * dy), cut_points[counter].z_solid));

                // Front face points
                cut_points[counter].face_front.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));
                cut_points[counter].face_front.push_back(cutVert((x2i - i * dx), (y2i - j * dy), cut_points[counter].z_solid));
              }
            }
          }
          ////////////////////////////////////////////////////////////////////////////////////////////
          //////   Case 2B: intersection point at the first line of cell in x-direction       ////////
          //////                        is greater than cell's y bound.                       ////////
          ////////////////////////////////////////////////////////////////////////////////////////////
          if (y1i >= (j + 1) * dy) {
            for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
              icell_cent = WGD->domain.cell(i, j, k);
              if (WGD->icellflag[icell_cent] != 2) {// If the cell is not terrain
                // Check to see if the cell is already a cut-cell
                if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                  WGD->icellflag[icell_cent] = 7;
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                  it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                  counter = std::distance(cut_cell_id.begin(), it);
                  if (counter == cut_cell_id.size()) {
                    cut_points.push_back(cutCell(icell_cent));
                    cut_cell_id.push_back(icell_cent);
                    counter = cut_cell_id.size() - 1;
                  }
                } else {
                  continue;
                }
              }

              if (WGD->icellflag[icell_cent] == 7) {
                // Add to the intersect points
                cut_points[counter].intersect.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));

                cut_points[counter].z_solid = WGD->domain.dz_array[k];
                if (k == PolyB->k_cut_end) {
                  if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                    // Define corner points
                    cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
                    // Define solid height of the cell
                    cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                  }
                }

                // Define solid points on each face
                // Below face points
                cut_points[counter].face_below.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));
                cut_points[counter].face_below.push_back(cutVert((x2j - i * dx), (y2j - j * dy), 0.0));
                if (cut_points[counter].corner_id[4] != 0 || cut_points[counter].corner_id[5] != 0 || cut_points[counter].corner_id[6] != 0 || cut_points[counter].corner_id[7] != 0) {
                  // Above face points
                  cut_points[counter].face_above.push_back(cutVert((x2i - i * dx), (y2i - j * dy), cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert((x2j - i * dx), (y2j - j * dy), cut_points[counter].z_solid));
                }
                // Left face points
                cut_points[counter].face_left.push_back(cutVert((x2j - i * dx), (y2j - j * dy), 0.0));
                cut_points[counter].face_left.push_back(cutVert((x2j - i * dx), (y2j - j * dy), cut_points[counter].z_solid));

                // Front face points
                cut_points[counter].face_front.push_back(cutVert((x2i - i * dx), (y2i - j * dy), 0.0));
                cut_points[counter].face_front.push_back(cutVert((x2i - i * dx), (y2i - j * dy), cut_points[counter].z_solid));
              }
            }
          }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        //////   Case 3: intersection point at the first line of cell in x-direction is at   ////////
        //////           lower y bound of cell and intersection point at the second line of  ////////
        //////           cell in x-direction is at higher y bound of cell.                   ////////
        /////////////////////////////////////////////////////////////////////////////////////////////
        if (y1i == j * dy && y2i == (j + 1) * dy) {
          if (x1j == i * dx && x2j == (i + 1) * dx) {
            for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
              icell_cent = WGD->domain.cell(i, j, k);
              if (WGD->icellflag[icell_cent] != 2) {// If the cell is not terrain
                // Check to see if the cell is already a cut-cell
                if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                  WGD->icellflag[icell_cent] = 7;
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                  it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                  counter = std::distance(cut_cell_id.begin(), it);
                  if (counter == cut_cell_id.size()) {
                    cut_points.push_back(cutCell(icell_cent));
                    cut_cell_id.push_back(icell_cent);
                    counter = cut_cell_id.size() - 1;
                  }
                } else {
                  continue;
                }
              }

              if (WGD->icellflag[icell_cent] == 7) {
                // Add to the intersect points
                cut_points[counter].intersect.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                if (k == PolyB->k_cut_end) {
                  if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                    // Define corner points
                    cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
                    // Define solid height of the cell
                    cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                  }
                }
              }
            }
          }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        //////   Case 4: intersection point at the first line of cell in x-direction is at   ////////
        //////           higher y bound of cell and intersection point at the second line of ////////
        //////           cell in x-direction is at lower y bound of cell.                    ////////
        /////////////////////////////////////////////////////////////////////////////////////////////
        if (y1i == (j + 1) * dy && y2i == j * dy) {
          if (x1j == (i + 1) * dx && x2j == i * dx) {
            for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
              icell_cent = WGD->domain.cell(i, j, k);
              ;
              if (WGD->icellflag[icell_cent] != 2) {// If the cell is not terrain
                // Check to see if the cell is already a cut-cell
                if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                  WGD->icellflag[icell_cent] = 7;
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                  it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                  counter = std::distance(cut_cell_id.begin(), it);
                  if (counter == cut_cell_id.size()) {
                    cut_points.push_back(cutCell(icell_cent));
                    cut_cell_id.push_back(icell_cent);
                    counter = cut_cell_id.size() - 1;
                  }
                } else {
                  continue;
                }
              }

              if (WGD->icellflag[icell_cent] == 7) {
                // Add to the intersect points
                cut_points[counter].intersect.push_back(cutVert((x1i - i * dx), (y1i - j * dy), 0.0));
                if (k == PolyB->k_cut_end) {
                  if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                    // Define corner points
                    cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
                    // Define solid height of the cell
                    cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                  }
                }
              }
            }
          }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        //////   Case 5: intersection points at the first and second lines of cell in        ////////
        //////                    x-direction are out of y bounds of cell.                   ////////
        /////////////////////////////////////////////////////////////////////////////////////////////
        if ((y1i > (j + 1) * dy || y1i < j * dy) && (y2i > (j + 1) * dy || y2i < j * dy)) {
          if (x1j > i * dx && x1j < (i + 1) * dx && x2j > i * dx && x2j < (i + 1) * dx) {
            for (auto k = PolyB->k_start; k <= PolyB->k_cut_end; k++) {
              icell_cent = WGD->domain.cell(i, j, k);
              if (WGD->icellflag[icell_cent] != 2) {// If the cell is not terrain
                // Check to see if the cell is already a cut-cell
                if ((WGD->icellflag[icell_cent] != 7 && WGD->ibuilding_flag[icell_cent] == building_number) || WGD->icellflag[icell_cent] == 1) {
                  WGD->icellflag[icell_cent] = 7;
                  cut_points.push_back(cutCell(icell_cent));
                  cut_cell_id.push_back(icell_cent);
                  counter = cut_cell_id.size() - 1;
                } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
                  it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
                  counter = std::distance(cut_cell_id.begin(), it);
                  if (counter == cut_cell_id.size()) {
                    cut_points.push_back(cutCell(icell_cent));
                    cut_cell_id.push_back(icell_cent);
                    counter = cut_cell_id.size() - 1;
                  }
                } else {
                  continue;
                }
              }

              if (WGD->icellflag[icell_cent] == 7) {
                // Add to the intersect points
                cut_points[counter].intersect.push_back(cutVert((x1j - i * dx), (y1j - j * dy), 0.0));

                cut_points[counter].z_solid = WGD->domain.dz_array[k];
                if (k == PolyB->k_cut_end) {
                  if (PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end] < WGD->domain.dz_array[k]) {
                    // Define corner points
                    cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
                    // Define solid height of the cell
                    cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[PolyB->k_cut_end];
                  }
                }

                // Define solid points on each face
                // Below face points
                cut_points[counter].face_below.push_back(cutVert((x1j - i * dx), (y1j - j * dy), 0.0));
                cut_points[counter].face_below.push_back(cutVert((x2j - i * dx), (y2j - j * dy), 0.0));
                if (cut_points[counter].corner_id[4] != 0 || cut_points[counter].corner_id[5] != 0 || cut_points[counter].corner_id[6] != 0 || cut_points[counter].corner_id[7] != 0) {
                  // Above face points
                  cut_points[counter].face_above.push_back(cutVert((x1j - i * dx), (y1j - j * dy), cut_points[counter].z_solid));
                  cut_points[counter].face_above.push_back(cutVert((x2j - i * dx), (y2j - j * dy), cut_points[counter].z_solid));
                }

                // Right face points
                cut_points[counter].face_right.push_back(cutVert((x1j - i * dx), (y1j - j * dy), 0.0));
                cut_points[counter].face_right.push_back(cutVert((x1j - i * dx), (y1j - j * dy), cut_points[counter].z_solid));

                // Left face points
                cut_points[counter].face_left.push_back(cutVert((x2j - i * dx), (y2j - j * dy), 0.0));
                cut_points[counter].face_left.push_back(cutVert((x2j - i * dx), (y2j - j * dy), cut_points[counter].z_solid));
              }
            }
          }
        }
      }
    }
  }


  if (cut_points.size() > 0) {
    for (auto id = 0u; id < cut_points.size(); id++) {
      auto [i, j, k] = WGD->domain.getCellIdx(cut_cell_id[id]);
      for (auto jj = 0; jj < 2; jj++) {
        y_face = (j + jj) * dy;// Center of cell y coordinate
        for (auto ii = 0; ii < 2; ii++) {
          x_face = (i + ii) * dx;// Center of cell x coordinate
          vert_id = 0;// Node index
          start_poly = vert_id;
          num_crossing = 0;
          while (vert_id < PolyB->polygonVertices.size() - 1) {
            if ((PolyB->polygonVertices[vert_id].y_poly <= y_face && PolyB->polygonVertices[vert_id + 1].y_poly > y_face) || (PolyB->polygonVertices[vert_id].y_poly > y_face && PolyB->polygonVertices[vert_id + 1].y_poly <= y_face)) {
              ray_intersect = (y_face - PolyB->polygonVertices[vert_id].y_poly) / (PolyB->polygonVertices[vert_id + 1].y_poly - PolyB->polygonVertices[vert_id].y_poly);
              if (x_face < (PolyB->polygonVertices[vert_id].x_poly + ray_intersect * (PolyB->polygonVertices[vert_id + 1].x_poly - PolyB->polygonVertices[vert_id].x_poly))) {
                num_crossing += 1;
              }
            }
            vert_id += 1;
            if (PolyB->polygonVertices[vert_id].x_poly == PolyB->polygonVertices[start_poly].x_poly && PolyB->polygonVertices[vert_id].y_poly == PolyB->polygonVertices[start_poly].y_poly) {
              vert_id += 1;
              start_poly = vert_id;
            }
          }
          // if num_crossing is odd = point is oustside of the polygon
          // if num_crossing is even = point is inside of the polygon
          if ((num_crossing % 2) != 0) {
            if (ii == 0 && jj == 0)// Left bottom corner of the cell
            {
              cut_points[id].face_below.push_back(cutVert(0.0, 0.0, 0.0));
              cut_points[id].face_behind.push_back(cutVert(0.0, 0.0, 0.0));
              cut_points[id].face_behind.push_back(cutVert(0.0, 0.0, cut_points[id].z_solid));
              cut_points[id].face_right.push_back(cutVert(0.0, 0.0, 0.0));
              cut_points[id].face_right.push_back(cutVert(0.0, 0.0, cut_points[id].z_solid));
              if (cut_points[id].corner_id[4] == 1) {
                cut_points[id].face_above.push_back(cutVert(0.0, 0.0, cut_points[id].z_solid));
              }
            }

            if (ii == 0 && jj == 1)// Left top corner of the cell
            {
              cut_points[id].face_below.push_back(cutVert(0.0, dy, 0.0));
              cut_points[id].face_behind.push_back(cutVert(0.0, dy, 0.0));
              cut_points[id].face_behind.push_back(cutVert(0.0, dy, cut_points[id].z_solid));
              cut_points[id].face_left.push_back(cutVert(0.0, dy, 0.0));
              cut_points[id].face_left.push_back(cutVert(0.0, dy, cut_points[id].z_solid));
              if (cut_points[id].corner_id[5] == 1) {
                cut_points[id].face_above.push_back(cutVert(0.0, dy, cut_points[id].z_solid));
              }
            }

            if (ii == 1 && jj == 1)// Right top corner of the cell
            {
              cut_points[id].face_below.push_back(cutVert(dx, dy, 0.0));
              cut_points[id].face_front.push_back(cutVert(dx, dy, 0.0));
              cut_points[id].face_front.push_back(cutVert(dx, dy, cut_points[id].z_solid));
              cut_points[id].face_left.push_back(cutVert(dx, dy, 0.0));
              cut_points[id].face_left.push_back(cutVert(dx, dy, cut_points[id].z_solid));
              if (cut_points[id].corner_id[6] == 1) {
                cut_points[id].face_above.push_back(cutVert(dx, dy, cut_points[id].z_solid));
              }
            }

            if (ii == 1 && jj == 0)// Right bottom corner of the cell
            {
              cut_points[id].face_below.push_back(cutVert(dx, 0.0, 0.0));
              cut_points[id].face_front.push_back(cutVert(dx, 0.0, 0.0));
              cut_points[id].face_front.push_back(cutVert(dx, 0.0, cut_points[id].z_solid));
              cut_points[id].face_right.push_back(cutVert(dx, 0.0, 0.0));
              cut_points[id].face_right.push_back(cutVert(dx, 0.0, cut_points[id].z_solid));
              if (cut_points[id].corner_id[7] == 1) {
                cut_points[id].face_above.push_back(cutVert(dx, 0.0, cut_points[id].z_solid));
              }
            }
          }
        }
      }
    }
  }

  int k = PolyB->k_cut_end;// Cell at the top of the column

  if (PolyB->height_eff - WGD->domain.z_face[k] < WGD->domain.dz_array[k]) {
    for (auto j = PolyB->j_start; j <= PolyB->j_end; j++) {
      for (auto i = PolyB->i_start; i <= PolyB->i_end; i++) {
        icell_cent = WGD->domain.cell(i, j, k);
        if (WGD->icellflag[WGD->domain.cellAdd(icell_cent, 0, 0, -1)] == 0
            && WGD->icellflag[WGD->domain.cellAdd(icell_cent, 0, 0, -1)] != 7
            && WGD->ibuilding_flag[WGD->domain.cellAdd(icell_cent, 0, 0, -1)] == building_number) {// If not a solid cell and belong to the same building
          // Check to see if the cell is already a cut-cell
          if (WGD->icellflag[icell_cent] != 7 || WGD->icellflag[icell_cent] == 1) {
            WGD->icellflag[icell_cent] = 7;
            cut_points.push_back(cutCell(icell_cent));
            cut_cell_id.push_back(icell_cent);
            counter = cut_cell_id.size() - 1;
          } else if (WGD->icellflag[icell_cent] == 7) {// If the cell is already marked as a cut-cell, add to the points
            it = std::find(cut_cell_id.begin(), cut_cell_id.end(), icell_cent);
            counter = std::distance(cut_cell_id.begin(), it);
            if (counter == cut_cell_id.size()) {
              cut_points.push_back(cutCell(icell_cent));
              cut_cell_id.push_back(icell_cent);
              counter = cut_cell_id.size() - 1;
            }
          } else {
            continue;
          }

          // Define corner points
          cut_points[counter].corner_id[4] = cut_points[counter].corner_id[5] = cut_points[counter].corner_id[6] = cut_points[counter].corner_id[7] = 0;
          cut_points[counter].z_solid = PolyB->height_eff - WGD->domain.z_face[k];// Define solid height of the cell
          if (WGD->icellflag[icell_cent] == 7) {
            cut_points[counter].intersect.push_back(cutVert(dx, dy, cut_points[counter].z_solid));// Add to the intersect points

            // Define solid points on each face
            // Below face points
            cut_points[counter].face_below.push_back(cutVert(0.0, 0.0, 0.0));
            cut_points[counter].face_below.push_back(cutVert(0.0, dy, 0.0));
            cut_points[counter].face_below.push_back(cutVert(dx, 0.0, 0.0));
            cut_points[counter].face_below.push_back(cutVert(dx, dy, 0.0));

            // Behind face points
            cut_points[counter].face_behind.push_back(cutVert(0.0, 0.0, 0.0));
            cut_points[counter].face_behind.push_back(cutVert(0.0, dy, 0.0));
            cut_points[counter].face_behind.push_back(cutVert(0.0, 0.0, cut_points[counter].z_solid));
            cut_points[counter].face_behind.push_back(cutVert(0.0, dy, cut_points[counter].z_solid));

            // Front face points
            cut_points[counter].face_front.push_back(cutVert(dx, 0.0, 0.0));
            cut_points[counter].face_front.push_back(cutVert(dx, dy, 0.0));
            cut_points[counter].face_front.push_back(cutVert(dx, 0.0, cut_points[counter].z_solid));
            cut_points[counter].face_front.push_back(cutVert(dx, dy, cut_points[counter].z_solid));

            // Right face points
            cut_points[counter].face_right.push_back(cutVert(0.0, 0.0, 0.0));
            cut_points[counter].face_right.push_back(cutVert(dx, 0.0, 0.0));
            cut_points[counter].face_right.push_back(cutVert(0.0, 0.0, cut_points[counter].z_solid));
            cut_points[counter].face_right.push_back(cutVert(dx, 0.0, cut_points[counter].z_solid));

            // Left face points
            cut_points[counter].face_left.push_back(cutVert(0.0, dy, 0.0));
            cut_points[counter].face_left.push_back(cutVert(dx, dy, 0.0));
            cut_points[counter].face_left.push_back(cutVert(0.0, dy, cut_points[counter].z_solid));
            cut_points[counter].face_left.push_back(cutVert(dx, dy, cut_points[counter].z_solid));
          }
        }
      }
    }
  }

  float distance_x, distance_y, distance_z;
  float distance;// Distance of cut face from the center of the cut-cell

  if (cut_points.size() > 0) {
    for (auto id = 0u; id < cut_points.size(); id++) {
      auto [i, j, k] = WGD->domain.getCellIdx(cut_cell_id[id]);
      // Reorder solid points of each face and calculate the area (set the related solver coefficients)

      // Behind face
      if (cut_points[id].face_behind.size() != 0) {
        reorderPoints(cut_points[id].face_behind, 0);
        cut_points[id].s_behind = calculateArea(WGD, cut_points[id].face_behind, cut_cell_id[id], 0);
      }

      // Front face
      if (cut_points[id].face_front.size() != 0) {
        reorderPoints(cut_points[id].face_front, 1);
        cut_points[id].s_front = calculateArea(WGD, cut_points[id].face_front, cut_cell_id[id], 1);
      }

      // Right face
      if (cut_points[id].face_right.size() != 0) {
        reorderPoints(cut_points[id].face_right, 2);
        cut_points[id].s_right = calculateArea(WGD, cut_points[id].face_right, cut_cell_id[id], 2);
      }

      // Left face
      if (cut_points[id].face_left.size() != 0) {
        reorderPoints(cut_points[id].face_left, 3);
        cut_points[id].s_left = calculateArea(WGD, cut_points[id].face_left, cut_cell_id[id], 3);
      }

      // Below face
      if (cut_points[id].face_below.size() != 0) {
        reorderPoints(cut_points[id].face_below, 4);
        cut_points[id].s_below = calculateArea(WGD, cut_points[id].face_below, cut_cell_id[id], 4);
      }

      // Above face
      if (cut_points[id].face_above.size() != 0) {
        reorderPoints(cut_points[id].face_above, 5);
        cut_points[id].s_above = calculateArea(WGD, cut_points[id].face_above, cut_cell_id[id], 5);
      }

      // Calculate the cut area in the cell using solid areas of each cell faces
      S_cut = sqrt(pow(cut_points[id].s_behind - cut_points[id].s_front, 2.0) + pow(cut_points[id].s_right - cut_points[id].s_left, 2.0)
                   + pow(cut_points[id].s_below - cut_points[id].s_above, 2.0));

      // Calculate normal unit vectors to the cut surface
      if (S_cut != 0.0) {
        cut_points[id].ni = (cut_points[id].s_behind - cut_points[id].s_front) / S_cut;
        cut_points[id].nj = (cut_points[id].s_right - cut_points[id].s_left) / S_cut;
        cut_points[id].nk = (cut_points[id].s_below - cut_points[id].s_above) / S_cut;
      }

      solid_V_frac = 0.0;

      // Calculate solid volume fraction of the cut-cell

      if (cut_points[id].face_behind.size() != 0) {
        solid_V_frac += (cut_points[id].face_behind[0].x_cut * (-1) * cut_points[id].s_behind) / (3 * dx * dy * WGD->domain.dz_array[k]);
      }

      if (cut_points[id].face_front.size() != 0) {
        solid_V_frac += (cut_points[id].face_front[0].x_cut * (1) * cut_points[id].s_front) / (3 * dx * dy * WGD->domain.dz_array[k]);
      }

      if (cut_points[id].face_right.size() != 0) {
        solid_V_frac += (cut_points[id].face_right[0].y_cut * (-1) * cut_points[id].s_right) / (3 * dx * dy * WGD->domain.dz_array[k]);
      }

      if (cut_points[id].face_left.size() != 0) {
        solid_V_frac += (cut_points[id].face_left[0].y_cut * (1) * cut_points[id].s_left) / (3 * dx * dy * WGD->domain.dz_array[k]);
      }

      if (cut_points[id].face_below.size() != 0) {
        solid_V_frac += (cut_points[id].face_below[0].z_cut * (-1) * cut_points[id].s_below) / (3 * dx * dy * WGD->domain.dz_array[k]);
      }

      if (cut_points[id].face_above.size() != 0) {
        solid_V_frac += (cut_points[id].face_above[0].z_cut * (1) * cut_points[id].s_above) / (3 * dx * dy * WGD->domain.dz_array[k]);
      }

      solid_V_frac += ((cut_points[id].intersect[0].x_cut * (cut_points[id].ni) * S_cut)
                       + (cut_points[id].intersect[0].y_cut * (cut_points[id].nj) * S_cut)
                       + (cut_points[id].intersect[0].z_cut * (cut_points[id].nk) * S_cut))
                      / (3 * dx * dy * WGD->domain.dz_array[k]);


      distance_x = (cut_points[id].intersect[0].x_cut - WGD->domain.x[i]) * cut_points[id].ni;
      distance_y = (cut_points[id].intersect[0].y_cut - WGD->domain.y[j]) * cut_points[id].nj;
      distance_z = (cut_points[id].intersect[0].z_cut - WGD->domain.z[k]) * cut_points[id].nk;

      // Calculate distance of the cell center from the cut surface
      distance = sqrt(pow(distance_x, 2.0) + pow(distance_y, 2.0) + pow(distance_z, 2.0));

      // If the cell center is not in solid
      if (WGD->center_id[cut_cell_id[id]] == 1) {
        WGD->wall_distance[cut_cell_id[id]] = distance;
      } else {// If the cell center is inside solid
        WGD->wall_distance[cut_cell_id[id]] = -distance;
      }

      // Set the building volume fraction of the cell by subtracting the solid volume fraction of the cell
      WGD->building_volume_frac[cut_cell_id[id]] -= solid_V_frac;

      WGD->ni[cut_cell_id[id]] = cut_points[id].ni;
      WGD->nj[cut_cell_id[id]] = cut_points[id].nj;
      WGD->nk[cut_cell_id[id]] = cut_points[id].nk;

      if (WGD->building_volume_frac[cut_cell_id[id]] < 0.0) {
        WGD->building_volume_frac[cut_cell_id[id]] = 0.0;
      }

      // Double check the solver coefficients and set the cells with below threshhold volume fraction to fully solid
      if (WGD->building_volume_frac[cut_cell_id[id]] <= 0.1) {
        WGD->icellflag[cut_cell_id[id]] = 0;
        WGD->e[cut_cell_id[id]] = 1.0;
        WGD->f[cut_cell_id[id]] = 1.0;
        WGD->g[cut_cell_id[id]] = 1.0;
        WGD->h[cut_cell_id[id]] = 1.0;
        WGD->m[cut_cell_id[id]] = 1.0;
        WGD->n[cut_cell_id[id]] = 1.0;
      }

      // Reset the solver coefficients values for fully solid cells
      if (WGD->e[cut_cell_id[id]] == 0.0 && WGD->f[cut_cell_id[id]] == 0.0 && WGD->g[cut_cell_id[id]] == 0.0
          && WGD->h[cut_cell_id[id]] == 0.0 && WGD->m[cut_cell_id[id]] == 0.0 && WGD->n[cut_cell_id[id]] == 0.0) {
        WGD->icellflag[cut_cell_id[id]] = 0;
        WGD->e[cut_cell_id[id]] = 1.0;
        WGD->f[cut_cell_id[id]] = 1.0;
        WGD->g[cut_cell_id[id]] = 1.0;
        WGD->h[cut_cell_id[id]] = 1.0;
        WGD->m[cut_cell_id[id]] = 1.0;
        WGD->n[cut_cell_id[id]] = 1.0;
      }
    }
  }
}


void CutBuilding::reorderPoints(std::vector<cutVert> &face_points, int index)
{

  Vector3Float centroid;
  std::vector<float> angle(face_points.size(), 0.0);
  Vector3Float sum;

  sum[0] = 0.0;
  sum[1] = 0.0;
  sum[2] = 0.0;

  // Calculate centroid of points
  for (auto i = 0u; i < face_points.size(); i++) {
    sum[0] += face_points[i].x_cut;
    sum[1] += face_points[i].y_cut;
    sum[2] += face_points[i].z_cut;
  }

  centroid[0] = sum[0] / face_points.size();
  centroid[1] = sum[1] / face_points.size();
  centroid[2] = sum[2] / face_points.size();

  // Calculate angle between each point and centroid
  for (auto i = 0u; i < face_points.size(); i++) {
    if (index == 0 || index == 1) {
      angle[i] = (180 / M_PI) * atan2((face_points[i].z_cut - centroid[2]), (face_points[i].y_cut - centroid[1]));
    }
    if (index == 2 || index == 3) {
      angle[i] = (180 / M_PI) * atan2((face_points[i].z_cut - centroid[2]), (face_points[i].x_cut - centroid[0]));
    }
    if (index == 4 || index == 5) {
      angle[i] = (180 / M_PI) * atan2((face_points[i].y_cut - centroid[1]), (face_points[i].x_cut - centroid[0]));
    }
  }
  // Call sort to sort points based on the angles (from -180 to 180)
  mergeSort(angle, face_points);
}


void CutBuilding::mergeSort(std::vector<float> &angle, std::vector<cutVert> &face_points)
{
  // if the size of the array is 1, it is already sorted
  if (angle.size() == 1)
    return;

  // make left and right sides of the data
  std::vector<float> angleL, angleR;
  std::vector<cutVert> face_points_L, face_points_R;

  angleL.resize(angle.size() / 2);
  angleR.resize(angle.size() - angle.size() / 2);
  face_points_L.resize(face_points.size() / 2);
  face_points_R.resize(face_points.size() - face_points.size() / 2);

  // copy data from the main data set to the left and right children
  size_t lC = 0, rC = 0;
  for (unsigned int i = 0; i < angle.size(); i++) {
    if (i < angle.size() / 2) {
      angleL[lC] = angle[i];
      face_points_L[lC++] = face_points[i];
    } else {
      angleR[rC] = angle[i];
      face_points_R[rC++] = face_points[i];
    }
  }

  // recursively sort the children
  mergeSort(angleL, face_points_L);
  mergeSort(angleR, face_points_R);

  // compare the sorted children to place the data into the main array
  lC = rC = 0;
  for (auto i = 0u; i < face_points.size(); i++) {
    if (rC == angleR.size() || (lC != angleL.size() && angleL[lC] < angleR[rC])) {
      angle[i] = angleL[lC];
      face_points[i] = face_points_L[lC++];
    } else {
      angle[i] = angleR[rC];
      face_points[i] = face_points_R[rC++];
    }
  }

  return;
}

float CutBuilding::calculateArea(WINDSGeneralData *WGD, std::vector<cutVert> &face_points, int cutcell_index, int index)
{
  float S = 0.0;
  float coeff = 0.0;
  int k = 0;
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  if (face_points.size() != 0) {
    k = cutcell_index / ((WGD->domain.nx() - 1) * (WGD->domain.ny() - 1));
    // calculate area fraction coefficient for each face of the cut-cell
    for (auto i = 0u; i < face_points.size() - 1; i++) {
      coeff += (0.5 * (face_points[i + 1].y_cut + face_points[i].y_cut) * (face_points[i + 1].z_cut - face_points[i].z_cut)) / (dy * WGD->domain.dz_array[k])
               + (0.5 * (face_points[i + 1].x_cut + face_points[i].x_cut) * (face_points[i + 1].z_cut - face_points[i].z_cut)) / (dx * WGD->domain.dz_array[k])
               + (0.5 * (face_points[i + 1].x_cut + face_points[i].x_cut) * (face_points[i + 1].y_cut - face_points[i].y_cut)) / (dx * dy);
    }

    coeff += (0.5 * (face_points[0].y_cut + face_points[face_points.size() - 1].y_cut)
              * (face_points[0].z_cut - face_points[face_points.size() - 1].z_cut))
               / (dy * WGD->domain.dz_array[k])
             + (0.5 * (face_points[0].x_cut + face_points[face_points.size() - 1].x_cut) * (face_points[0].z_cut - face_points[face_points.size() - 1].z_cut))
                 / (dx * WGD->domain.dz_array[k])
             + (0.5 * (face_points[0].x_cut + face_points[face_points.size() - 1].x_cut) * (face_points[0].y_cut - face_points[face_points.size() - 1].y_cut))
                 / (dx * dy);
  }
  coeff = 1.0 - coeff;// Calculate area fraction coefficient for fluid part

  if (coeff <= 0.05) {
    coeff = 0.0;
  } else if (coeff > 1.0) {
    coeff = 1.0;
  }

  // Set the related solver coefficient values to the solid area fraction of the face

  // Behind face
  if (index == 0) {
    S = (1.0 - coeff) * (dy * WGD->domain.dz_array[k]);
    if (WGD->f[cutcell_index] == 1.0) {
      WGD->f[cutcell_index] = coeff;
    } else {
      WGD->f[cutcell_index] -= (1.0 - coeff);
      if (WGD->f[cutcell_index] < 0.0) {
        WGD->f[cutcell_index] = 0.0;
      }
    }
  }

  // Front face
  if (index == 1) {
    S = (1.0 - coeff) * (dy * WGD->domain.dz_array[k]);
    if (WGD->e[cutcell_index] == 1.0) {
      WGD->e[cutcell_index] = coeff;
    } else {
      WGD->e[cutcell_index] -= (1.0 - coeff);
      if (WGD->e[cutcell_index] < 0.0) {
        WGD->e[cutcell_index] = 0.0;
      }
    }
  }

  // Right face
  if (index == 2) {
    S = (1.0 - coeff) * (dx * WGD->domain.dz_array[k]);
    if (WGD->h[cutcell_index] == 1.0) {
      WGD->h[cutcell_index] = coeff;
    } else {
      WGD->h[cutcell_index] -= (1.0 - coeff);
      if (WGD->h[cutcell_index] < 0.0) {
        WGD->h[cutcell_index] = 0.0;
      }
    }
  }

  // Left face
  if (index == 3) {
    S = (1.0 - coeff) * (dx * WGD->domain.dz_array[k]);
    if (WGD->g[cutcell_index] == 1.0) {
      WGD->g[cutcell_index] = coeff;
    } else {
      WGD->g[cutcell_index] -= (1.0 - coeff);
      if (WGD->g[cutcell_index] < 0.0) {

        WGD->g[cutcell_index] = 0.0;
      }
    }
  }

  // Below face
  if (index == 4) {
    S = (1.0 - coeff) * (dx * dy);
    if (WGD->n[cutcell_index] == 1.0) {
      WGD->n[cutcell_index] = coeff;
    } else {
      WGD->n[cutcell_index] -= (1.0 - coeff);
      if (WGD->n[cutcell_index] < 0.0) {
        WGD->n[cutcell_index] = 0.0;
      }
    }
  }

  // Above face
  if (index == 5) {
    S = (1.0 - coeff) * (dx * dy);
    if (WGD->m[cutcell_index] == 1.0) {
      WGD->m[cutcell_index] = coeff;
    } else {
      WGD->m[cutcell_index] -= (1.0 - coeff);
      if (WGD->m[cutcell_index] < 0.0) {
        WGD->m[cutcell_index] = 0.0;
      }
    }
  }

  return S;
}
