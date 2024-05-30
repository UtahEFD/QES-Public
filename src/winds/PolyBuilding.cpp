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

/**
 * @file PolyBuilding.cpp
 * @brief Designed for the general building shape (polygons).
 *
 * It's an inheritance of the building class (has all the features defined in that class).
 * In this class, first, the polygon buildings will be defined and then different
 * parameterizations related to each building will be applied.
 *
 * @sa Building
 * @sa ParseInterface
 */

#include "PolyBuilding.h"
#include "CutBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


PolyBuilding::PolyBuilding(const WINDSInputData *WID, WINDSGeneralData *WGD, int id)
  : Building()
{
  polygonVertices = WID->buildingsParams->SHPData->m_polygons[id];
  H = WID->buildingsParams->SHPData->m_features[WID->buildingsParams->shpHeightField][id] * WID->buildingsParams->heightFactor;
  base_height = WGD->base_height[id];

  ID = id;
}

PolyBuilding::PolyBuilding(const std::vector<polyVert> &iSP, float iH, float iBH, int iID)
  : Building()
{
  polygonVertices = iSP;
  H = iH;
  base_height = iBH;
  ID = iID;

  height_eff = base_height + H;
}

void PolyBuilding::setPolyBuilding(WINDSGeneralData *WGD)
{

  building_cent_x = 0.0;// x-coordinate of the centroid of the building
  building_cent_y = 0.0;// y-coordinate of the centroid of the building
  height_eff = H + base_height;// Effective height of the building

  // Calculate the centroid coordinates of the building (average of all nodes coordinates)
  for (auto i = 0u; i < polygonVertices.size() - 1; i++) {
    building_cent_x += polygonVertices[i].x_poly;
    building_cent_y += polygonVertices[i].y_poly;
  }
  building_cent_x /= polygonVertices.size() - 1;
  building_cent_y /= polygonVertices.size() - 1;

  i_building_cent = std::round(building_cent_x / WGD->dx) - 1;// Index of building centroid in x-direction
  j_building_cent = std::round(building_cent_y / WGD->dy) - 1;// Index of building centroid in y-direction

  // checking if polygon is rectangular
  if (polygonVertices.size() == 5) {
    float tol = 1.0 * M_PI / 180.0;
    std::vector<float> normal_x;
    std::vector<float> normal_y;

    normal_x.resize(polygonVertices.size() - 1, 0.0);
    normal_y.resize(polygonVertices.size() - 1, 0.0);

    // Calculating outward normal to face
    for (size_t id = 0; id < polygonVertices.size() - 1; id++) {
      normal_x[id] = polygonVertices[id + 1].y_poly - polygonVertices[id].y_poly;
      normal_y[id] = polygonVertices[id].x_poly - polygonVertices[id + 1].x_poly;
      float norm = pow(normal_x[id] * normal_x[id] + normal_y[id] * normal_y[id], -0.5);
      normal_x[id] *= norm;
      normal_y[id] *= norm;
    }

    // checking internal angle
    for (size_t id = 0; id < polygonVertices.size() - 2; id++) {
      float scalProd = normal_x[id] * normal_x[id + 1] + normal_y[id] * normal_y[id + 1];
      if (std::abs(scalProd) < std::abs(cos(0.5 * M_PI - tol))) {
        rectangular_flag = true;
      } else {
        rectangular_flag = false;
        break;
      }
    }
  } else {
    rectangular_flag = false;
  }

  return;
}


/**
 *
 * This function defines bounds of the polygon building and sets the icellflag values
 * for building cells. It applies the Stair-step method to define building bounds.
 *
 */
void PolyBuilding::setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_number)
{

  int mesh_type_flag = WID->simParams->meshTypeFlag;
  float ray_intersect;
  int num_crossing, start_poly;
  unsigned int vert_id;

  // Loop to calculate maximum and minimum of x and y values of the building
  x_min = x_max = polygonVertices[0].x_poly;
  y_min = y_max = polygonVertices[0].y_poly;
  for (auto id = 1u; id < polygonVertices.size(); id++) {
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


  i_start = (x_min / WGD->dx);// Index of building start location in x-direction
  i_end = (x_max / WGD->dx) + 1;// Index of building end location in x-direction
  j_start = (y_min / WGD->dy);// Index of building start location in y-direction
  j_end = (y_max / WGD->dy) + 1;// Index of building end location in y-direction

  // Define start index of the building in z-direction
  for (auto k = 1u; k < WGD->z.size(); k++) {
    k_start = k;
    if (base_height <= WGD->z_face[k + 1]) {
      break;
    }
  }

  // Define end index of the building in z-direction
  for (auto k = 0u; k < WGD->z.size(); k++) {
    k_end = k + 1;
    if (height_eff < WGD->z[k + 1]) {
      break;
    }
  }

  // Define cut end index of the building in z-direction
  for (auto k = 0u; k < WGD->z.size(); k++) {
    k_cut_end = k;
    if (height_eff <= WGD->z_face[k + 1]) {
      break;
    }
  }

  // Find out which cells are going to be inside the polygone
  // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
  // Check the center of each cell, if it's inside, set that cell to building
  for (auto j = j_start; j <= j_end; j++) {
    y_cent = (j + 0.5) * WGD->dy;// Center of cell y coordinate
    for (auto i = i_start; i <= i_end; i++) {
      x_cent = (i + 0.5) * WGD->dx;// Center of cell x coordinate
      vert_id = 0;// Node index
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
        for (auto k = k_start; k < k_end; k++) {
          int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
          if (WID->simParams->readCoefficientsFlag == 0 && WGD->icellflag[icell_cent] != 7) {
            WGD->icellflag[icell_cent] = 0;
          }
	  if (WGD->ibuilding_flag[icell_cent] == -1){
            WGD->ibuilding_flag[icell_cent] = building_number;
	  }
        }
        WGD->icellflag_footprint[i + j * (WGD->nx - 1)] = 0;
      }
    }
  }

  if (mesh_type_flag == 1 && WID->simParams->readCoefficientsFlag == 0)// Cut-cell method for buildings
  {
    cutBuilding->setCutCellFlags(WGD, this, building_number);
  }
}
