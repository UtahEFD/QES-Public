/*
 * QES-Winds
 *
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
 *
 */


#include "PolyBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


PolyBuilding::PolyBuilding(const WINDSInputData* WID, WINDSGeneralData* WGD, int id)
              : Building()
{
  polygonVertices = WID->simParams->shpPolygons[id];
  H = WID->simParams->shpBuildingHeight[id];
  base_height = WGD->base_height[id];

}

/**
 *
 */
void PolyBuilding::setPolyBuilding(WINDSGeneralData* WGD)
{

  building_cent_x = 0.0;               // x-coordinate of the centroid of the building
  building_cent_y = 0.0;               // y-coordinate of the centroid of the building
  height_eff = H+base_height;       // Effective height of the building

  // Calculate the centroid coordinates of the building (average of all nodes coordinates)
  for (auto i=0; i<polygonVertices.size()-1; i++)
  {
    building_cent_x += polygonVertices[i].x_poly;
    building_cent_y += polygonVertices[i].y_poly;
  }
  building_cent_x /= polygonVertices.size()-1;
  building_cent_y /= polygonVertices.size()-1;

  i_building_cent = std::round(building_cent_x/WGD->dx)-1;   // Index of building centroid in x-direction
  j_building_cent = std::round(building_cent_y/WGD->dy)-1;   // Index of building centroid in y-direction

    x1 = x2 = y1 = y2 = 0.0;

  xi.resize (polygonVertices.size(),0.0);      // Difference of x values of the centroid and each node
  yi.resize (polygonVertices.size(),0.0);     // Difference of y values of the centroid and each node

  polygon_area = 0.0;

  for (auto id=0; id<polygonVertices.size(); id++)
  {
    xi[id] = (polygonVertices[id].x_poly-building_cent_x)*cos(upwind_dir)+(polygonVertices[id].y_poly-building_cent_y)*sin(upwind_dir);
    yi[id] = -(polygonVertices[id].x_poly-building_cent_x)*sin(upwind_dir)+(polygonVertices[id].y_poly-building_cent_y)*cos(upwind_dir);
  }

  // Loop to calculate polygon area, projections of x and y values of each point wrt upwind wind
  for (auto id=0; id<polygonVertices.size()-1; id++)
  {
    polygon_area += 0.5*(polygonVertices[id].x_poly*polygonVertices[id+1].y_poly-polygonVertices[id].y_poly*polygonVertices[id+1].x_poly);
    // Find maximum and minimum x and y values in rotated coordinates
    if (xi[id] < x1)
    {
      x1 = xi[id];        // Minimum x
    }
    if (xi[id] > x2)
    {
      x2 = xi[id];        // Maximum x
    }
    if (yi[id] < y1)
    {
      y1 = yi[id];         // Minimum y
    }
    if (yi[id] > y2)
    {
      y2 = yi[id];         // Maximum y
    }
  }

  polygon_area = abs(polygon_area);
  width_eff = polygon_area/(x2-x1);           // Effective width of the building
  length_eff = polygon_area/(y2-y1);          // Effective length of the building

}


/**
*
* This function defines bounds of the polygon building and sets the icellflag values
* for building cells. It applies the Stair-step method to define building bounds.
*
*/
void PolyBuilding::setCellFlags(const WINDSInputData* WID, WINDSGeneralData* WGD, int building_number)
{

  int mesh_type_flag = WID->simParams->meshTypeFlag;
  float ray_intersect;
  int num_crossing, vert_id, start_poly;


  // Loop to calculate maximum and minimum of x and y values of the building
  x_min = x_max = polygonVertices[0].x_poly;
  y_min = y_max = polygonVertices[0].y_poly;
  for (auto id = 1; id < polygonVertices.size(); id++)
  {
    if (polygonVertices[id].x_poly > x_max)
    {
      x_max = polygonVertices[id].x_poly;
    }
    if (polygonVertices[id].x_poly < x_min)
    {
      x_min = polygonVertices[id].x_poly;
    }
    if (polygonVertices[id].y_poly > y_max)
    {
      y_max = polygonVertices[id].y_poly;
    }
    if (polygonVertices[id].y_poly < y_min)
    {
      y_min = polygonVertices[id].y_poly;
    }
  }


  i_start = (x_min/WGD->dx);       // Index of building start location in x-direction
  i_end = (x_max/WGD->dx)+1;       // Index of building end location in x-direction
  j_start = (y_min/WGD->dy);       // Index of building start location in y-direction
  j_end = (y_max/WGD->dy)+1;       // Index of building end location in y-direction

  // Define start index of the building in z-direction
  for (auto k = 1; k < WGD->z.size(); k++)
  {
    k_start = k;
    if (base_height <= WGD->z_face[k])
    {
      break;
    }
  }

  // Define end index of the building in z-direction
  for (auto k = 0; k < WGD->z.size(); k++)
  {
    k_end = k+1;
    if (height_eff < WGD->z[k+1])
    {
      break;
    }
  }

  // Define cut end index of the building in z-direction
  for (auto k = 0; k < WGD->z.size(); k++)
  {
    k_cut_end = k+1;
    if (height_eff <= WGD->z_face[k+1])
    {
      break;
    }
  }

  // Find out which cells are going to be inside the polygone
  // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
  // Check the center of each cell, if it's inside, set that cell to building
  for (auto j=j_start; j<=j_end; j++)
  {
    y_cent = (j+0.5)*WGD->dy;         // Center of cell y coordinate
    for (auto i=i_start; i<=i_end; i++)
    {
      x_cent = (i+0.5)*WGD->dx;       // Center of cell x coordinate
      vert_id = 0;               // Node index
      start_poly = vert_id;
      num_crossing = 0;
      while (vert_id < polygonVertices.size()-1)
      {
        if ( (polygonVertices[vert_id].y_poly<=y_cent && polygonVertices[vert_id+1].y_poly>y_cent) ||
             (polygonVertices[vert_id].y_poly>y_cent && polygonVertices[vert_id+1].y_poly<=y_cent) )
        {
          ray_intersect = (y_cent-polygonVertices[vert_id].y_poly)/(polygonVertices[vert_id+1].y_poly-polygonVertices[vert_id].y_poly);
          if (x_cent < (polygonVertices[vert_id].x_poly+ray_intersect*(polygonVertices[vert_id+1].x_poly-polygonVertices[vert_id].x_poly)))
          {
            num_crossing += 1;
          }
        }
        vert_id += 1;
        if (polygonVertices[vert_id].x_poly == polygonVertices[start_poly].x_poly &&
            polygonVertices[vert_id].y_poly == polygonVertices[start_poly].y_poly)
        {
          vert_id += 1;
          start_poly = vert_id;
        }
      }
      // if num_crossing is odd = cell is oustside of the polygon
      // if num_crossing is even = cell is inside of the polygon
      if ( (num_crossing%2) != 0 )
      {
        for (auto k=k_start; k<k_end; k++)
        {
          int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
          if (WID->simParams->readCoefficientsFlag == 0)
          {
            WGD->icellflag[icell_cent] = 0;
          }
          WGD->ibuilding_flag[icell_cent] = building_number;
        }

      }

    }
  }

  
}
