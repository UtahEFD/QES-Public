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
          WGD->icellflag[icell_cent] = 0;
          WGD->ibuilding_flag[icell_cent] = building_number;
        }

      }

    }
  }

}
