#include "PolyBuilding.h"

// These take care of the circular reference
#include "URBInputData.h"
#include "URBGeneralData.h"


PolyBuilding::PolyBuilding(const URBInputData* UID, URBGeneralData* UGD, int id)
              : Building()
{
  polygonVertices = UID->simParams->shpPolygons[id];
  H = UID->simParams->shpBuildingHeight[id];
  base_height = UGD->base_height[id];

}

/**
 *
 */
void PolyBuilding::setPolyBuilding(URBGeneralData* UGD)
{

  building_cent_x = 0;               // x-coordinate of the centroid of the building
  building_cent_y = 0;               // y-coordinate of the centroid of the building
  height_eff = H+base_height;       // Effective height of the building

  // Calculate the centroid coordinates of the building (average of all nodes coordinates)
  for (auto i=0; i<polygonVertices.size()-1; i++)
  {
    building_cent_x += polygonVertices[i].x_poly;
    building_cent_y += polygonVertices[i].y_poly;
  }
  building_cent_x /= polygonVertices.size()-1;
  building_cent_y /= polygonVertices.size()-1;

  i_building_cent = std::round(building_cent_x/UGD->dx)-1;   // Index of building centroid in x-direction
  j_building_cent = std::round(building_cent_y/UGD->dy)-1;   // Index of building centroid in y-direction

}


/**
*
* This function defines bounds of the polygon building and sets the icellflag values
* for building cells. It applies the Stair-step method to define building bounds.
*
*/
void PolyBuilding::setCellFlags(const URBInputData* UID, URBGeneralData* UGD)
{

  int mesh_type_flag = UID->simParams->meshTypeFlag;
  float ray_intersect;
  int num_crossing, vert_id, start_poly;


  if (mesh_type_flag == 0)
  {
    // Loop to calculate maximum and minimum of x and y values of the building
    x_min = x_max = polygonVertices[0].x_poly;
    y_min = x_max = polygonVertices[0].y_poly;
    for (auto id=1; id<polygonVertices.size(); id++)
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

    i_start = (x_min/UGD->dx);       // Index of building start location in x-direction
    i_end = (x_max/UGD->dx)+1;       // Index of building end location in x-direction
    j_start = (y_min/UGD->dy);       // Index of building start location in y-direction
    j_end = (y_max/UGD->dy)+1;       // Index of building end location in y-direction

    // Define start index of the building in z-direction
    for (auto k=1; k<UGD->z.size(); k++)
    {
      k_start = k;
      if (base_height <= UGD->z[k])
      {
        break;
      }
    }

    // Define end index of the building in z-direction
    for (auto k=0; k<UGD->z.size(); k++)
    {
      k_end = k+1;
      if (height_eff < UGD->z[k+1])
      {
        break;
      }
    }

    // Find out which cells are going to be inside the polygone
    // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
    // Check the center of each cell, if it's inside, set that cell to building
    for (auto j=j_start; j<j_end-1; j++)
    {
      y_cent = (j+0.5)*UGD->dy;         // Center of cell y coordinate
      for (auto i=i_start; i<i_end-1; i++)
      {
        x_cent = (i+0.5)*UGD->dx;       // Center of cell x coordinate
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
          //std::cout<< "i:  " << i << "\t\t"<< "j:  " << j << std::endl;
          for (auto k=k_start; k<k_end; k++)
          {
            int icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
            UGD->icellflag[icell_cent] = 0;

            UGD->ibuilding_flag[icell_cent] = UGD->allBuildingsV.size()-1;
          }
        }
      }
    }
  }

}
