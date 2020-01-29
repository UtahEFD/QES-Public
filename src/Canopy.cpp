#include "Canopy.h"

#include "URBInputData.h"
#include "URBGeneralData.h"

void Canopy::canopyDefineBoundary(URBGeneralData* UGD,int cellFlagToUse)
{

  float ray_intersect;
  unsigned int num_crossing, vert_id, start_poly;

  // Define start index of the canopy in z-direction
  for (auto k=1; k<UGD->z.size(); k++) {
    k_start = k;
    if (base_height <= UGD->z[k]) {
      break;
    }
  }
  
  // Define end index of the canopy in z-direction   
  // Note that 0u means 0 unsigned
  for (auto k=0; k<UGD->z.size(); k++) {
    k_end = k+1;
    if (base_height+H < UGD->z[k+1]) {
      break;
    }
  }

  // Loop to calculate maximum and minimum of x and y values of the building
  x_min = x_max = polygonVertices[0].x_poly;
  y_min = x_max = polygonVertices[0].y_poly;
  for (auto id=1; id<polygonVertices.size(); id++) {
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
  
  i_start = (x_min/UGD->dx);       // Index of canopy start location in x-direction
  i_end = (x_max/UGD->dx)+1;       // Index of canopy end location in x-direction
  j_start = (y_min/UGD->dy);       // Index of canopy end location in y-direction
  j_end = (y_max/UGD->dy)+1;       // Index of canopy start location in y-direction
  
  // Find out which cells are going to be inside the polygone
  // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
  // Check the center of each cell, if it's inside, set that cell to building
  for (auto j=j_start; j<j_end; j++) {
    // Center of cell y coordinate
    y_cent = (j+0.5)*UGD->dy;         
    for (auto i=i_start; i<i_end; i++) {
      x_cent = (i+0.5)*UGD->dx;
      // Node index
      vert_id = 0;               
      start_poly = vert_id;
      num_crossing = 0;
      while (vert_id < polygonVertices.size()-1) {
	if ( (polygonVertices[vert_id].y_poly<=y_cent && polygonVertices[vert_id+1].y_poly>y_cent) ||
	     (polygonVertices[vert_id].y_poly>y_cent && polygonVertices[vert_id+1].y_poly<=y_cent) ) {
	  ray_intersect = (y_cent-polygonVertices[vert_id].y_poly)/
	    (polygonVertices[vert_id+1].y_poly-polygonVertices[vert_id].y_poly);
	  if (x_cent < (polygonVertices[vert_id].x_poly+ray_intersect*
			(polygonVertices[vert_id+1].x_poly-polygonVertices[vert_id].x_poly))) {
	    num_crossing += 1;
	  }
	}
	vert_id += 1;
	if (polygonVertices[vert_id].x_poly == polygonVertices[start_poly].x_poly &&
	    polygonVertices[vert_id].y_poly == polygonVertices[start_poly].y_poly) {
	  vert_id += 1;
	  start_poly = vert_id;
	}
      }
      
      // define icellflag @ (x,y) for all z(k) in [k_start...k_end]
      // if num_crossing is odd = cell is oustside of the polygon
      // if num_crossing is even = cell is inside of the polygon
      if ( (num_crossing%2) != 0 ) {
	for (auto k=k_start; k<k_end; k++) {
	  int icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
	  if( UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2) {
	    // Canopy cell
	    UGD->icellflag[icell_cent] = cellFlagToUse;             
	  }
	}
      } // end define icellflag!
    }
  }
  
  for (auto j=j_start; j<j_end-1; j++) {
    for (auto i=i_start; i<i_end-1; i++) {
      for (auto k=k_start; k<k_end; k++) {
	int icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
	// if the cell is defined as canopy
	if (UGD->icellflag[icell_cent] == cellFlagToUse) {
          int id = i+j*(UGD->nx-1);		       
          // define height of the canopy (above the surface)
          UGD->canopy_top[id] = base_height+H; 
        }
      }
    }
  }
}
