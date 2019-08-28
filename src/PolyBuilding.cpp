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
    for (auto j=j_start; j<j_end; j++)
    {
      y_cent = (j+0.5)*UGD->dy;         // Center of cell y coordinate
      for (auto i=i_start; i<i_end; i++)
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


/**
*
* This function applies the upwind cavity in front of the building to buildings defined as polygons.
* This function reads in building features like nodes, building height and base height and uses
* features of the building defined in the class constructor and setCellsFlag function. It defines
* cells in the upwind area and applies the approperiate parameterization to them.
* More information: "Improvements to a fast-response urban wind model, M. Nelson et al. (2008)"
*
*/
void PolyBuilding::upwindCavity (const URBInputData* UID, URBGeneralData* UGD)
{
  float tol = 10.0*M_PI/180.0;         // Upwind cavity is applied if the angle
                                       // is in 10 degree of the perpendicular direction
  float retarding_factor = 0.4;        // In the outer region, velocities are reduced by 40% (Bagal et al. (2004))
  float height_factor = 0.6;           // Height of both elipsoids (inner and outer) extends to 0.6H
  float length_factor = 0.4;
  int k_top;
  float uh_rotation, vh_rotation;     // Velocity components at the height of the building after rotation
  float vortex_height;
  float retarding_height;
  float x_average, y_average;
  int upwind_i_start, upwind_i_end, upwind_j_start, upwind_j_end;
  float z_front;
  float x_u, y_u, x_v, y_v, x_w, y_w;
  float x_intersect_u, x_ellipse_u, xrz_u;
  float x_intersect_v, x_ellipse_v, xrz_v;
  float x_intersect_w, x_ellipse_w;
  float rz_end;
  float u_rotation, v_rotation;
  std::vector<float> perpendicular_dir;
  std::vector<float> effective_gamma;
  std::vector<float> face_length;
  std::vector<float> Lf_face;

  upwind_rel_dir.resize (polygonVertices.size(), 0.0);      // Upwind reletive direction for each face
  xf1.resize (polygonVertices.size(),0.0);
  xf2.resize (polygonVertices.size(),0.0);
  yf1.resize (polygonVertices.size(),0.0);
  yf2.resize (polygonVertices.size(),0.0);
  perpendicular_dir.resize (polygonVertices.size(), 0.0);
  effective_gamma.resize (polygonVertices.size(), 0.0);
  int counter = 0;

  int index_building_face = i_building_cent + j_building_cent*UGD->nx + (k_end)*UGD->nx*UGD->ny;
  //std::cout << "k_end: " << k_end << std::endl;
  u0_h = UGD->u0[index_building_face];         // u velocity at the height of building at the centroid
  v0_h = UGD->v0[index_building_face];         // v velocity at the height of building at the centroid
  // Wind direction of initial velocity at the height of building at the centroid
  upwind_dir = atan2(v0_h,u0_h);
  for (auto id=0; id<polygonVertices.size()-1; id++)
  {
    xf1[id] = 0.5*(polygonVertices[id].x_poly-polygonVertices[id+1].x_poly)*cos(upwind_dir)+
              0.5*(polygonVertices[id].y_poly-polygonVertices[id+1].y_poly)*sin(upwind_dir);
    yf1[id] = -0.5*(polygonVertices[id].x_poly-polygonVertices[id+1].x_poly)*sin(upwind_dir)+
              0.5*(polygonVertices[id].y_poly-polygonVertices[id+1].y_poly)*cos(upwind_dir);
    xf2[id] = 0.5*(polygonVertices[id+1].x_poly-polygonVertices[id].x_poly)*cos(upwind_dir)+
              0.5*(polygonVertices[id+1].y_poly-polygonVertices[id].y_poly)*sin(upwind_dir);
    yf2[id] = -0.5*(polygonVertices[id+1].x_poly-polygonVertices[id].x_poly)*sin(upwind_dir)+
              0.5*(polygonVertices[id+1].y_poly-polygonVertices[id].y_poly)*cos(upwind_dir);
    // Calculate upwind reletive direction for each face
    upwind_rel_dir[id] = atan2(yf2[id]-yf1[id],xf2[id]-xf1[id])+0.5*M_PI;
    if (upwind_rel_dir[id] > M_PI+0.0001)
    {
      upwind_rel_dir[id] -= 2*M_PI;
    }

    if (abs(upwind_rel_dir[id]) > M_PI-tol)
    {
      perpendicular_dir[id] = atan2(polygonVertices[id+1].y_poly-polygonVertices[id].y_poly,polygonVertices[id+1].x_poly-polygonVertices[id].x_poly)+0.5*M_PI;
      if (perpendicular_dir[id] <= -M_PI)
      {
        perpendicular_dir[id] += 2*M_PI;
      }

      if (perpendicular_dir[id] >= 0.75*M_PI)
      {
        effective_gamma[id] = perpendicular_dir[id]-M_PI;
      }
      else if (perpendicular_dir[id] >= 0.25*M_PI)
      {
        effective_gamma[id] = perpendicular_dir[id]-0.5*M_PI;
      }
      else if (perpendicular_dir[id] < -0.75*M_PI)
      {
        effective_gamma[id] = perpendicular_dir[id]+M_PI;
      }
      else if (perpendicular_dir[id] < -0.25*M_PI)
      {
        effective_gamma[id] = perpendicular_dir[id]+0.5*M_PI;
      }
      uh_rotation = u0_h*cos(effective_gamma[id])+v0_h*sin(effective_gamma[id]);
      vh_rotation = -u0_h*sin(effective_gamma[id])+v0_h*cos(effective_gamma[id]);
      face_length.push_back(sqrt(pow(xf2[id]-xf1[id],2.0)+pow(yf2[id]-yf1[id],2.0)));
      Lf_face.push_back(abs(UGD->lengthf_coeff*face_length[counter]*cos(upwind_rel_dir[id])/(1+0.8*face_length[counter]/height_eff)));

      // High-rise Modified Vortex Parameterization (HMVP) (Bagal et al. (2004))
      if (UID->simParams->upwindCavityFlag == 3)
      {
        vortex_height = MIN_S(face_length[counter],height_eff);
        retarding_height = height_eff;
      }
      else
      {
        vortex_height = height_eff;
        retarding_height = height_eff;
      }

      // Defining index related to the height that upwind cavity is being applied
      for (auto k=k_start; k<UGD->z.size(); k++)
      {
        k_top = k+1;
        if (height_factor*retarding_height + base_height <= UGD->z[k])
        {
          break;
        }
      }
      // Defining limits of the upwind cavity in x and y directions
      upwind_i_start = MAX_S(std::round(MIN_S(polygonVertices[id].x_poly, polygonVertices[id+1].x_poly)/UGD->dx)-std::round(1*Lf_face[counter]/UGD->dx)-1, 1);
      upwind_i_end = MIN_S(std::round(MAX_S(polygonVertices[id].x_poly, polygonVertices[id+1].x_poly)/UGD->dx)+std::round(1*Lf_face[counter]/UGD->dx), UGD->nx-2);
      upwind_j_start = MAX_S(std::round(MIN_S(polygonVertices[id].y_poly, polygonVertices[id+1].y_poly)/UGD->dy)-std::round(1*Lf_face[counter]/UGD->dy)-1, 1);
      upwind_j_end = MIN_S(std::round(MAX_S(polygonVertices[id].y_poly, polygonVertices[id+1].y_poly)/UGD->dy)+std::round(1*Lf_face[counter]/UGD->dy), UGD->ny-2);
      x_average = 0.5*(polygonVertices[id].x_poly+polygonVertices[id+1].x_poly);        // x-location of middle of the face
      y_average = 0.5*(polygonVertices[id].y_poly+polygonVertices[id+1].y_poly);        // y-location of middle of the face

      // Apply the upwind parameterization
      for (auto k=k_start; k<k_top; k++)
      {
        z_front = UGD->z[k]-base_height;            // Height from the base of the building
        for (auto j=upwind_j_start; j<upwind_j_end; j++)
        {
          for (auto i=upwind_i_start; i<upwind_i_end; i++)
          {
            x_u = (i*UGD->dx-x_average)*cos(upwind_dir)+((j+0.5)*UGD->dy-y_average)*sin(upwind_dir);      // x-location of u velocity
            y_u = -(i*UGD->dx-x_average)*sin(upwind_dir)+((j+0.5)*UGD->dy-y_average)*cos(upwind_dir);     // y-location of u velocity
            x_v = ((i+0.5)*UGD->dx-x_average)*cos(upwind_dir)+(j*UGD->dy-y_average)*sin(upwind_dir);      // x-location of v velocity
            y_v = -((i+0.5)*UGD->dx-x_average)*sin(upwind_dir)+(j*UGD->dy-y_average)*cos(upwind_dir);      // y-location of v velocity
            x_w = ((i+0.5)*UGD->dx-x_average)*cos(upwind_dir)+((j+0.5)*UGD->dy-y_average)*sin(upwind_dir);      // x-location of w velocity
            y_w = -((i+0.5)*UGD->dx-x_average)*sin(upwind_dir)+((j+0.5)*UGD->dy-y_average)*cos(upwind_dir);     // y-location of w velocity

            if ( (abs(y_u)<=abs(yf2[id])) && (height_factor*vortex_height>z_front))
            {
              // Intersection of a parallel line to the line goes through (xf1,yf1) and (xf2,yf2) and the y=y_u
              x_intersect_u = ((xf2[id]-xf1[id])/(yf2[id]-yf1[id]))*(y_u-yf1[id])+xf1[id];
              x_ellipse_u = -Lf_face[counter]*sqrt((1-pow(y_u/abs(yf2[id]), 2.0))*(1-pow(z_front/(height_factor*vortex_height), 2.0)));
              xrz_u = -Lf_face[counter]*sqrt((1-pow(y_u/abs(yf2[id]), 2.0))*(1-pow(z_front/(height_factor*retarding_height), 2.0)));
              rz_end = length_factor*x_ellipse_u;
              icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
              icell_face = i+j*UGD->nx+k*UGD->nx*UGD->ny;
              if (UID->simParams->upwindCavityFlag == 1)            // Rockle parameterization
              {
                if ( (x_u-x_intersect_u>=x_ellipse_u) && (x_u-x_intersect_u<=0.1*UGD->dxy) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  UGD->u0[icell_face] = 0.0;
                }
              }
              else
              {
                if ( (x_u-x_intersect_u>=xrz_u) && (x_u-x_intersect_u<rz_end) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  if (UID->simParams->upwindCavityFlag == 3)        // High-rise Modified Vortex Parameterization (HMVP) (Bagal et al. (2004))
                  {
                    UGD->u0[icell_face] *= ((x_u-x_intersect_u-xrz_u)*(retarding_factor-1.0)/(rz_end-xrz_u)+1.0);
                  }
                  else          // Modified Vortex Parameterization (MVP)
                  {
                    UGD->u0[icell_face] *= retarding_factor;
                  }
                }
                if ( (x_u-x_intersect_u >= length_factor*x_ellipse_u) && (x_u-x_intersect_u <= 0.1*UGD->dxy) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  u_rotation = UGD->u0[icell_face]*cos(effective_gamma[id]);
                  v_rotation = UGD->u0[icell_face]*sin(effective_gamma[id]);
                  if (abs(perpendicular_dir[id]) >= 0.25*M_PI && abs(perpendicular_dir[id]) <= 0.75*M_PI)
                  {
                    v_rotation = -vh_rotation*(-height_factor*cos(M_PI*z_front/(0.5*vortex_height))+0.05)
                                  *(-height_factor*sin(M_PI*abs(x_u-x_intersect_u)/(length_factor*Lf_face[counter])));
                  }
                  else
                  {
                    u_rotation = -uh_rotation*(-height_factor*cos(M_PI*z_front/(0.5*vortex_height))+0.05)
                                  *(-height_factor*sin(M_PI*abs(x_u-x_intersect_u)/(length_factor*Lf_face[counter])));
                  }
                  UGD->u0[icell_face] = u_rotation*cos(-effective_gamma[id])+v_rotation*sin(-effective_gamma[id]);
                }
              }
            }
            // v velocity
            if ( (abs(y_v)<=abs(yf2[id])) && (height_factor*vortex_height>z_front))
            {
              x_intersect_v = ((xf2[id]-xf1[id])/(yf2[id]-yf1[id]))*(y_v-yf1[id])+xf1[id];
              x_ellipse_v = -Lf_face[counter]*sqrt((1-pow(y_v/abs(yf2[id]), 2.0))*(1-pow(z_front/(height_factor*vortex_height), 2.0)));
              xrz_v = -Lf_face[counter]*sqrt((1-pow(y_v/abs(yf2[id]), 2.0))*(1-pow(z_front/(height_factor*retarding_height), 2.0)));
              rz_end = length_factor*x_ellipse_v;
              icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
              icell_face = i+j*UGD->nx+k*UGD->nx*UGD->ny;
              if (UID->simParams->upwindCavityFlag == 1)        // Rockle parameterization
              {
                if ( (x_v-x_intersect_v>=x_ellipse_v) && (x_v-x_intersect_v<=0.1*UGD->dxy) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  UGD->v0[icell_face] = 0.0;
                }
              }
              else
              {
                if ( (x_v-x_intersect_v>=xrz_v) && (x_v-x_intersect_v<rz_end) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  if (UID->simParams->upwindCavityFlag == 3)      // High-rise Modified Vortex Parameterization (HMVP) (Bagal et al. (2004))
                  {
                    UGD->v0[icell_face] *= ((x_v-x_intersect_v-xrz_v)*(retarding_factor-1.0)/(rz_end-xrz_v)+1.0);
                  }
                  else            // Modified Vortex Parameterization (MVP)
                  {
                    UGD->v0[icell_face] *= retarding_factor;
                  }
                }
                if ( (x_v-x_intersect_v >= length_factor*x_ellipse_v) && (x_v-x_intersect_v <= 0.1*UGD->dxy) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  u_rotation = UGD->v0[icell_face]*sin(effective_gamma[id]);
                  v_rotation = UGD->v0[icell_face]*cos(effective_gamma[id]);
                  if (abs(perpendicular_dir[id]) >= 0.25*M_PI && abs(perpendicular_dir[id]) <= 0.75*M_PI)
                  {
                    v_rotation = -vh_rotation*(-height_factor*cos(M_PI*z_front/(0.5*vortex_height))+0.05)
                                  *(-height_factor*sin(M_PI*abs(x_v-x_intersect_v)/(length_factor*Lf_face[counter])));
                  }
                  else
                  {
                    u_rotation = -uh_rotation*(-height_factor*cos(M_PI*z_front/(0.5*vortex_height))+0.05)
                                  *(-height_factor*sin(M_PI*abs(x_v-x_intersect_v)/(length_factor*Lf_face[counter])));
                  }
                  UGD->v0[icell_face] = -u_rotation*sin(-effective_gamma[id])+v_rotation*cos(-effective_gamma[id]);
                }
              }
            }

            // w velocity
            if ( (abs(y_w)<=abs(yf2[id])) && (height_factor*vortex_height>z_front))
            {
              x_intersect_w = ((xf2[id]-xf1[id])/(yf2[id]-yf1[id]))*(y_w-yf1[id])+xf1[id];
              x_ellipse_w = -Lf_face[counter]*sqrt((1-pow(y_w/abs(yf2[id]), 2.0))*(1-pow(z_front/(height_factor*vortex_height), 2.0)));
              icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
              icell_face = i+j*UGD->nx+k*UGD->nx*UGD->ny;
              if (UID->simParams->upwindCavityFlag == 1)        // Rockle parameterization
              {
                if ( (x_w-x_intersect_w>=x_ellipse_w) && (x_w-x_intersect_w<=0.1*UGD->dxy) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  UGD->w0[icell_face] = 0.0;
                  if (i < UGD->nx-1 && j < UGD->ny-1 && k < UGD->nz-2)
                  {
                    UGD->icellflag[icell_cent] = 3;
                  }
                }
              }
              else
              {
                if ( (x_w-x_intersect_w>=x_ellipse_w) && (x_w-x_intersect_w < length_factor*x_ellipse_w) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  UGD->w0[icell_face] *= retarding_factor;
                  if (i < UGD->nx-1 && j < UGD->ny-1 && k < UGD->nz-2)
                  {
                    UGD->icellflag[icell_cent] = 3;
                  }
                }
                if ( (x_w-x_intersect_w >= length_factor*x_ellipse_w) && (x_w-x_intersect_w <= 0.1*UGD->dxy) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
                {
                  UGD->w0[icell_face] = -sqrt(pow(u0_h,2.0)+pow(v0_h,2.0))*(0.1*cos(M_PI*abs(x_w-x_intersect_w)/(length_factor*Lf_face[counter]))-0.05);
                  if (i < UGD->nx-1 && j < UGD->ny-1 && k < UGD->nz-2)
                  {
                    UGD->icellflag[icell_cent] = 3;
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


/**
*
* This function applies wake behind the building parameterization to buildings defined as polygons.
* The parameterization has two parts: near wake and far wake. This function reads in building features
* like nodes, building height and base height and uses features of the building defined in the class
* constructor ans setCellsFlag function. It defines cells in each wake area and applies the approperiate
* parameterization to them.
*
*/
void PolyBuilding::polygonWake (const URBInputData* UID, URBGeneralData* UGD, int building_id)
{

  std::vector<float> Lr_face, Lr_node;
  std::vector<int> perpendicular_flag;
  Lr_face.resize (polygonVertices.size(), -1.0);       // Length of wake for each face
  Lr_node.resize (polygonVertices.size(), 0.0);       // Length of wake for each node
  perpendicular_flag.resize (polygonVertices.size(), 0);
  upwind_rel_dir.resize (polygonVertices.size(), 0.0);      // Upwind reletive direction for each face
  float z_build;            // z value of each building point from its base height
  float yc, xc;
  float Lr_local, Lr_local_u, Lr_local_v, Lr_local_w;   // Local length of the wake for each velocity component
  float x_wall, x_wall_u, x_wall_v, x_wall_w;
  float y_norm, canyon_factor;
  int x_id_min;

  float Lr_ave;             // Average length of Lr
  float total_seg_length;       // Length of each edge
  int index_previous, index_next;       // Indices of previous and next nodes
  int stop_id = 0;
  int kk;
  float tol = 0.01*M_PI/180.0;
  float farwake_exp = 1.5;
  float farwake_factor = 3;
  float epsilon = 10e-10;
  int u_wake_flag, v_wake_flag, w_wake_flag;
  int i_u, j_u, i_v, j_v, i_w, j_w;         // i and j indices for x, y and z directions
  float xp, yp;
  float xu, yu, xv, yv, xw, yw;
  float dn_u, dn_v, dn_w;             // Length of cavity zone
  float farwake_vel;
  std::vector<double> u_temp, v_temp;
  u_temp.resize (UGD->nx*UGD->ny, 0.0);
  v_temp.resize (UGD->nx*UGD->ny, 0.0);
  std::vector<double> u0_modified, v0_modified;
  std::vector<int> u0_mod_id, v0_mod_id;

  int index_building_face = i_building_cent + j_building_cent*UGD->nx + (k_end)*UGD->nx*UGD->ny;
  u0_h = UGD->u0[index_building_face];         // u velocity at the height of building at the centroid
  v0_h = UGD->v0[index_building_face];         // v velocity at the height of building at the centroid

  // Wind direction of initial velocity at the height of building at the centroid
  upwind_dir = atan2(v0_h,u0_h);

  x1 = x2 = y1 = y2 = 0.0;
  xi.resize (polygonVertices.size(),0.0);      // Difference of x values of the centroid and each node
  yi.resize (polygonVertices.size(),0.0);     // Difference of y values of the centroid and each node
  polygon_area = 0.0;

  // Loop to calculate polygon area, differences in x and y values
  for (auto id=0; id<polygonVertices.size(); id++)
  {
    polygon_area += 0.5*(polygonVertices[id].x_poly*polygonVertices[id+1].y_poly-polygonVertices[id].y_poly*polygonVertices[id+1].x_poly);
    xi[id] = (polygonVertices[id].x_poly-building_cent_x)*cos(upwind_dir)+(polygonVertices[id].y_poly-building_cent_y)*sin(upwind_dir);
    yi[id] = -(polygonVertices[id].x_poly-building_cent_x)*sin(upwind_dir)+(polygonVertices[id].y_poly-building_cent_y)*cos(upwind_dir);

    // Find maximum and minimum differences in x and y values
    if (xi[id] < x1)
    {
      x1 = xi[id];        // Minimum x difference
    }
    if (xi[id] > x2)
    {
      x2 = xi[id];        // Maximum x difference
    }
    if (yi[id] < y1)
    {
      y1 = yi[id];         // Minimum y difference
    }
    if (yi[id] > y2)
    {
      y2 = yi[id];         // Maximum y difference
    }
  }

  polygon_area = abs(polygon_area);
  width_eff = polygon_area/(x2-x1);           // Effective width of the building
  length_eff = polygon_area/(y2-y1);          // Effective length of the building
  L_over_H = length_eff/height_eff;           // Length over height
  W_over_H = width_eff/height_eff;            // Width over height

  // Checking bounds of length over height and width over height
  if (L_over_H > 3.0)
  {
    L_over_H = 3.0;
  }
  if (L_over_H < 0.3)
  {
    L_over_H = 0.3;
  }
  if (W_over_H > 10.0)
  {
    W_over_H = 10.0;
  }

  // Calculating length of the downwind wake based on Fackrell (1984) formulation
  Lr = 1.8*height_eff*W_over_H/(pow(L_over_H,0.3)*(1+0.24*W_over_H));

  for (auto id=0; id<polygonVertices.size()-1; id++)
  {
    // Calculate upwind reletive direction for each face
    upwind_rel_dir[id] = atan2(yi[id+1]-yi[id],xi[id+1]-xi[id])+0.5*M_PI;
    if (upwind_rel_dir[id] > M_PI+0.0001)
    {
      upwind_rel_dir[id] -= 2*M_PI;
    }
    // Finding faces that are eligible for applying the far-wake parameterizations
    // angle between two points should be in -180 to 0 degree
    if ( abs(upwind_rel_dir[id]) < 0.5*M_PI)
    {
      // Calculate length of the far wake zone for each face
      Lr_face[id] = Lr*cos(upwind_rel_dir[id]);
    }
  }

  Lr_ave = total_seg_length = 0.0;
  // This loop interpolates the value of Lr for eligible faces to nodes of those faces
  for (auto id=0; id<polygonVertices.size()-1; id++)
  {
    // If the face is eligible for parameterization
    if (Lr_face[id] > 0.0)
    {
      index_previous = (id+polygonVertices.size()-2)%(polygonVertices.size()-1);     // Index of previous face
      index_next = (id+1)%(polygonVertices.size()-1);           // Index of next face
      if (Lr_face[index_previous] < 0.0 && Lr_face[index_next] < 0.0)
      {
        Lr_node[id] = Lr_face[id];
        Lr_node[id+1] = Lr_face[id];
      }
      else if (Lr_face[index_previous] < 0.0)
      {
        Lr_node[id] = Lr_face[id];
        Lr_node[id+1] = ((yi[index_next]-yi[index_next+1])*Lr_face[index_next]+(yi[id]-yi[index_next])*Lr_face[id])/(yi[id]-yi[index_next+1]);
      }
      else if (Lr_face[index_next] < 0.0)
      {
        Lr_node[id] = ((yi[id]-yi[index_next])*Lr_face[id]+(yi[index_previous]-yi[id])*Lr_face[index_previous])/(yi[index_previous]-yi[index_next]);
        Lr_node[id+1] = Lr_face[id];
      }
      else
      {
        Lr_node[id] = ((yi[id]-yi[index_next])*Lr_face[id]+(yi[index_previous]-yi[id])*Lr_face[index_previous])/(yi[index_previous]-yi[index_next]);
        Lr_node[id+1] = ((yi[index_next]-yi[index_next+1])*Lr_face[index_next]+(yi[id]-yi[index_next])*Lr_face[id])/(yi[id]-yi[index_next+1]);
      }
      Lr_ave += Lr_face[id]*(yi[id]-yi[index_next]);
      total_seg_length += (yi[id]-yi[index_next]);
    }

    if ((polygonVertices[id+1].x_poly > polygonVertices[0].x_poly-0.1) && (polygonVertices[id+1].x_poly < polygonVertices[0].x_poly+0.1)
         && (polygonVertices[id+1].y_poly > polygonVertices[0].y_poly-0.1) && (polygonVertices[id+1].y_poly < polygonVertices[0].y_poly+0.1))
    {
      stop_id = id;
      break;
    }

  }

  Lr = Lr_ave/total_seg_length;
  for (auto k = k_start; k < k_end; k++)
  {
    kk = k;
    if (0.75*(H-base_height)+base_height <= UGD->z[k])
    {
      break;
    }
  }

  for (auto k=k_end-1; k>=k_start; k--)
  {
    z_build = UGD->z[k] - base_height;
    for (auto id=0; id<=stop_id; id++)
    {
      if (abs(upwind_rel_dir[id]) < 0.5*M_PI)
      {
        if (abs(upwind_rel_dir[id]) < tol)
        {
          perpendicular_flag[id]= 1;
          x_wall = xi[id];
        }
        for (auto y_id=0; y_id <= 2*ceil(abs(yi[id]-yi[id+1])/UGD->dxy); y_id++)
        {
          yc = yi[id]-0.5*y_id*UGD->dxy;
          Lr_local = Lr_node[id]+(yc-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
          // Checking to see whether the face is perpendicular to the wind direction
          if(perpendicular_flag[id] == 0)
          {
            x_wall = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yc-yi[id])+xi[id];
          }
          if (yc >= 0.0)
          {
            y_norm = y2;
          }
          else
          {
            y_norm = y1;
          }
          canyon_factor = 1.0;
          x_id_min = -1;
          for (auto x_id=1; x_id <= ceil(Lr_local/UGD->dxy); x_id++)
          {
            xc = x_id*UGD->dxy;
            int i = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx;
            int j = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy;
            if ( i>=UGD->nx-2 && i<=0 && j>=UGD->ny-2 && j<=0)
            {
              break;
            }
            int icell_cent = i+j*(UGD->nx-1)+kk*(UGD->nx-1)*(UGD->ny-1);
            if ( UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2 && x_id_min < 0)
            {
              x_id_min = x_id;
            }
            if ( (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2) && x_id_min > 0)
            {
              canyon_factor = xc/Lr;

              break;
            }
          }
          x_id_min = -1;
          for (auto x_id=1; x_id <= 2*ceil(farwake_factor*Lr_local/UGD->dxy); x_id++)
          {
            u_wake_flag = 1;
            v_wake_flag = 1;
            w_wake_flag = 1;
            xc = 0.5*x_id*UGD->dxy;
            int i = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx;
            int j = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy;
            if (i>=UGD->nx-2 && i<=0 && j>=UGD->ny-2 && j<=0)
            {
              break;
            }
            icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
            if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2 && x_id_min < 0)
            {
              x_id_min = x_id;
            }
            if (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2)
            {
              if (x_id_min >= 0)
              {
                if (UGD->ibuilding_flag[icell_cent] == building_id)
                {
                  x_id_min = -1;
                }
                else if (canyon_factor < 1.0)
                {
                  break;
                }
                else if (UGD->icellflag[i+j*(UGD->nx-1)+kk*(UGD->nx-1)*(UGD->ny-1)] == 0 || UGD->icellflag[i+j*(UGD->nx-1)+kk*(UGD->nx-1)*(UGD->ny-1)] == 2)
                {
                  break;
                }

              }
            }

            if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
            {
              i_u = std::round(((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx);
              j_u = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy;
              if (i_u<UGD->nx-2 && i_u>0 && j_u<UGD->ny-2 && j_u>0)
              {
                xp = i_u*UGD->dx-building_cent_x;
                yp = (j_u+0.5)*UGD->dy-building_cent_y;
                xu = xp*cos(upwind_dir)+yp*sin(upwind_dir);
                yu = -xp*sin(upwind_dir)+yp*cos(upwind_dir);
                Lr_local_u = Lr_node[id]+(yu-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
                if (perpendicular_flag[id] > 0)
                {
                  x_wall_u = xi[id];

                }
                else
                {
                  x_wall_u = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yu-yi[id])+ xi[id];
                }

                xu -= x_wall_u;
                if (abs(yu) < abs(y_norm) && abs(y_norm) > epsilon && z_build < height_eff && height_eff > epsilon)
                {
                  dn_u = sqrt((1.0-pow((yu/y_norm), 2.0))*(1.0-pow((z_build/height_eff),2.0))*pow((canyon_factor*Lr_local_u),2.0));
                }
                else
                {
                  dn_u = 0.0;
                }
                if (xu > farwake_factor*dn_u)
                {
                  u_wake_flag = 0;
                }
                icell_cent = i_u + j_u*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                icell_face = i_u + j_u*UGD->nx+k*UGD->nx*UGD->ny;
                if (dn_u > 0.0 && u_wake_flag == 1 && yu <= yi[id] && yu >= yi[id+1] && UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                {
                  // Far wake zone
                  if (xu > dn_u)
                  {
                    farwake_vel = UGD->u0[icell_face]*(1.0-pow((dn_u/(xu+UGD->wake_factor*dn_u)),farwake_exp));
                    if (canyon_factor == 1.0)
                    {
                      u0_modified.push_back(farwake_vel);
                      u0_mod_id.push_back(icell_face);
                      UGD->w0[i+j*UGD->nx+k*UGD->nx*UGD->ny] = 0.0;
                    }
                  }
                  // Cavity zone
                  else
                  {
                    UGD->u0[icell_face] = -u0_h*MIN_S(pow((1.0-xu/(UGD->cavity_factor*dn_u)),2.0),1.0)*MIN_S(sqrt(1.0-abs(yu/y_norm)),1.0);
                    UGD->w0[i+j*UGD->nx+k*UGD->nx*UGD->ny] = 0.0;
                  }
                }
              }

              i_v = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx;
              j_v = std::round(((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy);
              if (i_v<UGD->nx-2 && i_v>0 && j_v<UGD->ny-2 && j_v>0)
              {
                xp = (i_v+0.5)*UGD->dx-building_cent_x;
                yp = j_v*UGD->dy-building_cent_y;
                xv = xp*cos(upwind_dir)+yp*sin(upwind_dir);
                yv = -xp*sin(upwind_dir)+yp*cos(upwind_dir);
                Lr_local_v = Lr_node[id]+(yv-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
                if (perpendicular_flag[id] > 0)
                {
                  x_wall_v = xi[id];
                }
                else
                {
                  x_wall_v = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yv-yi[id]) + xi[id];
                }
                xv -= x_wall_v;

                if (abs(yv) < abs(y_norm) && abs(y_norm) > epsilon && z_build < height_eff && height_eff > epsilon)
                {
                  dn_v = sqrt((1.0-pow((yv/y_norm), 2.0))*(1.0-pow((z_build/height_eff),2.0))*pow((canyon_factor*Lr_local_v),2.0));
                }
                else
                {
                  dn_v = 0.0;
                }
                if (xv > farwake_factor*dn_v)
                {
                  v_wake_flag = 0;
                }
                icell_cent = i_v + j_v*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                icell_face = i_v + j_v*UGD->nx+k*UGD->nx*UGD->ny;
                if (dn_v > 0.0 && v_wake_flag == 1 && yv <= yi[id] && yv >= yi[id+1] && UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                {
                  // Far wake zone
                  if (xv > dn_v)
                  {
                    farwake_vel = UGD->v0[icell_face]*(1.0-pow((dn_v/(xv+UGD->wake_factor*dn_v)),farwake_exp));
                    if (canyon_factor == 1)
                    {
                      v0_modified.push_back(farwake_vel);
                      v0_mod_id.push_back(icell_face);
                      UGD->w0[i+j*UGD->nx+k*UGD->nx*UGD->ny] = 0.0;
                    }
                  }
                  // Cavity zone
                  else
                  {
                    UGD->v0[icell_face] = -v0_h*MIN_S(pow((1.0-xv/(UGD->cavity_factor*dn_v)),2.0),1.0)*MIN_S(sqrt(1.0-abs(yv/y_norm)),1.0);
                    UGD->w0[i+j*UGD->nx+k*UGD->nx*UGD->ny] = 0.0;
                  }
                }
              }

              i_w = ceil(((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx)-1;
              j_w = ceil(((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy)-1;
              if (i_w<UGD->nx-2 && i_w>0 && j_w<UGD->ny-2 && j_w>0)
              {
                xp = (i_w+0.5)*UGD->dx-building_cent_x;
                yp = (j_w+0.5)*UGD->dy-building_cent_y;
                xw = xp*cos(upwind_dir)+yp*sin(upwind_dir);
                yw = -xp*sin(upwind_dir)+yp*cos(upwind_dir);
                Lr_local_w = Lr_node[id]+(yw-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
                if (perpendicular_flag[id] > 0)
                {
                  x_wall_w = xi[id];
                }
                else
                {
                  x_wall_w = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yw-yi[id]) + xi[id];
                }
                xw -= x_wall_w;
                if (abs(yw) < abs(y_norm) && abs(y_norm) > epsilon && z_build < height_eff && height_eff > epsilon)
                {
                  dn_w = sqrt((1.0-pow(yw/y_norm, 2.0))*(1.0-pow(z_build/height_eff,2.0))*pow(canyon_factor*Lr_local_w,2.0));
                }
                else
                {
                  dn_w = 0.0;
                }

                if (xw > farwake_factor*dn_w)
                {
                  w_wake_flag = 0;
                }
                icell_cent = i_w + j_w*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                icell_face = i_w + j_w*UGD->nx+k*UGD->nx*UGD->ny;
                if (dn_w > 0.0 && w_wake_flag == 1 && yw <= yi[id] && yw >= yi[id+1] && UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                {
                  if (xw > dn_w)
                  {
                    if (canyon_factor == 1)
                    {
                      UGD->icellflag[icell_cent] = 5;
                    }
                  }
                  else
                  {
                    UGD->icellflag[icell_cent] = 4;
                  }
                }
                if (u_wake_flag == 0 && v_wake_flag == 0 && w_wake_flag == 0)
                {
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  for (auto x_id=0; x_id < u0_mod_id.size(); x_id++)
  {
    UGD->u0[u0_mod_id[x_id]] = u0_modified[x_id];
  }

  for (auto y_id=0; y_id < v0_mod_id.size(); y_id++)
  {
    UGD->v0[v0_mod_id[y_id]] = v0_modified[y_id];
  }

  u0_mod_id.clear();
  v0_mod_id.clear();
  u0_modified.clear();
  v0_modified.clear();
}





/**
*
* This function applies the street canyon parameterization to the qualified space between buildings defined as polygons.
* This function reads in building features like nodes, building height and base height and uses
* features of the building defined in the class constructor and setCellsFlag function. It defines
* cells qualified in the space between buildings and applies the approperiate parameterization to them.
* More information: "Improvements to a fast-response urban wind model, M. Nelson et al. (2008)"
*
*/
void PolyBuilding::streetCanyon (URBGeneralData *UGD)
{
  float tol = 0.01*M_PI/180.0;
  float angle_tol = 3*M_PI/4;
  float x_wall, x_wall_u, x_wall_v, x_wall_w;
  float xc, yc;
  int top_flag, canyon_flag;
  int k_ref;
  int reverse_flag;
  int x_id_min, x_id_max;
  int number_u, number_v;
  float u_component, v_component;
  float s;
  float velocity_mag;
  float canyon_dir;
  int d_build;
  int i_u, j_v;
  float x_u, y_u, x_v, y_v, x_w, y_w;
  float x_pos;
  float x_p, y_p;
  float cross_dir, x_ave, y_ave;
  float x_down, y_down;
  float segment_length;
  float downwind_rel_dir, along_dir;
  float cross_vel_mag, along_vel_mag;
  std::vector<int> perpendicular_flag;
  std::vector<float> perpendicular_dir;
  int i_build;


  for (auto build_id = 0; build_id < UGD->allBuildingsV.size(); build_id++)
  {
    i_build = UGD->building_id[build_id];
    xi.resize (UGD->allBuildingsV[i_build]->polygonVertices.size(),0.0);      // Difference of x values of the centroid and each node
    yi.resize (UGD->allBuildingsV[i_build]->polygonVertices.size(),0.0);     // Difference of y values of the centroid and each node
    upwind_rel_dir.resize (UGD->allBuildingsV[i_build]->polygonVertices.size(), 0.0);      // Upwind reletive direction for each face
    perpendicular_flag.resize (UGD->allBuildingsV[i_build]->polygonVertices.size(), 0);
    perpendicular_dir.resize (UGD->allBuildingsV[i_build]->polygonVertices.size(), 0.0);

    int index_building_face = UGD->allBuildingsV[i_build]->i_building_cent + UGD->allBuildingsV[i_build]->j_building_cent*UGD->nx + (UGD->allBuildingsV[i_build]->k_end)*UGD->nx*UGD->ny;
    UGD->allBuildingsV[i_build]->u0_h = UGD->u0[index_building_face];         // u velocity at the height of building at the centroid
    UGD->allBuildingsV[i_build]->v0_h = UGD->v0[index_building_face];         // v velocity at the height of building at the centroid

    // Wind direction of initial velocity at the height of building at the centroid
    UGD->allBuildingsV[i_build]->upwind_dir = atan2(UGD->allBuildingsV[i_build]->v0_h,UGD->allBuildingsV[i_build]->u0_h);

    for (auto id=0; id<UGD->allBuildingsV[i_build]->polygonVertices.size()-1; id++)
    {
      xi[id] = (UGD->allBuildingsV[i_build]->polygonVertices[id].x_poly-UGD->allBuildingsV[i_build]->building_cent_x)*cos(UGD->allBuildingsV[i_build]->upwind_dir)
              +(UGD->allBuildingsV[i_build]->polygonVertices[id].y_poly-UGD->allBuildingsV[i_build]->building_cent_y)*sin(UGD->allBuildingsV[i_build]->upwind_dir);
      yi[id] = -(UGD->allBuildingsV[i_build]->polygonVertices[id].x_poly-UGD->allBuildingsV[i_build]->building_cent_x)*sin(UGD->allBuildingsV[i_build]->upwind_dir)
              +(UGD->allBuildingsV[i_build]->polygonVertices[id].y_poly-UGD->allBuildingsV[i_build]->building_cent_y)*cos(UGD->allBuildingsV[i_build]->upwind_dir);
    }


    for (auto id=0; id<UGD->allBuildingsV[i_build]->polygonVertices.size()-1; id++)
    {
      // Calculate upwind reletive direction for each face
      upwind_rel_dir[id] = atan2(yi[id+1]-yi[id],xi[id+1]-xi[id])+0.5*M_PI;

      if (upwind_rel_dir[id] > M_PI)
      {
        upwind_rel_dir[id] -= 2*M_PI;
      }

      // angle between two points should be in -180 to 0 degree
      if ( abs(upwind_rel_dir[id]) < 0.5*M_PI-0.0001)
      {
        // Checking to see whether the face is perpendicular to the wind direction
        if (abs(upwind_rel_dir[id]) > M_PI-tol || abs(upwind_rel_dir[id]) < tol)
        {
          perpendicular_flag[id] = 1;
          x_wall = xi[id];
        }

        perpendicular_dir[id] = atan2(UGD->allBuildingsV[i_build]->polygonVertices[id+1].y_poly-UGD->allBuildingsV[i_build]->polygonVertices[id].y_poly,
                                UGD->allBuildingsV[i_build]->polygonVertices[id+1].x_poly-UGD->allBuildingsV[i_build]->polygonVertices[id].x_poly)+0.5*M_PI;
        if (perpendicular_dir[id] > M_PI)
        {
          perpendicular_dir[id] -= 2*M_PI;
        }
        for (auto y_id=0; y_id <= 2*ceil(abs(yi[id]-yi[id+1])/UGD->dxy); y_id++)
        {
          yc = MIN_S(yi[id],yi[id+1])+0.5*y_id*UGD->dxy;
          top_flag = 0;
          if(perpendicular_flag[id] == 0)
          {
            x_wall = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yc-yi[id])+xi[id];
          }

          for (auto k = UGD->allBuildingsV[i_build]->k_end-1; k >= UGD->allBuildingsV[i_build]->k_start; k--)
          {
            canyon_flag = 0;
            s = 0.0;
            reverse_flag = 0;
            x_id_min = -1;
            for (auto x_id = 1; x_id <= 2*ceil(UGD->allBuildingsV[i_build]->Lr/UGD->dxy); x_id++)
            {
              xc = 0.5*x_id*UGD->dxy;
              int i = ceil(((xc+x_wall)*cos(UGD->allBuildingsV[i_build]->upwind_dir)-yc*sin(UGD->allBuildingsV[i_build]->upwind_dir)
                              +UGD->allBuildingsV[i_build]->building_cent_x)/UGD->dx)-1;
              int j = ceil(((xc+x_wall)*sin(UGD->allBuildingsV[i_build]->upwind_dir)+yc*cos(UGD->allBuildingsV[i_build]->upwind_dir)
                              +UGD->allBuildingsV[i_build]->building_cent_y)/UGD->dy)-1;
              icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);

              if (i>=UGD->nx-2 && i<=0 && j>=UGD->ny-2 && j<=0)
              {
                break;
              }

              icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);

              if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2 && x_id_min < 0)
              {
                x_id_min = x_id;
              }
              if ( (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2) && x_id_min >= 0)
              {
                canyon_flag = 1;
                x_id_max = x_id-1;
                s = 0.5*(x_id_max-x_id_min)*UGD->dxy;

                if (top_flag == 0)
                {
                  k_ref = k+1;
                  int ic = ceil(((0.5*x_id_max*UGD->dxy+x_wall)*cos(UGD->allBuildingsV[i_build]->upwind_dir)-yc*sin(UGD->allBuildingsV[i_build]->upwind_dir)
                                  +UGD->allBuildingsV[i_build]->building_cent_x-0.001)/UGD->dx)-1;
                  int jc = ceil(((0.5*x_id_max*UGD->dxy+x_wall)*sin(UGD->allBuildingsV[i_build]->upwind_dir)+yc*cos(UGD->allBuildingsV[i_build]->upwind_dir)
                                  +UGD->allBuildingsV[i_build]->building_cent_y-0.001)/UGD->dy)-1;
                  icell_cent = ic+jc*(UGD->nx-1)+k_ref*(UGD->nx-1)*(UGD->ny-1);

                  int icell_face = ic+jc*UGD->nx+k_ref*UGD->nx*UGD->ny;
                  if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                  {
                    number_u = 0;
                    number_v = 0;
                    u_component = 0.0;
                    v_component = 0.0;
                    if (UGD->icellflag[icell_cent-1] != 0 && UGD->icellflag[icell_cent-1] != 2)
                    {
                      number_u += 1;
                      u_component += UGD->u0[icell_face];
                    }
                    if (UGD->icellflag[icell_cent+1] != 0 && UGD->icellflag[icell_cent+1] != 2)
                    {
                      number_u += 1;
                      u_component += UGD->u0[icell_face+1];
                    }
                    if (UGD->icellflag[icell_cent-(UGD->nx-1)] != 0 && UGD->icellflag[icell_cent-(UGD->nx-1)] != 2)
                    {
                      number_v += 1;
                      v_component += UGD->v0[icell_face];
                    }
                    if (UGD->icellflag[icell_cent+(UGD->nx-1)] != 0 && UGD->icellflag[icell_cent+(UGD->nx-1)] != 2)
                    {
                      number_v += 1;
                      v_component += UGD->v0[icell_face+UGD->nx];
                    }

                    if ( u_component != 0.0 && number_u > 0)
                    {
                      u_component /= number_u;
                    }
                    else
                    {
                      u_component = 0.0;
                    }
                    if ( v_component != 0.0 && number_v > 0)
                    {
                      v_component /= number_v;
                    }
                    else
                    {
                      v_component = 0.0;
                    }

                    if (number_u == 0 && number_v == 0)
                    {
                      canyon_flag = 0;
                      top_flag = 0;
                      s = 0.0;
                      break;
                    }
                    else if (number_u > 0 && number_v > 0)
                    {
                      velocity_mag = sqrt(pow(u_component,2.0)+pow(v_component,2.0));
                      canyon_dir = atan2(v_component,u_component);
                    }
                    else if (number_u > 0)
                    {
                      velocity_mag = abs(u_component);
                      if (u_component > 0.0)
                      {
                        canyon_dir = 0.0;
                      }
                      else
                      {
                        canyon_dir = M_PI;
                      }
                    }
                    else
                    {
                      velocity_mag = abs(v_component);
                      if (v_component > 0.0)
                      {
                        canyon_dir = 0.5*M_PI;
                      }
                      else
                      {
                        canyon_dir = -0.5*M_PI;
                      }
                    }

                    top_flag = 1;
                    icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);

                    if (abs(s) > 0.0)
                    {
                      if ( (UGD->ibuilding_flag[icell_cent] >= 0) && (UGD->allBuildingsV[UGD->ibuilding_flag[icell_cent]]->H < UGD->allBuildingsV[i_build]->H) && (UGD->z_face[k]/s < 0.65) )
                      {
                        canyon_flag = 0;
                        top_flag = 0;
                        s = 0.0;
                        break;
                      }
                    }
                  }
                  else
                  {
                    canyon_flag = 0;
                    top_flag = 0;
                    s = 0.0;
                    break;
                  }
                  if (velocity_mag > UGD->max_velmag)
                  {
                    canyon_flag = 0;
                    top_flag = 0;
                    s = 0.0;
                    break;
                  }
                }

                icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                if (UGD->ibuilding_flag[icell_cent] >= 0)
                {
                  d_build = UGD->ibuilding_flag[icell_cent];
                }
                int i = ceil(((xc-0.5*UGD->dxy+x_wall)*cos(UGD->allBuildingsV[i_build]->upwind_dir)-yc*sin(UGD->allBuildingsV[i_build]->upwind_dir)
                              +UGD->allBuildingsV[i_build]->building_cent_x-0.001)/UGD->dx)-1;
                int j = ceil(((xc-0.5*UGD->dxy+x_wall)*sin(UGD->allBuildingsV[i_build]->upwind_dir)+yc*cos(UGD->allBuildingsV[i_build]->upwind_dir)
                              +UGD->allBuildingsV[i_build]->building_cent_y-0.001)/UGD->dy)-1;
                for (auto j_id = 0; j_id < UGD->allBuildingsV[d_build]->polygonVertices.size()-1; j_id++)
                {
                  cross_dir = atan2(UGD->allBuildingsV[d_build]->polygonVertices[j_id+1].y_poly-UGD->allBuildingsV[d_build]->polygonVertices[j_id].y_poly,
                                    UGD->allBuildingsV[d_build]->polygonVertices[j_id+1].x_poly-UGD->allBuildingsV[d_build]->polygonVertices[j_id].x_poly)+0.5*M_PI;

                  if (cross_dir > M_PI+0.001)
                  {
                    cross_dir -= 2*M_PI;
                  }
                  x_ave = 0.5*(UGD->allBuildingsV[d_build]->polygonVertices[j_id+1].x_poly+UGD->allBuildingsV[d_build]->polygonVertices[j_id].x_poly);
                  y_ave = 0.5*(UGD->allBuildingsV[d_build]->polygonVertices[j_id+1].y_poly+UGD->allBuildingsV[d_build]->polygonVertices[j_id].y_poly);

                  x_down = ((i+0.5)*UGD->dx-x_ave)*cos(cross_dir) + ((j+0.5)*UGD->dy-y_ave)*sin(cross_dir);
                  y_down = -((i+0.5)*UGD->dx-x_ave)*sin(cross_dir) + ((j+0.5)*UGD->dy-y_ave)*cos(cross_dir);

                  if (abs(x_down) < 0.75*UGD->dxy)
                  {
                    segment_length = sqrt(pow(UGD->allBuildingsV[d_build]->polygonVertices[j_id+1].x_poly-UGD->allBuildingsV[d_build]->polygonVertices[j_id].x_poly, 2.0)
                                          +pow(UGD->allBuildingsV[d_build]->polygonVertices[j_id+1].y_poly-UGD->allBuildingsV[d_build]->polygonVertices[j_id].y_poly, 2.0));
                    if (abs(y_down) <= 0.5*segment_length)
                    {
                      downwind_rel_dir = canyon_dir-cross_dir;
                      if (downwind_rel_dir > M_PI+0.001)
                      {
                        downwind_rel_dir -= 2*M_PI;
                      }
                      if (downwind_rel_dir <= -M_PI)
                      {
                        downwind_rel_dir += 2*M_PI;
                      }
                      if (abs(downwind_rel_dir) < 0.5*M_PI+0.001)
                      {
                        reverse_flag = 1;
                        if (downwind_rel_dir >= 0.0)
                        {
                          along_dir = cross_dir-0.5*M_PI;
                        }
                        else
                        {
                          along_dir = cross_dir+0.5*M_PI;
                        }
                      }
                      else
                      {
                        reverse_flag = 0;
                        if (downwind_rel_dir >= 0.0)
                        {
                          along_dir = cross_dir+0.5*M_PI;
                        }
                        else
                        {
                          along_dir = cross_dir-0.5*M_PI;
                        }
                      }
                      if (along_dir > M_PI+0.001)
                      {
                        along_dir -= 2*M_PI;
                      }
                      if (along_dir <= -M_PI)
                      {
                        along_dir += 2*M_PI;
                      }
                      break;
                    }
                  }
                }
                if (cross_dir <= -M_PI)
                {
                  cross_dir += 2*M_PI;
                }
                if (reverse_flag == 1)
                {
                  if (cos(cross_dir-perpendicular_dir[id]) < -cos(angle_tol))
                  {
                    canyon_flag = 0;
                    s = 0;
                    top_flag = 0;
                  }
                }
                else
                {
                  if (cos(cross_dir-perpendicular_dir[id]) > cos(angle_tol))
                  {
                    canyon_flag = 0;
                    s = 0;
                    top_flag = 0;
                  }
                }
                break;
              }
            }


            if (canyon_flag == 1 && s > 0.9*UGD->dxy)
            {
              along_vel_mag = abs(velocity_mag*cos(canyon_dir-along_dir))*log(UGD->z[k]/UGD->z0)/log(UGD->z[k_ref]/UGD->z0);
              cross_vel_mag = abs(velocity_mag*cos(canyon_dir-cross_dir));
              for (auto x_id = x_id_min; x_id <= x_id_max; x_id++)
              {
                xc = 0.5*x_id*UGD->dxy;
                int i = ceil(((xc+x_wall)*cos(UGD->allBuildingsV[i_build]->upwind_dir)-yc*sin(UGD->allBuildingsV[i_build]->upwind_dir)
                                +UGD->allBuildingsV[i_build]->building_cent_x-0.001)/UGD->dx)-1;
                int j = ceil(((xc+x_wall)*sin(UGD->allBuildingsV[i_build]->upwind_dir)+yc*cos(UGD->allBuildingsV[i_build]->upwind_dir)
                                +UGD->allBuildingsV[i_build]->building_cent_y-0.001)/UGD->dy)-1;
                icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                {
                  i_u = std::round(((xc+x_wall)*cos(UGD->allBuildingsV[i_build]->upwind_dir)-yc*sin(UGD->allBuildingsV[i_build]->upwind_dir)
                                      +UGD->allBuildingsV[i_build]->building_cent_x)/UGD->dx);

                  x_p = i_u*UGD->dx-UGD->allBuildingsV[i_build]->building_cent_x;
                  y_p = (j+0.5)*UGD->dy-UGD->allBuildingsV[i_build]->building_cent_y;
                  x_u = x_p*cos(UGD->allBuildingsV[i_build]->upwind_dir)+y_p*sin(UGD->allBuildingsV[i_build]->upwind_dir);
                  y_u = -x_p*sin(UGD->allBuildingsV[i_build]->upwind_dir)+y_p*cos(UGD->allBuildingsV[i_build]->upwind_dir);

                  if(perpendicular_flag[id] == 0)
                  {
                    x_wall_u = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(y_u-yi[id])+xi[id];
                  }
                  else
                  {
                    x_wall_u = xi[id];
                  }
                  x_pos = x_u-x_wall_u;
                  if (x_pos <= s && x_pos > -0.5*UGD->dxy)
                  {
                    icell_face = i_u+j*UGD->nx+k*UGD->nx*UGD->ny;
                    UGD->u0[icell_face] = along_vel_mag*cos(along_dir)+cross_vel_mag*(2*x_pos/s)*2*(1-x_pos/s)*cos(cross_dir);
                  }

                  j_v = std::round(((xc+x_wall)*sin(UGD->allBuildingsV[i_build]->upwind_dir)+yc*cos(UGD->allBuildingsV[i_build]->upwind_dir)
                                      +UGD->allBuildingsV[i_build]->building_cent_y)/UGD->dy);
                  x_p = (i+0.5)*UGD->dx-UGD->allBuildingsV[i_build]->building_cent_x;
                  y_p = j_v*UGD->dy-UGD->allBuildingsV[i_build]->building_cent_y;
                  x_v = x_p*cos(UGD->allBuildingsV[i_build]->upwind_dir)+y_p*sin(UGD->allBuildingsV[i_build]->upwind_dir);
                  y_v = -x_p*sin(UGD->allBuildingsV[i_build]->upwind_dir)+y_p*cos(UGD->allBuildingsV[i_build]->upwind_dir);
                  if(perpendicular_flag[id] == 0)
                  {
                    x_wall_v = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(y_v-yi[id])+xi[id];
                  }
                  else
                  {
                    x_wall_v = xi[id];
                  }
                  x_pos = x_v-x_wall_v;
                  if (x_pos <= s && x_pos > -0.5*UGD->dxy)
                  {
                    icell_face = i+j_v*UGD->nx+k*UGD->nx*UGD->ny;
                    UGD->v0[icell_face] = along_vel_mag*sin(along_dir)+cross_vel_mag*(2*x_pos/s)*2*(1-x_pos/s)*sin(cross_dir);
                  }

                  x_p = (i+0.5)*UGD->dx-UGD->allBuildingsV[i_build]->building_cent_x;
                  y_p = (j+0.5)*UGD->dy-UGD->allBuildingsV[i_build]->building_cent_y;
                  x_w = x_p*cos(UGD->allBuildingsV[i_build]->upwind_dir)+y_p*sin(UGD->allBuildingsV[i_build]->upwind_dir);
                  y_w = -x_p*sin(UGD->allBuildingsV[i_build]->upwind_dir)+y_p*cos(UGD->allBuildingsV[i_build]->upwind_dir);
                  if(perpendicular_flag[id] == 0)
                  {
                    x_wall_w = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(y_w-yi[id])+xi[id];
                  }
                  else
                  {
                    x_wall_w = xi[id];
                  }
                  x_pos = x_w-x_wall_w;

                  if (x_pos <= s && x_pos > -0.5*UGD->dxy)
                  {
                    icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                    if (UGD->icellflag[icell_cent-(UGD->nx-1)*(UGD->ny-1)] != 0 && UGD->icellflag[icell_cent-(UGD->nx-1)*(UGD->ny-1)] != 2)
                    {
                      icell_face = i+j*UGD->nx+k*UGD->nx*UGD->ny;
                      if (reverse_flag == 0)
                      {
                        UGD->w0[icell_face] = -abs(0.5*cross_vel_mag*(1-2*x_pos/s))*(1-2*(s-x_pos)/s);
                      }
                      else
                      {
                        UGD->w0[icell_face] = abs(0.5*cross_vel_mag*(1-2*x_pos/s))*(1-2*(s-x_pos)/s);
                      }
                    }
                    UGD->icellflag[icell_cent] = 6;
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





/**
*
* This function applies the sidewall parameterization to the qualified space on the side of buildings defined as polygons.
* This function reads in building features like nodes, building height and base height and uses
* features of the building defined in the class constructor and setPolyBuilding and setCellsFlag functions. It defines
* cells qualified on the side of buildings and applies the approperiate parameterization to them.
* More information: "Comprehensive Evaluation of Fast-Response, Reynolds-Averaged Navier–Stokes, and Large-Eddy Simulation
* Methods Against High-Spatial-Resolution Wind-Tunnel Data in Step-Down Street Canyons, A. N. Hayati et al. (2017)"
*
*/
void PolyBuilding::sideWall (const URBInputData* UID, URBGeneralData* UGD)
{
  float tol = 10*M_PI/180.0;          // Sidewall is applied if outward normal of the face is in +/-10 degree perpendicular
                                      // to the local wind
  int side_wall_flag = 0;             // If 1, indicates that there are faces that are nominally parallel with the wind
  std::vector<float> face_rel_dir;    /**< Face relative angle to the perpendicular direction of the local wind */
  face_rel_dir.resize (polygonVertices.size(), 0.0);
  float R_scale_side;                 /**< Vortex size scaling factor */
  float R_cx_side;                    /**< Downwind length of the half-ellipse that defines the vortex recirculation region */
  float vd;                           /**< Half of lateral width of the elliptical recirculation region */
  float y_pref;
  int right_flag, left_flag;          // 1, dependent face eligible for parameterization; 0, not eligible
  int index_previous, index_next;     // Previous or next vertex
  float x_start_left, x_end_left, x_start_right, x_end_right;       // Start and end point of each left/right faces in x-direction
  float y_start_left, y_end_left, y_start_right, y_end_right;       // Start and end point of each left/right faces in y-direction
  float face_length;                // Length of the face
  float face_dir;                   // Direction of the face
  int i_start_right, j_start_right;         // i and j indices of the starting point for right face
  int i_start_left, j_start_left;           // i and j indices of the starting point for left face
  float u0_right, v0_right;                 // u0 and v0 values for the right face
  float u0_left, v0_left;                   // u0 and v0 values for the left face
  float x_p, y_p;
  float shell_width, shell_width_calc;
  float x,y;
  float x_u, x_v, y_u, y_v;
  float xp_u, xp_v, xp_c, yp_u, yp_v, yp_c;
  float internal_BL_width;
  int x_id_max, y_id_max;

  int index_building_face = i_building_cent + j_building_cent*UGD->nx + (k_end)*UGD->nx*UGD->ny;
  u0_h = UGD->u0[index_building_face];         // u velocity at the height of building at the centroid
  v0_h = UGD->v0[index_building_face];         // v velocity at the height of building at the centroid

  // Wind direction of initial velocity at the height of building at the centroid
  upwind_dir = atan2(v0_h,u0_h);

  xi.resize (polygonVertices.size(),0.0);      // Difference of x values of the centroid and each node
  yi.resize (polygonVertices.size(),0.0);     // Difference of y values of the centroid and each node

  // Loop to calculate x and y values of each polygon point in rotated coordinates
  for (auto id = 0; id < polygonVertices.size(); id++)
  {
    xi[id] = (polygonVertices[id].x_poly-building_cent_x)*cos(upwind_dir)+(polygonVertices[id].y_poly-building_cent_y)*sin(upwind_dir);
    yi[id] = -(polygonVertices[id].x_poly-building_cent_x)*sin(upwind_dir)+(polygonVertices[id].y_poly-building_cent_y)*cos(upwind_dir);
  }

  for (auto id = 0; id < polygonVertices.size()-1; id++)
  {
    // Face relative angle to the perpendicular angle to the local wind
    face_rel_dir[id] = atan2(yi[id+1]-yi[id], xi[id+1]-xi[id]) + 0.5*M_PI;
    if (face_rel_dir[id] > M_PI)
    {
      face_rel_dir[id] -= 2*M_PI;
    }
    if (abs(face_rel_dir[id]) >= 0.5*M_PI-tol && abs(face_rel_dir[id]) <= 0.5*M_PI+tol)
    {
      side_wall_flag = 1;         // Indicates that there are faces that are nominally parallel with the wind
    }
  }

  if (side_wall_flag == 1)
  {
    small_dimension = MIN_S(width_eff, H);
    long_dimension = MAX_S(width_eff, H);
    R_scale_side = pow(small_dimension, (2.0/3.0))*pow(long_dimension, (1.0/3.0));
    R_cx_side = 0.9*R_scale_side;
    vd = 0.5*0.22*R_scale_side;
    y_pref = vd/sqrt(0.5*R_cx_side);

    for (auto id = 0; id < polygonVertices.size()-1; id++)
    {
      // +/-10 degree perpendicular to the local wind
      if (abs(face_rel_dir[id]) >= 0.5*M_PI-tol && abs(face_rel_dir[id]) <= 0.5*M_PI+tol)
      {
        right_flag = 0;
        left_flag = 0;
        if (face_rel_dir[id] > 0.0)
        {
          index_previous = (id+polygonVertices.size()-2)%(polygonVertices.size()-1);
          // Finding the left face eligible for the parameterization
          if (abs(face_rel_dir[index_previous]) >= M_PI-tol)
          {
            left_flag = 1;
            x_start_left = polygonVertices[id].x_poly;
            y_start_left = polygonVertices[id].y_poly;
            x_end_left = polygonVertices[id+1].x_poly;
            y_end_left = polygonVertices[id+1].y_poly;
            face_length = sqrt(pow(x_start_left-x_end_left, 2.0) + pow(y_start_left-y_end_left, 2.0));
            face_dir = atan2(y_end_left-y_start_left,x_end_left-x_start_left);
          }
        }
        else
        {
          index_next = (id+1)%(polygonVertices.size()-1);
          // Finding the right face eligible for the parameterization
          if (abs(face_rel_dir[index_next]) >= M_PI-tol)
          {
            right_flag = 1;
            x_start_right = polygonVertices[id+1].x_poly;
            y_start_right = polygonVertices[id+1].y_poly;
            x_end_right = polygonVertices[id].x_poly;
            y_end_right = polygonVertices[id].y_poly;
            face_length = sqrt(pow(x_start_right-x_end_right, 2.0) + pow(y_start_right-y_end_right, 2.0));
            face_dir = atan2(y_end_right-y_start_right, x_end_right-x_start_right);
          }
        }
        // Loop through all points might be eligible for the parameterization
        for (auto k=k_end-1; k>=k_start; k--)
        {
          if (right_flag == 1)          // If the right face is eligible for the parameterization
          {
            i_start_right = ceil(x_start_right/UGD->dx)-1;
            j_start_right = ceil(y_start_right/UGD->dy)-1;
            for (auto j = MAX_S(0,j_start_right-1); j <= MIN_S(UGD->ny-2, j_start_right+1); j++)
            {
              for (auto i = MAX_S(0,i_start_right-1); i <= MIN_S(UGD->nx-2, i_start_right+1); i++)
              {
                icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
                icell_face = i + j*(UGD->nx) + k*(UGD->nx)*(UGD->ny);
                // If the cell is solid (building or terrain)
                if (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2)
                {
                  u0_right = UGD->u0[icell_face];
                  v0_right = UGD->v0[icell_face];
                }
                // If the cell is air, upwind or canopy vegetation
                else if (UGD->icellflag[icell_cent] == 1 || UGD->icellflag[icell_cent] == 3 || UGD->icellflag[icell_cent] == 9)
                {
                }
                // If the cell is anything else (not eligible for the sidewall)
                else
                {
                  right_flag = 0;
                }
              }
            }
            if (right_flag == 1)
            {
              x_id_max = ceil(MAX_S(face_length,R_cx_side)/(0.5*UGD->dxy));
              for (auto x_id = 1; x_id <= x_id_max; x_id++)
              {
                x_p = 0.5*x_id*UGD->dxy;
                shell_width = y_pref*sqrt(x_p);
                y_id_max = (ceil(shell_width/(0.5*UGD->dxy))+2);
                for (auto y_id = 1; y_id <= y_id_max; y_id++)
                {
                  y_p = -0.5*y_id*UGD->dxy;
                  x = x_start_right + x_p*cos(face_dir) - y_p*sin(face_dir);
                  y = y_start_right + x_p*sin(face_dir) + y_p*cos(face_dir);
                  int i = ceil(x/UGD->dx) - 1;
                  int j = ceil(y/UGD->dy) - 1;
                  icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
                  icell_face = i + j*(UGD->nx) + k*(UGD->nx)*(UGD->ny);
                  if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                  {
                    x_u = i*UGD->dx;
                    y_u = (j+0.5)*UGD->dy;
                    xp_u = (x_u-x_start_right)*cos(face_dir) + (y_u-y_start_right)*sin(face_dir);
                    yp_u = -(x_u-x_start_right)*sin(face_dir) + (y_u-y_start_right)*cos(face_dir);
                    shell_width_calc = 1-pow((0.5*R_cx_side-xp_u)/(0.5*R_cx_side), 2.0);
                    if (shell_width_calc > 0.0)
                    {
                      shell_width = vd*sqrt(shell_width_calc);
                    }
                    else
                    {
                      shell_width = 0.0;
                    }
                    internal_BL_width = y_pref*sqrt(xp_u);
                    if (abs(yp_u) <= shell_width)
                    {
                      UGD->u0[icell_face] = -u0_right*abs((shell_width-abs(yp_u))/vd);
                    }
                    else if (abs(yp_u) <= internal_BL_width)
                    {
                      UGD->u0[icell_face] = u0_right*log((abs(yp_u)+UGD->z0)/UGD->z0)/log((internal_BL_width+UGD->z0)/UGD->z0);
                    }

                    x_v = (i+0.5)*UGD->dx;
                    y_v = j*UGD->dy;
                    xp_v = (x_v-x_start_right)*cos(face_dir) + (y_v-y_start_right)*sin(face_dir);
                    yp_v = -(x_v-x_start_right)*sin(face_dir) + (y_v-y_start_right)*cos(face_dir);
                    shell_width_calc = 1-pow((0.5*R_cx_side-xp_v)/(0.5*R_cx_side), 2.0);
                    if (shell_width_calc > 0.0)
                    {
                      shell_width = vd*sqrt(shell_width_calc);
                    }
                    else
                    {
                      shell_width = 0.0;
                    }
                    internal_BL_width = y_pref*sqrt(xp_v);
                    if (abs(yp_v) <= shell_width)
                    {
                      UGD->v0[icell_face] = -v0_right*abs((shell_width-abs(yp_v))/vd);
                    }
                    else if (abs(yp_v) <= internal_BL_width)
                    {
                      UGD->v0[icell_face] = v0_right*log((abs(yp_v)+UGD->z0)/UGD->z0)/log((internal_BL_width+UGD->z0)/UGD->z0);
                    }

                    xp_c = (x_v-x_start_right)*cos(face_dir) + (y_u-y_start_right)*sin(face_dir);
                    yp_c = -(x_v-x_start_right)*sin(face_dir) + (y_u-y_start_right)*cos(face_dir);
                    shell_width_calc = 1-pow((0.5*R_cx_side-xp_c)/(0.5*R_cx_side), 2.0);
                    internal_BL_width = y_pref*sqrt(xp_c);
                    if (abs(yp_c) <= shell_width || abs(yp_c) <= internal_BL_width)
                    {
                      UGD->icellflag[icell_cent] = 10;
                    }
                  }
                }
              }
            }
          }
          if (left_flag == 1)           // If the left face is eligible for the parameterization
          {
            i_start_left = ceil(x_start_left/UGD->dx)-1;
            j_start_left = ceil(y_start_left/UGD->dy)-1;
            for (auto j = MAX_S(0,j_start_left-1); j <= MIN_S(UGD->ny-2, j_start_left+1); j++)
            {
              for (auto i = MAX_S(0,i_start_left-1); i <= MIN_S(UGD->nx-2, i_start_left+1); i++)
              {
                icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
                icell_face = i + j*(UGD->nx) + k*(UGD->nx)*(UGD->ny);
                // If the cell is solid (building or terrain)
                if (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2)
                {
                  u0_left = UGD->u0[icell_face];
                  v0_left = UGD->v0[icell_face];
                }
                // If the cell is air, upwind or canopy vegetation
                else if (UGD->icellflag[icell_cent] == 1 || UGD->icellflag[icell_cent] == 3 || UGD->icellflag[icell_cent] == 9)
                {
                }
                // If the cell anything else (not eligible for the parameterization)
                else
                {
                  left_flag = 0;
                }
              }
            }
            if (left_flag == 1)
            {
              x_id_max = ceil(MAX_S(face_length,R_cx_side)/(0.5*UGD->dxy));
              for (auto x_id = 1; x_id <= x_id_max; x_id++)
              {
                x_p = 0.5*x_id*UGD->dxy;
                shell_width = y_pref*sqrt(x_p);
                y_id_max = (ceil(shell_width/(0.5*UGD->dxy))+2);
                for (auto y_id = 1; y_id <= y_id_max; y_id++)
                {
                  y_p = 0.5*y_id*UGD->dxy;
                  x = x_start_left + x_p*cos(face_dir) - y_p*sin(face_dir);
                  y = y_start_left + x_p*sin(face_dir) + y_p*cos(face_dir);
                  int i = ceil(x/UGD->dx) - 1;
                  int j = ceil(y/UGD->dy) - 1;
                  icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
                  icell_face = i + j*(UGD->nx) + k*(UGD->nx)*(UGD->ny);
                  if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                  {
                    x_u = i*UGD->dx;
                    y_u = (j+0.5)*UGD->dy;
                    xp_u = (x_u-x_start_left)*cos(face_dir) + (y_u-y_start_left)*sin(face_dir);
                    yp_u = -(x_u-x_start_left)*sin(face_dir) + (y_u-y_start_left)*cos(face_dir);
                    shell_width_calc = 1-pow((0.5*R_cx_side-xp_u)/(0.5*R_cx_side), 2.0);
                    if (shell_width_calc > 0.0)
                    {
                      shell_width = vd*sqrt(shell_width_calc);
                    }
                    else
                    {
                      shell_width = 0.0;
                    }
                    internal_BL_width = y_pref*sqrt(xp_u);
                    if (abs(yp_u) <= shell_width)
                    {
                      UGD->u0[icell_face] = -u0_left*abs((shell_width-abs(yp_u))/vd);
                    }
                    else if (abs(yp_u) <= internal_BL_width)
                    {
                      UGD->u0[icell_face] = u0_left*log((abs(yp_u)+UGD->z0)/UGD->z0)/log((internal_BL_width+UGD->z0)/UGD->z0);
                    }

                    x_v = (i+0.5)*UGD->dx;
                    y_v = j*UGD->dy;
                    xp_v = (x_v-x_start_left)*cos(face_dir) + (y_v-y_start_left)*sin(face_dir);
                    yp_v = -(x_v-x_start_left)*sin(face_dir) + (y_v-y_start_left)*cos(face_dir);
                    shell_width_calc = 1-pow((0.5*R_cx_side-xp_v)/(0.5*R_cx_side), 2.0);
                    if (shell_width_calc > 0.0)
                    {
                      shell_width = vd*sqrt(shell_width_calc);
                    }
                    else
                    {
                      shell_width = 0.0;
                    }
                    internal_BL_width = y_pref*sqrt(xp_v);
                    if (abs(yp_v) <= shell_width)
                    {
                      UGD->v0[icell_face] = -v0_left*abs((shell_width-abs(yp_v))/vd);
                    }
                    else if (abs(yp_v) <= internal_BL_width)
                    {
                      UGD->v0[icell_face] = v0_left*log((abs(yp_v)+UGD->z0)/UGD->z0)/log((internal_BL_width+UGD->z0)/UGD->z0);
                    }

                    xp_c = (x_v-x_start_left)*cos(face_dir) + (y_u-y_start_left)*sin(face_dir);
                    yp_c = -(x_v-x_start_left)*sin(face_dir) + (y_u-y_start_left)*cos(face_dir);
                    shell_width_calc = 1-pow((0.5*R_cx_side-xp_c)/(0.5*R_cx_side), 2.0);
                    internal_BL_width = y_pref*sqrt(xp_c);
                    if (abs(yp_c) <= shell_width || abs(yp_c) <= internal_BL_width)
                    {
                      UGD->icellflag[icell_cent] = 10;
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
