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
  upwind_rel_dir.resize (polygonVertices.size(), 0.0);      // Upwind reletive direction for each face

  // Calculate the centroid coordinates of the building (average of all nodes coordinates)
  for (auto i=0; i<polygonVertices.size()-1; i++)
  {
    building_cent_x += polygonVertices[i].x_poly;
    building_cent_y += polygonVertices[i].y_poly;
  }
  building_cent_x /= polygonVertices.size()-1;
  building_cent_y /= polygonVertices.size()-1;

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

  int i_building_cent = std::round(building_cent_x/UGD->dx);   // Index of building centroid in x-direction
  int j_building_cent = std::round(building_cent_y/UGD->dy);   // Index of building centroid in y-direction
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

  for (auto id=0; id<polygonVertices.size()-1; id++)
  {
    // Calculate upwind reletive direction for each face
    upwind_rel_dir[id] = atan2(yi[id+1]-yi[id],xi[id+1]-xi[id])+0.5*M_PI;
    if (upwind_rel_dir[id] > M_PI+0.0001)
    {
      upwind_rel_dir[id] -= 2*M_PI;
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

    // Find out which cells are going to be inside the polygone
    // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
    // Check the center of each cell, if it's inside, set that cell to building
    for (auto j=j_start; j<j_end; j++)
    {
      y_cent = (j+0.5)*UGD->dy;         // Center of cell y coordinate
      for (auto i=i_start; i<i_end; i++)
      {
        x_cent = (i+0.5)*UGD->dx;
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
            int icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
            UGD->icellflag[icell_cent] = 0;
          }
        }

      }
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
void PolyBuilding::polygonWake (const URBInputData* UID, URBGeneralData* UGD)
{

  std::vector<float> Lr_face, Lr_node;
  std::vector<int> perpendicular_flag;
  Lr_face.resize (polygonVertices.size(), -1.0);       // Length of wake for each face
  Lr_node.resize (polygonVertices.size(), 0.0);       // Length of wake for each node
  perpendicular_flag.resize (polygonVertices.size(), 0);
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

  for (auto id=0; id<polygonVertices.size()-1; id++)
  {
    // Finding faces that are eligible for applying the far-wake parameterizations
    // angle between two points should be in -180 to 0 degree
    if ( abs(upwind_rel_dir[id]) < 0.5*M_PI)
    {
      // Calculate length of the far wake zone for each face
      Lr_face[id] = Lr*cos(upwind_rel_dir[id]);
      // Checking to see whether the face is perpendicular to the wind direction
      if (abs(upwind_rel_dir[id]) < tol)
      {
        perpendicular_flag[id]= 1;
        x_wall = xi[id];
      }
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

  for (auto k=k_start; k<k_end; k++)
  {
    kk = k;
    if (0.75*H+base_height <= UGD->z[k])
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
        for (auto y_id=0; y_id <= 2*ceil(abs(yi[id]-yi[id+1])/UGD->dxy); y_id++)
        {
          yc = yi[id]-0.5*y_id*UGD->dxy;
          Lr_local = Lr_node[id]+(yc-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
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
                x_id_min = -1;
                icell_cent = i+j*(UGD->nx-1)+kk*(UGD->nx-1)*(UGD->ny-1);
                if (canyon_factor < 1.0 || UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2)
                {
                  break;
                }
              }
            }
            int icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);

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
                /*if (Lr_local_u > Lr)
                {
                  Lr_local_u = Lr;
                }*/
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
                /*if (Lr_local_v > Lr)
                {
                  Lr_local_v = Lr;
                }*/
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
                //std::cout << "dn_w:    "  << dn_w << std::endl;
                if (dn_w > 0.0 && w_wake_flag == 1 && yw <= yi[id] && yw >= yi[id+1] && UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                {
                  if (xw > dn_w)
                  {
                    if (canyon_factor == 1)
                    {
                      if (UGD->icellflag[icell_cent] == 4)
                      {
                        UGD->icellflag[icell_cent] = 8;
                      }
                      else
                      {
                        UGD->icellflag[icell_cent] = 5;
                      }
                    }
                  }
                  else
                  {
                    UGD->icellflag[icell_cent] = 4;
                    //std::cout << "icell:    " << icell_cent << std::endl;
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

  xf1.resize (polygonVertices.size(),0.0);
  xf2.resize (polygonVertices.size(),0.0);
  yf1.resize (polygonVertices.size(),0.0);
  yf2.resize (polygonVertices.size(),0.0);
  perpendicular_dir.resize (polygonVertices.size(), 0.0);
  effective_gamma.resize (polygonVertices.size(), 0.0);
  int counter = 0;

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
      face_length.push_back(sqrt(pow(xi[id+1]-xi[id],2.0)+pow(yi[id+1]-yi[id],2.0)));
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
      upwind_i_start = MAX_S(std::round(MIN_S(polygonVertices[id].x_poly, polygonVertices[id+1].x_poly)/UGD->dx)-std::round(Lf_face[counter]/UGD->dx)-1, 1);
      upwind_i_end = MIN_S(std::round(MAX_S(polygonVertices[id].x_poly, polygonVertices[id+1].x_poly)/UGD->dx)+std::round(Lf_face[counter]/UGD->dx), UGD->nx-2);
      upwind_j_start = MAX_S(std::round(MIN_S(polygonVertices[id].y_poly, polygonVertices[id+1].y_poly)/UGD->dy)-std::round(Lf_face[counter]/UGD->dy)-1, 1);
      upwind_j_end = MIN_S(std::round(MAX_S(polygonVertices[id].y_poly, polygonVertices[id+1].y_poly)/UGD->dy)+std::round(Lf_face[counter]/UGD->dy), UGD->ny-2);
      x_average = 0.5*(polygonVertices[id].x_poly+polygonVertices[id+1].x_poly);        // x-location of middle of the face
      y_average = 0.5*(polygonVertices[id].y_poly+polygonVertices[id+1].y_poly);        // y-location of middle of the face

      // Apply the upwind parameterization
      for (auto k=k_start; k<k_top; k++)
      {
        z_front = UGD->z[k]-base_height;            // Height from the base of the building
        for (auto i=upwind_i_start; i<upwind_i_end; i++)
        {
          for (auto j=upwind_j_start; j<upwind_j_end; j++)
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
              //
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
                if ( (x_u-x_intersect_u>=xrz_u) && (x_u-x_intersect_u<=rz_end) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
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
                  if (abs(upwind_rel_dir[id]) >= 0.25*M_PI && abs(upwind_rel_dir[id]) <= 0.75*M_PI)
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
                if ( (x_v-x_intersect_v>=xrz_v) && (x_v-x_intersect_v<=rz_end) && (UGD->icellflag[icell_cent] != 0) && (UGD->icellflag[icell_cent] != 2))
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
                  u_rotation = UGD->v0[icell_face]*cos(effective_gamma[id]);
                  v_rotation = UGD->v0[icell_face]*sin(effective_gamma[id]);
                  if (abs(upwind_rel_dir[id]) >= 0.25*M_PI && abs(upwind_rel_dir[id]) <= 0.75*M_PI)
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
                  UGD->v0[icell_face] *= retarding_factor;
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
