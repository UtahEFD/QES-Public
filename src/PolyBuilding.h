#pragma	once

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include "Building.h"

using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;

#include "PolygonVertex.h"
#define MIN_S(x,y) ((x) < (y) ? (x) : (y))

/**
*
* This class is designed for the general building shape (polygons).
* It's an inheritance of the building class (has all the features defined in that class).
* In this class, first, the polygone buildings will be defined and then different
* parameterizations related to each building will be applied. For now, it only includes
* wake behind the building parameterization.
*
*/

class PolyBuilding : public Building
{
private:

    float x_min, x_max, y_min, y_max;      // Minimum and maximum values of x
                                           // and y for a building
    float x_cent, y_cent;                  /**< Coordinates of center of a cell */
    float polygon_area;                    /**< Polygon area */
    std::vector<float> xi, yi;
    int icell_cent, icell_face;
    float x1, x2, y1, y2;

    /** 
     * 
     */
    void finishBuildingPolybuiding( args... ) 
    {
        // Calculate the centroid coordinates of the building (average of all nodes coordinates)
      for (auto i=0; i<polygonVertices.size()-1; i++)
      {
        building_cent_x += polygonVertices[i].x_poly;
        building_cent_y += polygonVertices[i].y_poly;
      }
      building_cent_x /= polygonVertices.size()-1;
      building_cent_y /= polygonVertices.size()-1;

      // Define start index of the building in z-direction
      for (auto k=1; k<z.size(); k++)
      {
        k_start = k;
        if (baseHeight <= z[k])
        {
          break;
        }
      }

      // Define end index of the building in z-direction
      for (auto k=0; k<z.size(); k++)
      {
        k_end = k+1;
        if (height_eff < z[k+1])
        {
          break;
        }
      }

      int i_building_cent = std::round(building_cent_x/dx);   // Index of building centroid in x-direction
      int j_building_cent = std::round(building_cent_y/dy);   // Index of building centroid in y-direction
      int index_building_face = i_building_cent + j_building_cent*nx + (k_end)*nx*ny;
      u0_h = u0[index_building_face];         // u velocity at the height of building at the centroid
      v0_h = v0[index_building_face];         // v velocity at the height of building at the centroid
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
    }
    

public:

    // Need to fix this constructor
    PolyBuilding()
        : Building()
    {
    }

    /**
    *
    * Converts data about rect building into something that poly can
    * building can use
    * 
    * This constructor creates a polygon type building using
    * rectangular building information...
    *
    */
    PolyBuilding( float xStart, float yStart, float bldElevation, float baseHeight,
                  int nx, int ny, int nz, float dx, float dy, float dz )
        : Building()
    {
        // extract the vertices from this definition here and make the
        // poly building...

        finishBuildingPolybuiding();
    }
    

    /**
    *
    * This constructor creates a polygon type building: calculates and initializes
    * all the features specific to this type of the building. This function reads in
    * nodes of the polygon along with height and base height of the building.
    *
    */
    PolyBuilding( const std::vector< polyVert > &polygonVertices, float bldElevation, float baseHeight,
                  int nx, int ny, int nz, float dx, float dy, float dz, std::vector<double> &u0, std::vector<double> &v0, std::vector<float> z)
                  : Building()
    {
      building_cent_x = 0;               // x-coordinate of the centroid of the building
      building_cent_y = 0;               // y-coordinate of the centroid of the building
      height_eff = bldElevation+baseHeight;       // Effective height of the building

      finishBuildingPolybuiding();
    }

    virtual void parseValues()
    {
        // let it parse XML here if we get there...
    }


    // make sure virtual functions from "Building" get implemented
    // here ...
    

    /**
    *
    * This function defines bounds of the polygon building and sets the icellflag values
    * for building cells. It applies the Stair-step method to define building bounds.
    *
    */
    void setCellFlags(URBGeneralData *ugd, float dx, float dy, float dz, std::vector<float> z, int nx, int ny, int nz, std::vector<int> &icellflag, int mesh_type_flag,
                        std::vector< polyVert > &polygonVertices, float baseHeight, float bldElevation)
    {

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

        i_start = (x_min/dx);       // Index of building start location in x-direction
        i_end = (x_max/dx)+1;       // Index of building end location in x-direction
        j_start = (y_min/dy);       // Index of building start location in y-direction
        j_end = (y_max/dy)+1;       // Index of building end location in y-direction

        // Find out which cells are going to be inside the polygone
        // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
        // Check the center of each cell, if it's inside, set that cell to building
        for (int j=j_start; j<j_end; j++)
        {
          y_cent = (j+0.5)*dy;         // Center of cell y coordinate
          for (int i=i_start; i<i_end; i++)
          {
            x_cent = (i+0.5)*dx;
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
              for (int k=k_start; k<k_end; k++)
              {
                int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
                icellflag[icell_cent] = 0;
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
    void polygonWake (std::vector< polyVert > &polygonVertices, float bldElevation, float baseHeight,float dx, float dy,
                      float dz, std::vector<float> z, int nx, int ny, int nz, float cavity_factor, float wake_factor, float dxy,
                      std::vector<int> &icellflag, std::vector<double> &u0, std::vector<double> &v0, std::vector<double> &w0, float max_velmag)
    {

      std::vector<float> Lr_face, Lr_node;
      std::vector<float> upwind_rel_dir;
      std::vector<int> perpendicular_flag;
      Lr_face.resize (polygonVertices.size(), -1.0);       // Length of wake for each face
      Lr_node.resize (polygonVertices.size(), 0.0);       // Length of wake for each node
      upwind_rel_dir.resize (polygonVertices.size(), 0.0);      // Upwind reletive direction for each face
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
      u_temp.resize (nx*ny, 0.0);
      v_temp.resize (nx*ny, 0.0);
      std::vector<double> u0_modified, v0_modified;
      std::vector<int> u0_mod_id, v0_mod_id;

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
          // Checking to see whether the face is perpendicular to the wind direction
          if (abs(upwind_rel_dir[id]) < tol)
          {
            perpendicular_flag[id]= 1;
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
        if (0.75*bldElevation+baseHeight <= z[k])
        {
          break;
        }
      }

      for (auto k=k_end-1; k>=k_start; k--)
      {
        z_build = z[k] - baseHeight;
        for (auto id=0; id<=stop_id; id++)
        {
          if (abs(upwind_rel_dir[id]) < 0.5*M_PI)
          {
            for (auto y_id=0; y_id <= 2*ceil(abs(yi[id]-yi[id+1])/dxy); y_id++)
            {
              yc = yi[id]-0.5*y_id*dxy;
              Lr_local = Lr_node[id]+(yc-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
              if(perpendicular_flag[id] > 0)
              {
                x_wall = xi[id];
              }
              else
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
              for (auto x_id=1; x_id <= ceil(Lr_local/dxy); x_id++)
              {
                xc = x_id*dxy;
                int i = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/dx;
                int j = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/dy;
                if ( i>=nx-2 && i<=0 && j>=ny-2 && j<=0)
                {
                  break;
                }
                icell_cent = i+j*(nx-1)+kk*(nx-1)*(ny-1);
                if (icellflag[icell_cent] != 0 && icellflag[icell_cent] != 2 && x_id_min < 0)
                {
                  x_id_min = x_id;
                }
                if (icellflag[icell_cent] == 0 && x_id_min > 0)
                {
                  canyon_factor = xc/Lr;
                  break;
                }
              }
              x_id_min = -1;
              for (auto x_id=1; x_id <= 2*ceil(farwake_factor*Lr_local/dxy); x_id++)
              {
                u_wake_flag = 1;
                v_wake_flag = 1;
                w_wake_flag = 1;
                xc = 0.5*x_id*dxy;
                int i = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/dx;
                int j = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/dy;
                if (i>=nx-2 && i<=0 && j>=ny-2 && j<=0)
                {
                  break;
                }
                icell_cent = i+j*(nx-1)+k*(nx-1)*(ny-1);
                if (icellflag[icell_cent] != 0 && icellflag[icell_cent] != 2 && x_id_min < 0)
                {
                  x_id_min = x_id;
                }
                if (icellflag[icell_cent] == 0)
                {
                  if (x_id_min >= 0)
                  {
                    x_id_min = -1;
                    icell_cent = i+j*(nx-1)+kk*(nx-1)*(ny-1);
                    if (canyon_factor < 1.0 || icellflag[icell_cent] == 0)
                    {
                      break;
                    }
                  }
                }
                icell_cent = i+j*(nx-1)+k*(nx-1)*(ny-1);

                if (icellflag[icell_cent] != 0 && icellflag[icell_cent] != 2)
                {
                  i_u = std::round(((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/dx);
                  j_u = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/dy;
                  if (i_u<nx-2 && i_u>0 && j_u<ny-2 && j_u>0)
                  {
                    xp = i_u*dx-building_cent_x;
                    yp = (j_u+0.5)*dy-building_cent_y;
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
                    icell_cent = i_u + j_u*(nx-1)+k*(nx-1)*(ny-1);
                    icell_face = i_u + j_u*nx+k*nx*ny;
                    if (dn_u > 0.0 && u_wake_flag == 1 && yu <= yi[id] && yu >= yi[id+1] && icellflag[icell_cent] != 0 && icellflag[icell_cent] != 2)
                    {
                      // Far wake zone
                      if (xu > dn_u)
                      {
                        farwake_vel = u0[icell_face]*(1.0-pow((dn_u/(xu+wake_factor*dn_u)),farwake_exp));
                        if (canyon_factor == 1.0)
                        {
                          u0_modified.push_back(farwake_vel);
                          u0_mod_id.push_back(icell_face);
                          w0[i+j*nx+k*nx*ny] = 0.0;
                        }
                      }
                      // Cavity zone
                      else
                      {
                        u0[icell_face] = -u0_h*MIN_S(pow((1.0-xu/(cavity_factor*dn_u)),2.0),1.0)*MIN_S(sqrt(1.0-abs(yu/y_norm)),1.0);
                        w0[i+j*nx+k*nx*ny] = 0.0;
                      }
                    }
                  }

                  i_v = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/dx;
                  j_v = std::round(((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/dy);
                  if (i_v<nx-2 && i_v>0 && j_v<ny-2 && j_v>0)
                  {
                    xp = (i_v+0.5)*dx-building_cent_x;
                    yp = j_v*dy-building_cent_y;
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
                    icell_cent = i_v + j_v*(nx-1)+k*(nx-1)*(ny-1);
                    icell_face = i_v + j_v*nx+k*nx*ny;
                    if (dn_v > 0.0 && v_wake_flag == 1 && yv <= yi[id] && yv >= yi[id+1] && icellflag[icell_cent] != 0 && icellflag[icell_cent] != 2)
                    {
                      // Far wake zone
                      if (xv > dn_v)
                      {
                        farwake_vel = v0[icell_face]*(1.0-pow((dn_v/(xv+wake_factor*dn_v)),farwake_exp));
                        if (canyon_factor == 1)
                        {
                          v0_modified.push_back(farwake_vel);
                          v0_mod_id.push_back(icell_face);
                          w0[i+j*nx+k*nx*ny] = 0.0;
                        }
                      }
                      // Cavity zone
                      else
                      {
                        v0[icell_face] = -v0_h*MIN_S(pow((1.0-xv/(cavity_factor*dn_v)),2.0),1.0)*MIN_S(sqrt(1.0-abs(yv/y_norm)),1.0);
                        w0[i+j*nx+k*nx*ny] = 0.0;
                      }
                    }
                  }

                  i_w = ceil(((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/dx)-1;
                  j_w = ceil(((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/dy)-1;
                  if (i_w<nx-2 && i_w>0 && j_w<ny-2 && j_w>0)
                  {
                    xp = (i_w+0.5)*dx-building_cent_x;
                    yp = (j_w+0.5)*dy-building_cent_y;
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
                    icell_cent = i_w + j_w*(nx-1)+k*(nx-1)*(ny-1);
                    icell_face = i_w + j_w*nx+k*nx*ny;
                    //std::cout << "dn_w:    "  << dn_w << std::endl;
                    if (dn_w > 0.0 && w_wake_flag == 1 && yw <= yi[id] && yw >= yi[id+1] && icellflag[icell_cent] != 0 && icellflag[icell_cent] != 2)
                    {
                      if (xw > dn_w)
                      {
                        if (canyon_factor == 1)
                        {
                          if (icellflag[icell_cent] == 4)
                          {
                            icellflag[icell_cent] = 12;
                          }
                          else
                          {
                            icellflag[icell_cent] = 5;
                          }
                        }
                      }
                      else
                      {
                        icellflag[icell_cent] = 4;
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
        u0[u0_mod_id[x_id]] = u0_modified[x_id];
      }

      for (auto y_id=0; y_id < v0_mod_id.size(); y_id++)
      {
        v0[v0_mod_id[y_id]] = v0_modified[y_id];
      }

      u0_mod_id.clear();
      v0_mod_id.clear();
      u0_modified.clear();
      v0_modified.clear();
    }
};
