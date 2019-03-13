#pragma	once

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>

using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;

#include "PolygonVertex.h"

class PolyBuilding
{
public:

    PolyBuilding() 
    {
    }
    
    PolyBuilding( const std::vector< polyVert > &polygonVertices, float bldElevation )
    {
        // ????
    }

    void setCellsFlag (float dx, float dy, float dz, int nx, int ny, int nz, std::vector<int> &icellflag, int mesh_type_flag,
                        std::vector< polyVert > &polygonVertices, float baseHeight, float bldElevation)
    {

      int i_start, i_end, j_start, j_end, k_start, k_end;
      float x_min, x_max, y_min, y_max;
      float x_cent, y_cent;
      float ray_intersect;
      int num_crossing, vert_id, start_poly;

      if (mesh_type_flag == 0)
      {
        x_min = x_min = polygonVertices[0].x_poly;
        y_min = x_max = polygonVertices[0].y_poly;
        for (int id=1; id<polygonVertices.size(); id++)
        {
          if (polygonVertices[id].x_poly>x_max)
          {
            x_max = polygonVertices[id].x_poly;
          }
          if (polygonVertices[id].x_poly<x_min)
          {
            x_min = polygonVertices[id].x_poly;
          }
          if (polygonVertices[id].y_poly>y_max)
          {
            y_max = polygonVertices[id].y_poly;
          }
          if (polygonVertices[id].y_poly<y_min)
          {
            y_min = polygonVertices[id].y_poly;
          }
        }

        i_start = (x_min/dx);       /// Index of building start location in x-direction
        i_end = (x_max/dx)+1;       /// Index of building end location in x-direction
        j_start = (y_min/dy);       /// Index of building start location in y-direction
        j_end = (y_max/dy)+1;       /// Index of building end location in y-direction
        k_start = (baseHeight/dz)+1;		  /// Index of building start location in z-direction
        k_end = (bldElevation+baseHeight)/dz+1;             /// Index of building end location in z-direction

        for (int j=j_start; j<j_end; j++)
        {
          y_cent = (j+0.5)*dy;
          for (int i=i_start; i<i_end; i++)
          {
            x_cent = (i+0.5)*dx;
            vert_id = 0;
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

};
