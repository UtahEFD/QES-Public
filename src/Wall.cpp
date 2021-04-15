/****************************************************************************
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
 ****************************************************************************/


/**
 * @file Wall.cpp
 * @brief :brief here:
 *
 * :long desc here if necessary:
 */
#include "Wall.h"

#include "WINDSGeneralData.h"
#include "WINDSInputData.h"


void Wall::defineWalls(WINDSGeneralData *WGD)
{

  float dx = WGD->dx;
  float dy = WGD->dy;
  float dz = WGD->dz;
  int nx = WGD->nx;
  int ny = WGD->ny;
  int nz = WGD->nz;

  for (auto i=0; i<nx-1; i++)
  {
    for (auto j=0; j<ny-1; j++)
    {
      for (auto k=1; k<nz-1; k++)
      {
        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        if (WGD->e[icell_cent] < 0.05)
        {
          WGD->e[icell_cent] = 0.0;
        }
        if (WGD->f[icell_cent] < 0.05)
        {
          WGD->f[icell_cent] = 0.0;
        }
        if (WGD->g[icell_cent] < 0.05)
        {
          WGD->g[icell_cent] = 0.0;
        }
        if (WGD->h[icell_cent] < 0.05)
        {
          WGD->h[icell_cent] = 0.0;
        }
        if (WGD->m[icell_cent] < 0.05)
        {
          WGD->m[icell_cent] = 0.0;
        }
        if (WGD->n[icell_cent] < 0.05)
        {
          WGD->n[icell_cent] = 0.0;
        }
      }
    }
  }


  for (auto i=0; i<nx-1; i++)
  {
    for (auto j=0; j<ny-1; j++)
    {
      for (auto k=1; k<nz-1; k++)
      {
        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        if (WGD->icellflag[icell_cent] == 7 && WGD->building_volume_frac[icell_cent] <= 0.1)
        {
          //std::cout << "icell_cent:  " << icell_cent << std::endl;
          WGD->icellflag[icell_cent] = 0;
          WGD->e[icell_cent] = 1.0;
          WGD->f[icell_cent] = 1.0;
          WGD->g[icell_cent] = 1.0;
          WGD->h[icell_cent] = 1.0;
          WGD->m[icell_cent] = 1.0;
          WGD->n[icell_cent] = 1.0;
        }
      }
    }
  }

  for (auto i=0; i<nx-1; i++)
  {
    for (auto j=0; j<ny-1; j++)
    {
      for (auto k=1; k<nz-1; k++)
      {
        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        if (WGD->e[icell_cent] == 0.0 && WGD->f[icell_cent] == 0.0 && WGD->g[icell_cent] == 0.0
            && WGD->h[icell_cent] == 0.0 && WGD->m[icell_cent] == 0.0 && WGD->n[icell_cent] == 0.0 &&
            WGD->icellflag[icell_cent] == 7)
        {
          WGD->icellflag[icell_cent] = 0;
          WGD->e[icell_cent] = 1.0;
          WGD->f[icell_cent] = 1.0;
          WGD->g[icell_cent] = 1.0;
          WGD->h[icell_cent] = 1.0;
          WGD->m[icell_cent] = 1.0;
          WGD->n[icell_cent] = 1.0;
        }

      }
    }
  }

  for (auto i=0; i<nx-1; i++)
  {
    for (auto j=0; j<ny-1; j++)
    {
      for (auto k=1; k<nz-2; k++)
      {
        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        int icell_face = i + j*nx + k*nx*ny;

        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
        {

          // Wall below
          if (WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 0 || WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 2)
          {
            //WGD->wall_below_indices.push_back(icell_face);
            WGD->n[icell_cent] = 0.0;
          }
          // Wall above
          if (WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 0 || WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 2)
          {
            //WGD->wall_above_indices.push_back(icell_face);
            WGD->m[icell_cent] = 0.0;
          }
          // Wall in back
          if (WGD->icellflag[icell_cent-1] == 0 || WGD->icellflag[icell_cent-1] == 2)
          {
            if (i>0)
            {
              //WGD->wall_back_indices.push_back(icell_face);
              WGD->f[icell_cent] = 0.0;
            }
          }
          // Wall in front
          if (WGD->icellflag[icell_cent+1] == 0 || WGD->icellflag[icell_cent+1] == 2)
          {
            //WGD->wall_front_indices.push_back(icell_face);
            WGD->e[icell_cent] = 0.0;
          }
          // Wall on right
          if (WGD->icellflag[icell_cent-(nx-1)] == 0 || WGD->icellflag[icell_cent-(nx-1)] == 2)
          {
            if (j>0)
            {
              //WGD->wall_right_indices.push_back(icell_face);
              WGD->h[icell_cent] = 0.0;
            }
          }
          // Wall on left
          if (WGD->icellflag[icell_cent+(nx-1)] == 0 || WGD->icellflag[icell_cent+(nx-1)] == 2)
          {
            //WGD->wall_left_indices.push_back(icell_face);
            WGD->g[icell_cent] = 0.0;
          }
        }

        if (WGD->icellflag[icell_cent] == 1 || WGD->icellflag[icell_cent] == 7 )
        {
          if (WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 7 || (WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 8 && WGD->n[icell_cent] == 1))
          {
            WGD->n[icell_cent] = WGD->m[icell_cent-(nx-1)*(ny-1)];
          }

          if (WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 7 || (WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 8 && WGD->m[icell_cent] == 1))
          {
            WGD->m[icell_cent] = WGD->n[icell_cent+(nx-1)*(ny-1)];
          }

          if (WGD->icellflag[icell_cent-1] == 7 || (WGD->icellflag[icell_cent-1] == 8 && WGD->f[icell_cent] == 1))
          {
            if (i>0)
            {
              WGD->f[icell_cent] = WGD->e[icell_cent-1];
            }
          }

          if (WGD->icellflag[icell_cent+1] == 7 || (WGD->icellflag[icell_cent+1] == 8 && WGD->e[icell_cent] == 1))
          {
            WGD->e[icell_cent] = WGD->f[icell_cent+1];
          }

          if (WGD->icellflag[icell_cent-(nx-1)] == 7 || (WGD->icellflag[icell_cent-(nx-1)] == 8 && WGD->h[icell_cent] == 1))
          {
            if (j>0)
            {
              WGD->h[icell_cent] = WGD->g[icell_cent-(nx-1)];
            }
          }

          if (WGD->icellflag[icell_cent+(nx-1)] == 7 || (WGD->icellflag[icell_cent+(nx-1)] == 8 && WGD->g[icell_cent] == 1))
          {
            WGD->g[icell_cent] = WGD->h[icell_cent+(nx-1)];
          }
        }

        if (WGD->icellflag[icell_cent] == 1 || WGD->icellflag[icell_cent] == 8 )
        {
          if (WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 7 || WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 8)
          {
            WGD->n[icell_cent] = WGD->m[icell_cent-(nx-1)*(ny-1)];
          }

          if (WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 7 || WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 8)
          {
            WGD->m[icell_cent] = WGD->n[icell_cent+(nx-1)*(ny-1)];
          }

          if (WGD->icellflag[icell_cent-1] == 7 || WGD->icellflag[icell_cent-1] == 8)
          {
            if (i>0)
            {
              WGD->f[icell_cent] = WGD->e[icell_cent-1];
            }
          }

          if (WGD->icellflag[icell_cent+1] == 7 || WGD->icellflag[icell_cent+1] == 8)
          {
            WGD->e[icell_cent] = WGD->f[icell_cent+1];
          }

          if (WGD->icellflag[icell_cent-(nx-1)] == 7 || WGD->icellflag[icell_cent-(nx-1)] == 8)
          {
            if (j>0)
            {
              WGD->h[icell_cent] = WGD->g[icell_cent-(nx-1)];
            }
          }

          if (WGD->icellflag[icell_cent+(nx-1)] == 7 || WGD->icellflag[icell_cent+(nx-1)] == 8)
          {
            WGD->g[icell_cent] = WGD->h[icell_cent+(nx-1)];
          }
        }

      }
    }
  }

  for (auto i=0; i<nx-1; i++)
  {
    for (auto j=0; j<ny-1; j++)
    {
      for (auto k=1; k<nz-1; k++)
      {
        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        if (WGD->e[icell_cent] == 0.0 && WGD->f[icell_cent] == 0.0 && WGD->g[icell_cent] == 0.0
            && WGD->h[icell_cent] == 0.0 && WGD->m[icell_cent] == 0.0 && WGD->n[icell_cent] == 0.0 &&
            WGD->icellflag[icell_cent] == 7)
        {
          WGD->icellflag[icell_cent] = 0;
          WGD->e[icell_cent] = 1.0;
          WGD->f[icell_cent] = 1.0;
          WGD->g[icell_cent] = 1.0;
          WGD->h[icell_cent] = 1.0;
          WGD->m[icell_cent] = 1.0;
          WGD->n[icell_cent] = 1.0;
        }

        if (WGD->e[icell_cent] == 0.0 && WGD->f[icell_cent] == 0.0 && WGD->g[icell_cent] == 0.0
            && WGD->h[icell_cent] == 0.0 && WGD->m[icell_cent] == 0.0 && WGD->n[icell_cent] == 0.0 &&
            WGD->icellflag[icell_cent] == 8)
        {
          WGD->icellflag[icell_cent] = 2;
          WGD->e[icell_cent] = 1.0;
          WGD->f[icell_cent] = 1.0;
          WGD->g[icell_cent] = 1.0;
          WGD->h[icell_cent] = 1.0;
          WGD->m[icell_cent] = 1.0;
          WGD->n[icell_cent] = 1.0;
        }

      }
    }
  }

  for (auto i=0; i<nx-1; i++)
  {
    for (auto j=0; j<ny-1; j++)
    {
      for (auto k=1; k<nz-2; k++)
      {
        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        int icell_face = i + j*nx + k*nx*ny;

        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
        {

          // Wall below
          if (WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 0 || WGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 2)
          {
            WGD->wall_below_indices.push_back(icell_face);
            WGD->n[icell_cent] = 0.0;
          }
          // Wall above
          if (WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 0 || WGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 2)
          {
            WGD->wall_above_indices.push_back(icell_face);
            WGD->m[icell_cent] = 0.0;
          }
          // Wall in back
          if (WGD->icellflag[icell_cent-1] == 0 || WGD->icellflag[icell_cent-1] == 2)
          {
            if (i>0)
            {
              WGD->wall_back_indices.push_back(icell_face);
              WGD->f[icell_cent] = 0.0;
            }
          }
          // Wall in front
          if (WGD->icellflag[icell_cent+1] == 0 || WGD->icellflag[icell_cent+1] == 2)
          {
            WGD->wall_front_indices.push_back(icell_face);
            WGD->e[icell_cent] = 0.0;
          }
          // Wall on right
          if (WGD->icellflag[icell_cent-(nx-1)] == 0 || WGD->icellflag[icell_cent-(nx-1)] == 2)
          {
            if (j>0)
            {
              WGD->wall_right_indices.push_back(icell_face);
              WGD->h[icell_cent] = 0.0;
            }
          }
          // Wall on left
          if (WGD->icellflag[icell_cent+(nx-1)] == 0 || WGD->icellflag[icell_cent+(nx-1)] == 2)
          {
            WGD->wall_left_indices.push_back(icell_face);
            WGD->g[icell_cent] = 0.0;
          }
        }
      }
    }
  }
}


void Wall::wallLogBC (WINDSGeneralData *WGD)
{

  float dx = WGD->dx;
  float dy = WGD->dy;
  float dz = WGD->dz;
  int nx = WGD->nx;
  int ny = WGD->ny;
  int nz = WGD->nz;
  const float z0 = WGD->z0;
  std::vector<float> &u0 = WGD->u0;
  std::vector<float> &v0 = WGD->v0;
  std::vector<float> &w0 = WGD->w0;




  float ustar_wall;    // velocity gradient at the wall
  float new_ustar;     // new ustar value calculated
  float vel_mag1;      // velocity magnitude at the nearest cell to wall in perpendicular direction
  float vel_mag2;      // velocity magnitude at the second cell near wall in perpendicular direction
  float dist1;         // distance of the center of the nearest cell in perpendicular direction from wall
  float dist2;         // distance of the center of second near cell in perpendicular direction from wall
  float wind_dir;      // wind direction in parallel planes to wall

  // Total size of wall indices
  int wall_size = WGD->wall_right_indices.size()+WGD->wall_left_indices.size()+
                  WGD->wall_above_indices.size()+WGD->wall_below_indices.size()+
                  WGD->wall_front_indices.size()+WGD->wall_back_indices.size();

  std::vector<float> ustar;
  ustar.resize(wall_size, 0.0);
  std::vector<int> index;
  index.resize(wall_size, 0.0);
  int j;

  ustar_wall = 0.1;
  wind_dir = 0.0;
  vel_mag1 = 0.0;
  vel_mag2 = 0.0;

  dist1 = 0.5*dz;
  dist2 = 1.5*dz;

  // apply log law fix to the cells with wall below
  for (size_t i=0; i < WGD->wall_below_indices.size(); i++)
  {
    ustar_wall = 0.1;       // reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(v0[WGD->wall_below_indices[i]+nx*ny],u0[WGD->wall_below_indices[i]+nx*ny]);
      vel_mag2 = sqrt(pow(u0[WGD->wall_below_indices[i]+nx*ny],2.0)+pow(v0[WGD->wall_below_indices[i]+nx*ny],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/WGD->vk)*log(dist2/dist1);
      w0[WGD->wall_below_indices[i]] = 0;        // normal component of velocity set to zero
      // parallel components of velocity to wall
      u0[WGD->wall_below_indices[i]] = vel_mag1*cos(wind_dir);
      v0[WGD->wall_below_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = WGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    index[i] = WGD->wall_below_indices[i];
    ustar[i] = ustar_wall;
  }

  // apply log law fix to the cells with wall above
  for (size_t i=0; i < WGD->wall_above_indices.size(); i++)
  {
    ustar_wall = 0.1;       // reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(v0[WGD->wall_above_indices[i]-nx*ny],u0[WGD->wall_above_indices[i]-nx*ny]);
      vel_mag2 = sqrt(pow(u0[WGD->wall_above_indices[i]-nx*ny],2.0)+pow(v0[WGD->wall_above_indices[i]-nx*ny],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/WGD->vk)*log(dist2/dist1);
      w0[WGD->wall_above_indices[i]] = 0;          // normal component of velocity set to zero
      // parallel components of velocity to wall
      u0[WGD->wall_above_indices[i]] = vel_mag1*cos(wind_dir);
      v0[WGD->wall_above_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = WGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+WGD->wall_below_indices.size();
    index[j] = WGD->wall_above_indices[i];
    ustar[j] = ustar_wall;
  }

  dist1 = 0.5*dx;
  dist2 = 1.5*dx;

  // apply log law fix to the cells with wall in back
  for (size_t i=0; i < WGD->wall_back_indices.size(); i++)
  {
    ustar_wall = 0.1;
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[WGD->wall_back_indices[i]+1],v0[WGD->wall_back_indices[i]+1]);
      vel_mag2 = sqrt(pow(v0[WGD->wall_back_indices[i]+1],2.0)+pow(w0[WGD->wall_back_indices[i]+1],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/WGD->vk)*log(dist2/dist1);
      u0[WGD->wall_back_indices[i]] = 0;        // normal component of velocity set to zero
      // parallel components of velocity to wall
      v0[WGD->wall_back_indices[i]] = vel_mag1*cos(wind_dir);
      w0[WGD->wall_back_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = WGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+WGD->wall_below_indices.size()+WGD->wall_above_indices.size();
    index[j] = WGD->wall_back_indices[i];
    ustar[j] = ustar_wall;
  }


  // apply log law fix to the cells with wall in front
  for (size_t i=0; i < WGD->wall_front_indices.size(); i++)
  {
    ustar_wall = 0.1;       // reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[WGD->wall_front_indices[i]-1],v0[WGD->wall_front_indices[i]-1]);
      vel_mag2 = sqrt(pow(v0[WGD->wall_front_indices[i]-1],2.0)+pow(w0[WGD->wall_front_indices[i]-1],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/WGD->vk)*log(dist2/dist1);
      u0[WGD->wall_front_indices[i]] = 0;        // normal component of velocity set to zero
      // parallel components of velocity to wall
      v0[WGD->wall_front_indices[i]] = vel_mag1*cos(wind_dir);
      w0[WGD->wall_front_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = WGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+WGD->wall_below_indices.size()+WGD->wall_above_indices.size()+WGD->wall_back_indices.size();
    index[j] = WGD->wall_front_indices[i];
    ustar[j] = ustar_wall;
  }


  dist1 = 0.5*dy;
  dist2 = 1.5*dy;

  // apply log law fix to the cells with wall to right
  for (size_t i=0; i < WGD->wall_right_indices.size(); i++)
  {
    ustar_wall = 0.1;          // reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[WGD->wall_right_indices[i]+nx],u0[WGD->wall_right_indices[i]+nx]);
      vel_mag2 = sqrt(pow(u0[WGD->wall_right_indices[i]+nx],2.0)+pow(w0[WGD->wall_right_indices[i]+nx],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/WGD->vk)*log(dist2/dist1);
      v0[WGD->wall_right_indices[i]] = 0;        // normal component of velocity set to zero
      // parallel components of velocity to wall
      u0[WGD->wall_right_indices[i]] = vel_mag1*cos(wind_dir);
      w0[WGD->wall_right_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = WGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+WGD->wall_below_indices.size()+WGD->wall_above_indices.size()+WGD->wall_back_indices.size()+WGD->wall_front_indices.size();
    index[j] = WGD->wall_right_indices[i];
    ustar[j] = ustar_wall;
  }

  // apply log law fix to the cells with wall to left
  for (size_t i=0; i < WGD->wall_left_indices.size(); i++)
  {
    ustar_wall = 0.1;       // reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[WGD->wall_left_indices[i]-nx],u0[WGD->wall_left_indices[i]-nx]);
      vel_mag2 = sqrt(pow(u0[WGD->wall_left_indices[i]-nx],2.0)+pow(w0[WGD->wall_left_indices[i]-nx],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/WGD->vk)*log(dist2/dist1);
      v0[WGD->wall_left_indices[i]] = 0;          // normal component of velocity set to zero
      // parallel components of velocity to wall
      u0[WGD->wall_left_indices[i]] = vel_mag1*cos(wind_dir);
      w0[WGD->wall_left_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = WGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+WGD->wall_below_indices.size()+WGD->wall_above_indices.size()+WGD->wall_back_indices.size()+WGD->wall_front_indices.size()+WGD->wall_right_indices.size();
    index[j] = WGD->wall_left_indices[i];
    ustar[j] = ustar_wall;
  }
}

void Wall::setVelocityZero (WINDSGeneralData *WGD)
{
  for (auto k = 0; k < WGD->nz-1; k++)
  {
    for (auto j = 1; j < WGD->ny-1; j++)
    {
      for (auto i = 1; i < WGD->nx-1; i++)
      {
        int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        int icell_face = i + j*WGD->nx + k*WGD->nx*WGD->ny;
        if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2)
        {
          WGD->u0[icell_face] = 0.0;                    // Set velocity inside the building to zero
          WGD->u0[icell_face+1] = 0.0;
          WGD->v0[icell_face] = 0.0;                    // Set velocity inside the building to zero
          WGD->v0[icell_face+WGD->nx] = 0.0;
          WGD->w0[icell_face] = 0.0;                    // Set velocity inside the building to zero
          WGD->w0[icell_face+WGD->nx*WGD->ny] = 0.0;
        }
        if (WGD->icellflag[icell_cent] == 7 || WGD->icellflag[icell_cent] == 8)
        {
          WGD->u0[icell_face] = pow(WGD->dx, 2.0)*WGD->f[icell_cent]*WGD->u0[icell_face];
          WGD->v0[icell_face] = pow(WGD->dy, 2.0)*WGD->h[icell_cent]*WGD->v0[icell_face];
          WGD->w0[icell_face] = (WGD->dz_array[k]*0.5*(WGD->dz_array[k]+WGD->dz_array[k-1]))*WGD->n[icell_cent]*WGD->w0[icell_face];
          if (WGD->icellflag[icell_cent+1] != 7 && WGD->icellflag[icell_cent+1] != 8)
          {
            WGD->u0[icell_face+1] = pow(WGD->dx, 2.0)*WGD->e[icell_cent]*WGD->u0[icell_face+1];
          }
          if (WGD->icellflag[icell_cent+(WGD->nx-1)] != 7 && WGD->icellflag[icell_cent+(WGD->nx-1)] != 8)
          {
            WGD->v0[icell_face+WGD->nx] = pow(WGD->dy, 2.0)*WGD->g[icell_cent]*WGD->v0[icell_face+WGD->nx];
          }
          if (WGD->icellflag[icell_cent+(WGD->nx-1)*(WGD->ny-1)] != 7 && WGD->icellflag[icell_cent-(WGD->nx-1)*(WGD->ny-1)] != 8)
          {
            WGD->w0[icell_face+(WGD->nx*WGD->ny)] = (WGD->dz_array[k]*0.5*(WGD->dz_array[k]+WGD->dz_array[k+1]))*WGD->m[icell_cent]*WGD->w0[icell_face+(WGD->nx*WGD->ny)];
          }
        }
      }
    }
  }
}


void Wall::solverCoefficients (WINDSGeneralData *WGD)
{
  // New boundary condition implementation
  // This needs to be done only once
  for (auto k = 1; k < WGD->nz-2; k++)
  {
    for (auto j = 0; j < WGD->ny-1; j++)
    {
      for (auto i = 0; i < WGD->nx-1; i++)
      {
        int icell_cent = i + j*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
        WGD->e[icell_cent] = WGD->e[icell_cent]/(WGD->dx*WGD->dx);
        WGD->f[icell_cent] = WGD->f[icell_cent]/(WGD->dx*WGD->dx);
        WGD->g[icell_cent] = WGD->g[icell_cent]/(WGD->dy*WGD->dy);
        WGD->h[icell_cent] = WGD->h[icell_cent]/(WGD->dy*WGD->dy);
        WGD->m[icell_cent] = WGD->m[icell_cent]/(WGD->dz_array[k]*0.5*(WGD->dz_array[k]+WGD->dz_array[k+1]));
        WGD->n[icell_cent] = WGD->n[icell_cent]/(WGD->dz_array[k]*0.5*(WGD->dz_array[k]+WGD->dz_array[k-1]));
      }
    }
  }

}
