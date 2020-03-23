#include "Wall.h"

#include "URBGeneralData.h"
#include "URBInputData.h"


void Wall::defineWalls(URBGeneralData *UGD)
{

  float dx = UGD->dx;
  float dy = UGD->dy;
  float dz = UGD->dz;
  int nx = UGD->nx;
  int ny = UGD->ny;
  int nz = UGD->nz;

  for (auto i=0; i<nx-1; i++)
  {
    for (auto j=0; j<ny-1; j++)
    {
      for (auto k=1; k<nz-1; k++)
      {
        int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        if (UGD->e[icell_cent] < 0.05)
        {
          UGD->e[icell_cent] = 0.0;
        }
        if (UGD->f[icell_cent] < 0.05)
        {
          UGD->f[icell_cent] = 0.0;
        }
        if (UGD->g[icell_cent] < 0.05)
        {
          UGD->g[icell_cent] = 0.0;
        }
        if (UGD->h[icell_cent] < 0.05)
        {
          UGD->h[icell_cent] = 0.0;
        }
        if (UGD->m[icell_cent] < 0.05)
        {
          UGD->m[icell_cent] = 0.0;
        }
        if (UGD->n[icell_cent] < 0.05)
        {
          UGD->n[icell_cent] = 0.0;
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
        if (UGD->icellflag[icell_cent] == 7 && UGD->building_volume_frac[icell_cent] <= 0.1)
        {
          //std::cout << "icell_cent:  " << icell_cent << std::endl;
          UGD->icellflag[icell_cent] = 0;
          UGD->e[icell_cent] = 1.0;
          UGD->f[icell_cent] = 1.0;
          UGD->g[icell_cent] = 1.0;
          UGD->h[icell_cent] = 1.0;
          UGD->m[icell_cent] = 1.0;
          UGD->n[icell_cent] = 1.0;
        }
        if (UGD->icellflag[icell_cent] == 8 && UGD->terrain_volume_frac[icell_cent] <= 0.1)
        {
          //std::cout << "icell_cent:  " << icell_cent << std::endl;
          UGD->icellflag[icell_cent] = 2;
          UGD->e[icell_cent] = 1.0;
          UGD->f[icell_cent] = 1.0;
          UGD->g[icell_cent] = 1.0;
          UGD->h[icell_cent] = 1.0;
          UGD->m[icell_cent] = 1.0;
          UGD->n[icell_cent] = 1.0;
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
        if (UGD->e[icell_cent] == 0.0 && UGD->f[icell_cent] == 0.0 && UGD->g[icell_cent] == 0.0
            && UGD->h[icell_cent] == 0.0 && UGD->m[icell_cent] == 0.0 && UGD->n[icell_cent] == 0.0 &&
            UGD->icellflag[icell_cent] == 7)
        {
          UGD->icellflag[icell_cent] = 0;
          UGD->e[icell_cent] = 1.0;
          UGD->f[icell_cent] = 1.0;
          UGD->g[icell_cent] = 1.0;
          UGD->h[icell_cent] = 1.0;
          UGD->m[icell_cent] = 1.0;
          UGD->n[icell_cent] = 1.0;
        }
        if (UGD->e[icell_cent] == 0.0 && UGD->f[icell_cent] == 0.0 && UGD->g[icell_cent] == 0.0
            && UGD->h[icell_cent] == 0.0 && UGD->m[icell_cent] == 0.0 && UGD->n[icell_cent] == 0.0 &&
            UGD->icellflag[icell_cent] == 8)
        {
          UGD->icellflag[icell_cent] = 2;
          UGD->e[icell_cent] = 1.0;
          UGD->f[icell_cent] = 1.0;
          UGD->g[icell_cent] = 1.0;
          UGD->h[icell_cent] = 1.0;
          UGD->m[icell_cent] = 1.0;
          UGD->n[icell_cent] = 1.0;
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

        if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
        {

          /// Wall below
          if (UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 0 || UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 2)
          {
            //UGD->wall_below_indices.push_back(icell_face);
            UGD->n[icell_cent] = 0.0;
          }
          /// Wall above
          if (UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 0 || UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 2)
          {
            //UGD->wall_above_indices.push_back(icell_face);
            UGD->m[icell_cent] = 0.0;
          }
          /// Wall in back
          if (UGD->icellflag[icell_cent-1] == 0 || UGD->icellflag[icell_cent-1] == 2)
          {
            if (i>0)
            {
              //UGD->wall_back_indices.push_back(icell_face);
              UGD->f[icell_cent] = 0.0;
            }
          }
          /// Wall in front
          if (UGD->icellflag[icell_cent+1] == 0 || UGD->icellflag[icell_cent+1] == 2)
          {
            //UGD->wall_front_indices.push_back(icell_face);
            UGD->e[icell_cent] = 0.0;
          }
          /// Wall on right
          if (UGD->icellflag[icell_cent-(nx-1)] == 0 || UGD->icellflag[icell_cent-(nx-1)] == 2)
          {
            if (j>0)
            {
              //UGD->wall_right_indices.push_back(icell_face);
              UGD->h[icell_cent] = 0.0;
            }
          }
          /// Wall on left
          if (UGD->icellflag[icell_cent+(nx-1)] == 0 || UGD->icellflag[icell_cent+(nx-1)] == 2)
          {
            //UGD->wall_left_indices.push_back(icell_face);
            UGD->g[icell_cent] = 0.0;
          }
        }

        if (UGD->icellflag[icell_cent] == 1 || UGD->icellflag[icell_cent] == 7 )
        {
          if (UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 7 || (UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 8 && UGD->n[icell_cent] == 1))
          {
            UGD->n[icell_cent] = UGD->m[icell_cent-(nx-1)*(ny-1)];
          }

          if (UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 7 || (UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 8 && UGD->m[icell_cent] == 1))
          {
            UGD->m[icell_cent] = UGD->n[icell_cent+(nx-1)*(ny-1)];
          }

          if (UGD->icellflag[icell_cent-1] == 7 || (UGD->icellflag[icell_cent-1] == 8 && UGD->f[icell_cent] == 1))
          {
            if (i>0)
            {
              UGD->f[icell_cent] = UGD->e[icell_cent-1];
            }
          }

          if (UGD->icellflag[icell_cent+1] == 7 || (UGD->icellflag[icell_cent+1] == 8 && UGD->e[icell_cent] == 1))
          {
            UGD->e[icell_cent] = UGD->f[icell_cent+1];
          }

          if (UGD->icellflag[icell_cent-(nx-1)] == 7 || (UGD->icellflag[icell_cent-(nx-1)] == 8 && UGD->h[icell_cent] == 1))
          {
            if (j>0)
            {
              UGD->h[icell_cent] = UGD->g[icell_cent-(nx-1)];
            }
          }

          if (UGD->icellflag[icell_cent+(nx-1)] == 7 || (UGD->icellflag[icell_cent+(nx-1)] == 8 && UGD->g[icell_cent] == 1))
          {
            UGD->g[icell_cent] = UGD->h[icell_cent+(nx-1)];
          }
        }

        if (UGD->icellflag[icell_cent] == 1 || UGD->icellflag[icell_cent] == 8 )
        {
          if (UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 7 || UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 8)
          {
            UGD->n[icell_cent] = UGD->m[icell_cent-(nx-1)*(ny-1)];
          }

          if (UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 7 || UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 8)
          {
            UGD->m[icell_cent] = UGD->n[icell_cent+(nx-1)*(ny-1)];
          }

          if (UGD->icellflag[icell_cent-1] == 7 || UGD->icellflag[icell_cent-1] == 8)
          {
            if (i>0)
            {
              UGD->f[icell_cent] = UGD->e[icell_cent-1];
            }
          }

          if (UGD->icellflag[icell_cent+1] == 7 || UGD->icellflag[icell_cent+1] == 8)
          {
            UGD->e[icell_cent] = UGD->f[icell_cent+1];
          }

          if (UGD->icellflag[icell_cent-(nx-1)] == 7 || UGD->icellflag[icell_cent-(nx-1)] == 8)
          {
            if (j>0)
            {
              UGD->h[icell_cent] = UGD->g[icell_cent-(nx-1)];
            }
          }

          if (UGD->icellflag[icell_cent+(nx-1)] == 7 || UGD->icellflag[icell_cent+(nx-1)] == 8)
          {
            UGD->g[icell_cent] = UGD->h[icell_cent+(nx-1)];
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
        if (UGD->e[icell_cent] == 0.0 && UGD->f[icell_cent] == 0.0 && UGD->g[icell_cent] == 0.0
            && UGD->h[icell_cent] == 0.0 && UGD->m[icell_cent] == 0.0 && UGD->n[icell_cent] == 0.0 &&
            UGD->icellflag[icell_cent] == 7)
        {
          UGD->icellflag[icell_cent] = 0;
          UGD->e[icell_cent] = 1.0;
          UGD->f[icell_cent] = 1.0;
          UGD->g[icell_cent] = 1.0;
          UGD->h[icell_cent] = 1.0;
          UGD->m[icell_cent] = 1.0;
          UGD->n[icell_cent] = 1.0;
        }

        if (UGD->e[icell_cent] == 0.0 && UGD->f[icell_cent] == 0.0 && UGD->g[icell_cent] == 0.0
            && UGD->h[icell_cent] == 0.0 && UGD->m[icell_cent] == 0.0 && UGD->n[icell_cent] == 0.0 &&
            UGD->icellflag[icell_cent] == 8)
        {
          UGD->icellflag[icell_cent] = 2;
          UGD->e[icell_cent] = 1.0;
          UGD->f[icell_cent] = 1.0;
          UGD->g[icell_cent] = 1.0;
          UGD->h[icell_cent] = 1.0;
          UGD->m[icell_cent] = 1.0;
          UGD->n[icell_cent] = 1.0;
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

        if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
        {

          /// Wall below
          if (UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 0 || UGD->icellflag[icell_cent-(nx-1)*(ny-1)] == 2)
          {
            UGD->wall_below_indices.push_back(icell_face);
            UGD->n[icell_cent] = 0.0;
          }
          /// Wall above
          if (UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 0 || UGD->icellflag[icell_cent+(nx-1)*(ny-1)] == 2)
          {
            UGD->wall_above_indices.push_back(icell_face);
            UGD->m[icell_cent] = 0.0;
          }
          /// Wall in back
          if (UGD->icellflag[icell_cent-1] == 0 || UGD->icellflag[icell_cent-1] == 2)
          {
            if (i>0)
            {
              UGD->wall_back_indices.push_back(icell_face);
              UGD->f[icell_cent] = 0.0;
            }
          }
          /// Wall in front
          if (UGD->icellflag[icell_cent+1] == 0 || UGD->icellflag[icell_cent+1] == 2)
          {
            UGD->wall_front_indices.push_back(icell_face);
            UGD->e[icell_cent] = 0.0;
          }
          /// Wall on right
          if (UGD->icellflag[icell_cent-(nx-1)] == 0 || UGD->icellflag[icell_cent-(nx-1)] == 2)
          {
            if (j>0)
            {
              UGD->wall_right_indices.push_back(icell_face);
              UGD->h[icell_cent] = 0.0;
            }
          }
          /// Wall on left
          if (UGD->icellflag[icell_cent+(nx-1)] == 0 || UGD->icellflag[icell_cent+(nx-1)] == 2)
          {
            UGD->wall_left_indices.push_back(icell_face);
            UGD->g[icell_cent] = 0.0;
          }
        }
      }
    }
  }
}


void Wall::wallLogBC (URBGeneralData *UGD)
{

  float dx = UGD->dx;
  float dy = UGD->dy;
  float dz = UGD->dz;
  int nx = UGD->nx;
  int ny = UGD->ny;
  int nz = UGD->nz;
  const float z0 = UGD->z0;
  std::vector<float> &u0 = UGD->u0;
  std::vector<float> &v0 = UGD->v0;
  std::vector<float> &w0 = UGD->w0;




  float ustar_wall;              /**< velocity gradient at the wall */
  float new_ustar;              /**< new ustar value calculated */
  float vel_mag1;               /**< velocity magnitude at the nearest cell to wall in perpendicular direction */
  float vel_mag2;                /**< velocity magnitude at the second cell near wall in perpendicular direction */
  float dist1;                  /**< distance of the center of the nearest cell in perpendicular direction from wall */
  float dist2;                  /**< distance of the center of second near cell in perpendicular direction from wall */
  float wind_dir;               /**< wind direction in parallel planes to wall */

  // Total size of wall indices
  int wall_size = UGD->wall_right_indices.size()+UGD->wall_left_indices.size()+
                  UGD->wall_above_indices.size()+UGD->wall_below_indices.size()+
                  UGD->wall_front_indices.size()+UGD->wall_back_indices.size();

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

  /// apply log law fix to the cells with wall below
  for (size_t i=0; i < UGD->wall_below_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(v0[UGD->wall_below_indices[i]+nx*ny],u0[UGD->wall_below_indices[i]+nx*ny]);
      vel_mag2 = sqrt(pow(u0[UGD->wall_below_indices[i]+nx*ny],2.0)+pow(v0[UGD->wall_below_indices[i]+nx*ny],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/UGD->vk)*log(dist2/dist1);
      w0[UGD->wall_below_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[UGD->wall_below_indices[i]] = vel_mag1*cos(wind_dir);
      v0[UGD->wall_below_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = UGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    index[i] = UGD->wall_below_indices[i];
    ustar[i] = ustar_wall;
  }

  /// apply log law fix to the cells with wall above
  for (size_t i=0; i < UGD->wall_above_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(v0[UGD->wall_above_indices[i]-nx*ny],u0[UGD->wall_above_indices[i]-nx*ny]);
      vel_mag2 = sqrt(pow(u0[UGD->wall_above_indices[i]-nx*ny],2.0)+pow(v0[UGD->wall_above_indices[i]-nx*ny],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/UGD->vk)*log(dist2/dist1);
      w0[UGD->wall_above_indices[i]] = 0;          /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[UGD->wall_above_indices[i]] = vel_mag1*cos(wind_dir);
      v0[UGD->wall_above_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = UGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+UGD->wall_below_indices.size();
    index[j] = UGD->wall_above_indices[i];
    ustar[j] = ustar_wall;
  }

  dist1 = 0.5*dx;
  dist2 = 1.5*dx;

  /// apply log law fix to the cells with wall in back
  for (size_t i=0; i < UGD->wall_back_indices.size(); i++)
  {
    ustar_wall = 0.1;
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[UGD->wall_back_indices[i]+1],v0[UGD->wall_back_indices[i]+1]);
      vel_mag2 = sqrt(pow(v0[UGD->wall_back_indices[i]+1],2.0)+pow(w0[UGD->wall_back_indices[i]+1],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/UGD->vk)*log(dist2/dist1);
      u0[UGD->wall_back_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      v0[UGD->wall_back_indices[i]] = vel_mag1*cos(wind_dir);
      w0[UGD->wall_back_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = UGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+UGD->wall_below_indices.size()+UGD->wall_above_indices.size();
    index[j] = UGD->wall_back_indices[i];
    ustar[j] = ustar_wall;
  }


  /// apply log law fix to the cells with wall in front
  for (size_t i=0; i < UGD->wall_front_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[UGD->wall_front_indices[i]-1],v0[UGD->wall_front_indices[i]-1]);
      vel_mag2 = sqrt(pow(v0[UGD->wall_front_indices[i]-1],2.0)+pow(w0[UGD->wall_front_indices[i]-1],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/UGD->vk)*log(dist2/dist1);
      u0[UGD->wall_front_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      v0[UGD->wall_front_indices[i]] = vel_mag1*cos(wind_dir);
      w0[UGD->wall_front_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = UGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+UGD->wall_below_indices.size()+UGD->wall_above_indices.size()+UGD->wall_back_indices.size();
    index[j] = UGD->wall_front_indices[i];
    ustar[j] = ustar_wall;
  }


  dist1 = 0.5*dy;
  dist2 = 1.5*dy;

  /// apply log law fix to the cells with wall to right
  for (size_t i=0; i < UGD->wall_right_indices.size(); i++)
  {
    ustar_wall = 0.1;          /// reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[UGD->wall_right_indices[i]+nx],u0[UGD->wall_right_indices[i]+nx]);
      vel_mag2 = sqrt(pow(u0[UGD->wall_right_indices[i]+nx],2.0)+pow(w0[UGD->wall_right_indices[i]+nx],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/UGD->vk)*log(dist2/dist1);
      v0[UGD->wall_right_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[UGD->wall_right_indices[i]] = vel_mag1*cos(wind_dir);
      w0[UGD->wall_right_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = UGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+UGD->wall_below_indices.size()+UGD->wall_above_indices.size()+UGD->wall_back_indices.size()+UGD->wall_front_indices.size();
    index[j] = UGD->wall_right_indices[i];
    ustar[j] = ustar_wall;
  }

  /// apply log law fix to the cells with wall to left
  for (size_t i=0; i < UGD->wall_left_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (auto iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[UGD->wall_left_indices[i]-nx],u0[UGD->wall_left_indices[i]-nx]);
      vel_mag2 = sqrt(pow(u0[UGD->wall_left_indices[i]-nx],2.0)+pow(w0[UGD->wall_left_indices[i]-nx],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/UGD->vk)*log(dist2/dist1);
      v0[UGD->wall_left_indices[i]] = 0;          /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[UGD->wall_left_indices[i]] = vel_mag1*cos(wind_dir);
      w0[UGD->wall_left_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = UGD->vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+UGD->wall_below_indices.size()+UGD->wall_above_indices.size()+UGD->wall_back_indices.size()+UGD->wall_front_indices.size()+UGD->wall_right_indices.size();
    index[j] = UGD->wall_left_indices[i];
    ustar[j] = ustar_wall;
  }
}

void Wall::setVelocityZero (URBGeneralData *UGD)
{
  for (auto k = 0; k < UGD->nz-1; k++)
  {
    for (auto j = 1; j < UGD->ny-1; j++)
    {
      for (auto i = 1; i < UGD->nx-1; i++)
      {
        int icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        int icell_face = i + j*UGD->nx + k*UGD->nx*UGD->ny;
        if (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2)
        {
          UGD->u0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
          UGD->u0[icell_face+1] = 0.0;
          UGD->v0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
          UGD->v0[icell_face+UGD->nx] = 0.0;
          UGD->w0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
          UGD->w0[icell_face+UGD->nx*UGD->ny] = 0.0;
        }
      }
    }
  }
}


void Wall::solverCoefficients (URBGeneralData *UGD)
{
  // New boundary condition implementation
  // This needs to be done only once
  for (auto k = 1; k < UGD->nz-2; k++)
  {
    for (auto j = 0; j < UGD->ny-1; j++)
    {
      for (auto i = 0; i < UGD->nx-1; i++)
      {
        int icell_cent = i + j*(UGD->nx-1) + k*(UGD->nx-1)*(UGD->ny-1);
        UGD->e[icell_cent] = UGD->e[icell_cent]/(UGD->dx*UGD->dx);
        UGD->f[icell_cent] = UGD->f[icell_cent]/(UGD->dx*UGD->dx);
        UGD->g[icell_cent] = UGD->g[icell_cent]/(UGD->dy*UGD->dy);
        UGD->h[icell_cent] = UGD->h[icell_cent]/(UGD->dy*UGD->dy);
        UGD->m[icell_cent] = UGD->m[icell_cent]/(UGD->dz_array[k]*0.5*(UGD->dz_array[k]+UGD->dz_array[k+1]));
        UGD->n[icell_cent] = UGD->n[icell_cent]/(UGD->dz_array[k]*0.5*(UGD->dz_array[k]+UGD->dz_array[k-1]));
      }
    }
  }

}
