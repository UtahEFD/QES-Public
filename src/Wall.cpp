#include "URBGeneralData.h"
#include "URBInputData.h"
#include "Wall.h"

void Wall::defineWalls(URBGeneralData *ugd)
{

  int dx = ugd->dx;
  int dy = ugd->dy;
  int dz = ugd->dz;
  int nx = ugd->nx;
  int ny = ugd->ny;
  int nz = ugd->nz;
  std::vector<int> &wall_right_indices = ugd->wall_right_indices;
  std::vector<int> &wall_left_indices = ugd->wall_left_indices;
  std::vector<int> &wall_above_indices = ugd->wall_above_indices;
  std::vector<int> &wall_below_indices = ugd->wall_below_indices;
  std::vector<int> &wall_front_indices = ugd->wall_front_indices;
  std::vector<int> &wall_back_indices = ugd->wall_back_indices;

  std::vector<int> &icellflag = ugd->icellflag;
  std::vector<float> &n = ugd->n;
  std::vector<float> &m = ugd->m;
  std::vector<float> &e = ugd->e;
  std::vector<float> &f = ugd->f;
  std::vector<float> &g = ugd->g;
  std::vector<float> &h = ugd->h;


  for (int i=0; i<nx-1; i++)
  {
    for (int j=0; j<ny-1; j++)
    {
      for (int k=1; k<nz-2; k++)
      {
        icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);
        icell_face = i + j*nx + k*nx*ny;

        if (icellflag[icell_cent] !=0 && icellflag[icell_cent] !=2)
        {

          /// Wall below
          if (icellflag[icell_cent-(nx-1)*(ny-1)]==0 || icellflag[icell_cent-(nx-1)*(ny-1)]==2)
          {
            wall_below_indices.push_back(icell_face);
            n[icell_cent] = 0.0;
          }
          /// Wall above
          if (icellflag[icell_cent+(nx-1)*(ny-1)]==0 || icellflag[icell_cent+(nx-1)*(ny-1)]==2)
          {
            wall_above_indices.push_back(icell_face);
            m[icell_cent] = 0.0;
          }
          /// Wall in back
          if (icellflag[icell_cent-1]==0 || icellflag[icell_cent-1]==2)
          {
            if (i>0)
            {
              wall_back_indices.push_back(icell_face);
              f[icell_cent] = 0.0;
            }
          }
          /// Wall in front
          if (icellflag[icell_cent+1]==0 || icellflag[icell_cent+1]==2)
          {
            wall_front_indices.push_back(icell_face);
            e[icell_cent] = 0.0;
          }
          /// Wall on right
          if (icellflag[icell_cent-(nx-1)]==0 || icellflag[icell_cent-(nx-1)]==2)
          {
            if (j>0)
            {
              wall_right_indices.push_back(icell_face);
              h[icell_cent] = 0.0;
            }
          }
          /// Wall on left
          if (icellflag[icell_cent+(nx-1)]==0 || icellflag[icell_cent+(nx-1)]==2)
          {
            wall_left_indices.push_back(icell_face);
            g[icell_cent] = 0.0;
          }
        }
      }
    }
  }

}


void Wall::wallLogBC (URBGeneralData *ugd)
{

  int dx = ugd->dx;
  int dy = ugd->dy;
  int dz = ugd->dz;
  int nx = ugd->nx;
  int ny = ugd->ny;
  int nz = ugd->nz;
  const float z0 = ugd->z0;
  std::vector<double> &u0 = ugd->u0;
  std::vector<double> &v0 = ugd->v0;
  std::vector<double> &w0 = ugd->w0;

  std::vector<int> &wall_right_indices = ugd->wall_right_indices;
  std::vector<int> &wall_left_indices = ugd->wall_left_indices;
  std::vector<int> &wall_above_indices = ugd->wall_above_indices;
  std::vector<int> &wall_below_indices = ugd->wall_below_indices;
  std::vector<int> &wall_front_indices = ugd->wall_front_indices;
  std::vector<int> &wall_back_indices = ugd->wall_back_indices;



  float ustar_wall;              /**< velocity gradient at the wall */
  float new_ustar;              /**< new ustar value calculated */
  float vel_mag1;               /**< velocity magnitude at the nearest cell to wall in perpendicular direction */
  float vel_mag2;                /**< velocity magnitude at the second cell near wall in perpendicular direction */
  float dist1;                  /**< distance of the center of the nearest cell in perpendicular direction from wall */
  float dist2;                  /**< distance of the center of second near cell in perpendicular direction from wall */
  float wind_dir;               /**< wind direction in parallel planes to wall */

  // Total size of wall indices
  int wall_size = wall_right_indices.size()+wall_left_indices.size()+
                  wall_above_indices.size()+wall_below_indices.size()+
                  wall_front_indices.size()+wall_back_indices.size();

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
  for (size_t i=0; i < wall_below_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(v0[wall_below_indices[i]+nx*ny],u0[wall_below_indices[i]+nx*ny]);
      vel_mag2 = sqrt(pow(u0[wall_below_indices[i]+nx*ny],2.0)+pow(v0[wall_below_indices[i]+nx*ny],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      w0[wall_below_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[wall_below_indices[i]] = vel_mag1*cos(wind_dir);
      v0[wall_below_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    index[i] = wall_below_indices[i];
    ustar[i] = ustar_wall;
  }

  /// apply log law fix to the cells with wall above
  for (size_t i=0; i < wall_above_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(v0[wall_above_indices[i]-nx*ny],u0[wall_above_indices[i]-nx*ny]);
      vel_mag2 = sqrt(pow(u0[wall_above_indices[i]-nx*ny],2.0)+pow(v0[wall_above_indices[i]-nx*ny],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      w0[wall_above_indices[i]] = 0;          /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[wall_above_indices[i]] = vel_mag1*cos(wind_dir);
      v0[wall_above_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size();
    index[j] = wall_above_indices[i];
    ustar[j] = ustar_wall;
  }

  dist1 = 0.5*dx;
  dist2 = 1.5*dx;

  /// apply log law fix to the cells with wall in back
  for (size_t i=0; i < wall_back_indices.size(); i++)
  {
    ustar_wall = 0.1;
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[wall_back_indices[i]+1],v0[wall_back_indices[i]+1]);
      vel_mag2 = sqrt(pow(v0[wall_back_indices[i]+1],2.0)+pow(w0[wall_back_indices[i]+1],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      u0[wall_back_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      v0[wall_back_indices[i]] = vel_mag1*cos(wind_dir);
      w0[wall_back_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size()+wall_above_indices.size();
    index[j] = wall_back_indices[i];
    ustar[j] = ustar_wall;
  }


  /// apply log law fix to the cells with wall in front
  for (size_t i=0; i < wall_front_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[wall_front_indices[i]-1],v0[wall_front_indices[i]-1]);
      vel_mag2 = sqrt(pow(v0[wall_front_indices[i]-1],2.0)+pow(w0[wall_front_indices[i]-1],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      u0[wall_front_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      v0[wall_front_indices[i]] = vel_mag1*cos(wind_dir);
      w0[wall_front_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size()+wall_above_indices.size()+wall_back_indices.size();
    index[j] = wall_front_indices[i];
    ustar[j] = ustar_wall;
  }


  dist1 = 0.5*dy;
  dist2 = 1.5*dy;

  /// apply log law fix to the cells with wall to right
  for (size_t i=0; i < wall_right_indices.size(); i++)
  {
    ustar_wall = 0.1;          /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[wall_right_indices[i]+nx],u0[wall_right_indices[i]+nx]);
      vel_mag2 = sqrt(pow(u0[wall_right_indices[i]+nx],2.0)+pow(w0[wall_right_indices[i]+nx],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      v0[wall_right_indices[i]] = 0;        /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[wall_right_indices[i]] = vel_mag1*cos(wind_dir);
      w0[wall_right_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size()+wall_above_indices.size()+wall_back_indices.size()+wall_front_indices.size();
    index[j] = wall_right_indices[i];
    ustar[j] = ustar_wall;
  }

  /// apply log law fix to the cells with wall to left
  for (size_t i=0; i < wall_left_indices.size(); i++)
  {
    ustar_wall = 0.1;       /// reset default value for velocity gradient
    for (int iter=0; iter<20; iter++)
    {
      wind_dir = atan2(w0[wall_left_indices[i]-nx],u0[wall_left_indices[i]-nx]);
      vel_mag2 = sqrt(pow(u0[wall_left_indices[i]-nx],2.0)+pow(w0[wall_left_indices[i]-nx],2.0));
      vel_mag1 = vel_mag2 - (ustar_wall/vk)*log(dist2/dist1);
      v0[wall_left_indices[i]] = 0;          /// normal component of velocity set to zero
      /// parallel components of velocity to wall
      u0[wall_left_indices[i]] = vel_mag1*cos(wind_dir);
      w0[wall_left_indices[i]] = vel_mag1*sin(wind_dir);
      new_ustar = vk*vel_mag1/log(dist1/z0);
      ustar_wall = new_ustar;
    }
    j = i+wall_below_indices.size()+wall_above_indices.size()+wall_back_indices.size()+wall_front_indices.size()+wall_right_indices.size();
    index[j] = wall_left_indices[i];
    ustar[j] = ustar_wall;
  }
}

void Wall::setVelocityZero (URBGeneralData *ugd)
{
  for (auto k = 1; k < ugd->nz-1; k++)
  {
    for (auto j = 1; j < ugd->ny-1; j++)
    {
      for (auto i = 1; i < ugd->nx-1; i++)
      {
        int icell_cent = i + j*(ugd->nx-1) + k*(ugd->nx-1)*(ugd->ny-1);
        int icell_face = i + j*ugd->nx + k*ugd->nx*ugd->ny;
        if (ugd->icellflag[icell_cent] == 0 || ugd->icellflag[icell_cent] == 2)
        {
          ugd->u0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
          ugd->u0[icell_face+1] = 0.0;
          ugd->v0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
          ugd->v0[icell_face+ugd->nx] = 0.0;
          ugd->w0[icell_face] = 0.0;                    /// Set velocity inside the building to zero
          ugd->w0[icell_face+ugd->nx*ugd->ny] = 0.0;
        }
      }
    }
  }


void Wall::solverCoefficients (URBGeneralData *ugd)
{
  // New boundary condition implementation
  // This needs to be done only once
  for (auto k = 0; k < ugd->nz-1; k++)
  {
    for (auto j = 0; j < ugd->ny-1; j++)
    {
      for (auto i = 0; i < ugd->nx-1; i++)
      {
        int icell_cent = i + j*(ugd->nx-1) + k*(ugd->nx-1)*(ugd->ny-1);
        ugd->e[icell_cent] = ugd->e[icell_cent]/(ugd->dx*ugd->dx);
        ugd->f[icell_cent] = ugd->f[icell_cent]/(ugd->dx*ugd->dx);
        ugd->g[icell_cent] = ugd->g[icell_cent]/(ugd->dy*ugd->dy);
        ugd->h[icell_cent] = ugd->h[icell_cent]/(ugd->dy*ugd->dy);
        ugd->m[icell_cent] = ugd->m[icell_cent]/(ugd->dz_array[k]*0.5*(ugd->dz_array[k]+ugd->dz_array[k+1]));
        ugd->n[icell_cent] = ugd->n[icell_cent]/(ugd->dz_array[k]*0.5*(ugd->dz_array[k]+ugd->dz_array[k-1]));
      }
    }
  }

}
