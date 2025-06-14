/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Matthew Moody
 * Copyright (c) 2025 Jeremy Gibbs
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Brian Bailey
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file FIREOutput.cpp
 * @brief This class handles saving output files for Fire related variables
 * This is a specialized output class derived and inheriting from QESNetCDFOutput.
 */
#include "FIREOutput.h"

FIREOutput::FIREOutput(WINDSGeneralData *wgd, Fire *fire, std::string output_file)
  : QESNetCDFOutput(output_file)
{
  //std::cout << "[FireOutput] set up NetCDF file " << output_file << std::endl;
  output_fields = { "x", "y", "z", "u", "v", "w", "icell", "terrain", "burn", "fuel", "Force", "LS" };

  // copy of wgd pointer
  wgd_ = wgd;
  fire_ = fire;

  int nx = wgd_->domain.nx();
  int ny = wgd_->domain.ny();
  int nz = wgd_->domain.nz();

  long numcell_cout = (nx - 1) * (ny - 1) * (nz - 2);

  // Location of face centers in z-dir (without ghost cell)
  z_out.resize(nz - 2);
  for (auto k = 1; k < nz - 1; k++) {
    z_out[k - 1] = wgd_->domain.z[k];
  }

  x_out.resize(nx - 1);
  for (auto i = 0; i < nx - 1; i++) {
    x_out[i] = (i + 0.5) * wgd_->domain.dx();// Location of face centers in x-dir
  }

  y_out.resize(ny - 1);
  for (auto j = 0; j < ny - 1; j++) {
    y_out[j] = (j + 0.5) * wgd_->domain.dy();// Location of face centers in y-dir
  }

  // Output data container
  u_out.resize(numcell_cout, 0.0);
  v_out.resize(numcell_cout, 0.0);
  w_out.resize(numcell_cout, 0.0);
  icellflag_out.resize(numcell_cout, 0.0);

  // set cell-centered data dimensions
  // space dimensions
  NcDim NcDim_x = addDimension("xdim", nx - 1);
  NcDim NcDim_y = addDimension("ydim", ny - 1);
  NcDim NcDim_z = addDimension("zdim", nz - 2);

  //std::cout << "dimensions added" << std::endl;

  // create attributes space dimensions
  std::vector<NcDim> dim_vect_x;
  dim_vect_x.push_back(NcDim_x);
  createAttVector("x", "x-distance", "m", dim_vect_x, &x_out);
  std::vector<NcDim> dim_vect_y;
  dim_vect_y.push_back(NcDim_y);
  createAttVector("y", "y-distance", "m", dim_vect_y, &y_out);
  std::vector<NcDim> dim_vect_z;
  dim_vect_z.push_back(NcDim_z);
  createAttVector("z", "z-distance", "m", dim_vect_z, &z_out);

  // create 2D vector (x,y- time independent)
  std::vector<NcDim> dim_vect_2d;
  dim_vect_2d.push_back(NcDim_y);
  dim_vect_2d.push_back(NcDim_x);
  // create attributes
  createAttVector("terrain", "terrain height", "m", dim_vect_2d, &(wgd_->terrain));
  createAttVector("fuel", "fuel type", "--", dim_vect_2d, &(fire_->fuel_map));

  // create 3D vector (x,y,t)
  std::vector<NcDim> dim_vect_3d;
  dim_vect_3d.push_back(NcDim_t);
  dim_vect_3d.push_back(NcDim_y);
  dim_vect_3d.push_back(NcDim_x);
  // create attributes
  createAttVector("burn", "burn flag value", "--", dim_vect_3d, &(fire_->burn_out));
  createAttVector("Force", "ROS Forcing", "m/s", dim_vect_3d, &(fire_->Force));
  createAttVector("LS", "Level Set Value", "--", dim_vect_3d, &(fire_->front_map));
  //std::cout << "burn added" << std::endl;

  // create 4D vector (x,y,z,t)
  std::vector<NcDim> dim_vect_4d;
  dim_vect_4d.push_back(NcDim_t);
  dim_vect_4d.push_back(NcDim_z);
  dim_vect_4d.push_back(NcDim_y);
  dim_vect_4d.push_back(NcDim_x);
  // create attributes for velocity
  createAttVector("u", "x-component velocity", "m s-1", dim_vect_4d, &u_out);
  createAttVector("v", "y-component velocity", "m s-1", dim_vect_4d, &v_out);
  createAttVector("w", "z-component velocity", "m s-1", dim_vect_4d, &w_out);
  createAttVector("icell", "icell flag value", "--", dim_vect_4d, &icellflag_out);


  // create output fields
  addOutputFields();
}

// Save output at cell-centered values
void FIREOutput::save(QEStime timeOut)
{
  // get grid size (not output var size)
  int nx = wgd_->domain.nx();
  int ny = wgd_->domain.ny();
  int nz = wgd_->domain.nz();

  // set time
  timeCurrent = timeOut;

  // get cell-centered values
  for (auto k = 1; k < nz - 1; k++) {
    for (auto j = 0; j < ny - 1; j++) {
      for (auto i = 0; i < nx - 1; i++) {
        int icell_face = i + j * nx + k * nx * ny;
        int icell_cent = i + j * (nx - 1) + (k - 1) * (nx - 1) * (ny - 1);
        u_out[icell_cent] = 0.5 * (wgd_->u[icell_face + 1] + wgd_->u[icell_face]);
        v_out[icell_cent] = 0.5 * (wgd_->v[icell_face + nx] + wgd_->v[icell_face]);
        w_out[icell_cent] = 0.5 * (wgd_->w[icell_face + nx * ny] + wgd_->w[icell_face]);
        icellflag_out[icell_cent] = wgd_->icellflag[icell_cent + ((nx - 1) * (ny - 1))];
      }
    }
  }

  // save the fields to NetCDF files
  saveOutputFields();
};
