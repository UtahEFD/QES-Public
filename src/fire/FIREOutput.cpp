#include "FIREOutput.h"

FIREOutput::FIREOutput(WINDSGeneralData *wgd, Fire *fire, std::string output_file)
  : QESNetCDFOutput(output_file)
{

  std::cout << "[FireOutput] set up NetCDF file " << output_file << std::endl;
  output_fields = { "t", "x", "y", "z", "u", "v", "w", "icell", "terrain", "burn", "fuel" };

  // copy of wgd pointer
  wgd_ = wgd;
  fire_ = fire;

  int nx = wgd_->nx;
  int ny = wgd_->ny;
  int nz = wgd_->nz;

  long numcell_cout = (nx - 1) * (ny - 1) * (nz - 2);

  // Location of face centers in z-dir (without ghost cell)
  z_out.resize(nz - 2);
  for (auto k = 1; k < nz - 1; k++) {
    z_out[k - 1] = wgd_->z[k];
  }

  x_out.resize(nx - 1);
  for (auto i = 0; i < nx - 1; i++) {
    x_out[i] = (i + 0.5) * wgd_->dx;// Location of face centers in x-dir
  }

  y_out.resize(ny - 1);
  for (auto j = 0; j < ny - 1; j++) {
    y_out[j] = (j + 0.5) * wgd_->dy;// Location of face centers in y-dir
  }

  // Output data container
  u_out.resize(numcell_cout, 0.0);
  v_out.resize(numcell_cout, 0.0);
  w_out.resize(numcell_cout, 0.0);
  icellflag_out.resize(numcell_cout, 0.0);
  
  // set cell-centered data dimensions
  // space dimensions
  NcDim NcDim_x = addDimension("xdim", wgd_->nx - 1);
  NcDim NcDim_y = addDimension("ydim", wgd_->ny - 1);
  NcDim NcDim_z = addDimension("zdim", wgd_->nz - 2);

  std::cout << "dimensions added" << std::endl;


  /*
    // create attributes for time dimension
    std::vector<NcDim> dim_vect_tstr;
    dim_vect_tstr.push_back(NcDim_t);
    dim_vect_tstr.push_back(NcDim_tstr);
    createAttVector("times", "date time", "-", dim_vect_tstr, &timestamp);
    */

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


  /*
    // create attributes for time dimension
    std::vector<NcDim> dim_vect_t;
    dim_vect_t.push_back(NcDim_t);
    createAttScalar("t_f","time","s",dim_vect_t,&time);
    */

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

  std::cout << "burn added" << std::endl;

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
  int nx = wgd_->nx;
  int ny = wgd_->ny;
  int nz = wgd_->nz;

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
  /*
    // remove x, y, z and terrain
    // from output array after first save
    if (output_counter==0) {
        rmTimeIndepFields();
    }
    
    // increment for next time insertion
    output_counter +=1;
    */
};
