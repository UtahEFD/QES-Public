#include "URBOutput_VizFields.h"

URBOutput_VizFields::URBOutput_VizFields(URBGeneralData *ugd,std::string output_file)
  : URBOutput_Generic(output_file)
{
  std::cout<<"Getting output fields for Vizualization file"<<std::endl;
  //FM -> need to implement the outputFields options here...
  output_fields = {"t","x","y","z","u","v","w","icell","terrain"};
  /* output_fields = UID->fileOptions->outputFields;
     
     if (output_fields.empty() || output_fields[0]=="all") {
     output_fields.clear();
     output_fields = {"u","v","w","icell"};
     }
     
     validateFileOptions();     
  */

  int nx = ugd->nx;
  int ny = ugd->ny;
  int nz = ugd->nz;
  
  long numcell_cout = (nx-1)*(ny-1)*(nz-2);
  
  z_out.resize( nz-2 );
  for (auto k=1; k<nz-1; k++) {
    z_out[k-1] = (k-0.5)*ugd->dz; // Location of face centers in z-dir
  }
  
  x_out.resize( nx-1 );
  for (auto i=0; i<nx-1; i++) {
    x_out[i] = (i+0.5)*ugd->dx; // Location of face centers in x-dir
  }
  
  y_out.resize( ny-1 );
  for (auto j=0; j<ny-1; j++) {
    y_out[j] = (j+0.5)*ugd->dy; // Location of face centers in y-dir
  }

  // Output related data
  u_out.resize( numcell_cout, 0.0 );
  v_out.resize( numcell_cout, 0.0 );
  w_out.resize( numcell_cout, 0.0 );  
  icellflag_out.resize( numcell_cout, 0.0 );

  // set cell-centered data dimensions
  // time dimension 
  std::vector<NcDim> dim_time;
  dim_time.push_back(addDimension("t"));
  // create attributes
  createAttScalar("t","time","s",dim_time,&time);
  
  // space dimensions 
  std::vector<NcDim> dim_vect_z;
  dim_vect_z.push_back(addDimension("z",ugd->nz-2));
  std::vector<NcDim> dim_vect_y;
  dim_vect_y.push_back(addDimension("y",ugd->ny-1));
  std::vector<NcDim> dim_vect_x;
  dim_vect_x.push_back(addDimension("x",ugd->nx-1)); 
  // create attributes
  createAttVector("x","x-distance","m",dim_vect_x,&x_out);
  createAttVector("y","y-distance","m",dim_vect_y,&y_out);
  createAttVector("z","z-distance","m",dim_vect_z,&z_out);
  
  // 2D vector (time indep)
  std::vector<NcDim> dim_vect_2d;
  dim_vect_2d.push_back(dim_vect_y[0]);
  dim_vect_2d.push_back(dim_vect_x[0]);
  // create attributes
  createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd->terrain));
  
  // 3D vector (time dep)
  std::vector<NcDim> dim_vect_3d;
  dim_vect_3d.push_back(dim_time[0]);
  dim_vect_3d.push_back(dim_vect_z[0]);
  dim_vect_3d.push_back(dim_vect_y[0]);
  dim_vect_3d.push_back(dim_vect_x[0]);
  // create attributes
  createAttVector("u","x-component velocity","m s-1",dim_vect_3d,&u_out);
  createAttVector("v","y-component velocity","m s-1",dim_vect_3d,&v_out);
  createAttVector("w","z-component velocity","m s-1",dim_vect_3d,&w_out);
  createAttVector("icell","icell flag value","--",dim_vect_3d,&icellflag_out);
  
  // create output fields
  addOutputFields();

}

bool URBOutput_VizFields::validateFileOtions()
{
  //check all fileoption specificed to make sure it's possible...
  return true;
}

  
// Save output at cell-centered values
void URBOutput_VizFields::save(URBGeneralData *ugd)
{
  // get grid size (not output var size)
  int nx = ugd->nx;
  int ny = ugd->ny;
  int nz = ugd->nz;
    
  // set time
  time = (double)output_counter;
    
  // get cell-centered values
  for (auto k = 1; k < nz-1; k++){
    for (auto j = 0; j < ny-1; j++){
      for (auto i = 0; i < nx-1; i++){
	int icell_face = i + j*nx + k*nx*ny;
	int icell_cent = i + j*(nx-1) + (k-1)*(nx-1)*(ny-1);
	u_out[icell_cent] = 0.5*(ugd->u[icell_face+1]+ugd->u[icell_face]);
	v_out[icell_cent] = 0.5*(ugd->v[icell_face+nx]+ugd->v[icell_face]);
	w_out[icell_cent] = 0.5*(ugd->w[icell_face+nx*ny]+ugd->w[icell_face]);
	icellflag_out[icell_cent] = ugd->icellflag[icell_cent+((nx-1)*(ny-1))];
      }
    }
  }
  
  // save the fields to NetCDF files
  saveOutputFields();

  // remove x, y, z and terrain 
  // from output array after first save
  if (output_counter==0) {
    rmOutputField("x");
    rmOutputField("y");
    rmOutputField("z");
    rmOutputField("terrain");
  }
  
  // increment for next time insertion
  output_counter +=1;
};
