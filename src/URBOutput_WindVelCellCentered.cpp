#include "URBOutput_WindVelCellCentered.h"

URBOutput_WindVelCellCentered::URBOutput_WindVelCellCentered(URBGeneralData *ugd,std::string output_file)
  : URBOutput_Generic(output_file)
{
  std::cout<<"Getting output fields for Cell-Centered data"<<std::endl;
  //FM -> need to implement the outputFields options here...
  output_fields = {"t","x","y","z","u","v","w","icell"};
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
  std::vector<NcDim> dim_vect;
  dim_vect.push_back(addDimension("t"));
  dim_vect.push_back(addDimension("z",ugd->nz-2));
  dim_vect.push_back(addDimension("y",ugd->ny-1));
  dim_vect.push_back(addDimension("x",ugd->nx-1));
  createDimensions(dim_vect);

  // create attributes
  createAttScalar("t","time","s",dim_scalar_t,&time);
  createAttVector("x","x-distance","m",dim_scalar_x,&x_out);
  createAttVector("y","y-distance","m",dim_scalar_y,&y_out);
  createAttVector("z","z-distance","m",dim_scalar_z,&z_out);
  createAttVector("u","x-component velocity","m s-1",dim_vector,&u_out);
  createAttVector("v","y-component velocity","m s-1",dim_vector,&v_out);
  createAttVector("w","z-component velocity","m s-1",dim_vector,&w_out);
  createAttVector("icell","icell flag value","--",dim_vector,&icellflag_out);

  // create output fields
  addOutputFields();

}

bool URBOutput_WindVelCellCentered::validateFileOtions()
{
  //check all fileoption specificed to make sure it's possible...
  return true;
}

  
// Save output at cell-centered values
void URBOutput_WindVelCellCentered::save(URBGeneralData *ugd)
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

  // remove x, y, z from output array after first save
  if (output_counter==0) {
    rmOutputField("x");
    rmOutputField("y");
    rmOutputField("z");
  }
    
  // increment for next time insertion
  output_counter +=1;
};
