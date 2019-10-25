#include "URBOutput_Static.h"

URBOutput_Static::URBOutput_Static(URBGeneralData *ugd,std::string output_file)
  : URBOutput_Generic(output_file)
{
  std::cout<<"Getting output fields for Static data"<<std::endl;
  //FM -> need to implement the outputFields options here...
  output_fields = {"x","y","z","terrain"};
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
  
    // set cell-centered data dimensions
  std::vector<NcDim> dim_vect;
  dim_vect.push_back(addDimension("t",1));
  dim_vect.push_back(addDimension("z",ugd->nz-2));
  dim_vect.push_back(addDimension("y",ugd->ny-1));
  dim_vect.push_back(addDimension("x",ugd->nx-1));
  createDimensions(dim_vect);
    
  // create attributes
  createAttScalar("t","time","s",dim_scalar_t,&time);
  createAttVector("x","x-distance","m",dim_scalar_x,&x_out);
  createAttVector("y","y-distance","m",dim_scalar_y,&y_out);
  createAttVector("z","z-distance","m",dim_scalar_z,&z_out);
  createAttVector("terrain","terrain height","m",dim_vector_2d,&(ugd->terrain));
  
  // create output fields
  addOutputFields();
  
}

bool URBOutput_Static::validateFileOtions()
{
  //check all fileoption specificed to make sure it's possible...
  return true;
}

  
// Save output static values
void URBOutput_Static::save(URBGeneralData *ugd)
{
  if(output_counter==0) {
    // save the fields to NetCDF files
    saveOutputFields();
    // increment time (ensure data saved only once)
    output_counter +=1;
  } else {
    std::cout << "Static fields already saved" << std::endl;
  }
};
