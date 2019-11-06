#include "URBOutput_TURBInputFile.h"

URBOutput_TURBInputFile::URBOutput_TURBInputFile(URBGeneralData *ugd,std::string output_file)
  : URBOutput_Generic(output_file)
{
  std::cout<<"Setting fields of TURBInputFile file"<<std::endl;
  
  // set list of fields to save, no option available for this file
  output_fields = {"u","v","w","icell","terrain"};
  
  // set time data dimensions
  std::vector<NcDim> dim_scal_t;
  dim_scal_t.push_back(addDimension("t"));
  
  // set face-centered data dimensions
  // scalar dimension 
  std::vector<NcDim> dim_scal_z_fc;
  dim_scal_z_fc.push_back(addDimension("z_fc",ugd->nz-1));
  std::vector<NcDim> dim_scal_y_fc;
  dim_scal_y_fc.push_back(addDimension("y_fc",ugd->ny));
  std::vector<NcDim> dim_scal_x_fc;
  dim_scal_x_fc.push_back(addDimension("x_fc",ugd->nx));
  
  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_fc;
  dim_vect_fc.push_back(dim_scal_t[0]);
  dim_vect_fc.push_back(dim_scal_z_fc[0]);
  dim_vect_fc.push_back(dim_scal_y_fc[0]);
  dim_vect_fc.push_back(dim_scal_x_fc[0]);
  
  // set cell-centered data dimensions
  // scalar dimension
  std::vector<NcDim> dim_scal_z_cc;
  dim_scal_z_cc.push_back(addDimension("z_cc",ugd->nz-1));
  std::vector<NcDim> dim_scal_y_cc;
  dim_scal_y_cc.push_back(addDimension("y_cc",ugd->ny-1));
  std::vector<NcDim> dim_scal_x_cc;
  dim_scal_x_cc.push_back(addDimension("x_cc",ugd->nx-1));
  
  // 2D vector (surface, indep of time)
  std::vector<NcDim> dim_vect_2d;
  dim_vect_2d.push_back(dim_scal_y_cc[0]);
  dim_vect_2d.push_back(dim_scal_x_cc[0]);
  
  // 3D vector dimension (time indep)
  std::vector<NcDim> dim_vect_cc;
  dim_vect_cc.push_back(dim_scal_t[0]);
  dim_vect_cc.push_back(dim_scal_z_cc[0]);
  dim_vect_cc.push_back(dim_scal_y_cc[0]);
  dim_vect_cc.push_back(dim_scal_x_cc[0]);
  
  // create attributes 
  createAttScalar("t","time","s",dim_scal_t,&time);
  createAttVector("u","x-component velocity","m s-1",dim_vect_fc,&(ugd->u));
  createAttVector("v","y-component velocity","m s-1",dim_vect_fc,&(ugd->v));
  createAttVector("w","z-component velocity","m s-1",dim_vect_fc,&(ugd->w));
  createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd->terrain));
  createAttVector("icell","icell flag value","--",dim_vect_cc,&(ugd->icellflag));  

  /// attributes for coefficients for SOR solver
  //createAttVector("icell","icell flag value","--",dim_vect_cc,&(ugd->icellflag));  
  
  // create output fields
  addOutputFields();
}

bool URBOutput_TURBInputFile::validateFileOtions()
{
  //check all fileoption specificed to make sure it's possible...
  return true;
}

  
// Save output at cell-centered values
void URBOutput_TURBInputFile::save(URBGeneralData *ugd)
{
  
  // set time
  time = (double)output_counter;
  
  // save fields
  saveOutputFields();

  // remove terrain 
  // from output array after first save
  if (output_counter==0) {
    rmOutputField("terrain");
  }
  

  // increment for next time insertion
  output_counter +=1;

  
};
