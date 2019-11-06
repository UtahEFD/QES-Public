#include "URBOutput_WindVelFaceCentered.h"

URBOutput_WindVelFaceCentered::URBOutput_WindVelFaceCentered(URBGeneralData *ugd,std::string output_file)
  : URBOutput_Generic(output_file)
{
  std::cout<<"Getting output fields for Face-Centered data"<<std::endl;
  
  //FM -> need to implement the outputFields options here...
  output_fields = {"u","v","w"};
  
  /* output_fields = UID->fileOptions->outputFields;
     
     if (output_fields.empty() || output_fields[0]=="all") {
     output_fields.clear();
     output_fields = {"u","v","w","icell"};
     }
     
     validateFileOptions();     
  */
  
  // set face-centered data dimensions
  // scalar dimension 
  std::vector<NcDim> dim_scal_t;
  dim_scal_t.push_back(addDimension("t"));
  std::vector<NcDim> dim_scal_z;
  dim_scal_z.push_back(addDimension("z",ugd->nz-2));
  std::vector<NcDim> dim_scal_y;
  dim_scal_y.push_back(addDimension("y",ugd->ny-1));
  std::vector<NcDim> dim_scal_x;
  dim_scal_x.push_back(addDimension("x",ugd->nx));
  
  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_fc;
  dim_vect_fc.push_back(addDimension("t"));
  dim_vect_fc.push_back(addDimension("z",ugd->nz));
  dim_vect_fc.push_back(addDimension("y",ugd->ny));
  dim_vect_fc.push_back(addDimension("x",ugd->nx));
  
  
  // set cell-centered data dimensions
  // 2D vector (surface, indep of time)
  std::vector<NcDim> dim_vect_2d;
  dim_vect_2d.push_back(addDimension("y",ugd->ny-1));
  dim_vect_2d.push_back(addDimension("x",ugd->nx-1));

  // 3D vector dimension (time dep)
  std::vector<NcDim> dim_vect_cc;
  dim_vect_cc.push_back(addDimension("t"));
  dim_vect_cc.push_back(addDimension("z",ugd->nz-1));
  dim_vect_cc.push_back(addDimension("y",ugd->ny-1));
  dim_vect_cc.push_back(addDimension("x",ugd->nx-1));
        
  // create attributes
  createAttScalar("t","time","s",dim_scal_t,&time);
  createAttVector("u","x-component velocity","m s-1",dim_vect_fc,&(ugd->u));
  createAttVector("v","y-component velocity","m s-1",dim_vect_fc,&(ugd->v));
  createAttVector("w","z-component velocity","m s-1",dim_vect_fc,&(ugd->w));
  createAttVector("terrain","terrain height","m",dim_vect_2d,&(ugd->terrain));
  createAttVector("icell","icell flag value","--",dim_vect_cc,&(ugd->icellflag));
  

  // create output fields
  addOutputFields();
}

bool URBOutput_WindVelFaceCentered::validateFileOtions()
{
  //check all fileoption specificed to make sure it's possible...
  return true;
}

  
// Save output at cell-centered values
void URBOutput_WindVelFaceCentered::save(URBGeneralData *ugd)
{
  
  // set time
  time = (double)output_counter;
  
  // save fields
  saveOutputFields();
  
  // increment for next time insertion
  output_counter +=1;

};
