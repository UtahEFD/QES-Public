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
  std::vector<NcDim> dim_vect;
  dim_vect.push_back(addDimension("t"));
  dim_vect.push_back(addDimension("z",ugd->nz));
  dim_vect.push_back(addDimension("y",ugd->ny));
  dim_vect.push_back(addDimension("x",ugd->nx));
  createDimensions(dim_vect); 
      
  // create attributes
  createAttScalar("t","time","s",dim_scalar_t,&time);
  createAttVector("u","x-component velocity","m s-1",dim_vector,&(ugd->u));
  createAttVector("v","y-component velocity","m s-1",dim_vector,&(ugd->v));
  createAttVector("w","z-component velocity","m s-1",dim_vector,&(ugd->w));

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
