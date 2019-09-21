#include "URBOutput_WindVelFaceCentered.h"

URBOutput_WindVelFaceCentered::URBOutput_WindVelFaceCentered(URBGeneralData *ugd,std::string output_file)
  : URBOutput_Generic(output_file)
{
  /* FM -> need to implement the outputFields options here...
     std::cout<<"Getting output fields"<<std::endl;
     output_fields = UID->fileOptions->outputFields;
     
     if (output_fields.empty() || output_fields[0]=="all") {
     output_fields.clear();
     output_fields = {"u","v","w","icell"};
     }
     
     validateFileOptions();     
  */
  
  output_fields = {"u","v","w"};
  
  // set cell-centered dimensions
  NcDim t_dim = addDimension("t");
  NcDim z_dim = addDimension("z",ugd->nz);
  NcDim y_dim = addDimension("y",ugd->ny);
  NcDim x_dim = addDimension("x",ugd->nx);
  
  dim_scalar_t.push_back(t_dim);
  //dim_scalar_z.push_back(z_dim);
  //dim_scalar_y.push_back(y_dim);
  //dim_scalar_x.push_back(x_dim);
  dim_vector.push_back(t_dim);
  dim_vector.push_back(z_dim);
  dim_vector.push_back(y_dim);
  dim_vector.push_back(x_dim);
      
  // create attributes
  AttScalarDbl att_t = {&time,"t", "time","s", dim_scalar_t};
  //AttVectorDbl att_x = {&x_out,"x", "x-distance", "m", dim_scalar_x};
  //AttVectorDbl att_y = {&y_out,"y", "y-distance", "m", dim_scalar_y};
  //AttVectorDbl att_z = {&z_out,"z", "z-distance", "m", dim_scalar_z};
  AttVectorDbl att_u = {&(ugd->u),"u", "x-component velocity", "m s-1", dim_vector};
  AttVectorDbl att_v = {&(ugd->v),"v", "y-component velocity", "m s-1", dim_vector};
  AttVectorDbl att_w = {&(ugd->w),"w", "z-component velocity", "m s-1", dim_vector};
    
  // map the name to attributes
  map_att_scalar_dbl.emplace("t", att_t);
  //map_att_vector_dbl.emplace("x", att_x);
  //map_att_vector_dbl.emplace("y", att_y);
  //map_att_vector_dbl.emplace("z", att_z);
  map_att_vector_dbl.emplace("u", att_u);
  map_att_vector_dbl.emplace("v", att_v);
  map_att_vector_dbl.emplace("w", att_w);
    
  // we will always save time and grid lengths
  output_scalar_dbl.push_back(map_att_scalar_dbl["t"]);

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
