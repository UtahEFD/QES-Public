#pragma once

#include "UrbOutput_WindVel.h"

// Specialized output classes that can take URBGeneratlData or
// URBInputData, etc... and dump out reasonably..
class URBOutput_WindVelFaceCentered : public URBOutput_WindVel
{
  
  virtual bool validateFileOtions()
  {
    //check all fileoption specificed to make sure it's possible...
  }

  URBOutput_WindVelCellCentered(URBGeneralData *ugd)
    : NetCDFOutput(...)
  {
    validateFileOptions();
    
    
    // set cell-centered dimensions
    NcDim t_dim = addDimension("t");
    NcDim z_dim = addDimension("z",nz-2);
    NcDim y_dim = addDimension("y",ny-1);
    NcDim x_dim = addDimension("x",nx-1);
    
    dim_scalar_t.push_back(t_dim);
    dim_scalar_z.push_back(z_dim);
    dim_scalar_y.push_back(y_dim);
    dim_scalar_x.push_back(x_dim);
    dim_vector.push_back(t_dim);
    dim_vector.push_back(z_dim);
    dim_vector.push_back(y_dim);
    dim_vector.push_back(x_dim);
    dim_vector_2d.push_back(y_dim);
    dim_vector_2d.push_back(x_dim);
    
    // create attributes
    AttScalarDbl att_t = {&time,  "t", "time",      "s", dim_scalar_t};
    AttVectorDbl att_x = {&(ugd->x_out), "x", "x-distance", "m", dim_scalar_x};
    AttVectorDbl att_y = {&y_out, "y", "y-distance", "m", dim_scalar_y};
    AttVectorDbl att_z = {&z_out, "z", "z-distance", "m", dim_scalar_z};
    AttVectorDbl att_u = {&u_out, "u", "x-component velocity", "m s-1", dim_vector};
    AttVectorDbl att_v = {&v_out, "v", "y-component velocity", "m s-1", dim_vector};
    AttVectorDbl att_w = {&w_out, "w", "z-component velocity", "m s-1", dim_vector};
    AttVectorDbl att_h = {&terrain,  "terrain", "terrain height", "m", dim_vector_2d};
    AttVectorInt att_i = {&icellflag_out,  "icell", "icell flag value", "--", dim_vector};
    
    // map the name to attributes
    map_att_scalar_dbl.emplace("t", att_t);
    map_att_vector_dbl.emplace("x", att_x);
    map_att_vector_dbl.emplace("y", att_y);
    map_att_vector_dbl.emplace("z", att_z);
    map_att_vector_dbl.emplace("u", att_u);
    map_att_vector_dbl.emplace("v", att_v);
    map_att_vector_dbl.emplace("w", att_w);
    map_att_vector_dbl.emplace("terrain", att_h);
    map_att_vector_int.emplace("icell", att_i);
    
    // we will always save time and grid lengths
    output_scalar_dbl.push_back(map_att_scalar_dbl["t"]);
    output_vector_dbl.push_back(map_att_vector_dbl["x"]);
    output_vector_dbl.push_back(map_att_vector_dbl["y"]);
    output_vector_dbl.push_back(map_att_vector_dbl["z"]);
    output_vector_dbl.push_back(map_att_vector_dbl["terrain"]);
    
  }

  
  saveStaticFields() 
  {
    
  }
  
  updateTimeSeriesFields()
  {
    // repeats the save .... 
  }
  
};
