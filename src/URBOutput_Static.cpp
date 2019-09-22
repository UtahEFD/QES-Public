#include "URBOutput_Static.h"

URBOutput_Static::URBOutput_Static(URBGeneralData *ugd,std::string output_file)
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
  
  int nx = ugd->nx;
  int ny = ugd->ny;
  int nz = ugd->nz;
  
  long numcell_cout = (nx-1)*(ny-1)*(nz-2);

  z_out.resize( nz-2 );
  for (auto k=1; k<nz-1; k++)
    {
      z_out[k-1] = (k-0.5)*ugd->dz;;    /**< Location of face centers in z-dir */
    }
  
  x_out.resize( nx-1 );
  for (auto i=0; i<nx-1; i++)
    {
      x_out[i] = (i+0.5)*ugd->dx;          /**< Location of face centers in x-dir */
    }
  
  y_out.resize( ny-1 );
  for (auto j=0; j<ny-1; j++)
    {
      y_out[j] = (j+0.5)*ugd->dy;          /**< Location of face centers in y-dir */
    }

  // /////////////////////////////////////////
  // Output related data
  output_fields = {"x","y","z","terrain"};
  
  // set cell-centered dimensions
  NcDim t_dim = addDimension("t",1);
  NcDim z_dim = addDimension("z",ugd->nz-2);
  NcDim y_dim = addDimension("y",ugd->ny-1);
  NcDim x_dim = addDimension("x",ugd->nx-1);
  
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
  AttScalarDbl att_t = {&time,"t", "time","s", dim_scalar_t};
  AttVectorDbl att_x = {&x_out,"x", "x-distance", "m", dim_scalar_x};
  AttVectorDbl att_y = {&y_out,"y", "y-distance", "m", dim_scalar_y};
  AttVectorDbl att_z = {&z_out,"z", "z-distance", "m", dim_scalar_z};
  //AttVectorDbl att_u = {&u_out,"u", "x-component velocity", "m s-1", dim_vector};
  //AttVectorDbl att_v = {&v_out,"v", "y-component velocity", "m s-1", dim_vector};
  //AttVectorDbl att_w = {&w_out,"w", "z-component velocity", "m s-1", dim_vector};
  AttVectorDbl att_h = {&(ugd->terrain),"terrain", "terrain height", "m", dim_vector_2d};
  //AttVectorInt att_i = {&icellflag_out,"icell", "icell flag value", "--", dim_vector};
    
  // map the name to attributes
  map_att_scalar_dbl.emplace("t", att_t);
  map_att_vector_dbl.emplace("x", att_x);
  map_att_vector_dbl.emplace("y", att_y);
  map_att_vector_dbl.emplace("z", att_z);
  //map_att_vector_dbl.emplace("u", att_u);
  //map_att_vector_dbl.emplace("v", att_v);
  //map_att_vector_dbl.emplace("w", att_w);
  map_att_vector_dbl.emplace("terrain", att_h);
  //map_att_vector_int.emplace("icell", att_i);
    
  // we will always save time and grid lengths
  output_scalar_dbl.push_back(map_att_scalar_dbl["t"]);
  
  addOutputFields();
  
}

bool URBOutput_Static::validateFileOtions()
{
  //check all fileoption specificed to make sure it's possible...
  return true;
}

  
// Save output at cell-centered values
void URBOutput_Static::save(URBGeneralData *ugd)
{
  if(output_counter==0) {
    saveOutputFields();
    output_counter +=1;
  } else {
    std::cout << "Static fields already saved" << std::endl;
  }
};
