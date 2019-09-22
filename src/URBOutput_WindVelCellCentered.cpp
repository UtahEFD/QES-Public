#include "URBOutput_WindVelCellCentered.h"

URBOutput_WindVelCellCentered::URBOutput_WindVelCellCentered(URBGeneralData *ugd,std::string output_file)
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
  u_out.resize( numcell_cout, 0.0 );
  v_out.resize( numcell_cout, 0.0 );
  w_out.resize( numcell_cout, 0.0 );
  
  icellflag_out.resize( numcell_cout, 0.0 );

  output_fields = {"x","y","z","u","v","w","icell"};
  
  // set cell-centered dimensions
  NcDim t_dim = addDimension("t");
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
  // dim_vector_2d.push_back(y_dim);
  // dim_vector_2d.push_back(x_dim);
    
  // create attributes
  AttScalarDbl att_t = {&time,"t", "time","s", dim_scalar_t};
  AttVectorDbl att_x = {&x_out,"x", "x-distance", "m", dim_scalar_x};
  AttVectorDbl att_y = {&y_out,"y", "y-distance", "m", dim_scalar_y};
  AttVectorDbl att_z = {&z_out,"z", "z-distance", "m", dim_scalar_z};
  AttVectorDbl att_u = {&u_out,"u", "x-component velocity", "m s-1", dim_vector};
  AttVectorDbl att_v = {&v_out,"v", "y-component velocity", "m s-1", dim_vector};
  AttVectorDbl att_w = {&w_out,"w", "z-component velocity", "m s-1", dim_vector};
  // AttVectorDbl att_h = {&terrain,
  //"terrain", "terrain height", "m", dim_vector_2d};
  AttVectorInt att_i = {&icellflag_out,"icell", "icell flag value", "--", dim_vector};
    
  // map the name to attributes
  map_att_scalar_dbl.emplace("t", att_t);
  map_att_vector_dbl.emplace("x", att_x);
  map_att_vector_dbl.emplace("y", att_y);
  map_att_vector_dbl.emplace("z", att_z);
  map_att_vector_dbl.emplace("u", att_u);
  map_att_vector_dbl.emplace("v", att_v);
  map_att_vector_dbl.emplace("w", att_w);
  //map_att_vector_dbl.emplace("terrain", att_h);
  map_att_vector_int.emplace("icell", att_i);
    
  // we will always save time and grid lengths
  output_scalar_dbl.push_back(map_att_scalar_dbl["t"]);
  //output_vector_dbl.push_back(map_att_vector_dbl["x"]);
  //output_vector_dbl.push_back(map_att_vector_dbl["y"]);
  //output_vector_dbl.push_back(map_att_vector_dbl["z"]);
  //output_vector_dbl.push_back(map_att_vector_dbl["terrain"]);  
  
  addOutputFields();

  /*  
  // create list of fields to save
  for (size_t i=0; i<output_fields.size(); i++) {
    std::string key = output_fields[i];
    if (map_att_scalar_dbl.count(key)) {
      output_scalar_dbl.push_back(map_att_scalar_dbl[key]);
    } else if (map_att_vector_dbl.count(key)) {
      output_vector_dbl.push_back(map_att_vector_dbl[key]);
    } else if(map_att_vector_int.count(key)) {
      output_vector_int.push_back(map_att_vector_int[key]);
    }
  }
  
  // add vector double fields
  for ( AttScalarDbl att : output_scalar_dbl ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
  }

  // add vector double fields
  for ( AttVectorDbl att : output_vector_dbl ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
  }

  // add vector int fields
  for ( AttVectorInt att : output_vector_int ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
  }
  */

}

bool URBOutput_WindVelCellCentered::validateFileOtions()
{
  //check all fileoption specificed to make sure it's possible...
  return true;
}

  
// Save output at cell-centered values
void URBOutput_WindVelCellCentered::save(URBGeneralData *ugd)
{
  /* FM -> need to clean this .....
    int nx = ugd->nx;
    int ny = ugd->ny;
    int nz = ugd->nz;
    
    
    // output size and location
    std::vector<size_t> scalar_index;
    std::vector<size_t> scalar_size;
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;
    std::vector<size_t> vector_index_2d;
    std::vector<size_t> vector_size_2d;
    
    scalar_index = {static_cast<unsigned long>(output_counter)};
    scalar_size  = {1};
    vector_index = {static_cast<size_t>(output_counter), 0, 0, 0};
    vector_size  = {1, static_cast<unsigned long>(nz-2),
    static_cast<unsigned long>(ny-1), 
    static_cast<unsigned long>(nx-1)};
    vector_index_2d = {0, 0};
    vector_size_2d  = {static_cast<unsigned long>(ny-1), 
    static_cast<unsigned long>(nx-1)};
    
    
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
    
    /* better way (more robust) using:
    - output_vector_dbl[i].dimensions.size()
    - output_vector_dbl[i].dimensions[0].getSize()
  */
  
  saveOutputFields();
  
  /*
  // loop through 1D fields to save
  for (auto i=0u; i<output_scalar_dbl.size(); i++) {
    saveField1D(output_scalar_dbl[i].name, scalar_index, output_scalar_dbl[i].data);    
  }

  // loop through 2D double fields to save
  for (auto i=0u; i<output_vector_dbl.size(); i++) {
    
    // x,y,z, saved once with no time component
    if (i<3 && output_counter==0) {
      saveField2D(output_vector_dbl[i].name, *output_vector_dbl[i].data);
    } else {
      saveField2D(output_vector_dbl[i].name, vector_index,
			  vector_size, *output_vector_dbl[i].data);
    }
  }
    
  // loop through 2D int fields to save
  for (auto i=0u; i<output_vector_int.size(); i++) {
    saveField2D(output_vector_int[i].name, vector_index,
		vector_size, *output_vector_int[i].data);
  }
  */

  // FM -> need to change this!!
  // remove x, y, z from output array after first save
  if (output_counter==0) {
    output_vector_dbl.erase(output_vector_dbl.begin(),output_vector_dbl.begin()+2);
  }
  
  // increment for next time insertion
  output_counter +=1;
  
};
