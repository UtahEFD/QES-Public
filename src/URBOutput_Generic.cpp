#include "URBOutput_Generic.h"

URBOutput_Generic::URBOutput_Generic(std::string output_file) 
  : NetCDFOutput(output_file)
{
};

void URBOutput_Generic::addOutputFields()
{
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

};

void URBOutput_Generic::saveOutputFields()
{

  int nx_out=0;
  int ny_out=0;
  int nz_out=0;
 
  /* 
     query fot the dimenson of the output. 
     if 4D array -> can get nx,ny,nz
     if 2D array -> can get nx,ny
     if only 1D -> ambuguous desciption, cannot get dimensions
  */

  std::vector<int> dims_vec_dbl;
  for (auto i=0u; i<output_vector_dbl.size(); i++) {
    dims_vec_dbl.push_back(output_vector_dbl[i].dimensions.size());
  }
  
  // find the 4D array
  std::vector<int>::iterator it4;
  it4 = find (dims_vec_dbl.begin(), dims_vec_dbl.end(), 4);
  // find the 2D array
  std::vector<int>::iterator it2;
  it2 = find (dims_vec_dbl.begin(), dims_vec_dbl.end(), 2);
  
  if (it4 != dims_vec_dbl.end()) {
    // Get index of element from iterator
    int id = std::distance(dims_vec_dbl.begin(), it4);
    nx_out = output_vector_dbl[id].dimensions[3].getSize();
    ny_out = output_vector_dbl[id].dimensions[2].getSize();
    nz_out = output_vector_dbl[id].dimensions[1].getSize();
  } else if (it2 != dims_vec_dbl.end()) {
    // Get index of element from iterator                                                                                                                                                                         
    int id = std::distance(dims_vec_dbl.begin(), it2);
    nx_out = output_vector_dbl[id].dimensions[1].getSize();
    ny_out = output_vector_dbl[id].dimensions[0].getSize();
  } else {
    // abort ??
  }  

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
  vector_size  = {1, static_cast<unsigned long>(nz_out),static_cast<unsigned long>(ny_out), static_cast<unsigned long>(nx_out)};
  vector_index_2d = {0, 0};
  vector_size_2d  = {static_cast<unsigned long>(ny_out), static_cast<unsigned long>(nx_out)};
  
  // loop through 1D fields to save
  for (unsigned int i=0; i<output_scalar_dbl.size(); i++) {
    saveField1D(output_scalar_dbl[i].name, scalar_index, output_scalar_dbl[i].data);
  }
    
  // loop through double fields to save
  for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
      
    // 1D (x,z,y)
    if (output_vector_dbl[i].dimensions.size()==1) {
      saveField2D(output_vector_dbl[i].name, *output_vector_dbl[i].data);
    } 
    // 2D (terrain)
    else if (output_vector_dbl[i].dimensions.size()==2) {
      saveField2D(output_vector_dbl[i].name, vector_index_2d, vector_size_2d, *output_vector_dbl[i].data);
    }
    // 4D (u,v,w)
    else if (output_vector_dbl[i].dimensions.size()==4) {
      saveField2D(output_vector_dbl[i].name, vector_index, vector_size, *output_vector_dbl[i].data);
    }
  }
    
  // loop through 2D int fields to save
  for (unsigned int i=0; i<output_vector_int.size(); i++) {
    saveField2D(output_vector_int[i].name, vector_index, vector_size, *output_vector_int[i].data);
  }

};
