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

  /*
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
    
  // loop through 2D double fields to save
  for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
      
    // x,y,z, terrain saved once with no time component
    if (i<3 && output_counter==0) {
      saveField2D(output_vector_dbl[i].name, *output_vector_dbl[i].data);
    } else if (i==3 && output_counter==0) {
      saveField2D(output_vector_dbl[i].name, vector_index_2d,
			  vector_size_2d, *output_vector_dbl[i].data);
    } else {
      saveField2D(output_vector_dbl[i].name, vector_index,
			  vector_size, *output_vector_dbl[i].data);
    }
  }
    
  // loop through 2D int fields to save
  for (unsigned int i=0; i<output_vector_int.size(); i++) {
    saveField2D(output_vector_int[i].name, vector_index,
			vector_size, *output_vector_int[i].data);
  }
  */
};
