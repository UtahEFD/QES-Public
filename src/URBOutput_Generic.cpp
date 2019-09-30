#include "URBOutput_Generic.h"

URBOutput_Generic::URBOutput_Generic(std::string output_file) 
  : NetCDFOutput(output_file)
{
};

/*
  Attribute are create based and the type of the data
  -> attribute are store in map_att_*
  -> all possible attribute available for the derived class should be 
  created by its constructor. 
  -> attribute are pushed back to output_* based on what is selected 
  by output_fields 
  -> the methods allows to by type generic (as long as the data is 
  either int,float or double
 */

//----------------------------------------
// create attribute scalar
// -> float
void URBOutput_Generic::createAttSacalar(std::string name,
					 std::string long_name,
					 std::string units,
					 std::vector<NcDim> dims,
					 int* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarInt att = {&data,name,long_name,units,dims};
  map_att_scalar_flt.emplace(name,att);
  
}
// -> float
void URBOutput_Generic::createAttSacalar(std::string name,
					 std::string long_name,
					 std::string units,
					 std::vector<NcDim> dims,
					 float* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarFlt att = {&data,name,long_name,units,dims};
  map_att_scalar_flt.emplace(name,att);
  
}
// -> double
void URBOutput_Generic::createAttSacalar(std::string name,
					 std::string long_name,
					 std::string units,
					 std::vector<NcDim> dims,
					 double* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarDbl att = {&data,name,long_name,units,dims};
  map_att_scalar_dbl.emplace(name,att);
}

//----------------------------------------
// create attribute Vector
// -> int
void URBOutput_Generic::createAttVector(std::string name,
					 std::string long_name,
					 std::string units,
					 std::vector<NcDim> dims,
					 int* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorInt att = {&data,name,long_name,units,dims};
  map_att_vector_int.emplace(name,att);
  
}
// -> float
void URBOutput_Generic::createAttVector(std::string name,
					 std::string long_name,
					 std::string units,
					 std::vector<NcDim> dims,
					 float* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorFlt att = {&data,name,long_name,units,dims};
  map_att_vector_flt.emplace(name,att);
  
}
// -> double
void URBOutput_Generic::createAttVector(std::string name,
					 std::string long_name,
					 std::string units,
					 std::vector<NcDim> dims,
					 double* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorDbl att = {&data,name,long_name,units,dims};
  map_att_vector_dbl.emplace(name,att);
}

//----------------------------------------
void URBOutput_Generic::addOutputFields()
{
  
  // create list of fields to save base on output_fields
  for (size_t i=0; i<output_fields.size(); i++) {
    std::string key = output_fields[i];
    
    if (map_att_scalar_int.count(key)) {
      // scalar int
      output_scalar_int.push_back(map_att_scalar_int[key]);
    } else if (map_att_vector_dbl.count(key)) {
      // scalar flt
      output_vector_flt.push_back(map_att_vector_dbl[key]);
    } else if (map_att_vector_dbl.count(key)) {
      // scalar dbl
      output_vector_dbl.push_back(map_att_vector_dbl[key]);  
    } else if(map_att_vector_int.count(key)) {
      // vector int
      output_vector_int.push_back(map_att_vector_int[key]);
    } else if(map_att_vector_flt.count(key)) {
      // vector flt
      output_vector_flt.push_back(map_att_vector_flt[key]);    
    } else if (map_att_vector_dbl.count(key)) {
      // vector dbl
      output_vector_dbl.push_back(map_att_vector_dbl[key]);
    }
  }

  // add scalar fields
  // -> int
  for ( AttScalarInt att : output_scalar_int ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
  }
  // -> float
  for ( AttScalarFlt att : output_scalar_flt ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
  }
  // -> double
  for ( AttScalarDbl att : output_scalar_dbl ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
  }

  // add vector fields
  // -> int
  for ( AttVectorInt att : output_vector_int ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
  }
  // -> int
  for ( AttVectorFlt att : output_vector_flt ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
  }x
  // -> double
  for ( AttVectorDbl att : output_vector_dbl ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
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

  // FM need to change this as it is not generic for int/float/double
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
  vector_size  = {1, static_cast<unsigned long>(nz_out),
		  static_cast<unsigned long>(ny_out),
		  static_cast<unsigned long>(nx_out)};
  vector_index_2d = {0, 0};
  vector_size_2d  = {static_cast<unsigned long>(ny_out),
		     static_cast<unsigned long>(nx_out)};
  
  // loop through scalar fields to save
  // -> int
  for (unsigned int i=0; i<output_scalar_int.size(); i++) {
    saveField1D(output_scalar_int[i].name, scalar_index,
		output_scalar_int[i].data);
  }
  // -> float
  for (unsigned int i=0; i<output_scalar_flt.size(); i++) {
    saveField1D(output_scalar_flt[i].name, scalar_index,
		output_scalar_flt[i].data);
  }
  // -> double
  for (unsigned int i=0; i<output_scalar_dbl.size(); i++) {
    saveField1D(output_scalar_dbl[i].name, scalar_index,
		output_scalar_dbl[i].data);
  }

  
  // loop through vectore fields to save
  // -> int
  for (unsigned int i=0; i<output_vector_int.size(); i++) {
    // 1D (x or z or y)
    if (output_vector_int[i].dimensions.size()==1) {
      saveField2D(output_vector_int[i].name, *output_vector_int[i].data);
    } 
    // 2D (terrain)
    else if (output_vector_int[i].dimensions.size()==2) {
      saveField2D(output_vector_int[i].name, vector_index_2d,
		  vector_size_2d, *output_vector_int[i].data);
    }
    // 4D (u,v,w)
    else if (output_vector_int[i].dimensions.size()==4) {
      saveField2D(output_vector_int[i].name, vector_index,
		  vector_size, *output_vector_int[i].data);
    }
  }
  // -> float
  for (unsigned int i=0; i<output_vector_flt.size(); i++) { 
    // 1D (x or z or y)
    if (output_vector_flt[i].dimensions.size()==1) {
      saveField2D(output_vector_flt[i].name, *output_vector_flt[i].data);
    } 
    // 2D (terrain)
    else if (output_vector_flt[i].dimensions.size()==2) {
      saveField2D(output_vector_flt[i].name, vector_index_2d,
		  vector_size_2d, *output_vector_flt[i].data);
    }
    // 4D (u,v,w)
    else if (output_vector_flt[i].dimensions.size()==4) {
      saveField2D(output_vector_flt[i].name, vector_index,
		  vector_size, *output_vector_flt[i].data);
    }
  }
  // -> double
  for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
    // 1D (x or z or y)
    if (output_vector_dbl[i].dimensions.size()==1) {
      saveField2D(output_vector_dbl[i].name, *output_vector_dbl[i].data);
    } 
    // 2D (terrain)
    else if (output_vector_dbl[i].dimensions.size()==2) {
      saveField2D(output_vector_dbl[i].name, vector_index_2d,
		  vector_size_2d, *output_vector_dbl[i].data);
    }
    // 4D (u,v,w)
    else if (output_vector_dbl[i].dimensions.sizexs()==4) {
      saveField2D(output_vector_dbl[i].name, vector_index,
		  vector_size, *output_vector_dbl[i].data);
    }
  }

};
