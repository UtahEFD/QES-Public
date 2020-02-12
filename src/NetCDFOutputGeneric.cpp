
#include "NetCDFOutputGeneric.h"


NetCDFOutputGeneric::NetCDFOutputGeneric(std::string output_file,bool doFileOutput_val) 
  : NetCDFOutput(output_file,doFileOutput_val)
{
};

//----------------------------------------
// create attribute scalar
// -> int
void NetCDFOutputGeneric::createAttScalar(std::string name,
					std::string long_name,
					std::string units,
					std::vector<NcDim> dims,
					int* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarInt att = {data,name,long_name,units,dims};
  map_att_scalar_int.emplace(name,att);
  

}
// -> float
void NetCDFOutputGeneric::createAttScalar(std::string name,
					std::string long_name,
					std::string units,
					std::vector<NcDim> dims,
					float* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarFlt att = {data,name,long_name,units,dims};
  map_att_scalar_flt.emplace(name,att);
  
}
// -> double
void NetCDFOutputGeneric::createAttScalar(std::string name,
					std::string long_name,
					std::string units,
					std::vector<NcDim> dims,
					double* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarDbl att = {data,name,long_name,units,dims};
  map_att_scalar_dbl.emplace(name,att);
}

//----------------------------------------
// create attribute Vector
// -> int
void NetCDFOutputGeneric::createAttVector(std::string name,
					std::string long_name,
					std::string units,
					std::vector<NcDim> dims,
					std::vector<int>* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorInt att = {data,name,long_name,units,dims};
  map_att_vector_int.emplace(name,att);
  
}
// -> float
void NetCDFOutputGeneric::createAttVector(std::string name,
					std::string long_name,
					std::string units,
					std::vector<NcDim> dims,
					std::vector<float>* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorFlt att = {data,name,long_name,units,dims};
  map_att_vector_flt.emplace(name,att);
  
}
// -> double
void NetCDFOutputGeneric::createAttVector(std::string name,
					std::string long_name,
					std::string units,
					std::vector<NcDim> dims,
					std::vector<double>* data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorDbl att = {data,name,long_name,units,dims};
  map_att_vector_dbl.emplace(name,att);
}

//----------------------------------------
void NetCDFOutputGeneric::addOutputFields()
{
   /*
    This function add the  fields to the output vectors
    and link them to the NetCDF.

    Since the type is not know, one needs to loop through 
    the 6 output vector to find it.
        
    FMargairaz
  */
  
  // create list of fields to save base on output_fields
  for (size_t i=0; i<output_fields.size(); i++) {
    std::string key = output_fields[i];
        
    if (map_att_scalar_int.count(key)) {
      // scalar int
      output_scalar_int.push_back(map_att_scalar_int[key]);
    } else if (map_att_scalar_flt.count(key)) {
      // scalar flt
      output_scalar_flt.push_back(map_att_scalar_flt[key]);
    } else if (map_att_scalar_dbl.count(key)) {
      // scalar dbl
      output_scalar_dbl.push_back(map_att_scalar_dbl[key]);  
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
    addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
  }  
  // -> float
  for ( AttScalarFlt att : output_scalar_flt ) {    
    addField(att.name, att.units, att.long_name, att.dimensions, ncFloat);
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
    addField(att.name, att.units, att.long_name, att.dimensions, ncFloat);
  }
  // -> double
  for ( AttVectorDbl att : output_vector_dbl ) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
  }
  
};



void NetCDFOutputGeneric::rmOutputField(std::string name)
{
  /*
    This function remove a field from the output vectors
    Since the type is not know, one needs to loop through 
    the 6 output vector to find it.
    
    Note: the field CANNOT be added again.
    
    FMargairaz
  */

  // loop through scalar fields to remove
  // -> int
  for (unsigned int i=0; i<output_scalar_int.size(); i++) {
    if(output_scalar_int[i].name==name) {
      output_scalar_int.erase(output_scalar_int.begin()+i);
      return;
    }
  }
  // -> float
  for (unsigned int i=0; i<output_scalar_flt.size(); i++) {
    if(output_scalar_flt[i].name==name) {
      output_scalar_flt.erase(output_scalar_flt.begin()+i);
      return;
    } 
  }
  
  // -> double
  for (unsigned int i=0; i<output_scalar_dbl.size(); i++) {
    if(output_scalar_dbl[i].name==name) {
      output_scalar_dbl.erase(output_scalar_dbl.begin()+i);
      return;
    } 
  }
  
  // loop through vector fields to remove
  // -> int
  for (unsigned int i=0; i<output_vector_int.size(); i++) {
    if(output_vector_int[i].name==name) {
      output_vector_int.erase(output_vector_int.begin()+i);
      return;
    } 
  }
  // -> float
  for (unsigned int i=0; i<output_vector_flt.size(); i++) { 
    if(output_vector_flt[i].name==name) {
      output_vector_flt.erase(output_vector_flt.begin()+i);
      return;
    } 
  }
  // -> double
  for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
    if(output_vector_dbl[i].name==name) {
      output_vector_dbl.erase(output_vector_dbl.begin()+i);
      return;
    } 
  } 
};

void NetCDFOutputGeneric::rmTimeIndepFields()
{
  /*
    This function remove time indep field from the output vectors
    Since the types are not know, one needs to loop through 
    the 6 output vector to find it.
    
    Note: the fields CANNOT be added again.
    
    FMargairaz
  */

  // loop through scalar fields to remove
  // -> int
  for (unsigned int i=0; i<output_scalar_int.size(); i++) {
    if(output_scalar_int[i].dimensions[0].getName()!="t") {
      output_scalar_int.erase(output_scalar_int.begin()+i);
    }
  }
  // -> float
  for (unsigned int i=0; i<output_scalar_flt.size(); i++) {
    if(output_scalar_flt[i].dimensions[0].getName()!="t") {
      output_scalar_flt.erase(output_scalar_flt.begin()+i);
    } 
  }
  
  // -> double
  for (unsigned int i=0; i<output_scalar_dbl.size(); i++) {
    if(output_scalar_dbl[i].dimensions[0].getName()!="t") {
      output_scalar_dbl.erase(output_scalar_dbl.begin()+i);
    } 
  }
  
  // loop through vector fields to remove
  // -> int
  for (unsigned int i=0; i<output_vector_int.size(); i++) {
    if(output_vector_int[i].dimensions[0].getName()!="t") {
      output_vector_int.erase(output_vector_int.begin()+i);
    } 
  }
  // -> float
  for (unsigned int i=0; i<output_vector_flt.size(); i++) { 
    if(output_vector_flt[i].dimensions[0].getName()!="t") {
      output_vector_flt.erase(output_vector_flt.begin()+i);
    } 
  }
  // -> double
  for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
    if(output_vector_dbl[i].dimensions[0].getName()!="t") {
      output_vector_dbl.erase(output_vector_dbl.begin()+i);
    } 
  } 
};

void NetCDFOutputGeneric::saveOutputFields()
{
   /*
    This function save the fields from the output vectors
    Since the type is not know, one needs to loop through 
    the 6 output vector to find it.
    
    FMargairaz
   */

  // loop through scalar fields to save
  // -> int
  for (unsigned int i=0; i<output_scalar_int.size(); i++) {
    std::vector<size_t> scalar_index;
    scalar_index = {static_cast<unsigned long>(output_counter)};  
    saveField1D(output_scalar_int[i].name, scalar_index,
		output_scalar_int[i].data);
  }
  // -> float
  for (unsigned int i=0; i<output_scalar_flt.size(); i++) {
    std::vector<size_t> scalar_index;
    scalar_index = {static_cast<unsigned long>(output_counter)}; 
    saveField1D(output_scalar_flt[i].name, scalar_index,
		output_scalar_flt[i].data);
  }
  // -> double
  for (unsigned int i=0; i<output_scalar_dbl.size(); i++) {
    std::vector<size_t> scalar_index;
    scalar_index = {static_cast<unsigned long>(output_counter)}; 
    saveField1D(output_scalar_dbl[i].name, scalar_index,
		output_scalar_dbl[i].data);
  }
  
  // loop through vector fields to save
  // -> int
  for (unsigned int i=0; i<output_vector_int.size(); i++) {

    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;
    
    // if var is time dep -> special treatement for time
    if(output_vector_int[i].dimensions[0].getName()=="t"){
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for(unsigned int d=1;d<output_vector_int[i].dimensions.size();d++){
	int dim=output_vector_int[i].dimensions[d].getSize();
	vector_index.push_back(0);
	vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else{
      for(unsigned int d=0;d<output_vector_int[i].dimensions.size();d++){
	int dim=output_vector_int[i].dimensions[d].getSize();
	vector_index.push_back(0);
	vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    
    saveField2D(output_vector_int[i].name, vector_index,
		vector_size, *output_vector_int[i].data);
  }
  // -> float
  for (unsigned int i=0; i<output_vector_flt.size(); i++) { 
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;
    
    // if var is time dep -> special treatement for time
    if(output_vector_flt[i].dimensions[0].getName()=="t"){
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for(unsigned int d=1;d<output_vector_flt[i].dimensions.size();d++){
	int dim=output_vector_flt[i].dimensions[d].getSize();
	vector_index.push_back(0);
	vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else{
      for(unsigned int d=0;d<output_vector_flt[i].dimensions.size();d++){
	int dim=output_vector_flt[i].dimensions[d].getSize();
	vector_index.push_back(0);
	vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    
    saveField2D(output_vector_flt[i].name, vector_index,
		vector_size, *output_vector_flt[i].data);
  }
  // -> double
  for (unsigned int i=0; i<output_vector_dbl.size(); i++) {
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;
    
    // if var is time dep -> special treatement for time
    if(output_vector_dbl[i].dimensions[0].getName()=="t"){
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for(unsigned int d=1;d<output_vector_dbl[i].dimensions.size();d++){
	int dim=output_vector_dbl[i].dimensions[d].getSize();
	vector_index.push_back(0);
	vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else{
      for(unsigned int d=0;d<output_vector_dbl[i].dimensions.size();d++){
	int dim=output_vector_dbl[i].dimensions[d].getSize();
	vector_index.push_back(0);
	vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
   
    saveField2D(output_vector_dbl[i].name, vector_index,
		vector_size, *output_vector_dbl[i].data);
    
  }

};

