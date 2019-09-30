#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <netcdf>

#include "URBGeneralData.h"
#include "NetCDFOutput.h"

/**
 * This class handles saving output files.
 */

using namespace netCDF;
using namespace netCDF::exceptions;

struct AttVectorInt {
  std::vector<int>* data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

struct AttScalarFlt {
  float* data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
struct AttVectorFlt {
  std::vector<float>* data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

struct AttScalarDbl {
  double* data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
struct AttVectorDbl {
  std::vector<double>* data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

class URBOutput_Generic : public NetCDFOutput 
{
 public:
  URBOutput_Generic()
    {}
  URBOutput_Generic(std::string);
  virtual ~URBOutput_Generic()
    {}

  // create dimenson of output fields
  void createDimensions(std::vector<NcDim>);
  
  // create attribute scalar based on type of data
  void createAttScalar(std::string,std::string,std::string,
		       std::vector<NcDim>,int*);
  void createAttScalar(std::string,std::string,std::string,
		       std::vector<NcDim>,float*);
  void createAttScalar(std::string,std::string,std::string,
		       std::vector<NcDim>,double*);
  
  // create attribute vector based on type of data
  void createAttVector(std::string,std::string,std::string,
		       std::vector<NcDim>,int*);
  void createAttVector(std::string,std::string,std::string,
		       std::vector<NcDim>,float*);
  void createAttVector(std::string,std::string,std::string,
		       std::vector<NcDim>,double*);

  // add fields based on output_fields
  void addOutputFields();
  // save fields 
  void saveOutputFields();

 protected:
  
  virtual bool validateFileOtions() 
  {
    return true;
  };
  
  virtual void save(URBGeneralData*)
  {}
  
  int output_counter=0;
  double time=0;
    
  std::vector<NcDim> dim_scalar_t;
  std::vector<NcDim> dim_scalar_z;
  std::vector<NcDim> dim_scalar_y;
  std::vector<NcDim> dim_scalar_x;
  std::vector<NcDim> dim_vector;
  std::vector<NcDim> dim_vector_2d;

  std::vector<std::string> output_fields;

  std::map<std::string,AttScalarInt> map_att_scalar_int;
  std::map<std::string,AttScalarFlt> map_att_scalar_flt;
  std::map<std::string,AttScalarDbl> map_att_scalar_dbl;
  std::map<std::string,AttVectorInt> map_att_vector_int;
  std::map<std::string,AttVectorFlt> map_att_vector_flt;
  std::map<std::string,AttVectorDbl> map_att_vector_dbl;

  std::vector<AttScalarInt> output_scalar_int;
  std::vector<AttScalarFlt> output_scalar_flt;
  std::vector<AttScalarDbl> output_scalar_dbl;
  std::vector<AttVectorInt> output_vector_int;
  std::vector<AttVectorFlt> output_vector_flt;
  std::vector<AttVectorDbl> output_vector_dbl;

};

