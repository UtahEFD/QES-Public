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

struct AttScalarFlt {
  double* data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
struct AttVectorFlt {
  std::vector<double>* data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

struct AttVectorInt {
  std::vector<int>* data;
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
  
  void addOutputFields();
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

  std::map<std::string,AttScalarDbl> map_att_scalar_dbl;
  std::map<std::string,AttVectorDbl> map_att_vector_dbl;

  std::map<std::string,AttVectorInt> map_att_vector_int;

  std::vector<AttScalarDbl> output_scalar_dbl;
  std::vector<AttVectorDbl> output_vector_dbl;

  std::vector<AttVectorInt> output_vector_int;

};

