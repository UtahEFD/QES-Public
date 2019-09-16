#pragma once

#include <string>
#include <vector>
#include <map>
#include <netcdf>

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
struct AttVectorInt {
  std::vector<int>* data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};


class NetCDFOutput {
    
protected:
  // netCDF variables
  NcFile* outfile;
  std::map<std::string,NcVar> fields;
  
  virtual bool validateFileOtions() 
  {
    return true;
  }

  std::map<std::string,AttScalarDbl> map_att_scalar_dbl;
  std::map<std::string,AttVectorDbl> map_att_vector_dbl;
  std::map<std::string,AttVectorInt> map_att_vector_int;
  std::vector<AttScalarDbl> output_scalar_dbl;
  std::vector<AttVectorDbl> output_vector_dbl;
  std::vector<AttVectorInt> output_vector_int;
  
public:
  
  // initializer
  NetCDFOutput(FileOptions &fopts, std::string);
  
  // setter
  NcDim addDimension(std::string, int size=0);
  NcDim getDimension(std::string);
  void addField(std::string, std::string, std::string, std::vector<NcDim>, NcType);
  void saveField1D(std::string, const std::vector<size_t>, double*);
  void saveField2D(std::string, std::vector<double>&);
  void saveField2D(std::string, const std::vector<size_t>,
		   std::vector<size_t>, std::vector<double>&);
  void saveField2D(std::string, const std::vector<size_t>,
		   std::vector<size_t>, std::vector<int>&);
  
};

