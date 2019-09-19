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

class NetCDFOutput {
    
protected:
  // netCDF variables
  NcFile* outfile;
  std::map<std::string,NcVar> fields;
  
public:
  
  // initializer
  NetCDFOutput(std::string);
  
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

