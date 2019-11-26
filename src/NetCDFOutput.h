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
  NetCDFOutput()
    {}
  // initializer
  NetCDFOutput(std::string);
  virtual ~NetCDFOutput()
    {}

  // setter
  NcDim addDimension(std::string, int size=0);
  NcDim getDimension(std::string);
  void addField(std::string, std::string, std::string, std::vector<NcDim>, NcType);
  
  // save functions for 1D array (save 1D time)
  void saveField1D(std::string, const std::vector<size_t>, int*);
  void saveField1D(std::string, const std::vector<size_t>, float*);
  void saveField1D(std::string, const std::vector<size_t>, double*);
  
  // save functions for 2D array (save 1D array, eg: x,y,z )
  void saveField2D(std::string, std::vector<int>&);
  void saveField2D(std::string, std::vector<float>&);
  void saveField2D(std::string, std::vector<double>&);

  // save functions for *D 
  void saveField2D(std::string, const std::vector<size_t>,
		   std::vector<size_t>, std::vector<int>&);
  void saveField2D(std::string, const std::vector<size_t>,
		   std::vector<size_t>, std::vector<float>&);
  void saveField2D(std::string, const std::vector<size_t>,
		   std::vector<size_t>, std::vector<double>&);
  
};
