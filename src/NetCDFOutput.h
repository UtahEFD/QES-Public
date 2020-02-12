
#pragma once


#include <iostream>
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
  NetCDFOutput(std::string,bool);
  virtual ~NetCDFOutput()
    {}


    // bool value for determining whether the netcdf file should be output or no, a copy of the input value
    // !!! needs to be set at constructor time till a new way to deal with wanting a file or no is figured out
    // !!! each instance of an inherited class where this value is allowed to be true or false needs to
    //  have a call to check this value and if false, to return, in their constructor, and in their call to save
    //  The idea is to make it act like an empty constructor in that instance.
    bool doFileOutput;

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
