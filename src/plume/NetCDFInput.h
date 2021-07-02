//
//  Input.hpp
//  
//  This class handles netcdf input
//
//  Created by Jeremy Gibbs on 03/15/19.
//  Modified by Fabien Margairaz

#pragma once


#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <netcdf>


/**
 * This class handles reading input files.
 */

using namespace netCDF;
using namespace netCDF::exceptions;

class NetCDFInput {
    
 protected:
  // netCDF variables
  NcFile* infile;
  
 public:
    
  // initializer
  NetCDFInput()
    {}
  NetCDFInput(std::string);
  virtual ~NetCDFInput()
    {}
  
  // getters
  void getDimension(std::string, NcDim&);
  void getDimensionSize(std::string name, int&);
  void getVariable(std::string, NcVar&);
  
  // get variable for 1D 
  void getVariableData(std::string name,std::vector<int>&);
  void getVariableData(std::string name,std::vector<float>&);
  void getVariableData(std::string name,std::vector<double>&);
  
  // get variable for *D
  void getVariableData(std::string name,std::vector<size_t>,std::vector<size_t>,std::vector<int>&);
  void getVariableData(std::string name,std::vector<size_t>,std::vector<size_t>,std::vector<float>&);
  void getVariableData(std::string name,std::vector<size_t>,std::vector<size_t>,std::vector<double>&);

};