//
//  Input.cpp
//
//  This class handles netcdf input
//
//  Created by Jeremy Gibbs on 03/15/19.
//


#include "Input.hpp"


using namespace netCDF;
using namespace netCDF::exceptions;

Input ::Input(std::string input_file)
{

  std::cout << "[Input] \t Reading " << input_file << std::endl;
  infile = new NcFile(input_file, NcFile::read);
}

void Input ::getDimension(std::string name, NcDim &external)
{

  external = infile->getDim(name);
}

void Input ::getDimensionSize(std::string name, int &external)
{

  external = infile->getDim(name).getSize();
}

void Input ::getVariable(std::string name, NcVar &external)
{

  external = infile->getVar(name);
}

void Input ::getVariableData(std::string name, std::vector<double> &external)
{

  infile->getVar(name).getVar(&external[0]);
}

void Input ::getVariableData(std::string name, const std::vector<size_t> start, std::vector<size_t> count, std::vector<double> &external)
{

  infile->getVar(name).getVar(start, count, &external[0]);
}