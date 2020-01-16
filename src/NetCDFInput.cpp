//
//  NetCDFInput.cpp
//  
//  This class handles netcdf input
//
//  Created by Jeremy Gibbs on 03/15/19.
//  Modified by Fabien Margairaz

#include <iostream>
#include "NetCDFInput.h"

using namespace netCDF;
using namespace netCDF::exceptions;

NetCDFInput :: NetCDFInput(std::string input_file) {
    
    std::cout<<"[NetCDFInput] \t Reading "<<input_file<<std::endl;
    infile = new NcFile(input_file, NcFile::read);
}

void NetCDFInput :: getDimension(std::string name, NcDim& external) {
    
    external = infile->getDim(name);
}

void NetCDFInput :: getDimensionSize(std::string name, int& external) {
    
    external = infile->getDim(name).getSize();
}

void NetCDFInput :: getVariable(std::string name, NcVar& external) {
    
    external = infile->getVar(name);
}

// 1D -> int
void NetCDFInput :: getVariableData(std::string name, std::vector<int>& external) {
    
    infile->getVar(name).getVar(&external[0]);
}
// 1D -> float
void NetCDFInput :: getVariableData(std::string name, std::vector<float>& external) {
    
    infile->getVar(name).getVar(&external[0]);
}
// 1D -> double
void NetCDFInput :: getVariableData(std::string name, std::vector<double>& external) {
    
    infile->getVar(name).getVar(&external[0]);
}

// *D -> int
void NetCDFInput :: getVariableData(std::string name, const std::vector<size_t> start,
                              std::vector<size_t> count, std::vector<int>& external) {
    
    infile->getVar(name).getVar(start,count,&external[0]);
}
// *D -> float
void NetCDFInput :: getVariableData(std::string name, const std::vector<size_t> start,
                              std::vector<size_t> count, std::vector<float>& external) {
    
    infile->getVar(name).getVar(start,count,&external[0]);
}
// *D -> double
void NetCDFInput :: getVariableData(std::string name, const std::vector<size_t> start,
                              std::vector<size_t> count, std::vector<double>& external) {
    
    infile->getVar(name).getVar(start,count,&external[0]);
}