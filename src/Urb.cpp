//
//  Urb.cpp
//  
//  This class represents CUDA-URB fields
//
//  Created by Jeremy Gibbs on 03/18/19.
//

#include <iostream>
#include "Urb.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;

Urb :: Urb(Input* input) {
    std::cout<<"[Urb] \t\t Loading CUDA-URB fields "<<std::endl;
    
    // get input dimensions
    input->getDimensionSize("x",grid.nx);
    input->getDimensionSize("y",grid.ny);
    input->getDimensionSize("z",grid.nz);
    input->getDimensionSize("t",grid.nt);
    
    start = {0,0,0,0};
    count = {static_cast<unsigned long>(grid.nt),
             static_cast<unsigned long>(grid.nz),
             static_cast<unsigned long>(grid.ny),
             static_cast<unsigned long>(grid.nx)};
    
    // get input data and information
    grid.x.resize(grid.nx);
    grid.y.resize(grid.ny);
    grid.z.resize(grid.nz);
    grid.t.resize(grid.nt);
    grid.icell.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    wind.u.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    wind.v.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    wind.w.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    
    input->getVariableData("x",grid.x);
    input->getVariableData("y",grid.y);
    input->getVariableData("z",grid.z);
    input->getVariableData("t",grid.t);
    input->getVariableData("u",start,count,wind.u);
    input->getVariableData("v",start,count,wind.v);
    input->getVariableData("w",start,count,wind.w);
    input->getVariableData("icell",start,count,grid.icell);
    
}