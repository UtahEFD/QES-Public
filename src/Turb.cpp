//
//  Turb.hpp
//  
//  This class represents CUDA-TURB fields
//
//  Created by Jeremy Gibbs on 03/25/19.
//

#include <iostream>
#include "Turb.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;

Turb :: Turb(Input* input) {
    std::cout<<"[Turb] \t\t Loading CUDA-TURB fields "<<std::endl;
    
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
    tau.tau_11.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    tau.tau_22.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    tau.tau_33.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    tau.tau_12.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    tau.tau_13.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    tau.tau_23.resize(grid.nt*grid.nz*grid.ny*grid.nx);
        
    input->getVariableData("x",grid.x);
    input->getVariableData("y",grid.y);
    input->getVariableData("z",grid.z);
    input->getVariableData("t",grid.t);
    input->getVariableData("tau_11",start,count,tau.tau_11);
    input->getVariableData("tau_22",start,count,tau.tau_22);
    input->getVariableData("tau_33",start,count,tau.tau_33);
    input->getVariableData("tau_12",start,count,tau.tau_12);
    input->getVariableData("tau_13",start,count,tau.tau_13);
    input->getVariableData("tau_23",start,count,tau.tau_23);
    
}