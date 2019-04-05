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
    
    int c = grid.nt*grid.nz*grid.ny*grid.nx;
    start = {0,0,0,0};
    count = {static_cast<unsigned long>(grid.nt),
             static_cast<unsigned long>(grid.nz),
             static_cast<unsigned long>(grid.ny),
             static_cast<unsigned long>(grid.nx)};
    count2d = {static_cast<unsigned long>(grid.ny),
               static_cast<unsigned long>(grid.nx)};
    
    // get input data and information
    grid.x.resize(grid.nx);
    grid.y.resize(grid.ny);
    grid.z.resize(grid.nz);
    grid.t.resize(grid.nt);
    grid.terrain.resize(grid.ny*grid.nx);
    grid.icell.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    wind.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    
    input->getVariableData("x",grid.x);
    input->getVariableData("y",grid.y);
    input->getVariableData("z",grid.z);
    input->getVariableData("t",grid.t);
    input->getVariableData("icell",start,count,grid.icell);
    input->getVariableData("terrain",start,count2d,grid.terrain);

    grid.dx = grid.x[1]-grid.x[0];
    grid.dy = grid.y[1]-grid.y[0];
    grid.dz = grid.z[1]-grid.z[0];
    
    std::vector<double> u1(c),u2(c),u3(c);
    
    input->getVariableData("u",start,count,u1);
    input->getVariableData("v",start,count,u2);
    input->getVariableData("w",start,count,u3);
    
    int id;
    for (int n=0;n<grid.nt;n++) {
        for(int k=0;k<grid.nz;k++) {
            for(int j=0; j<grid.ny;j++) { 
                for(int i=0;i<grid.nx;i++){  
                    
                    id = i + j*grid.nx + k*grid.nx*grid.ny + n*grid.nx*grid.ny*grid.nz;
                    
                    wind.at(id).u = u1.at(id);
                    wind.at(id).v = u2.at(id);
                    wind.at(id).w = u3.at(id);
                }
            }
        }
    }
    
    // clean up temporary vectors
    u1.clear();
    u2.clear();
    u3.clear(); 
}