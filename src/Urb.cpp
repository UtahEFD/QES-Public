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
    
    int c = grid.nt*grid.nz*grid.ny*grid.nx;    // numValues
    start = {0,0,0,0};      // used for getVariableData() when it is a grid of values
    count = {static_cast<unsigned long>(grid.nt),
             static_cast<unsigned long>(grid.nz),
             static_cast<unsigned long>(grid.ny),
             static_cast<unsigned long>(grid.nx)};  // the number of values in each grid dimension of the input data, I guess the default is a 4D structure
    count2d = {static_cast<unsigned long>(grid.ny),
               static_cast<unsigned long>(grid.nx)};  // the number of values in each grid dimension of the input data, for a 2D structure
    
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

    // set the dx values to 1, and correct them if the grid is has more than one value
    grid.dx = 1;
    grid.dy = 1;
    grid.dz = 1;
    if(grid.nx > 1)
    {
        grid.dx = grid.x[1] - grid.x[0];
    }
    if(grid.ny > 1)
    {
        grid.dy = grid.y[1] - grid.y[0];
    }
    if(grid.nz > 1)
    {
        grid.dz = grid.z[1] - grid.z[0];
    }
    
    std::vector<double> u1(c),u2(c),u3(c);  // temporary vector to hold the velocities before putting them into the grid
    
    input->getVariableData("u",start,count,u1);
    input->getVariableData("v",start,count,u2);
    input->getVariableData("w",start,count,u3);
    
    // now take the input wind values stored in the vector, and put them into the Wind grid data structure
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


    // calculate the urb domain start and end values, needed for wall boundary condition application
    // I would guess that it is the first x value in the list of x points, and the last x value from the list of x points
    // same thing for each of the y and z values
    domainXstart = grid.x.at(0);
    domainXend = grid.x.at(grid.nx-1);
    domainYstart = grid.y.at(0);
    domainYend = grid.y.at(grid.ny-1);
    domainZstart = grid.z.at(0);
    domainZend = grid.z.at(grid.nz-1);


#if 1
    // the values for this are required, but the outputting of them is a useful debugging tool
    std::cout << "Urb nx = \"" << grid.nx << "\"\n";
    std::cout << "Urb ny = \"" << grid.ny << "\"\n";
    std::cout << "Urb nz = \"" << grid.nz << "\"\n";
    std::cout << "Urb nt = \"" << grid.nt << "\"\n";
    std::cout << "Urb dx = \"" << grid.dx << "\"\n";
    std::cout << "Urb dy = \"" << grid.dy << "\"\n";
    std::cout << "Urb dz = \"" << grid.dz << "\"\n";
    std::cout << "Urb domainXstart = \"" << domainXstart << "\"\n";
    std::cout << "Urb domainXend = \"" << domainXend << "\"\n";
    std::cout << "Urb domainYstart = \"" << domainYstart << "\"\n";
    std::cout << "Urb domainYend = \"" << domainYend << "\"\n";
    std::cout << "Urb domainZstart = \"" << domainZstart << "\"\n";
    std::cout << "Urb domainZend = \"" << domainZend << "\"\n";
#endif

}