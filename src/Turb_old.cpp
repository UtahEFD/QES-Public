//
//  Turb.hpp
//  
//  This class represents CUDA-TURB fields
//
//  Created by Jeremy Gibbs on 03/25/19.
//  Modified by Loren Atwood on 01/09/20.
//

#include <iostream>
#include "Turb_old.hpp"


using namespace netCDF;
using namespace netCDF::exceptions;

Turb :: Turb(Input* input) {
    std::cout<<"[Turb] \t\t Loading CUDA-TURB fields "<<std::endl;
    
    // get input dimensions
    input->getDimensionSize("x",nx);
    input->getDimensionSize("y",ny);
    input->getDimensionSize("z",nz);
    input->getDimensionSize("t",nt);
    
    start = {0,0,0,0};      // used for getVariableData() when it is a grid of values
    count = {static_cast<unsigned long>(nt),
             static_cast<unsigned long>(nz),
             static_cast<unsigned long>(ny),
             static_cast<unsigned long>(nx)};  // the number of values in each grid dimension of the input data, I guess the default is a 4D structure
    
    // set turb data and information storage sizes
    x.resize(nx);
    y.resize(ny);
    z.resize(nz);
    t.resize(nt);
    txx.resize(nt*nz*ny*nx);
    txy.resize(nt*nz*ny*nx);
    txz.resize(nt*nz*ny*nx);
    tyy.resize(nt*nz*ny*nx);
    tyz.resize(nt*nz*ny*nx);
    tzz.resize(nt*nz*ny*nx);
    sig_x.resize(nt*nz*ny*nx);
    sig_y.resize(nt*nz*ny*nx);
    sig_z.resize(nt*nz*ny*nx);
    CoEps.resize(nt*nz*ny*nx);
    tke.resize(nt*nz*ny*nx);

    // get input grid data
    input->getVariableData("x",x);
    input->getVariableData("y",y);
    input->getVariableData("z",z);
    input->getVariableData("t",t);
    
    // set the dx values to 1, and correct them if the grid is has more than one value
    dx = 1;
    dy = 1;
    dz = 1;
    if(nx > 1)
    {
        dx = x.at(1) - x.at(0);
    }
    if(ny > 1)
    {
        dy = y.at(1) - y.at(0);
    }
    if(nz > 1)
    {
        dz = z.at(1) - z.at(0);
    }
    
    // get input stress tensor and other turbulence data
    input->getVariableData("tau_11",start,count,txx);
    input->getVariableData("tau_12",start,count,txy);
    input->getVariableData("tau_13",start,count,txz);
    input->getVariableData("tau_22",start,count,tyy);
    input->getVariableData("tau_23",start,count,tyz);
    input->getVariableData("tau_33",start,count,tzz);
    input->getVariableData("sig_11",start,count,sig_x);
    input->getVariableData("sig_22",start,count,sig_y);
    input->getVariableData("sig_33",start,count,sig_z);
    input->getVariableData("CoEps",start,count,CoEps);
    input->getVariableData("tke",start,count,tke);
    
    
    // calculate the turb start and end values, needed for getting the domain end and start for all boundary condition application
    // I would guess that it is the first x value in the list of x points, and the last x value from the list of x points
    // same thing for each of the y and z values
    turbXstart = x.at(0);
    turbXend = x.at(nx-1);
    turbYstart = y.at(0);
    turbYend = y.at(ny-1);
    turbZstart = z.at(0);
    turbZend = z.at(nz-1);


#if 1
    // the values for some of this are required, but the outputting of them is a useful debugging tool
    std::cout << "Turb nx = \"" << nx << "\"" << std::endl;
    std::cout << "Turb ny = \"" << ny << "\"" << std::endl;
    std::cout << "Turb nz = \"" << nz << "\"" << std::endl;
    std::cout << "Turb nt = \"" << nt << "\"" << std::endl;
    std::cout << "Turb dx = \"" << dx << "\"" << std::endl;
    std::cout << "Turb dy = \"" << dy << "\"" << std::endl;
    std::cout << "Turb dz = \"" << dz << "\"" << std::endl;
    std::cout << "Turb turbXstart = \"" << turbXstart << "\"" << std::endl;
    std::cout << "Turb turbXend = \"" << turbXend << "\"" << std::endl;
    std::cout << "Turb turbYstart = \"" << turbYstart << "\"" << std::endl;
    std::cout << "Turb turbYend = \"" << turbYend << "\"" << std::endl;
    std::cout << "Turb turbZstart = \"" << turbZstart << "\"" << std::endl;
    std::cout << "Turb turbZend = \"" << turbZend << "\"" << std::endl;
#endif

}