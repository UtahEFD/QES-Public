//
//  Urb.cpp
//  
//  This class represents CUDA-URB fields
//
//  Created by Jeremy Gibbs on 03/18/19.
//  Modified by Loren Atwood on 01/09/20.
//

#include <iostream>
#include "Urb.hpp"


using namespace netCDF;
using namespace netCDF::exceptions;

Urb :: Urb(Input* input) {

    std::cout<<"[Urb] \t\t Loading CUDA-URB fields "<<std::endl;
    
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
    count2d = {static_cast<unsigned long>(ny),
               static_cast<unsigned long>(nx)};  // the number of values in each grid dimension of the input data, for a 2D structure
    
    // set urb data and information storage sizes
    x.resize(nx);
    y.resize(ny);
    z.resize(nz);
    t.resize(nt);
    terrain.resize(ny*nx);
    icell.resize(nt*nz*ny*nx);
    u.resize(nt*nz*ny*nx);
    v.resize(nt*nz*ny*nx);
    w.resize(nt*nz*ny*nx);
    
    // get input grid data and information
    input->getVariableData("x",x);
    input->getVariableData("y",y);
    input->getVariableData("z",z);
    input->getVariableData("t",t);
    input->getVariableData("icell",start,count,icell);
    input->getVariableData("terrain",start,count2d,terrain);

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
    
    // get input velocity data
    input->getVariableData("u",start,count,u);
    input->getVariableData("v",start,count,v);
    input->getVariableData("w",start,count,w);
    

    // calculate the urb start and end values, needed for getting the domain end and start for all boundary condition application
    // I would guess that it is the first x value in the list of x points, and the last x value from the list of x points
    // same thing for each of the y and z values
    urbXstart = x.at(0);
    urbXend = x.at(nx-1);
    urbYstart = y.at(0);
    urbYend = y.at(ny-1);
    urbZstart = z.at(0);
    urbZend = z.at(nz-1);


#if 1
    // the values for this are required, but the outputting of them is a useful debugging tool
    std::cout << "Urb nx = \"" << nx << "\"" << std::endl;
    std::cout << "Urb ny = \"" << ny << "\"" << std::endl;
    std::cout << "Urb nz = \"" << nz << "\"" << std::endl;
    std::cout << "Urb nt = \"" << nt << "\"" << std::endl;
    std::cout << "Urb dx = \"" << dx << "\"" << std::endl;
    std::cout << "Urb dy = \"" << dy << "\"" << std::endl;
    std::cout << "Urb dz = \"" << dz << "\"" << std::endl;
    std::cout << "Urb urbXstart = \"" << urbXstart << "\"" << std::endl;
    std::cout << "Urb urbXend = \"" << urbXend << "\"" << std::endl;
    std::cout << "Urb urbYstart = \"" << urbYstart << "\"" << std::endl;
    std::cout << "Urb urbYend = \"" << urbYend << "\"" << std::endl;
    std::cout << "Urb urbZstart = \"" << urbZstart << "\"" << std::endl;
    std::cout << "Urb urbZend = \"" << urbZend << "\"" << std::endl;
#endif

}