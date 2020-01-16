//
//  Turb.hpp
//  
//  This class represents CUDA-TURB fields
//
//  Created by Jeremy Gibbs on 03/25/19.
//  Modified by Loren Atwood on 01/09/20.
//

#include <iostream>
#include "Turb.hpp"


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



Turb :: Turb(NetCDFInput* input) {
    std::cout<<"[Turb] \t\t Loading CUDA-TURB fields "<<std::endl;
    
    // get input dimensions
    input->getDimensionSize("x",nx);
    input->getDimensionSize("y",ny);
    input->getDimensionSize("z",nz);
    input->getDimensionSize("t",nt);

    /* FM - note
       here used:
       nx, ny - cell-center (URB and TURB use nx, ny as number of faces)
       nz - cell center without the bottom ghost cell
    */

    // FM -> nz=nz-1 to adapt PLUME convention of grid size.
    nz--;

    
    start = {0,0,0,0};      // used for getVariableData() when it is a grid of values
    count = {static_cast<unsigned long>(nt),
             static_cast<unsigned long>(nz+1),
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

    // FM -> need to change this when grid is matching between modules
    std::vector<float> temp_x(nx);
    input->getVariableData("x",temp_x);
    for(int i=0;i<nx;i++)
      x.at(i)=temp_x.at(i);
    temp_x.clear();

    std::vector<float> temp_y(ny);
    input->getVariableData("y",temp_y);
    for(int j=0;j<ny;j++)
      y.at(j)=temp_y.at(j);
    temp_y.clear();

    std::vector<float> temp_z(nz+1);
    input->getVariableData("z",temp_z);
    for(int k=0;k<nz;k++)
      z.at(k)=temp_z.at(k+1);
    temp_z.clear();


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
    // use temporary vectors to hold the values before putting them into the grid because they need interpolated
    std::vector<double> temp_txx(nt*(nz+1)*ny*nx);
    std::vector<double> temp_txy(nt*(nz+1)*ny*nx);
    std::vector<double> temp_txz(nt*(nz+1)*ny*nx);
    std::vector<double> temp_tyy(nt*(nz+1)*ny*nx);
    std::vector<double> temp_tyz(nt*(nz+1)*ny*nx);
    std::vector<double> temp_tzz(nt*(nz+1)*ny*nx);
    std::vector<double> temp_CoEps(nt*(nz+1)*ny*nx);
    std::vector<double> temp_tke(nt*(nz+1)*ny*nx);
    
    input->getVariableData("tau11",start,count,temp_txx);
    input->getVariableData("tau12",start,count,temp_txy);
    input->getVariableData("tau13",start,count,temp_txz);
    input->getVariableData("tau22",start,count,temp_tyy);
    input->getVariableData("tau23",start,count,temp_tyz);
    input->getVariableData("tau33",start,count,temp_tzz);
    input->getVariableData("CoEps",start,count,temp_CoEps);
    input->getVariableData("tke",start,count,temp_tke);
    

    // now take the input tensor, inverted tensor, and variance values stored in the vectors, and put them into the data structures found inside Turb
    for (int n=0;n<nt;n++) {
      for(int k=0;k<nz;k++) {
        for(int j=0; j<ny;j++) {
          for(int i=0;i<nx;i++){

            // FM quick fix for missmatched grid
            // id1 -> Plume grid
            int id1 = i + j*nx + k*nx*ny + n*nx*ny*nz;
            // id2 -> TURB grid
            int id2 = i + j*nx + (k+1)*nx*ny + n*nx*ny*(nz+1);

            txx.at(id1) = temp_txx.at(id2);
            txy.at(id1) = temp_txy.at(id2);
            txz.at(id1) = temp_txz.at(id2);
            tyy.at(id1) = temp_tyy.at(id2);
            tyz.at(id1) = temp_tyz.at(id2);
            tzz.at(id1) = temp_tzz.at(id2);

            // FM -> why store sigmas?  LA -> useful for initial values
            sig_x.at(id1) = pow(txx.at(id1),0.5);
            sig_y.at(id1) = pow(tyy.at(id1),0.5);
            sig_z.at(id1) = pow(tzz.at(id1),0.5);

            CoEps.at(id1) = temp_CoEps.at(id2);
            tke.at(id1) = temp_tke.at(id2);
            
          }
        }
      }
    }

    // clean up temporary vectors
    temp_txx.clear();
    temp_txy.clear();
    temp_txz.clear();
    temp_tyy.clear();
    temp_tyz.clear();
    temp_tzz.clear();
    temp_CoEps.clear();
    temp_tke.clear();
    
    

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