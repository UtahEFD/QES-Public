//
//  Urb.cpp
//  
//  This class represents CUDA-URB fields
//
//  Created by Jeremy Gibbs on 03/18/19.
//  Modified by Loren Atwood on 01/09/20.
//

#include "Urb.hpp"


// LA are these needed here???
using namespace netCDF;
using namespace netCDF::exceptions;


Urb::Urb(Input* input, const bool& debug)
{

    std::cout<<"[Urb] \t\t Loading CUDA-URB fields "<<std::endl;
    
    // get input dimensions
    input->getDimensionSize("x",nx);
    input->getDimensionSize("y",ny);
    input->getDimensionSize("z",nz);
    input->getDimensionSize("t",nt);
    
    start = {0,0,0,0};      // used for getVariableData() when it is a grid of values
    countcc = {static_cast<unsigned long>(nt),
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
    icellflag.resize(nt*nz*ny*nx);
    u.resize(nt*nz*ny*nx);
    v.resize(nt*nz*ny*nx);
    w.resize(nt*nz*ny*nx);
    
    // get input grid data and information
    input->getVariableData("x",x);
    input->getVariableData("y",y);
    input->getVariableData("z",z);
    input->getVariableData("t",t);
    input->getVariableData("icell",start,countcc,icellflag);
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
    input->getVariableData("u",start,countcc,u);
    input->getVariableData("v",start,countcc,v);
    input->getVariableData("w",start,countcc,w);
    

    // calculate the urb start and end values, needed for getting the domain end and start for all boundary condition application
    // I would guess that it is the first x value in the list of x points, and the last x value from the list of x points
    // same thing for each of the y and z values
    urbXstart = x.at(0);
    urbXend = x.at(nx-1);
    urbYstart = y.at(0);
    urbYend = y.at(ny-1);
    urbZstart = z.at(0);
    urbZend = z.at(nz-1);


    if( debug == true )
    {
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
    }

}


Urb::Urb(NetCDFInput* input, const bool& debug)
{

    std::cout<<"[Urb] \t\t Loading CUDA-URB fields "<<std::endl;

    // face-center grid size
    int nx_fc,ny_fc,nz_fc;
    
    // get input dimensions
    input->getDimensionSize("x",nx_fc);
    input->getDimensionSize("y",ny_fc);
    input->getDimensionSize("z",nz_fc);
    input->getDimensionSize("t",nt);

    nx = nx_fc-1;
    ny = ny_fc-1;
    nz = nz_fc-2;
    
    start = {0,0,0,0};      // used for getVariableData() when it is a grid of values
    countfc = {static_cast<unsigned long>(nt),
               static_cast<unsigned long>(nz_fc),
               static_cast<unsigned long>(ny_fc),
               static_cast<unsigned long>(nx_fc)};  // the number of values in each grid dimension of the input data, I guess the default is a 4D structure
    countcc = {static_cast<unsigned long>(nt) ,
               static_cast<unsigned long>(nz+1),
               static_cast<unsigned long>(ny),
               static_cast<unsigned long>(nx)};
    count2d = {static_cast<unsigned long>(ny),
               static_cast<unsigned long>(nx)};  // the number of values in each grid dimension of the input data, for a 2D structure
    
    // set urb data and information storage sizes
    x.resize(nx);
    y.resize(ny);
    z.resize(nz);
    t.resize(nt);
    terrain.resize(ny*nx);
    icellflag.resize(nt*nz*ny*nx);
    u.resize(nt*nz*ny*nx);
    v.resize(nt*nz*ny*nx);
    w.resize(nt*nz*ny*nx);
    

    // get input grid data and information

    // FM -> need to change this when grid is matching between modules
    std::vector<float> temp_x(nx);
    input->getVariableData("x_cc",temp_x);
    for(int i=0;i<nx;i++)
      x.at(i)=temp_x.at(i);
    temp_x.clear();

    std::vector<float> temp_y(ny);
    input->getVariableData("y_cc",temp_y);
    for(int j=0;j<ny;j++)
      y.at(j)=temp_y.at(j);
    temp_y.clear();

    std::vector<float> temp_z(nz+1);
    input->getVariableData("z_cc",temp_z);
    for(int k=0;k<nz;k++)
      z.at(k)=temp_z.at(k+1);
    temp_z.clear();
    
    input->getVariableData("t",t);
    
    
    std::vector<float> temp_icellflag(ny*nx*(nz+1));
    input->getVariableData("icell",start,countcc,temp_icellflag);
    for(int k=0;k<nz;k++) {
        for(int j=0;j<ny;j++) {
            for(int i=0;i<nx;i++){
                
                // FM quick fix for missmatched grid
                // id1 -> Plume grid
                int id1 = i + j*nx + k*nx*ny;
                // id2 -> URB grid
                int id2 = i + j*nx + (k+1)*nx*ny;
                
                icellflag.at(id1)=temp_icellflag.at(id2);
            }
        }
    }
    temp_icellflag.clear();
    

    // FM -> need to change this as URB is float only
    std::vector<float> temp_terrain(ny*nx);
    input->getVariableData("terrain",start,count2d,temp_terrain);
    for(int j=0; j<ny; j++) {
      for(int i=0; i<nx; i++) {
          int id = i + j*nx;
          terrain.at(id)=temp_terrain.at(id);
      }
    }
    temp_terrain.clear();



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
    // use temporary vectors to hold the velocities before putting them into the grid because they need interpolated
    std::vector<float> u1(nt*nz_fc*ny_fc*nx_fc);
    std::vector<float> u2(nt*nz_fc*ny_fc*nx_fc);
    std::vector<float> u3(nt*nz_fc*ny_fc*nx_fc);

    input->getVariableData("u",start,countfc,u1);
    input->getVariableData("v",start,countfc,u2);
    input->getVariableData("w",start,countfc,u3);

    // now take the input wind values stored in the temporary vectors, and put them into the Wind grid data structure
    for (int n=0;n<nt;n++) {
        for(int k=0;k<nz;k++) {
            for(int j=0; j<ny;j++) {
                for(int i=0;i<nx;i++){
                    
                    // FM quick fix for missmatched grid
                    // id1 -> Plume grid
                    int id1 = i + j*nx + k*nx*ny + n*nx*ny*nz;
                    // id2 -> URB grid
                    int id2 = i + j*nx_fc + (k+1)*nx_fc*ny_fc + n*nx_fc*ny_fc*nz_fc;
                    
                    // interpolation of the face-center velocity field
                    u.at(id1) = 0.5*(u1.at(id2)+u1.at(id2+1));
                    v.at(id1) = 0.5*(u2.at(id2)+u2.at(id2+nx_fc));
                    w.at(id1) = 0.5*(u3.at(id2)+u3.at(id2+nx_fc*ny_fc));
                }
            }
        }
    }
    
    // clean up temporary vectors
    u1.clear();
    u2.clear();
    u3.clear();

    

    // calculate the urb start and end values, needed for getting the domain end and start for all boundary condition application
    // I would guess that it is the first x value in the list of x points, and the last x value from the list of x points
    // same thing for each of the y and z values
    urbXstart = x.at(0);
    urbXend = x.at(nx-1);
    urbYstart = y.at(0);
    urbYend = y.at(ny-1);
    urbZstart = z.at(0);
    urbZend = z.at(nz-1);


    if( debug == true )
    {
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
    }

}
