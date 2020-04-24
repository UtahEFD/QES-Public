//
//  URBData.cpp
//
//  This class represents CUDA-URB fields
//
//  Created by Jeremy Gibbs on 03/18/19.
//  Modified by Fabien Margairaz

#include <iostream>
#include "URBGeneralData.h"

using namespace netCDF;
using namespace netCDF::exceptions;

URBGeneralData :: URBGeneralData(Args* arguments) {
    
    std::cout<<"[WINDS Data] \t Loading QES-winds fields "<<std::endl;

    // fullname passed to URBGeneralData
    input = new NetCDFInput(arguments->inputUrbFile);

    // nx,ny - face centered value (consistant with URB)
    input->getDimensionSize("x",nx);
    input->getDimensionSize("y",ny);
    // nz - face centered value + bottom ghost (consistant with URB)
    input->getDimensionSize("z",nz);
    // nt - number of time instance in data
    input->getDimensionSize("t",nt);

    numcell_cent    = (nx-1)*(ny-1)*(nz-1);        /**< Total number of cell-centered values in domain */
    numcell_face    = nx*ny*nz;                    /**< Total number of face-centered values in domain */
  
    // get grid information
    x.resize(nx-1);
    y.resize(ny-1);
    z.resize( nz-1 );
    z_face.resize( nz-1 );
    dz_array.resize( nz-1, 0.0 );
    
    input->getVariableData("x_cc",x);
    dx = x[1] - x[0]; /**< Grid resolution in x-direction */
        
    input->getVariableData("y_cc",y);
    dy = y[1] - y[0]; /**< Grid resolution in x-direction */
    dxy = MIN_S(dx, dy);
    
    input->getVariableData("z_cc",z);
    // check if dz_array is in the NetCDF file 
    NcVar NcVar_dz;
    input->getVariable("dz", NcVar_dz);
    if(!NcVar_dz.isNull()) { 
        input->getVariableData("dz",dz_array);
        dz = *std::min_element(dz_array.begin() , dz_array.end());
    } else {
        dz = z[1] - z[0];
        for (size_t k=0; k<z.size(); k++) {
            dz_array[k] = dz;
        }
    }
    
    for (size_t k=1; k<z.size(); k++) {
        z_face[k] = z_face[k-1] + dz_array[k];  /**< Location of face centers in z-dir */
    }

    //get time variables
    t.resize(nt);
    input->getVariableData("t",t);
    
    // netCDF variables
    std::vector<size_t> start;
    std::vector<size_t> count_2d;

    start = {0,0};
    count_2d = {static_cast<unsigned long>(ny-1),
                static_cast<unsigned long>(nx-1)};
  
    // terrain (cell-center)
    terrain.resize((ny-1)*(nx-1),0.0);
    NcVar NcVar_terrain;
    input->getVariable("terrain", NcVar_terrain);
    if(!NcVar_terrain.isNull()) { // => terrain data in URB file
        input->getVariableData("terrain",start,count_2d,terrain);
    } else { // => no external terrain data provided
        std::cout << "[WINDS Data] \t no terrain data found -> assumed flat" << std::endl;
    }

    // icellflag (see .h for velues)
    icellflag.resize(numcell_cent,-1);
    /// coefficients for SOR solver
    e.resize(numcell_cent,1.0);
    f.resize(numcell_cent,1.0);
    g.resize(numcell_cent,1.0);
    h.resize(numcell_cent,1.0);
    m.resize(numcell_cent,1.0);
    n.resize(numcell_cent,1.0);

    // velocity fields
    u.resize(numcell_face,0.0);
    v.resize(numcell_face,0.0);
    w.resize(numcell_face,0.0);
  
    return;
}

void URBGeneralData::loadNetCDFData(int stepin)
{
  
    std::cout << "[WINDS Data] \t loading data at step " << stepin <<std::endl;
  
    // netCDF variables
    std::vector<size_t> start;
    std::vector<size_t> count_cc;
    std::vector<size_t> count_fc;

    start = {static_cast<unsigned long>(stepin),0,0,0};
    count_cc = {1,
                static_cast<unsigned long>(nz-1),
                static_cast<unsigned long>(ny-1),
                static_cast<unsigned long>(nx-1)};
    count_fc = {1,
                static_cast<unsigned long>(nz),
                static_cast<unsigned long>(ny),
                static_cast<unsigned long>(nx)};
  
    // cell-center variables
    // icellflag (see .h for velues)
    input->getVariableData("icell",start,count_cc,icellflag);
    /// coefficients for SOR solver
    NcVar NcVar_SORcoeff;
    input->getVariable("e", NcVar_SORcoeff);
    
    if(!NcVar_SORcoeff.isNull()) { 
        input->getVariableData("e",start,count_cc,e);
        input->getVariableData("f",start,count_cc,f);
        input->getVariableData("g",start,count_cc,g);
        input->getVariableData("h",start,count_cc,h);
        input->getVariableData("m",start,count_cc,m);
        input->getVariableData("n",start,count_cc,n); 
    } else { 
        std::cout << "[URBData] \t no SORcoeff data found -> assumed e,f,g,h,m,n=1" << std::endl;
    }
  
    // face-center variables
    input->getVariableData("u",start,count_fc,u);
    input->getVariableData("v",start,count_fc,v);
    input->getVariableData("w",start,count_fc,w);

    return;
}
