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
  std::cout<<"[URBData] \t Loading CUDA-URB fields "<<std::endl;

  // fullname passed to URBGeneralData
  input = new NetCDFInput(arguments->inputFileURB);

  // nx,ny - face centered value (consistant with URB)
  input->getDimensionSize("x",nx);
  input->getDimensionSize("y",ny);
  // nz - face centered value + bottom ghost (consistant with URB)
  input->getDimensionSize("z",nz);
  // nt - number of time instance in data
  input->getDimensionSize("t",nt);

  // netCDF variables
  std::vector<size_t> start;
  std::vector<size_t> count_2d;

  start = {0,0};
  count_2d = {static_cast<unsigned long>(ny-1),
	      static_cast<unsigned long>(nx-1)};
  
  // get grid information
  size_t count;
  x_cc.resize(nx-1);
  y_cc.resize(ny-1);
  z_cc.resize(nz-1);
  input->getVariableData("x_cc",x_cc);
  input->getVariableData("y_cc",y_cc);
  input->getVariableData("z_cc",z_cc);

  // derived variables
  dz = z_cc[1] - z_cc[0];
  dy = y_cc[1] - y_cc[0];
  dx = x_cc[1] - x_cc[0];

  //get time variables
  t.resize(nt);
  input->getVariableData("t",t);
  
  // terrain (cell-center)
  terrain.resize((ny-1)*(nx-1),0.0);
  NcVar NcVar_terrain;
  input->getVariable("terrain", NcVar_terrain);
  if(!NcVar_terrain.isNull()) { // => terrain data in URB file
    input->getVariableData("terrain",start,count_2d,terrain);
  } else { // => no external terrain data provided
    std::cout << "[URBData] \t no terrain data found -> assumed flat" << std::endl;
  }

  // cell-center count
  count=(nz-1)*(ny-1)*(nx-1);
  // icellflag (see .h for velues)
  icellflag.resize(count,-1);
  /// coefficients for SOR solver
  e.resize(count,1.0);
  f.resize(count,1.0);
  g.resize(count,1.0);
  h.resize(count,1.0);
  m.resize(count,1.0);
  n.resize(count,1.0);

  // face-center count
  count=(nz)*(ny)*(nx);
  // velocity fields
  u.resize(count,0.0);
  v.resize(count,0.0);
  w.resize(count,0.0);
  
  return;
}

void URBGeneralData::loadNetCDFData(int stepin)
{
  
  std::cout << "[URBData] \t loading data at step " << stepin <<std::endl;
  
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
