/** \file "WRFInput.cpp" input data header file. 
    \author Pete Willemsen, Matthieu 

    Copyright (C) 2019 Pete Willemsen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include <netcdf>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

#include "WRFInput.h"

WRFInput::WRFInput()
{
}

WRFInput::~WRFInput()
{
}

void WRFInput::readDomainInfo()
{
    
// Read domain dimensions, terrain elevation, wind data, land-use and Z0 from WRF output. 
// Possibility to crop domain borders by adding a varargin (expected format: [Xstart,Xend;YStart,Yend]).
// Layers of data are recorder in the following format: (row,col) = (ny,nx).

//SimData.Clock = ncread(WRFFile,'Times'); % Time in vector format
                                                
    // Read the WRF NetCDF file as read-only (default option in the
    // following call).
//    try
//    {
        // Open the file for read access
    NcFile wrfInputFile( "/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21", NcFile::read );

        // Retrieve the variable named "Times"
        NcVar simDataClock = wrfInputFile.getVar("Times");
        if (simDataClock.isNull()) return;

        std::cout << "Number of attributes: " << wrfInputFile.getAttCount() << std::endl;

        // data.getVar(dataIn);
        // Check the values...
   
        // The netCDF file is automatically closed by the NcFile destructor

        // XSTART, XEND
        // XEND = double(ncreadatt(WRFFile,'/','WEST-EAST_GRID_DIMENSION'))-1;
        
        // YSTART, YEND
        // YEND = double(ncreadatt(WRFFile,'/','SOUTH-NORTH_GRID_DIMENSION'))-1;
        // nx, ny,

        std::multimap<std::string,NcGroupAtt> globalAttributes = wrfInputFile.getAtts();
        for (auto i=globalAttributes.cbegin(); i!=globalAttributes.cend(); i++) {
            std::cout << "Attribute Name: " << i->first << std::endl;
        }
        
        // Pull out DX and DY
        double cellSize[2];
        auto gblAttIter = globalAttributes.find("DX");
        gblAttIter->second.getValues( cellSize );

        gblAttIter = globalAttributes.find("DY");
        gblAttIter->second.getValues( cellSize+1 );
        std::cout << "DX = " << cellSize[0] << ", DY = " << cellSize[1] << std::endl;
           
        // double dx, dy;
        //dx = wrfInputFile.double(ncreadatt(WRFFile,'/','DX'));
        // dy = wrfInputFile.getAtt(,'/','DY'));

        // Relief = ncread(WRFFile,'HGT');
        // LU = ncread(WRFFile,'LU_INDEX');

                return;
//    }catch(NcException& e)
//     {
//       e.what();
//       cout<<"FAILURE*************************************"<<endl;
//       return;
//     }

}

