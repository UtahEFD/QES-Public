/** \file "WRFInput.cpp" input data header file. 
1;95;0c    \author Pete Willemsen, Matthieu 

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

        // The netCDF file is automatically closed by the NcFile destructor
    NcFile wrfInputFile( "/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21", NcFile::read );

        // Retrieve the variable named "Times": char Times(Time, DateStrLen) ;
        NcVar simDataClock = wrfInputFile.getVar("Times");
        if (simDataClock.isNull()) return;

        // This is a string???
        // simDataClock.getVal

        std::cout << "Number of attributes: " << wrfInputFile.getAttCount() << std::endl;
        std::multimap<std::string,NcGroupAtt> globalAttributes = wrfInputFile.getAtts();
        // You can see all of the attributes with the following snippet:
        // for (auto i=globalAttributes.cbegin(); i!=globalAttributes.cend(); i++) {
        // std::cout << "Attribute Name: " << i->first << std::endl;
        // }
        
        int xDim[2], yDim[2];
        auto gblAttIter = globalAttributes.find("WEST-EAST_GRID_DIMENSION");
        xDim[0] = 1;

        // Grab the Stored end of the dimension and subtract 1.
        // xDim+1 is a pointer reference to the 2nd array value.
        // Same happens for yDim below.
        gblAttIter->second.getValues( xDim+1 );
        xDim[1] -= 1;

        gblAttIter = globalAttributes.find("SOUTH-NORTH_GRID_DIMENSION");
        yDim[0] = 1;
        gblAttIter->second.getValues( yDim+1 );
        yDim[1] -= 1;

        std::cout << "XDim = " << xDim[0] << ", " << xDim[1] << std::endl;
        std::cout << "YDim = " << yDim[0] << ", " << yDim[1] << std::endl;

        // Compute nx and ny
        nx = xDim[1] - xDim[0] + 1;
        ny = yDim[1] - yDim[0] + 1;

        // Pull out DX and DY
        double cellSize[2];
        gblAttIter = globalAttributes.find("DX");
        gblAttIter->second.getValues( cellSize );

        gblAttIter = globalAttributes.find("DY");
        gblAttIter->second.getValues( cellSize+1 );
        std::cout << "DX = " << cellSize[0] << ", DY = " << cellSize[1] << std::endl;

           
// % If new domain borders are defined
// if nargin == 3 
        
//    NewDomainCorners = varargin{1};
    
//    XSTART_New = NewDomainCorners(1,1); XEND_New = NewDomainCorners(1,2); 
//    YSTART_New = NewDomainCorners(2,1); YEND_New = NewDomainCorners(2,2);
    
//    SimData.nx = XEND_New - XSTART_New +1;
//    SimData.ny = YEND_New - YSTART_New +1; 
    
//    SimData.OLD_XSTART = SimData.XSTART; SimData.OLD_XEND = SimData.XEND;
//    SimData.OLD_YSTART = SimData.YSTART; SimData.OLD_YEND = SimData.YEND;
        
//    SimData.XSTART = XSTART_New; SimData.XEND = XEND_New;
//    SimData.YSTART = YSTART_New; SimData.YEND = YEND_New;
      
// end


// Relief = ncread(WRFFile,'HGT');
// SimData.Relief = Relief(SimData.XSTART:SimData.XEND,SimData.YSTART:SimData.YEND,1)'; 

#if 0
        // Relief
        double hgt;
        gblAttIter = globalAttributes.find("HGT");
        gblAttIter->second.getValues( &hgt );
        std::cout << "HGT = " << hgt << std::endl;


        // % Wind data    
        // SimData = WindFunc(SimData); 

        // LU = ncread(WRFFile,'LU_INDEX');
        double lu_index;
        gblAttIter = globalAttributes.find("LU_INDEX");
        gblAttIter->second.getValues( &lu_index );
        std::cout << "LU_INDEX = " << lu_index << std::endl;
#endif

//    }catch(NcException& e)
//     {
//       e.what();
//       cout<<"FAILURE*************************************"<<endl;
//       return;
//     }

}

void WRFInput::readWindData()
{
    // This function computes velocity magnitude, direction and vertical
    // coordinates from WRF velocity components U,V and geopotential height.
    // Values are interpolated at each corresponding cell center.

    // Extraction of the Wind data vertical position
#if 0

PHB = ncread(SimData.WRFFile,'PHB');
PHB = double(PHB(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT));

PH = ncread(SimData.WRFFile,'PH');
PH = double(PH(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT));

Height = (PHB + PH)./9.81; % Converting to meters
SimData.NbAlt = size(Height,3) - 1;




% Wind components
Ustagg = ncread(SimData.WRFFile,'U');
Ustagg = Ustagg(SimData.XSTART:SimData.XEND +1, SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT);

Vstagg = ncread(SimData.WRFFile,'V');
Vstagg = Vstagg(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND +1, :, SimData.TIMEVECT);


%% Centering values %%

SimData.NbAlt = size(Height,3) - 1;

U = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
for x = 1:SimData.nx
    U(x,:,:,:) = .5*(Ustagg(x,:,:,:) + Ustagg(x+1,:,:,:));
end

V = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
for y = 1:SimData.ny   
    V(:,y,:,:) = .5*(Vstagg(:,y,:,:) + Vstagg(:,y+1,:,:));
end

SimData.CoordZ = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
for k = 1:SimData.NbAlt
    SimData.CoordZ(:,:,k,:) = .5*(Height(:,:,k,:) + Height(:,:,k+1,:));
end

%% Velocity and direction %%

SimData.WS = sqrt(U.^2 + V.^2);

SimData.WD = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));

for x = 1:SimData.nx
    for y = 1:SimData.ny
        for Alt = 1:SimData.NbAlt
            for Time = 1:numel(SimData.TIMEVECT)
                if U(x,y,Alt,Time) > 0
                    SimData.WD(x,y,Alt,Time) = 270-(180/pi)*atan(V(x,y,Alt,Time)/U(x,y,Alt,Time));
                else
                    SimData.WD(x,y,Alt,Time) = 90-(180/pi)*atan(V(x,y,Alt,Time)/U(x,y,Alt,Time));
                end
            end
        end
    end
end

%% Permutation to set dimensions as (row,col,etc) = (ny,nx,nz,nt) %%

SimData.WD = permute(SimData.WD,[2 1 3 4]);
SimData.WS = permute(SimData.WS,[2 1 3 4]);
SimData.CoordZ = permute(SimData.CoordZ,[2 1 3 4]);
#endif

}
