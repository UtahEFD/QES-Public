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

#include <fstream>
#include <cmath>

#include "WRFInput.h"

WRFInput::WRFInput(const std::string& filename)
    : wrfInputFile( filename, NcFile::read ),
      m_minWRFAlt( 22 ), m_maxWRFAlt( 330 ), m_maxTerrainSize( 10001 ), m_maxNbStat( 156 ),
      m_TerrainFlag(1), m_BdFlag(0), m_VegFlag(0), m_Z0Flag(2)
{
    std::cout <<"there are "<<wrfInputFile.getVarCount()<<" variables"<<std::endl;;
    std::cout <<"there are "<<wrfInputFile.getAttCount()<<" attributes"<<std::endl;;
    
    std::cout <<"there are "<<wrfInputFile.getGroupCount()<<" groups"<<std::endl;;
    std::cout <<"there are "<<wrfInputFile.getTypeCount()<<" types"<<std::endl;;
    
    // Read in all the dm
    std::cout <<"there are "<< wrfInputFile.getDimCount()<<" dimensions"<<std::endl;;
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
    // wrfInputFile( "/scratch/Downloads/RXCwrfout_d07_2012-11-11_15-21", NcFile::read );

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

        std::cout << "(nx, ny) = (" << nx << ", " << ny << ")" << std::endl;

        // Pull out DX and DY
        double cellSize[2];
        gblAttIter = globalAttributes.find("DX");
        gblAttIter->second.getValues( cellSize );

        gblAttIter = globalAttributes.find("DY");
        gblAttIter->second.getValues( cellSize+1 );

        m_dx = cellSize[0];
        m_dy = cellSize[1];
        std::cout << "DX = " << m_dx << ", DY = " << m_dy << std::endl;

           
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

        //
        // Relief
        //
        NcVar reliefVar = wrfInputFile.getVar("HGT");
        std::cout << "relief dim count: " << reliefVar.getDimCount() << std::endl;
        std::vector<NcDim> dims = reliefVar.getDims();
        long totalDim = 1;
        for (int i=0; i<dims.size(); i++) {
            std::cout << "Dim: " << dims[i].getName() << ", ";
            if (dims[i].isUnlimited())
                std::cout << "Unlimited (" << dims[i].getSize() << ")" << std::endl;
            else
                std::cout << dims[i].getSize() << std::endl;
            totalDim *= dims[i].getSize();
        }
        std::cout << "relief att count: " << reliefVar.getAttCount() << std::endl;
        std::map<std::string, NcVarAtt> reliefVar_attrMap = reliefVar.getAtts();
        for (std::map<std::string, NcVarAtt>::const_iterator ci=reliefVar_attrMap.begin();
             ci!=reliefVar_attrMap.end(); ++ci) {
            std::cout << "Relief Attr: " << ci->first << std::endl;
    }

    // this is a 114 x 114 x 41 x 360 dim array...
    // slice by slice 114 x 114 per slice; 41 slices; 360 times

            // How do I extract out the start to end in and and y?
            // PHB = double(PHB(SimData.XSTART:SimData.XEND,
            // SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT));

    // we want all slice data (xstart:xend X ystart:yend) at all
    // slices but only for the first 2 time series (TIMEVECT)
    double* reliefData = new double[ totalDim ];
    reliefVar.getVar( reliefData );
    
    // whole thing is in... now
    long subsetDim = nx * ny;
    relief = new double[ subsetDim ];
    for (auto l=0; l<subsetDim; l++)
        relief[l] = reliefData[l];
        
    std::cout << "first 10 for relief" << std::endl;
    for (auto l=0; l<10; l++)
        std::cout << relief[l] << std::endl;

    std::cout << "last 10 for relief" << std::endl;
    for (auto l=subsetDim-10-1; l<subsetDim; l++)
        std::cout << relief[l] << std::endl;

    delete [] reliefData;


    // % Wind data    
    // SimData = WindFunc(SimData); 
    readWindData();

    //
    // LU = ncread(WRFFile,'LU_INDEX');
    //
    NcVar LUIndexVar = wrfInputFile.getVar("LU_INDEX");
    std::cout << "LUIndex dim count: " << LUIndexVar.getDimCount() << std::endl;
    dims.clear();
    dims = LUIndexVar.getDims();
    totalDim = 1;
    for (int i=0; i<dims.size(); i++) {
        std::cout << "Dim: " << dims[i].getName() << ", ";
        if (dims[i].isUnlimited())
            std::cout << "Unlimited (" << dims[i].getSize() << ")" << std::endl;
        else
            std::cout << dims[i].getSize() << std::endl;
        totalDim *= dims[i].getSize();
    }
    std::cout << "LUIndex att count: " << LUIndexVar.getAttCount() << std::endl;
    std::map<std::string, NcVarAtt> LUIndexVar_attrMap = LUIndexVar.getAtts();
    for (std::map<std::string, NcVarAtt>::const_iterator ci=LUIndexVar_attrMap.begin();
         ci!=LUIndexVar_attrMap.end(); ++ci) {
        std::cout << "LUIndex Attr: " << ci->first << std::endl;
    }

    // this is a 114 x 114 x 41 x 360 dim array...
    // slice by slice 114 x 114 per slice; 41 slices; 360 times

    // we want all slice data (xstart:xend X ystart:yend) at all
    // slices but only for the first 2 time series (TIMEVECT)
    double* LUIndexData = new double[ totalDim ];
    LUIndexVar.getVar( LUIndexData );
    
    // whole thing is in... now
    subsetDim = nx * ny;
    double* LUIndexSave = new double[ subsetDim ];
    for (auto l=0; l<subsetDim; l++)
        LUIndexSave[l] = LUIndexData[l];
        
    std::cout << "first 10 for LUIndex" << std::endl;
    for (auto l=0; l<10; l++)
        std::cout << LUIndexSave[l] << std::endl;

    std::cout << "last 10 for LUIndex" << std::endl;
    for (auto l=subsetDim-10-1; l<subsetDim; l++)
        std::cout << LUIndexSave[l] << std::endl;

    delete [] LUIndexData;

//    }catch(NcException& e)
//     {
//       e.what();
//       cout<<"FAILURE*************************************"<<endl;
//       return;
//     }

    if (nx * ny > m_maxTerrainSize) {
        smoothDomain();
    }
    
}


void WRFInput::roughnessLength()
{
#if 0
% Computes a roughness length array covering each point of the grid
% In case 1 INPUT file must be WRF RESTART file
% In case 2 INPUT file must be WRF OUTPUT
% In case 3 INPUT variable is a constant value

switch Z0Flag
    
    case 1   %%% WRF RESTART file
        
        Z0 = ncread(SimData.Z0DataSource ,'Z0');
        Z0 = Z0(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND)';
        
    case 2   %%% WRF OUTPUT file
        
        Z0 = zeros(size(SimData.LU,1),size(SimData.LU,2));
        
        for i = 1:size(SimData.LU,1)
            for j = 1:size(SimData.LU,2) 
                
                switch SimData.LU(i,j) 
                    
%%%%%%%%%%%%%%%%%%%%%%%%%     USGS LAND COVER     %%%%%%%%%%%%%%%%%%%%%%%%% 

%                     case 1   %%% Urban and built-up land
%                         Z0(i,j) = 0.8;
%                     case 2   %%% Dryland cropland and pasture
%                         Z0(i,j) = 0.15;
%                     case 3   %%% Irrigated cropland and pasture
%                         Z0(i,j) = 0.1;
%                     case 4   %%% Mixed dryland/irrigated cropland and pasture
%                         Z0(i,j) = 0.15;
%                     case 5   %%% Cropland/grassland mosaic
%                         Z0(i,j) = 0.14;
%                     case 6   %%% Cropland/woodland mosaic
%                         Z0(i,j) = 0.2;
%                     case 7   %%% Grassland
%                         Z0(i,j) = 0.12;
%                     case 8   %%% Shrubland
%                         Z0(i,j) = 0.05;
%                     case 9   %%% Mixed shrubland/grassland
%                         Z0(i,j) = 0.06;
%                     case 10   %%% Savanna
%                         Z0(i,j) = 0.15;
%                     case 11   %%% Deciduous broadleaf forest
%                         Z0(i,j) = 0.5;
%                     case 12   %%% Deciduous needleleaf forest
%                         Z0(i,j) = 0.5;
%                     case 13   %%% Evergreeen broadleaf forest
%                         Z0(i,j) = 0.5;
%                     case 14   %%% Evergreen needleleaf forest
%                         Z0(i,j) = 0.5;
%                     case 15   %%% Mixed forest
%                         Z0(i,j) = 0.5;
%                     case 16   %%% Water bodies
%                         Z0(i,j) = 0.0001;
%                     case 17   %%% Herbaceous wetland
%                         Z0(i,j) = 0.2;
%                     case 18   %%% Wooded wetland
%                         Z0(i,j) = 0.4;
%                     case 19   %%% Barren or sparsely vegetated
%                         Z0(i,j) = 0.01;
%                     case 20   %%% Herbaceous tundra
%                         Z0(i,j) = 0.1;
%                     case 21   %%% Wooded tundra
%                         Z0(i,j) = 0.3;
%                     case 22   %%% Mixed tundra
%                         Z0(i,j) = 0.15;
%                     case 23   %%% Bare ground tundra
%                         Z0(i,j) = 0.1;
%                     case 24   %%% Snow or ice
%                         Z0(i,j) = 0.001;

%%%%%%%%%%%%%%%%%%%%%%%%%      MODIS-WINTER      %%%%%%%%%%%%%%%%%%%%%%%%%%

                    case 1   %%% Evergreen needleleaf forest
                        Z0(i,j) = 0.5;
                    case 2   %%% Evergreeen broadleaf forest
                        Z0(i,j) = 0.5;
                    case 3   %%% Deciduous needleleaf forest
                        Z0(i,j) = 0.5;
                    case 4   %%% Deciduous broadleaf forest
                        Z0(i,j) = 0.5;
                    case 5   %%% Mixed forests
                        Z0(i,j) = 0.5;
                    case 6   %%% Closed Shrublands
                        Z0(i,j) = 0.1;
                    case 7   %%% Open Shrublands
                        Z0(i,j) = 0.1;
                    case 8   %%% Woody Savannas
                        Z0(i,j) = 0.15;
                    case 9   %%% Savannas
                        Z0(i,j) = 0.15;
                    case 10   %%% Grasslands
                        Z0(i,j) = 0.075;
                    case 11   %%% Permanent wetlands
                        Z0(i,j) = 0.3;
                    case 12   %%% Croplands
                        Z0(i,j) = 0.075;
                    case 13   %%% Urban and built-up land
                        Z0(i,j) = 0.5;
                    case 14   %%% Cropland/natural vegetation mosaic
                        Z0(i,j) = 0.065;
                    case 15   %%% Snow or ice
                        Z0(i,j) = 0.01;
                    case 16   %%% Barren or sparsely vegetated
                        Z0(i,j) = 0.065;
                    case 17   %%% Water
                        Z0(i,j) = 0.0001;
                    case 18   %%% Wooded tundra
                        Z0(i,j) = 0.15;
                    case 19   %%% Mixed tundra
                        Z0(i,j) = 0.1;
                    case 20   %%% Barren tundra
                        Z0(i,j) = 0.06;
                    case 21   %%% Lakes
                        Z0(i,j) = 0.0001;
                end
            end
        end
        
    case 3   %%% User-defined constant
        
        Z0 = repmat(SimData.Z0DataSource, SimData.XEND-SimData.XSTART+1, SimData.YEND-SimData.YSTART+1)';
end
    #endif
}



void WRFInput::readWindData()
{
    // This function computes velocity magnitude, direction and vertical
    // coordinates from WRF velocity components U,V and geopotential height.
    // Values are interpolated at each corresponding cell center.

    // Extraction of the Wind data vertical position
    NcVar phbVar = wrfInputFile.getVar("PHB");
    std::cout << "PHB dim count: " << phbVar.getDimCount() << std::endl;
    std::vector<NcDim> dims = phbVar.getDims();
    long totalDim = 1;
    for (auto i=0; i<dims.size(); i++) {
        std::cout << "Dim: " << dims[i].getName() << ", ";
        if (dims[i].isUnlimited())
            std::cout << "Unlimited (" << dims[i].getSize() << ")" << std::endl;
        else
            std::cout << dims[i].getSize() << std::endl;
        totalDim *= dims[i].getSize();
    }
    std::cout << "PHB att count: " << phbVar.getAttCount() << std::endl;
    std::map<std::string, NcVarAtt> phbVar_attrMap = phbVar.getAtts();
    for (std::map<std::string, NcVarAtt>::const_iterator ci=phbVar_attrMap.begin();
         ci!=phbVar_attrMap.end(); ++ci) {
        std::cout << "PHB Attr: " << ci->first << std::endl;
    }


    // ////////////////////
    // Test reading in of this data into an array we can easily deal
    // with
#if 0 
    // WRF multi-dim format
    int timeDim = dims[0].getSize();
    int zDim = dims[1].getSize();
    int yDim = dims[2].getSize();
    int xDim = dims[3].getSize();
    
    std::cout << "PHB Dims: t=" << timeDim << "< z=" << zDim << ", y=" << yDim << ", x=" << xDim << std::endl;
    double* allPHBData = new double[ timeDim * zDim * yDim * xDim ];
    phbVar.getVar( allPHBData );

    dumpWRFDataArray("PHB", allPHBData, timeDim, zDim, yDim, xDim);
#endif
    // ////////////////////

    // this is a 114 x 114 x 41 x 360 dim array...
    // slice by slice 114 x 114 per slice; 41 slices; 360 times

    // Need to extract out the start to end in and and y?
    // PHB = double(PHB(SimData.XSTART:SimData.XEND,
    // SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT));
    // can use getVar to do this.

    int nz = 41;
    long subsetDim = 2 * nz * ny * nx;

    // PHB(Time, bottom_top_stag, south_north, west_east) ;
    double* phbData = new double[ subsetDim ];
    std::vector< size_t > starts = { 0, 0, 0, 0 };
    std::vector< size_t > counts = { 2, 41, 114, 114 };   // depends on order of dims
    phbVar.getVar( starts, counts, phbData );
                     
    dumpWRFDataArray("PHB Subset", phbData, 2, 41, 114, 114);

    // 
    // Extraction of the Wind data vertical position
    // 
    NcVar phVar = wrfInputFile.getVar("PH");

    double* phData = new double[ subsetDim ];
    phVar.getVar( starts, counts, phData );
    

    // 
    /// Height
    // 
    double* heightData = new double[ subsetDim ];
    for (auto l=0; l<subsetDim; l++) {
        heightData[l] = (phbData[l] + phData[l]) / 9.81;
    }
    
    // Extraction of the Ustagg
    // Ustagg = ncread(SimData.WRFFile,'U');
    // Ustagg = Ustagg(SimData.XSTART:SimData.XEND +1, SimData.YSTART:SimData.YEND, :, SimData.TIMEVECT);
    NcVar uStaggered = wrfInputFile.getVar("U");

    std::vector<NcDim> ustagg_dims = uStaggered.getDims();
    for (int i=0; i<ustagg_dims.size(); i++) {
        std::cout << "Dim: " << ustagg_dims[i].getName() << ", ";
        if (ustagg_dims[i].isUnlimited())
            std::cout << "Unlimited (" << ustagg_dims[i].getSize() << ")" << std::endl;
        else
            std::cout << ustagg_dims[i].getSize() << std::endl;
    }
    
    // time, Z, Y, X is order
    starts.clear(); counts.clear();
    starts = { 0, 0, 0, 0 };
    counts = { 2, 40, 114, 115 };
    subsetDim = 1;
    for (auto i=0; i<counts.size(); i++)  {
        subsetDim *= (counts[i] - starts[i]);
    }
    
    double* uStaggeredData = new double[ subsetDim ];
    uStaggered.getVar( starts, counts, uStaggeredData );
    dumpWRFDataArray("Ustagg", uStaggeredData, 2, 40, 114, 115);

    // 
    // Vstagg = ncread(SimData.WRFFile,'V');
    // Vstagg = Vstagg(SimData.XSTART:SimData.XEND, SimData.YSTART:SimData.YEND +1, :, SimData.TIMEVECT);
    //
    NcVar vStaggered = wrfInputFile.getVar("V");
    
    starts.clear();  counts.clear();
    starts = { 0, 0, 0, 0 };
    counts = { 2, 40, 115, 114 };
    subsetDim = 1;
    for (auto i=0; i<counts.size(); i++) 
        subsetDim *= (counts[i] - starts[i]);
    
    double* vStaggeredData = new double[ subsetDim ];
    vStaggered.getVar( starts, counts, vStaggeredData );

    
    // 
    // %% Centering values %%
    // SimData.NbAlt = size(Height,3) - 1;
    //
    int nbAlt = 40;  // zDim - 1 but hack for now -- need to be computed
    
    // ///////////////////////////////////
    // U = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
    // for x = 1:SimData.nx
    //   U(x,:,:,:) = .5*(Ustagg(x,:,:,:) + Ustagg(x+1,:,:,:));
    // end
    // ///////////////////////////////////    

    // Just make sure we've got the write dims here
    nx = 114;
    ny = 114;
    
    std::vector<double> U( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {

                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    auto idxP1x = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + (x+1);

                    U[idx] = 0.5 * ( uStaggeredData[idx] + uStaggeredData[idxP1x] );
                }
            }
        }
    }
    dumpWRFDataArray("U", U.data(), 2, 40, 114, 114);
    
    // V = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
    // for y = 1:SimData.ny
    //    V(:,y,:,:) = .5*(Vstagg(:,y,:,:) + Vstagg(:,y+1,:,:));
    // end
    std::vector<double> V( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {
                    
                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    auto idxP1y = t * (nbAlt * ny * nx) + z * (ny * nx) + (y+1) * (nx) + x;

                    V[idx] = 0.5 * ( vStaggeredData[idx] + vStaggeredData[idxP1y] );
                }
            }
        }
    }
    dumpWRFDataArray("V", V.data(), 2, 40, 114, 114);
    
    // SimData.CoordZ = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
    // for k = 1:SimData.NbAlt
    //    SimData.CoordZ(:,:,k,:) = .5*(Height(:,:,k,:) + Height(:,:,k+1,:));
    // end
    std::vector<double> coordZ( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {

                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    auto idxP1z = t * (nbAlt * ny * nx) + (z+1) * (ny * nx) + y * (nx) + x;

                    coordZ[idx] = 0.5 * ( heightData[idx] + heightData[idxP1z] );
                }
            }
        }
    }

    // %% Velocity and direction %%
    // SimData.WS = sqrt(U.^2 + V.^2);
    std::vector<double> WS( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {
                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    WS[idx] = sqrt( U[idx]*U[idx] + V[idx]*V[idx] );
                }
            }
        }

        
    }
    dumpWRFDataArray("WS", WS.data(), 2, nbAlt, ny, nx);
    

    // SimData.WD = zeros(SimData.nx,SimData.ny,SimData.NbAlt,numel(SimData.TIMEVECT));
    std::vector<double> WD( nx * ny * nbAlt * 2, 0.0 );
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {
            for (auto y=0; y<ny; y++) {
                for (auto x=0; x<nx; x++) {
                    auto idx = t * (nbAlt * ny * nx) + z * (ny * nx) + y * (nx) + x;
                    if (U[idx]> 0) {
                        WD[idx] = 270.0 - (180.0/c_PI) * atan( V[idx]/U[idx] );
                    }
                    else {
                        WD[idx] = 90.0 - (180.0/c_PI) * atan( V[idx]/U[idx] );
                    }
                }
            }
        }
    }
    dumpWRFDataArray("WD", WD.data(), 2, nbAlt, ny, nx);

#if 0
I shoould need this... hopefully
%% Permutation to set dimensions as (row,col,etc) = (ny,nx,nz,nt) %%

SimData.WD = permute(SimData.WD,[2 1 3 4]);
SimData.WS = permute(SimData.WS,[2 1 3 4]);
SimData.CoordZ = permute(SimData.CoordZ,[2 1 3 4]);
#endif

    std::cout << "Smoothing domain" << std::endl;

    // Updating dimensions
    int nx2 = floor(sqrt(m_maxTerrainSize * nx/(float)ny));
    int ny2 = floor(sqrt(m_maxTerrainSize * ny/(float)nx));
    
    // SimData.nx = nx2;
    // SimData.ny = ny2;
    m_dx = m_dx * nx/(float)nx2;
    m_dy = m_dy * ny/(float)ny2;

    std::cout << "Resizing from " << nx << " by " << ny << " to " << nx2 << " by " << ny2 << std::endl;

    // Terrain
    // SimData.Relief = imresize(SimData.Relief,[ny2,nx2]);
    std::vector<double> reliefResize( nx2 * ny2 * nbAlt * 2, 0.0 );

    //
    // Fake this for now with nearest neighbor... implement bicubic
    // later
    //

    float scaleX = nx / (float)nx2;
    float scaleY = ny / (float)ny2;
    
    for (auto y=0; y<ny2; y++) {
        for (auto x=0; x<nx2; x++) {
                    
            // Map this into the larger vector
            int largerX = (int)floor( x * scaleX );
            int largerY = (int)floor( y * scaleY );
            
            auto idxLarger = largerY * (nx) + largerX;
            auto idx = y * (nx2) + x;

            reliefResize[idx] = relief[idxLarger];
        }
    }


}

void WRFInput::setWRFDataPoint()
{
#if 0
% If MaxNbStat is smaller than the number of WRF data point available, then
% we operate a selection. Vertical height is selected between defined boundaries.
% For each stations, their horizontal and vertical coordinates, wind speed 
% and direction, along with the number of vertical altitude pts, are recorded 

if SimData.nx*SimData.ny > SimData.MaxNbStat
    
    nx2 = sqrt(SimData.MaxNbStat*SimData.nx/SimData.ny);
    ny2 = sqrt(SimData.MaxNbStat*SimData.ny/SimData.nx);
    
    WRF_RowY = (1:SimData.ny/ny2:SimData.ny);
    WRF_ColX = (1:SimData.nx/nx2:SimData.nx);
    
    WRF_RowY = unique(round(WRF_RowY));
    WRF_ColX = unique(round(WRF_ColX));
    
else
    
    WRF_RowY = (1:SimData.ny);
    WRF_ColX = (1:SimData.nx);
    
end

SimData.NbStat = numel(WRF_RowY)*numel(WRF_ColX);
StatData.CoordX = zeros(1,SimData.NbStat);
StatData.CoordY = zeros(1,SimData.NbStat);
StatData.nz = zeros(1,SimData.NbStat);

StatData.CoordZ = struct([]);
StatData.WS = struct([]);
StatData.WD = struct([]);


Stat = 1;
for y = WRF_RowY
    for x = WRF_ColX
        
        StatData.CoordX(Stat) = x;
        StatData.CoordY(Stat) = y;
        
        levelk_max = 0;
        for t = 1:numel(SimData.TIMEVECT)                     
            CoordZ_xyt = SimData.CoordZ(y,x,:,t);        
            [levelk] = find(CoordZ_xyt >= SimData.MinWRFAlt & CoordZ_xyt <= SimData.MaxWRFAlt);
            if numel(levelk) > numel(levelk_max)
                levelk_max = levelk; % If wind data heights change during time, higher height vector is selected
            end
        end
        
        StatData.CoordZ{Stat} = reshape(SimData.CoordZ(y,x,levelk_max,:),numel(levelk_max),size(SimData.CoordZ,4));
        StatData.nz(Stat) = size(StatData.CoordZ{Stat},1);
        
        StatData.WS{Stat} = reshape(SimData.WS(y,x,levelk_max,:),numel(levelk_max),size(SimData.WS,4));
        StatData.WD{Stat} = reshape(SimData.WD(y,x,levelk_max,:),numel(levelk_max),size(SimData.WD,4));   
        
        Stat = Stat + 1;
    end
end

SimData.maxCoordz = 0; % Will be used to set domain vertical dimension
SimData.minCoordz = 0;
for i = 1:SimData.NbStat
    SimData.minCoordz = min(min(min(SimData.maxCoordz,StatData.CoordZ{i})));
    SimData.maxCoordz = max(max(max(SimData.maxCoordz,StatData.CoordZ{i})));
end

fprintf('%i %s %g %s %g %s\n',SimData.NbStat,' WRF data points have been generated between ',SimData.minCoordz,' and ',SimData.maxCoordz,' meters AGL')
#endif


}


void WRFInput::dumpWRFDataArray(const std::string &name, double *data, int dimT, int dimZ, int dimY, int dimX)
{
    std::cout << "[" << name << "] WRF Data Dump" << std::endl << "==========================" << std::endl;

    // This output is analagous to Matlab's Columns 1 through dimY style of output
    for (auto t=0; t<dimT; t++) {
        for (auto z=0; z<dimZ; z++) {
            std::cout << "Slice: (t=" << t << ", z=" << z << ")" << std::endl;
            for (auto x=0; x<dimX; x++) {
                for (auto y=0; y<dimY; y++) {

                    auto idx = t * (dimZ * dimY * dimX) + z * (dimY * dimX) + y * (dimX) + x;
                    std::cout << data[idx] << ' ';
                    
                }
                std::cout << std::endl;
            }
        }
    }
}


void WRFInput::smoothDomain()
{
    // Smooth the following layers: topography, wind data, Z0 and
    // land-use using imresize (bicubic interpolation by default).
    // Land-use is interpolated with the "nearest point" method as we
    // must not change the categories values.
    
#if 0
    std::cout << "Smoothing domain" << std::endl;

    // Updating dimensions
    int nx2 = floor(sqrt(m_maxTerrainSize * nx/(float)ny));
    int ny2 = floor(sqrt(m_maxTerrainSize * ny/(float)nx));
    
    // SimData.nx = nx2;
    // SimData.ny = ny2;
    m_dx = m_dx * nx/(float)nx2;
    m_dy = m_dy * ny/(float)ny2;

    // Terrain
    // SimData.Relief = imresize(SimData.Relief,[ny2,nx2]);
    int nbAlt = 40;
    std::vector<double> reliefResize( nx_2 * ny_2 * nbAlt * 2, 0.0 );

    // fake this for now with nearest neighbor...
    for (auto t=0; t<2; t++) {
        for (auto z=0; z<nbAlt; z++) {

            for (auto y=0; y<ny2; y++) {
                for (auto x=0; x<nx2; x++) {
                    
                    // Map this into the larger vector

                }
            }

        }
    }
    
            
#endif



    // Wind velocity, direction and vertical position
    // SimData.WS = imresize(SimData.WS,[ny2,nx2]);
    // SimData.WD = imresize(SimData.WD,[ny2,nx2]);
    // SimData.CoordZ = imresize(SimData.CoordZ,[ny2,nx2]);

    // Roughness length
    // SimData.Z0 = imresize(SimData.Z0,[ny2,nx2]);

    // Land-use
    // SimData.LU = imresize(SimData.LU,[ny2,nx2],'nearest');

}


void WRFInput::minimizeDomainHeight()
{
    // Here we lower minimum altitude to 0 in order to save
    // computational space.  For the same reason, topography below 50
    // cm will not be considered.

    // SimData.OldTopoMin = min(min(SimData.Relief));
    // SimData.Relief = SimData.Relief - SimData.OldTopoMin;  % Lowering minimum altitude to 0 

    // IndLowRelief = find(SimData.Relief <= 15); % Relief below 0.5m is not considered
    // SimData.Relief(IndLowRelief) = 0;

    // SimData.NbTerrain = numel(SimData.Relief) - size(IndLowRelief,1);
    // SimData.NewTopoMax = max(max(SimData.Relief));
    // SimData.CoordZ = SimData.CoordZ - SimData.OldTopoMin;

}