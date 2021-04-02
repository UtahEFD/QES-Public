#pragma once

/** \file "WRFInput.h" Context header file. 
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

// Based on Mattieu's Matlab Scripts
// and paper from
//
// "One-Way Coupling of the WRF–QUIC Urban Dispersion Modeling
// System", A. K. KOCHANSKI, E. R. PARDYJAK AND R. STOLL,
// A. GOWARDHAN, M. J. BROWN, and W. J. STEENBURGH
//

#include <string>
#include <netcdf>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

#include "Sensor.h"

// Forward Decl
class WINDSGeneralData;

class profData 
{
public:
    float zCoord;
    float ws, wd;
};

class stationData 
{
public:
    stationData() {}
    ~stationData() {}

    float xCoord, yCoord;
    float z0;
    
    std::vector< std::vector< profData > > profiles;

private:
};

class WRFInput
{
public:

    const double c_PI = 3.14159265358979323846;

    WRFInput(const std::string& filename, double domainUTMx, double domainUTMy,
             int zoneUTM, std::string &zoneLetterUTM, float dimX, float dimY,
             int sensorSample,
             bool sensorsOnly=false);
    ~WRFInput();

    // Ouput a wind field from QES to the WRF file:
    // outputWindField

    std::string m_WRFFilename;
    
    // If set to true, ONLY the sensor data will be pulled from WRF
    // Atmos Mesh.
    bool m_processOnlySensorData;

    // Fire Mesh Related Data
    bool hasFireMesh;
    
    int fm_nx, fm_ny, fm_nz;
    float fm_dx, fm_dy, fm_dz;
    std::vector<double> fmHeight;

    // Atmospheric Mesh
    int atm_nx, atm_ny, atm_nz;
    double atm_dx, atm_dy;

    // Station (i.e. Sensor) Data
    std::vector< stationData > statData;

    float atm_minWRFAlt, fm_minWRFAlt;
    float atm_maxWRFAlt, fm_maxWRFAlt;

    int maxSensors;

    std::vector<double> u0_fmw, v0_fmw, ht_fmw;

    /**
     */
    void applyHalotoStationData(const float haloX, const float haloY);

    /**
     */
    void dumpStationData() const;


    /**
     */
    void extractWind( WINDSGeneralData *wgd );
    

    /**
     * Reading WRF data - ReadDomainInfo.m
     *
     * Read domain dimensions, terrain elevation, wind data, land-use
     * and Z0 from WRF output. Possibility to crop domain borders by
     * adding a varargin (expected format:
     * [Xstart,Xend;YStart,Yend]). Layers of data are recorder in the
     * following format: (row,col) = (ny,nx).
     * 
     * This function extracts the following variables :
     *     • Horizontal dimensions nx and ny (vertical dimension will be computed later, see part).
     *     • Time moments (in units provided by WRF).
     *     • Horizontal resolution dx and dy (vertical resolution is set to 1 in this version).
     * It is possible to crop domain borders by adding a variable while calling the function. This variable
     * should be an array with the following format:
     *     Xstart Xend
     *     Y start Y end
     * Finally, this function also extracts:
     *     • Terrain topography.
     *     • Wind data (more details in next subsection).
     *     • Land-use categories.
     *     • Roughness length.

	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
	\param[out] "data" Primitive data structure
     */
    void readDomainInfo();

    /**
     * Reading wind data - WindFunc.m
     * 
     * In a few steps, wind data vertical position, velocity and direction is computed:
     *     • WRF wind data altitude (in meters) is obtained from geopotential height (dividing the sum of
     * PH and PHB by the constant of gravitational acceleration g = 9.81m/s2
     *     • Wind components U and V components are extracted.
     * 
     *     • These three variables are interpolated at each WRF cell center (simple arithmetic mean).
     *     • Velocity magnitude is computed as sqrt(U^2 + V^2)
     *     • Velocity direction is computed from U and V vector components.
     *     • In the last step, velocity magnitude, direction and vertical
     *     coordinates are permuted to be recorded in format [row,col] =
     *     [ny,nx]. So are every other layers in this script (they are stored as
     *     [nx,ny] in WRF).
     */
    void readWindData();
            

    /**
     * Roughness Length
     *
     *
     */
    void roughnessLength();

    /**
     * Smoothing domain - Smooth.m
     * 
     * If the total number of horizontal cells exceeds the limit defined in the first part with MaxTerrainSize,
     * this function will generate a new domain respecting the constraint.
     * It is based on the matlab function imresize, which by default relies on a bicubic interpolation method
     * (it is possible to change the method, see imresize other options: bilinear, nearest point, Lanczos-2 and
     * Lanczos-3, which have not been tested yet).
     *     • First, new horizontal dimensions and resolutions are computed.
     *     • Terrain topography is interpolated to fit the new domain dimensions.
     *     • Idem for the wind velocity magnitude, direction and altitude values.
     *     • Idem for roughness length.
     *     • Land-use categories are interpolated to the "nearest point" as we must not change their values
     *       while averaging them.
     */
    void smoothDomain();
    

    /** 
     * Minimizing cell number - MinDomainHeight.m
     * 
     * This small function aims at reducing the number of cells by lowering the minimum domain elevation to
     * 0. For the same reason, it also sets every block below 50 cm to 0, recording the corresponding number
     * of blocks in NbTerrain. Wind vertical coordinate is modified
     * accordingly.
     */
    void minimizeDomainHeight();
    
        

    /** 
     * Selecting WRF data stations - SetWRFdataPt.m
     * 
     * If MaxNbStat is large enough, every WRF wind data points will be used. If this is not the case, a
     * number of points close to MaxNbStat are distributed evenly across the domain. In order to do so:
     *     • WRF wind data row and column (horizontal) indices are computed to assert a even distribution.
     *     • These horizontal coordinates are associated to each stations.
     *     • For each station, a vertical index is computed. It corresponds to wind data altitude contained
     * between defined boundaries (MinWRFAlt and MaxWRFAlt).
     *     • The corresponding vertical coordinates (in meters) are associated to each stations, as well as the
     * number of these coordinates.
     *     • Wind speed and direction at these altitudes are associated to each stations.
     *     • Finally, maximal vertical coordinate is stored, it will be used to define the domain height (this
     * value multiplied by 1.2).
     */
    void setWRFDataPoint();
    
    float lookupLandUse( int luIdx );

    void setHaloAdditions( int xA, int yA ) 
    {
        m_haloX_DimAddition = xA;
        m_haloY_DimAddition = yA;
    }
    

private:

    NcFile wrfInputFile;

    int nx, ny;
    float m_dx, m_dy;
    
    int xDim[2], yDim[2];

    int m_minWRFAlt;
    int m_maxWRFAlt;
    int m_maxTerrainSize;
    int m_maxNbStat;

    int m_TerrainFlag; // 0 = No relief; 1 = W/ relief.
    int m_BdFlag;      // 0 = No buildings; 1 = Defined by the user.
    int m_VegFlag;     // 0 = No vegetation; 1 = Extracted from WRF.
    int m_Z0Flag;      // 1 = Extracted from WRF Restart File; 2 = Defined by WRF land use; 3 = Constant value.
  
    // For later efforts
    // SimData.Z0DataSource = SimData.WRFFile;            %%% If
    // Z0Flag = 1, put WRF Restart File name; if Z0Flag = 2, put
    // WRFOUTPUT FILE name; if Z0Flag = 3, put a CONSTANT VALUE.

    double* relief;

    int m_haloX_DimAddition, m_haloY_DimAddition;

    void dumpWRFDataArray(const std::string &name, double *allData, int dimT, int dimZ, int dimY, int dimX);

    // PHB Struct from WRF
    // float PHB(Time, bottom_top_stag, south_north, west_east) ;
    //		PHB:FieldType = 104 ;
    //		PHB:MemoryOrder = "XYZ" ;
    //		PHB:description = "base-state geopotential" ;
    //		PHB:units = "m2 s-2" ;
    //		PHB:stagger = "Z" ;
    //		PHB:coordinates = "XLONG XLAT" ;

    // PH Struct
    // float PH(Time, bottom_top_stag, south_north, west_east) ;
    // 		PH:FieldType = 104 ;
    //		PH:MemoryOrder = "XYZ" ;
    //		PH:description = "perturbation geopotential" ;
    //		PH:units = "m2 s-2" ;
    //		PH:stagger = "Z" ;
    //		PH:coordinates = "XLONG XLAT" ;
};

    
