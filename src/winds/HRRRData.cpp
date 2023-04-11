/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/** @file HRRRData.cpp */

#include <string>
using namespace std;
#include <boost/date_time/posix_time/posix_time.hpp>
namespace bgreg = boost::gregorian;
namespace btime = boost::posix_time;
#include <iostream>
#include <math.h>
#include "util/NetCDFInput.h"
#include "util/GIStool.h"
#include "HRRRData.h"
#include "WINDSGeneralData.h"
#include "WINDSInputData.h"



HRRRData::HRRRData(std::string fileName, std::vector<std::string> HRRRFields)
{
   netcdf = new  NetCDFInput(fileName);
    
    netcdf->getDimension("time", timeDim);
    netcdf->getDimensionSize("time", timeSize);
    hrrrTime.resize( timeSize );
    hrrrTimeTrans.resize( timeSize );
    std::vector<size_t> startIdx = {0,0};
    std::vector<size_t> counts = {static_cast<unsigned long>(timeSize),1};
    netcdf->getVariableData("time", startIdx , counts, hrrrTime);

    for (auto t=0; t<timeSize; t++){
      hrrrTimeTrans[t] = hrrrTime[t]; 
    }

    netcdf->getDimension("x", xDim);
    netcdf->getDimensionSize("x", xSize);
    netcdf->getDimension("y", yDim);
    netcdf->getDimensionSize("y", ySize);
    hrrrX.resize( xSize );
    hrrrY.resize( ySize );
    std::vector<size_t> xStartIdx = {0,0};
    std::vector<size_t> xCounts = {static_cast<unsigned long>(xSize),1};
    netcdf->getVariableData("x", xStartIdx , xCounts, hrrrX);
    std::vector<size_t> yStartIdx = {0,0};
    std::vector<size_t> yCounts = {static_cast<unsigned long>(ySize),1};
    netcdf->getVariableData("y", yStartIdx , yCounts, hrrrY);
    
    siteLat.resize( xSize * ySize );
    siteLon.resize( xSize * ySize );
    
    std::vector<size_t> twoStartIdx = {0,0};
    std::vector<size_t> twoCounts = {static_cast<unsigned long>(ySize),static_cast<unsigned long>(xSize)};
    
    netcdf->getVariableData("latitude", twoStartIdx, twoCounts, siteLat);
    netcdf->getVariableData("longitude", twoStartIdx, twoCounts, siteLon);

    for (auto i = 0; i < xSize; i++){
      for (auto j = 0; j < ySize; j++){
	int id = i + j * xSize;
	siteLon[id] = siteLon[id] - 360;
      }
    }      
    
}



void HRRRData::findHRRRSensors(const WINDSInputData *WID, WINDSGeneralData *WGD){
    siteUTMx.resize( xSize * ySize );
    siteUTMy.resize( xSize * ySize );
    siteUTMzone.resize( xSize * ySize );
    float temp_utmx, temp_utmy;
    temp_utmx = float(WID->simParams->UTMx);
    temp_utmy = float(WID->simParams->UTMy);
    float lat_south, lat_north, lon_east, lon_west;
    GIStool::UTMConverter(lon_west,lat_south,temp_utmx,temp_utmy,WID->simParams->UTMZone,true,1);
    temp_utmx = float(WID->simParams->UTMx) + float((WGD->nx-1) * WGD->dx);
    temp_utmy = float(WID->simParams->UTMy) + float((WGD->ny-1) * WGD->dy);
    GIStool::UTMConverter(lon_east,lat_north,temp_utmx,temp_utmy,WID->simParams->UTMZone,true,1);
    
    for (auto i = 0; i < xSize; i++){
      for (auto j = 0; j < ySize; j++){
	int id = i + j * xSize;
	if (siteLat[id] >= lat_south && siteLat[id] <= lat_north
	    && siteLon[id] >= lon_west && siteLon[id] <= lon_east){
	  float temp_lat, temp_lon;
	  temp_lat = float(siteLat[id]);
	  temp_lon = float(siteLon[id]);
	  GIStool::UTMConverter(temp_lon,temp_lat,siteUTMx[id],siteUTMy[id],siteUTMzone[id] ,true,0);
	  hrrrSensorUTMx.push_back(double(siteUTMx[id]));
	  hrrrSensorUTMy.push_back(double(siteUTMy[id]));
	  hrrrSensorUTMzone.push_back(siteUTMzone[id]);
	  hrrrSensorID.push_back(id);
	}
      }
    }
}


void HRRRData::readData(int t){
  
  std::vector<size_t> stepStartIdx = {t,0,0};
  std::vector<size_t> stepCounts = {1,static_cast<unsigned long>(ySize),static_cast<unsigned long>(xSize)};
  hrrrU.resize( xSize * ySize );
  hrrrV.resize( xSize * ySize );
  hrrrZ0.resize( xSize * ySize );
  hrrrCloudCover.resize( xSize * ySize );
  hrrrShortRadiation.resize( xSize * ySize );
  hrrrUTop.resize( xSize * ySize );
  hrrrVTop.resize( xSize * ySize );
  hrrrPBLHeight.resize( xSize * ySize );
  netcdf->getVariableData("UGRD_10maboveground", stepStartIdx, stepCounts, hrrrU);
  netcdf->getVariableData("VGRD_10maboveground", stepStartIdx, stepCounts, hrrrV);
  netcdf->getVariableData("SFCR_surface", stepStartIdx, stepCounts, hrrrZ0);
  netcdf->getVariableData("TCDC_entireatmosphere", stepStartIdx, stepCounts, hrrrCloudCover);
  netcdf->getVariableData("DSWRF_surface", stepStartIdx, stepCounts, hrrrShortRadiation);
  netcdf->getVariableData("HPBL_surface", stepStartIdx, stepCounts, hrrrPBLHeight);
  netcdf->getVariableData("UGRD_700mb", stepStartIdx, stepCounts, hrrrUTop);
  netcdf->getVariableData("VGRD_700mb", stepStartIdx, stepCounts, hrrrVTop);
  

  hrrrSpeed.clear();
  hrrrDir.clear();
  hrrrSpeedTop.clear();
  hrrrDirTop.clear();
  for (size_t i = 0; i < hrrrSensorID.size(); i++) {
    hrrrSpeed.push_back(sqrt( pow(hrrrU[hrrrSensorID[i]], 2.0) + pow(hrrrV[hrrrSensorID[i]], 2.0) ));
    hrrrDir.push_back(fmod((atan2(hrrrV[hrrrSensorID[i]]/hrrrSpeed[i], hrrrU[hrrrSensorID[i]]/hrrrSpeed[i])*180.0/M_PI) , 360.0));

    hrrrSpeedTop.push_back(sqrt( pow(hrrrUTop[hrrrSensorID[i]], 2.0) + pow(hrrrVTop[hrrrSensorID[i]], 2.0) ));
    hrrrDirTop.push_back(fmod((atan2(hrrrVTop[hrrrSensorID[i]]/hrrrSpeedTop[i], hrrrUTop[hrrrSensorID[i]]/hrrrSpeedTop[i])*180.0/M_PI), 360.0));
  }

  
}
 

