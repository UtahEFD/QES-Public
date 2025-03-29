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
#include "WINDSGeneralData.h"
#include "WINDSInputData.h"
#include "plume/PLUMEInputData.h"


HRRRData::HRRRData(std::string fileName, std::vector<std::string> HRRRFields)
{
  netcdf = new  NetCDFInput(fileName);

  std::cout << "fileName:   " << fileName << std::endl;
  std::cout << "HRRRFields:   " << HRRRFields[0] << std::endl;
    
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
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();
  siteUTMx.resize( xSize * ySize );
  siteUTMy.resize( xSize * ySize );
  siteUTMzone.resize( xSize * ySize );
  float temp_utmx, temp_utmy;
  float hrrr_x = 6000.0;
  float hrrr_y = 6000.0;
  temp_utmx = float(WID->simParams->UTMx) - hrrr_x;
  temp_utmy = float(WID->simParams->UTMy) - hrrr_y;
  float lat_south, lat_north, lon_east, lon_west;
  GIStool::UTMConverter(lon_west,lat_south,temp_utmx,temp_utmy,WID->simParams->UTMZone,true,1);
  temp_utmx = float(WID->simParams->UTMx) + float((nx-1) * dx) + hrrr_x;
  temp_utmy = float(WID->simParams->UTMy) + float((ny-1) * dy) + hrrr_y;
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


void HRRRData::findHRRRSources(const PlumeInputData *PID, WINDSGeneralData *WGD){
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();
  
  siteUTMx.resize( xSize * ySize );
  siteUTMy.resize( xSize * ySize );
  siteUTMzone.resize( xSize * ySize );
  float temp_utmx, temp_utmy;
  temp_utmx = float(WGD->UTMx);
  temp_utmy = float(WGD->UTMy);
  float lat_south, lat_north, lon_east, lon_west;
  GIStool::UTMConverter(lon_west,lat_south,temp_utmx,temp_utmy,WGD->UTMZone,true,1);
  temp_utmx = float(WGD->UTMx) + float((nx-1) * dx);
  temp_utmy = float(WGD->UTMy) + float((ny-1) * dy);
  GIStool::UTMConverter(lon_east,lat_north,temp_utmx,temp_utmy,WGD->UTMZone,true,1);
    
  for (auto i = 0; i < xSize; i++){
    for (auto j = 0; j < ySize; j++){
      int id = i + j * xSize;
      if (siteLat[id] >= lat_south && siteLat[id] <= lat_north
	  && siteLon[id] >= lon_west && siteLon[id] <= lon_east){
	
	float temp_lat, temp_lon;
	temp_lat = float(siteLat[id]);
	temp_lon = float(siteLon[id]);
	GIStool::UTMConverter(temp_lon,temp_lat,siteUTMx[id],siteUTMy[id],siteUTMzone[id] ,true,0);
	hrrrSourceUTMx.push_back(double(siteUTMx[id]));
	hrrrSourceUTMy.push_back(double(siteUTMy[id]));
	hrrrSourceUTMzone.push_back(siteUTMzone[id]);
	hrrrSourceID.push_back(id);
      }
    }
  }

}


void HRRRData::readSensorData(int t){
  
  std::vector<size_t> stepStartIdx = {t,0,0};
  std::vector<size_t> stepCounts = {1,static_cast<unsigned long>(ySize),static_cast<unsigned long>(xSize)};
  hrrrU.resize( xSize * ySize );
  hrrrV.resize( xSize * ySize );
  hrrrZ0.resize( xSize * ySize );
  hrrrCloudCover.resize( xSize * ySize );
  hrrrShortRadiation.resize( xSize * ySize );
  hrrrSenHeatFlux.resize( xSize * ySize );
  hrrrUStar.resize( xSize * ySize );
  hrrrPotTemp.resize( xSize * ySize );
  netcdf->getVariableData("UGRD_10maboveground", stepStartIdx, stepCounts, hrrrU);
  netcdf->getVariableData("VGRD_10maboveground", stepStartIdx, stepCounts, hrrrV);
  netcdf->getVariableData("SFCR_surface", stepStartIdx, stepCounts, hrrrZ0);
  netcdf->getVariableData("TCDC_entireatmosphere", stepStartIdx, stepCounts, hrrrCloudCover);
  netcdf->getVariableData("DSWRF_surface", stepStartIdx, stepCounts, hrrrShortRadiation);

  netcdf->getVariableData("SHTFL_surface", stepStartIdx, stepCounts, hrrrSenHeatFlux);
  netcdf->getVariableData("FRICV_surface", stepStartIdx, stepCounts, hrrrUStar);
  netcdf->getVariableData("TMP_surface", stepStartIdx, stepCounts, hrrrPotTemp);
  
  float u_temp, v_temp;
  float rotcon_p = 0.622515;
  float lon_xx_p = -97.5;
  float angle2;
  hrrrSpeed.clear();
  hrrrDir.clear();
  for (size_t i = 0; i < hrrrSensorID.size(); i++) {
    u_temp = hrrrU[hrrrSensorID[i]];
    v_temp = hrrrV[hrrrSensorID[i]];
    angle2 = rotcon_p * (siteLon[hrrrSensorID[i]] - lon_xx_p) * 0.017453;
    hrrrU[hrrrSensorID[i]] = cos(angle2) * u_temp + sin(angle2) * v_temp;
    hrrrV[hrrrSensorID[i]] = -sin(angle2) * u_temp + cos(angle2) * v_temp;

    hrrrSpeed.push_back(sqrt( pow(hrrrU[hrrrSensorID[i]], 2.0) + pow(hrrrV[hrrrSensorID[i]], 2.0) ));
    hrrrDir.push_back(270 - fmod(360 + (atan2(hrrrV[hrrrSensorID[i]], hrrrU[hrrrSensorID[i]])*180.0/M_PI) , 360.0));
      
  }
  
}



void HRRRData::readAloftData(int t){

  std::vector<size_t> stepStartIdx = {t,0,0};
  std::vector<size_t> stepCounts = {1,static_cast<unsigned long>(ySize),static_cast<unsigned long>(xSize)};
  hrrrUTop.resize( xSize * ySize );
  hrrrVTop.resize( xSize * ySize );
  hrrrU700.resize( xSize * ySize );
  hrrrV700.resize( xSize * ySize );
  hrrrU850.resize( xSize * ySize );
  hrrrV850.resize( xSize * ySize );
  hrrrU925.resize( xSize * ySize );
  hrrrV925.resize( xSize * ySize );
  hrrrPBLHeight.resize( xSize * ySize );
  netcdf->getVariableData("HPBL_surface", stepStartIdx, stepCounts, hrrrPBLHeight);
  netcdf->getVariableData("UGRD_700mb", stepStartIdx, stepCounts, hrrrU700);
  netcdf->getVariableData("VGRD_700mb", stepStartIdx, stepCounts, hrrrV700);
  netcdf->getVariableData("UGRD_850mb", stepStartIdx, stepCounts, hrrrU850);
  netcdf->getVariableData("VGRD_850mb", stepStartIdx, stepCounts, hrrrV850);
  netcdf->getVariableData("UGRD_925mb", stepStartIdx, stepCounts, hrrrU925);
  netcdf->getVariableData("VGRD_925mb", stepStartIdx, stepCounts, hrrrV925);
  
  float u_temp, v_temp;
  float rotcon_p = 0.622515;
  float lon_xx_p = -97.5;
  float angle2;
  hrrrSpeed700.clear();
  hrrrDir700.clear();
  hrrrSpeed850.clear();
  hrrrDir850.clear();
  hrrrSpeed925.clear();
  hrrrDir925.clear();
  hrrrSpeedTop.clear();
  hrrrDirTop.clear();
  for (size_t i = 0; i < hrrrSensorID.size(); i++) {
    
    u_temp = hrrrU700[hrrrSensorID[i]];
    v_temp = hrrrV700[hrrrSensorID[i]];
    angle2 = rotcon_p * (siteLon[hrrrSensorID[i]] - lon_xx_p) * 0.017453;
    hrrrU700[hrrrSensorID[i]] = cos(angle2) * u_temp + sin(angle2) * v_temp;
    hrrrV700[hrrrSensorID[i]] = -sin(angle2) * u_temp + cos(angle2) * v_temp;
    hrrrSpeed700.push_back(sqrt( pow(hrrrU700[hrrrSensorID[i]], 2.0) + pow(hrrrV700[hrrrSensorID[i]], 2.0) ));
    hrrrDir700.push_back(270 - fmod(360 + (atan2(hrrrV700[hrrrSensorID[i]], hrrrU700[hrrrSensorID[i]])*180.0/M_PI), 360.0));
    hrrrSpeedTop.push_back(sqrt( pow(hrrrU700[hrrrSensorID[i]], 2.0) + pow(hrrrV700[hrrrSensorID[i]], 2.0) ));
    hrrrDirTop.push_back(270 - fmod(360 + (atan2(hrrrV700[hrrrSensorID[i]], hrrrU700[hrrrSensorID[i]])*180.0/M_PI), 360.0));

    u_temp = hrrrU850[hrrrSensorID[i]];
    v_temp = hrrrV850[hrrrSensorID[i]];
    hrrrU850[hrrrSensorID[i]] = cos(angle2) * u_temp + sin(angle2) * v_temp;
    hrrrV850[hrrrSensorID[i]] = -sin(angle2) * u_temp + cos(angle2) * v_temp;
    hrrrSpeed850.push_back(sqrt( pow(hrrrU850[hrrrSensorID[i]], 2.0) + pow(hrrrV850[hrrrSensorID[i]], 2.0) ));
    hrrrDir850.push_back(270 - fmod(360 + (atan2(hrrrV850[hrrrSensorID[i]], hrrrU850[hrrrSensorID[i]])*180.0/M_PI), 360.0));

    u_temp = hrrrU925[hrrrSensorID[i]];
    v_temp = hrrrV925[hrrrSensorID[i]];
    hrrrU925[hrrrSensorID[i]] = cos(angle2) * u_temp + sin(angle2) * v_temp;
    hrrrV925[hrrrSensorID[i]] = -sin(angle2) * u_temp + cos(angle2) * v_temp;
    hrrrSpeed925.push_back(sqrt( pow(hrrrU925[hrrrSensorID[i]], 2.0) + pow(hrrrV925[hrrrSensorID[i]], 2.0) ));
    hrrrDir925.push_back(270 - fmod(360 + (atan2(hrrrV925[hrrrSensorID[i]], hrrrU925[hrrrSensorID[i]])*180.0/M_PI), 360.0)); 
  }

}
 

void HRRRData::readSourceData(int t){
  
  std::vector<size_t> stepStartIdx = {t,0,0};
  std::vector<size_t> stepCounts = {1,static_cast<unsigned long>(ySize),static_cast<unsigned long>(xSize)};
  hrrrCon.resize( xSize * ySize );
  netcdf->getVariableData("MASSDEN_8maboveground", stepStartIdx, stepCounts, hrrrCon);

  hrrrC.clear();
  for (size_t i = 0; i < hrrrSourceID.size(); i++) {
    hrrrC.push_back(hrrrCon[hrrrSourceID[i]]);
      
  }
  
}
