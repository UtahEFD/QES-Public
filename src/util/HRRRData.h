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
/** @file HRRRData.h */

#pragma once

#include <string>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;
#include <iostream>
#include <cmath>
#include "NetCDFInput.h"
#include "GIStool.h"
// Forward declaration
class WINDSGeneralData;
class WINDSInputData;
class PlumeInputData;


/**
 * @class HRRRData
 * @brief Processing HRRR wind data into QES-Winds
 */

class HRRRData
{
  private:
  
  NcFile hrrrInputFile;
  std::vector<std::string> hrrrFields;
  
  public:
  
  NetCDFInput *netcdf;
  std::string fileName; /**< HRRR file name */
  std::vector<std::string> HRRRFields; /**< HRRR input fields */
  std::vector<double> hrrrTime, hrrrX, hrrrY;
  std::vector<double> siteLat, siteLon;
  std::vector<float> siteUTMx, siteUTMy;
  std::vector<int> siteUTMzone;
  std::vector<double>  hrrrSensorUTMx, hrrrSensorUTMy;
  std::vector<int> hrrrSensorID, hrrrSensorUTMzone;
  NcDim xDim, yDim, timeDim;
  int xSize, ySize, timeSize;
  std::vector<time_t> hrrrTimeTrans;
  std::vector<double> hrrrU, hrrrV, hrrrSpeed, hrrrDir, hrrrZ0;
  std::vector<double> hrrrCloudCover, hrrrShortRadiation;
  std::vector<double> hrrrPBLHeight, hrrrU700, hrrrV700, hrrrU850, hrrrV850, hrrrU925, hrrrV925, hrrrUTop, hrrrVTop;
  std::vector<double> hrrrSpeedTop, hrrrDirTop, hrrrSpeed700, hrrrDir700, hrrrSpeed850, hrrrDir850, hrrrSpeed925, hrrrDir925;

  std::vector<double>  hrrrSourceUTMx, hrrrSourceUTMy;
  std::vector<int> hrrrSourceID, hrrrSourceUTMzone;
  std::vector<float> hrrrCon, hrrrC;

  HRRRData(std::string fileName, std::vector<std::string> HRRRFields);

  ~HRRRData();

  void findHRRRSensors(const WINDSInputData *WID, WINDSGeneralData *WGD);

  void findHRRRSources(const PlumeInputData *PID, WINDSGeneralData *WGD);

  void readSensorData(int t);

  void readAloftData(int t);

  void readSourceData(int t);

};
