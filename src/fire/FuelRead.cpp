/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Matthew Moody
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

/**
 * @file FuelRead.cpp
 * @brief Loads fuel data.
 *
 * @sa Vector3
 */

#include "util/Geotiff.cpp"
#include "FuelRead.hpp"
#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"



FuelRead::FuelRead(const std::string &filename,
	std::tuple<int, int> dim,
	std::tuple<float, float> cellSize)
{
  std::cout<<"test"<<std::endl;
  /*
  // create object of Geotiff class
  Geotiff tiff((const char*) &filename);  
 
  // output a value from 2D array  
  float** rasterBandData = tiff.GetRasterBand(1) ; 
  cout << "value at row 10, column 10: " << rasterBandData[10][10] << endl; 
 
  // call other methods, like get the name of the Geotiff
  // passed-in, its length, and its projection string 
  cout << tiff.GetFileName() << endl ;
  cout << strlen( tiff.GetFileName() ) << endl ;
  cout << tiff.GetProjection() << endl;
   
  // dump out the Geotransform (6 element array of doubles) 
  double *gt = tiff.GetGeoTransform(); 
  cout << gt[0] << " " << gt[1] << " " << gt[2] << " " << gt[3] << " " << gt[4] << " " << gt[5] << endl; 
   
  // dump out Geotiff band NoData value (often it is -9999.0)
  cout << "No data value: " << tiff.GetNoDataValue() << endl;  
  
  // dump out array (band) dimensions of Geotiff data  
  int *dims = tiff.GetDimensions() ; 
  cout << dims[0] << " " << dims[1] << " " << dims[2] << endl;
  */
}
