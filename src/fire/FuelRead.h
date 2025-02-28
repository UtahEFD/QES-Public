/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Matthew Moody
 * Copyright (c) 2025 Jeremy Gibbs
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Brian Bailey
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file FuelRead.h
 * @brief This function reads fuel data from provided GEOTIF.
 */
#ifndef __FUEL_READ_HPP__
#define __FUEL_READ_HPP__ 1

#include <string>
#include "util/Vector3Float.h"
#include "util/Vector3Int.h"

#include "gdal_priv.h"
#include "cpl_conv.h"// for CPLMalloc()
#include "ogrsf_frmts.h"

#include "winds/Cell.h"
#include "winds/Edge.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>

// class WINDSGeneralData;
// class WINDSInputData;

/**
 * @class FuelRead
 */
class FuelRead
{
public:
  /**
   * Constructs a GIS Surface Fuel Model for use with QES.
   *
   * @param filename the filename containing the GIS data to load
   * @param dim a 2-tuple of ints representing the dimension of
   * the domain, as in {nx, ny}
   * @param cellSize a 2-tuple of floats representing the size of
   * each domain cell in the surface dimensions, as in {dx, dy}
   * @param UTMx the UTM origin in x
   * @param UTMy the UTM origin in y
   * @param OriginFlag :document this:
   * @param DEMDistanceX :document this:
   * @param DEMDistanceY :document this:
   * @return a string representing the results of the failed summation.
   */

  FuelRead(const std::string &filename,
           std::tuple<int, int> dim,
           std::tuple<float, float> cellSize);

  std::vector<int> fuelField;
};


#endif
