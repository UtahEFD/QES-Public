/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

/** @file FuelRead.hpp */
#ifndef __FUEL_READ_HPP__
#define __FUEL_READ_HPP__ 1

#include <string>
#include "winds/Triangle.h"
#include "util/Vector3.h"
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

class WINDSGeneralData;
class WINDSInputData;

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
