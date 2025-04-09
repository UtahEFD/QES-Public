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
/** @file HRRRInput.h */

#pragma once

#include "util/ParseInterface.h"
#include <string>
#include <netcdf>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

/**
 * @class HRRRInput
 * @brief Read in HRRR wind data into QES-Winds
 * it can be used in QES-Plume later to read in 
 * HRRR smoke data
 */

class HRRRInput : public ParseInterface
{
 private:
 public:

  std::string HRRRFile; /**< HRRR file name */
  //std::vector<std::string> inputFields; /**< HRRR input fields */
  int interpolationScheme = 0; /**< Interpolation scheme for Interpolation scheme for initial guess field (0-Barnes Scheme (default), 1-Nearest site, 2-Bilinear interpolation) */
  int stabilityClasses = 0;  /**< Defining method for stability classes (0-No stability (default), 1-Pasquill-Gifford classes, 2-Monin-Obukhov length (using surface fluxes) */

  virtual void parseValues()
  {
    //parseMultiPrimitives<std::string>(false, inputFields, "inputFields");
    parsePrimitive<int>(false, interpolationScheme, "interpolationScheme");
    parsePrimitive<int>(false, stabilityClasses, "stabilityClasses");
    HRRRFile = "";
    parsePrimitive<std::string>(false, HRRRFile, "HRRRFile");
    if (HRRRFile != ""){
      HRRRFile = QESfs::get_absolute_path(HRRRFile);
    }
    
  }
};
