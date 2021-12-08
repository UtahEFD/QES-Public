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

/** @file TURBParams.h */

#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <limits>

#include "util/ParseInterface.h"
#include "util/Vector3.h"

/**
 * @class TURBParams
 * @brief :document this:
 *
 * @sa ParseInterface
 */
class TURBParams : public ParseInterface
{
private:
protected:
public:
  int methodLocalMixing;
  bool save2file;
  std::string filename, varname;

  int mlSamplesPerAirCell;

  Vector3 *sigConst;
  bool flagNonLocalMixing;
  int buildingWallFlag, terrainWallFlag;
  float turbUpperBound;
  float backgroundMixing;


  TURBParams()
  {}
  ~TURBParams()
  {}

  virtual void parseValues()
  {
    methodLocalMixing = 0;
    parsePrimitive<int>(true, methodLocalMixing, "method");
    save2file = false;
    parsePrimitive<bool>(false, save2file, "save");
    filename = "";
    parsePrimitive<std::string>(false, filename, "LMfile");
    varname = "mixlength";// default name
    parsePrimitive<std::string>(false, varname, "varname");

    // defaults for local mixing sample rates -- used with ray
    // tracing methods
    if (methodLocalMixing == 3) {// OptiX
      mlSamplesPerAirCell = 2000;
    } else {
      mlSamplesPerAirCell = 500;// other ray-traced methods
    }
    parsePrimitive<int>(false, mlSamplesPerAirCell, "samples");
    if (methodLocalMixing == 3) {

      std::cout << "Setting samples per air cell for ray-traced mixing length to "
                << mlSamplesPerAirCell << std::endl;
    }


    if (methodLocalMixing < 0 || methodLocalMixing > 4) {
      std::cout << "[WARNING] unknown local mixing method -> "
                << "set method to 0 (height above terrain)" << std::endl;
      methodLocalMixing = 0;
    }

    if ((methodLocalMixing == 4 || save2file == true) && (filename == "")) {
      std::cout << "[WARNING] no local mixing file provided -> "
                << "set method to 0 (height above terrain)" << std::endl;
      methodLocalMixing = 0;
    }
    if (methodLocalMixing == 0 || methodLocalMixing == 4) {
      save2file = "false";
    }

    sigConst = nullptr;
    parseElement<Vector3>(false, sigConst, "sigmaConst");

    flagNonLocalMixing = false;
    parsePrimitive<bool>(false, flagNonLocalMixing, "nonLocalMixing");

    terrainWallFlag = 2;
    parsePrimitive<int>(false, terrainWallFlag, "terrainWallFlag");
    buildingWallFlag = 2;
    parsePrimitive<int>(false, buildingWallFlag, "buildingWallFlag");

    turbUpperBound = 100;
    parsePrimitive<float>(false, turbUpperBound, "turbUpperBound");

    backgroundMixing = 0.0;
    parsePrimitive<float>(false, backgroundMixing, "backgroundMixing");

  }
};
