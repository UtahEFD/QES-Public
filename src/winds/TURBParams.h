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
#include "util/QESout.h"

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

  TURBParams() : methodLocalMixing(0), save2file(false),
                 filename(""), varname("mixlength"),
                 flagNonLocalMixing(false),
                 terrainWallFlag(2), buildingWallFlag(2),
                 turbUpperBound(20.0), backgroundMixing(0.0)
  {}
  ~TURBParams()
  {}

  virtual void parseValues()
  {

    parsePrimitive<int>(true, methodLocalMixing, "method");
    parsePrimitive<bool>(false, save2file, "save");
    parsePrimitive<std::string>(false, filename, "LMfile");
    parsePrimitive<std::string>(false, varname, "varname");
    parsePrimitive<bool>(false, flagNonLocalMixing, "nonLocalMixing");
    parsePrimitive<int>(false, terrainWallFlag, "terrainWallFlag");
    parsePrimitive<int>(false, buildingWallFlag, "buildingWallFlag");
    parsePrimitive<float>(false, turbUpperBound, "turbUpperBound");
    parsePrimitive<float>(false, backgroundMixing, "backgroundMixing");

#ifdef HAS_OPTIX
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
#else
    if (methodLocalMixing == 3) {
      QESout::warning("Optix not supported -> set method to serial (methodLocalMixing = 1)");
      methodLocalMixing = 1;
    }
#endif

    if (methodLocalMixing < 0 || methodLocalMixing > 4) {
      std::string mess = "Unknown local mixing method -> ";
      mess += "set method to height above terrain (methodLocalMixing = 0)";
      QESout::warning(mess);
      methodLocalMixing = 0;
    }

    if ((methodLocalMixing == 4 || save2file) && (filename.empty())) {
      std::string mess = "No local mixing file provided -> ";
      mess += "set method to height above terrain(methodLocalMixing = 0) ";
      QESout::warning(mess);
      methodLocalMixing = 0;
    }
    if (methodLocalMixing == 0 || methodLocalMixing == 4) {
      save2file = "false";
    }

    if (!filename.empty()) {
      filename = QESfs::get_absolute_path(filename);
    }

    sigConst = nullptr;
    ParseVector<float> *sig_in = nullptr;
    parseElement<ParseVector<float>>(false, sig_in, "sigmaConst");
    if (sig_in) {
      if (sig_in->size() == 3) {
        sigConst = new Vector3((*(sig_in))[0], (*(sig_in))[1], (*(sig_in))[2]);
      } else {
        exit(EXIT_FAILURE);
      }
    }
  }
};
