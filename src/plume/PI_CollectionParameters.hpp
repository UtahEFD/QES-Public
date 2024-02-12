/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file CollectionParameters.hpp
 * @brief
 *
 * @note Child of ArgumentParsing
 * @sa ArgumentParsing
 */

//
//  CollectionParamters.hpp
//
//  This class handles xml collection box options
//  this is for collecting output from Lagrangian particle values to Eulerian values like concentration
//
//  Created by Jeremy Gibbs on 03/25/19.
//  Modified by Loren Atwood on 02/18/20.
//

#pragma once

#include "util/ParseInterface.h"
#include <string>
#include <cmath>

class PI_CollectionParameters : public ParseInterface
{

private:
public:
  int nBoxesX{}, nBoxesY{}, nBoxesZ{};
  float boxBoundsX1{}, boxBoundsY1{}, boxBoundsZ1{};
  float boxBoundsX2{}, boxBoundsY2{}, boxBoundsZ2{};
  float averagingStartTime{};// time to start concentration averaging, not the time to start output.
  float averagingPeriod{};// time averaging frequency and output frequency

  void parseValues() override
  {
    parsePrimitive<float>(true, averagingStartTime, "timeAvgStart");
    parsePrimitive<float>(true, averagingPeriod, "timeAvgFreq");
    parsePrimitive<float>(true, boxBoundsX1, "boxBoundsX1");
    parsePrimitive<float>(true, boxBoundsY1, "boxBoundsY1");
    parsePrimitive<float>(true, boxBoundsZ1, "boxBoundsZ1");
    parsePrimitive<float>(true, boxBoundsX2, "boxBoundsX2");
    parsePrimitive<float>(true, boxBoundsY2, "boxBoundsY2");
    parsePrimitive<float>(true, boxBoundsZ2, "boxBoundsZ2");
    parsePrimitive<int>(true, nBoxesX, "nBoxesX");
    parsePrimitive<int>(true, nBoxesY, "nBoxesY");
    parsePrimitive<int>(true, nBoxesZ, "nBoxesZ");

    // check some of the parsed values to see if they make sense
    checkParsedValues();
  }

  void checkParsedValues()
  {
    // make sure that all variables are greater than 0 except where they need to be at least 0
    if (averagingStartTime < 0) {
      std::cerr << "(CollectionParameters::checkParsedValues): input averagingStartTime must be greater than or equal to zero!";
      std::cerr << " averagingStartTime = \"" << averagingStartTime << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (averagingPeriod <= 0) {
      std::cerr << "(CollectionParameters::checkParsedValues): input averagingPeriod must be greater than zero!";
      std::cerr << " averagingPeriod = \"" << averagingPeriod << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (boxBoundsX1 < 0) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsX1 must be zero or greater!";
      std::cerr << " boxBoundsX1 = \"" << boxBoundsX1 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (boxBoundsY1 < 0) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsY1 must be zero or greater!";
      std::cerr << " boxBoundsY1 = \"" << boxBoundsY1 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (boxBoundsZ1 < 0) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsZ1 must be zero or greater!";
      std::cerr << " boxBoundsZ1 = \"" << boxBoundsZ1 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (boxBoundsX2 < 0) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsX2 must be zero or greater!";
      std::cerr << " boxBoundsX2 = \"" << boxBoundsX2 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (boxBoundsY2 < 0) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsY2 must be zero or greater!";
      std::cerr << " boxBoundsY2 = \"" << boxBoundsY2 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (boxBoundsZ2 < 0) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsZ2 must be zero or greater!";
      std::cerr << " boxBoundsZ2 = \"" << boxBoundsZ2 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (nBoxesX < 1) {
      std::cerr << "(CollectionParameters::checkParsedValues): input nBoxesX must be one or greater!";
      std::cerr << " nBoxesX = \"" << nBoxesX << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (nBoxesY < 1) {
      std::cerr << "(CollectionParameters::checkParsedValues): input nBoxesY must be one or greater!";
      std::cerr << " nBoxesY = \"" << nBoxesY << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (nBoxesZ < 1) {
      std::cerr << "(CollectionParameters::checkParsedValues): input nBoxesZ must be one or greater!";
      std::cerr << " nBoxesZ = \"" << nBoxesZ << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }

    // make sure the boxBounds1 is not greater than the boxBounds2 for each dimension
    if (boxBoundsX1 > boxBoundsX2) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsX1 must be smaller than or equal to input boxBoundsX2!";
      std::cerr << " boxBoundsX1 = \"" << boxBoundsX1 << "\", boxBoundsX2 = \"" << boxBoundsX2 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (boxBoundsY1 > boxBoundsY2) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsY1 must be smaller than or equal to input boxBoundsY2!";
      std::cerr << " boxBoundsY1 = \"" << boxBoundsY1 << "\", boxBoundsY2 = \"" << boxBoundsY2 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (boxBoundsZ1 > boxBoundsZ2) {
      std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsZ1 must be smaller than or equal to input boxBoundsZ2!";
      std::cerr << " boxBoundsZ1 = \"" << boxBoundsZ1 << "\", boxBoundsZ2 = \"" << boxBoundsZ2 << "\"" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
};
