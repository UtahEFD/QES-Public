/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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

/** @file FileOptions.h */

#pragma once

#include "util/ParseInterface.h"
#include <string>
#include <vector>

/**
 * @class FileOptions
 *
 * This class contains data and variables that set flags and
 * settngs read from the xml.
 */
class FileOptions : public ParseInterface
{
private:
public:
  int outputFlag;
  std::vector<std::string> outputFields;
  bool massConservedFlag;
  bool sensorVelocityFlag;
  bool staggerdVelocityFlag;

  FileOptions()
  {
    outputFlag = 0;
    massConservedFlag = false;
    sensorVelocityFlag = false;
    staggerdVelocityFlag = false;
  }

  virtual void parseValues()
  {
    parsePrimitive<int>(true, outputFlag, "outputFlag");
    parseMultiPrimitives<std::string>(false, outputFields, "outputFields");
    parsePrimitive<bool>(false, massConservedFlag, "massConservedFlag");
    parsePrimitive<bool>(false, sensorVelocityFlag, "sensorVelocityFlag");
    parsePrimitive<bool>(false, staggerdVelocityFlag, "staggerdVelocityFlag");
  }
};
