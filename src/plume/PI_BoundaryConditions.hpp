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

/** @file BoundaryCondition.hpp
 * @brief
 *
 * @note Child of ArgumentParsing
 * @sa ArgumentParsing
 */

#pragma once

#include <string>

#include "util/ParseInterface.h"


class PI_BoundaryConditions : public ParseInterface
{

private:
public:
  // current possible BCtypes are:
  // "exiting", "periodic", "reflection"

  std::string xBCtype;
  std::string yBCtype;
  std::string zBCtype;

  // possible reflection methods:
  /*
   * "doNothing" (default)   - nothing happen when particle enter wall
   * "setInactive"           - particle is set to inactive when entering a wall
   * "stairstepReflection"   - particle use full stair step reflection when entering a wall
   * "meshReflection"        - particle use triangular mesh for reflection
   */

  std::string wallReflection;

  virtual void parseValues()
  {
    parsePrimitive<std::string>(true, xBCtype, "xBCtype");
    parsePrimitive<std::string>(true, yBCtype, "yBCtype");
    parsePrimitive<std::string>(true, zBCtype, "zBCtype");


    wallReflection = "";
    parsePrimitive<std::string>(false, wallReflection, "wallReflection");

    if (wallReflection.empty()) {
      wallReflection = "doNothing";
    }
  }
};
