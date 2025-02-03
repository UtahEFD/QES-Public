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
 * @file FuelProperties.hpp
 * @brief Parses ignition points from XML
**/
#include "util/ParseInterface.h"

class ignition : public ParseInterface
{
private:
public:
  float xStart, yStart, length, width, baseHeight, height;

  virtual void parseValues()
  {
    parsePrimitive<float>(true, height, "height");
    parsePrimitive<float>(true, baseHeight, "baseHeight");
    parsePrimitive<float>(true, xStart, "xStart");
    parsePrimitive<float>(true, yStart, "yStart");
    parsePrimitive<float>(true, length, "length");
    parsePrimitive<float>(true, width, "width");
  }
};
