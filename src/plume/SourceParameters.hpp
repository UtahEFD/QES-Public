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

/** @file Sources.hpp
 * @brief This class contains data and variables that set flags and
 * settngs read from the xml.
 *
 * @note Child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include "Source.hpp"
#include "SourceGeometry_Cube.hpp"
#include "SourceGeometry_FullDomain.hpp"
#include "SourceGeometry_Line.hpp"
#include "SourceGeometry_Point.hpp"
#include "SourceGeometry_SphereShell.hpp"

#include "util/ParseInterface.h"


class SourceParameters : public ParseInterface
{
private:
public:
  int numSources;// number of sources, you fill in source information for each source next
  std::vector<ParseSource *> sources;// source type and the collection of all the different sources from input

  virtual void parseValues()
  {
    parsePrimitive<int>(false, numSources, "numSources");
    parseMultiElements(false, sources, "source");
  }
};
