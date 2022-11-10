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

#include "SourceType.hpp"
#include "SourcePoint.hpp"
#include "SourceLine.hpp"
#include "SourceCircle.hpp"
#include "SourceCube.hpp"
#include "SourceFullDomain.hpp"

#include "util/ParseInterface.h"


class Sources : public ParseInterface
{
private:
public:
  int numSources;// number of sources, you fill in source information for each source next
  std::vector<SourceType *> sources;// source type and the collection of all the different sources from input

  virtual void parseValues()
  {
    parsePrimitive<int>(true, numSources, "numSources");
    parseMultiPolymorphs(false, sources, Polymorph<SourceType, SourcePoint>("SourcePoint"));
    parseMultiPolymorphs(false, sources, Polymorph<SourceType, SourceLine>("SourceLine"));
    parseMultiPolymorphs(false, sources, Polymorph<SourceType, SourceCircle>("SourceCircle"));
    parseMultiPolymorphs(false, sources, Polymorph<SourceType, SourceCube>("SourceCube"));
    parseMultiPolymorphs(false, sources, Polymorph<SourceType, SourceFullDomain>("SourceFullDomain"));
  }
};
