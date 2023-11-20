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

/** @file SourceFullDomain.hpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once

#include "PlumeParameters.hpp"
#include "CollectionParameters.hpp"
#include "ParticleOutputParameters.hpp"
#include "PI_ParticleParameters.h"
#include "SourceParameters.hpp"
#include "BoundaryConditions.hpp"

#include "util/ParseInterface.h"

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class PlumeInputData : public ParseInterface
{

public:
  PlumeParameters *plumeParams = nullptr;
  CollectionParameters *colParams = nullptr;
  ParticleOutputParameters *partOutParams = nullptr;
  SourceParameters *sourceParams = nullptr;
  PI_ParticleParameters *particleParams = nullptr;
  BoundaryConditions *BCs = nullptr;


  PlumeInputData()
  {
    plumeParams = 0;
    colParams = 0;
    partOutParams = 0;
    sourceParams = 0;
    particleParams = 0;
  }

  PlumeInputData(const std::string fileName)
  {
    plumeParams = 0;
    colParams = 0;
    partOutParams = 0;
    sourceParams = 0;
    particleParams = 0;

    // read and parse the XML
    parseXML(fileName, "QESPlumeParameters");
  }

  virtual void parseValues()
  {
    parseElement<PlumeParameters>(true, plumeParams, "plumeParameters");
    parseElement<CollectionParameters>(true, colParams, "collectionParameters");
    parseElement<ParticleOutputParameters>(false, partOutParams, "particleOutputParameters");
    parseElement<SourceParameters>(false, sourceParams, "sourceParameters");
    parseElement<PI_ParticleParameters>(false, particleParams, "particleParameters");
    parseElement<BoundaryConditions>(true, BCs, "boundaryConditions");
  }
};
