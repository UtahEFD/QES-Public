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

/** @file Buildings.h */

#pragma once

#include "util/ParseInterface.h"
#include "util/ESRIShapefile.h"

#include "Building.h"
#include "RectangularBuilding.h"
#include "PolygonQUICBuilding.h"
#include "PolyBuilding.h"

/**
 * @class Buildings
 * @brief Contains data and variables that pertain to all buildings
 * along with a list of all buildings pulled from an input xml file.
 */
class Buildings : public ParseInterface
{
private:
public:
  int numBuildings = 0; /**< number of Building objects */
  int numPolygonNodes = 0; /**< number of polygon nodes */
  std::vector<Building *> buildings; /**< vector containing Building objects */
  float wallRoughness = 0.001; /**< wall roughness metric */
  int rooftopFlag = 1; /**< :Rooftop flag (0-none, 1-log profile (default), 2-vortex): */
  int upwindCavityFlag = 2; /**< :Upwind cavity flag (0-none, 1-Rockle, 2-MVP (default), 3-HMVP): */
  int streetCanyonFlag = 1; /**< :Street canyon flag (0-none, 1-Roeckle w/ Fackrel (default)): */
  int streetIntersectionFlag = 0; /**< :Street intersection flag (0-off, 1-on): */
  int wakeFlag = 2; /**< :Wake flag (0-none, 1-Rockle, 2-Modified Rockle (default), 3-Area Scaled): */
  int sidewallFlag = 1; /**< :Sidewall flag (0-off, 1-on (default)): */

  // SHP File parameters
  std::string shpFile; /**< SHP file name */
  std::string shpBuildingLayerName; /**< :document this: */
  std::string shpHeightField;
  ESRIShapefile *SHPData = nullptr; /**< :document this: */
  float heightFactor = 1.0; /**< :document this: */

  virtual void parseValues()
  {
    buildings.clear();

    parsePrimitive<float>(true, wallRoughness, "wallRoughness");
    parsePrimitive<int>(true, rooftopFlag, "rooftopFlag");
    parsePrimitive<int>(true, upwindCavityFlag, "upwindCavityFlag");
    parsePrimitive<int>(true, streetCanyonFlag, "streetCanyonFlag");
    parsePrimitive<int>(true, streetIntersectionFlag, "streetIntersectionFlag");
    parsePrimitive<int>(true, wakeFlag, "wakeFlag");
    parsePrimitive<int>(true, sidewallFlag, "sidewallFlag");

    parsePrimitive<int>(false, numBuildings, "numBuildings");
    parsePrimitive<int>(false, numPolygonNodes, "numPolygonNodes");
    parseMultiPolymorphs(false, buildings, Polymorph<Building, RectangularBuilding>("rectangularBuilding"));
    parseMultiPolymorphs(false, buildings, Polymorph<Building, PolygonQUICBuilding>("QUICBuilding"));
    parsePrimitive<float>(false, heightFactor, "heightFactor");

    shpFile = "";
    parsePrimitive<std::string>(false, shpFile, "SHPFile");
    shpBuildingLayerName = "buildings";// defaults
    parsePrimitive<std::string>(false, shpBuildingLayerName, "SHPBuildingLayer");
    shpHeightField = "H";// default;
    parsePrimitive<std::string>(false, shpHeightField, "SHPHeightField");

    SHPData = nullptr;
    if (shpFile != "") {
      shpFile = QESfs::get_absolute_path(shpFile);
      // Read polygon node coordinates and building height from shapefile
      //SHPData = new ESRIShapefile(shpFile, shpTreeLayerName, shpPolygons, shpFeatures);
      SHPData = new ESRIShapefile(shpFile, shpBuildingLayerName);
      // std::cout << shpPolygons.size() << " " << shpTreeHeight.size() << std::endl;
    }
  }
};
