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
  int numBuildings; /**< number of Building objects */
  int numPolygonNodes; /**< number of polygon nodes */
  std::vector<Building *> buildings; /**< vector containing Building objects */
  float wallRoughness; /**< wall roughness metric */

  // SHP File parameters
  std::string shpFile; /**< SHP file name */
  std::string shpBuildingLayerName; /**< :document this: */
  std::string shpHeightField;
  ESRIShapefile *SHPData = nullptr; /**< :document this: */
  float heightFactor = 1.0; /**< :document this: */

  virtual void parseValues()
  {
    parsePrimitive<int>(false, numBuildings, "numBuildings");
    parsePrimitive<int>(false, numPolygonNodes, "numPolygonNodes");
    parseMultiPolymorphs(false, buildings, Polymorph<Building, RectangularBuilding>("rectangularBuilding"));
    parseMultiPolymorphs(false, buildings, Polymorph<Building, PolygonQUICBuilding>("QUICBuilding"));
    parsePrimitive<float>(true, wallRoughness, "wallRoughness");
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
