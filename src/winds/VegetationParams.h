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


#pragma once

#include "util/ParseInterface.h"
#include "util/ESRIShapefile.h"

#include "Building.h"

#include "CanopyElement.h"
#include "CanopyHomogeneous.h"
#include "CanopyIsolatedTree.h"
#include "CanopyWindbreak.h"
#include "CanopyROC.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


class VegetationParams : public ParseInterface
{
private:
public:
  std::vector<Building *> canopies; /**< :document this: */

  int wakeFlag = 1; /**< :document this: */


  // SHP File parameters
  std::string shpFile; /**< :document this: */
  std::string shpTreeLayerName; /**< :document this: */
  ESRIShapefile *SHPData = nullptr; /**< :document this: */

  std::vector<std::vector<polyVert>> shpPolygons; /**< :document this: */
  std::map<std::string, std::vector<float>> shpFeatures; /**< :document this: */

  virtual void parseValues()
  {
    parsePrimitive<int>(false, wakeFlag, "wakeFlag");

    // read the input data for canopies
    parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyHomogeneous>("Homogeneous"));
    parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyIsolatedTree>("IsolatedTree"));
    parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyWindbreak>("Windbreak"));
    parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyROC>("ROC"));

    // add other type of canopy here

    shpFile = "";
    parsePrimitive<std::string>(false, shpFile, "SHPFile");

    shpTreeLayerName = "trees";// defaults
    parsePrimitive<std::string>(false, shpTreeLayerName, "SHPTreeLayer");

    //
    // Process ESRIShapeFile here, but leave extraction of poly
    // building for later in WINDSGeneralData
    //
    SHPData = nullptr;
    if (shpFile != "") {
      shpFile = QESfs::get_absolute_path(shpFile);
      // Read polygon node coordinates and building height from shapefile
      //SHPData = new ESRIShapefile(shpFile, shpTreeLayerName, shpPolygons, shpFeatures);
      SHPData = new ESRIShapefile(shpFile, shpTreeLayerName);
      // std::cout << shpPolygons.size() << " " << shpTreeHeight.size() << std::endl;
    }
  }
};
