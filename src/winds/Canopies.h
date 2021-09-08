#pragma once

#include "util/ParseInterface.h"
#include "Building.h"

#include "CanopyElement.h"
#include "CanopyHomogeneous.h"
#include "CanopyIsolatedTree.h"
#include "CanopyWindbreak.h"
#include "CanopyVineyard.h"
#include "ESRIShapefile.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"


class Canopies : public ParseInterface
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
    parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyVineyard>("Vineyard"));

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
      // Read polygon node coordinates and building height from shapefile
      //SHPData = new ESRIShapefile(shpFile, shpTreeLayerName, shpPolygons, shpFeatures);
      SHPData = new ESRIShapefile(shpFile, shpTreeLayerName);
      // std::cout << shpPolygons.size() << " " << shpTreeHeight.size() << std::endl;
    }
  }
};
