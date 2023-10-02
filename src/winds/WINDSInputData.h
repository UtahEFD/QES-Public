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
 ***************************************************************************/

/** @file WINDSInputData.h */

#pragma once

/*
 * A collection of data read from an XML. This contains
 * all root level information extracted from the xml.
 */

#include "util/ParseInterface.h"
#include "util/QESFileSystemHandler.h"

#include "SimulationParameters.h"
#include "FileOptions.h"
#include "MetParams.h"
#include "BuildingsParams.h"
#include "VegetationParams.h"

#include "fire/Fires.hpp"

#include "TURBParams.h"

/**
 * @class WINDSInputData
 * @brief Collection of data read from an XML.
 *
 * Contains all root level information extracted from the xml
 *
 * @sa ParseInterface
 */
class WINDSInputData : public ParseInterface
{
public:
  SimulationParameters *simParams = nullptr; /**< :document this: */
  FileOptions *fileOptions = nullptr; /**< :document this: */
  MetParams *metParams = nullptr; /**< :document this: */
  TURBParams *turbParams = nullptr; /**< :document this: */
  BuildingsParams *buildingsParams = nullptr; /**< :document this: */
  VegetationParams *vegetationParams = nullptr; /**< :document this: */
  Fires *fires = nullptr; /**< :document this: */

  WINDSInputData()
  {
    fileOptions = 0;
    metParams = 0;
    turbParams = 0;
    buildingsParams = 0;
    vegetationParams = 0;
    fires = 0;
  }

  WINDSInputData(const std::string fileName)
  {

    fileOptions = 0;
    metParams = 0;
    turbParams = 0;
    buildingsParams = 0;
    vegetationParams = 0;
    fires = 0;

    QESfs::set_file_path(fileName);
    // read and parse the XML
    parseXML(fileName, "QESWindsParameters");
  }

  /**
   * :document this:
   */
  virtual void parseValues()
  {
    parseElement<SimulationParameters>(true, simParams, "simulationParameters");
    parseElement<FileOptions>(false, fileOptions, "fileOptions");
    parseElement<MetParams>(false, metParams, "metParams");
    parseElement<TURBParams>(false, turbParams, "turbParams");
    parseElement<BuildingsParams>(false, buildingsParams, "buildingsParams");
    parseElement<VegetationParams>(false, vegetationParams, "vegetationParams");
    parseElement<Fires>(false, fires, "fires");
  }
};
