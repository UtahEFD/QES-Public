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

/** @file ParticleOutputParameters.hpp
 * @brief This class contains data and variables that set flags and
 * settngs read from the xml.
 *
 * @note Child of ParseInterface
 * @sa ParseInterface
 */

#pragma once


#include "util/ParseInterface.h"
#include "util/QESFileSystemHandler.h"

#include <string>
#include <netcdf>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;


class PI_ParticleOutputParameters : public ParseInterface
{
private:
public:
  float outputStartTime = -1.0;
  float outputEndTime = -1.0;
  float outputFrequency;
  std::vector<std::string> outputFields;

  /*int numSources;// number of sources, you fill in source information for each source next
  std::vector<ParseSource *> sources;// source type and the collection of all the different sources from input
  std::string HRRRFile; /**< HRRR file name 
  std::vector<std::string> inputFields; /**< HRRR input fields */

  virtual void parseValues()
  {
    parsePrimitive<float>(false, outputStartTime, "outputStartTime");
    parsePrimitive<float>(false, outputEndTime, "outputEndTime");
    parsePrimitive<float>(true, outputFrequency, "outputFrequency");
    parseMultiPrimitives<std::string>(false, outputFields, "outputFields");
    
    /* parsePrimitive<int>(false, numSources, "numSources");
    parseMultiElements(false, sources, "source");
    HRRRFile = "";
    parsePrimitive<std::string>(false, HRRRFile, "HRRRFile");
    std::cout << HRRRFile <<std::endl;
    std::cout << HRRRFile <<std::endl;
    parseMultiPrimitives<std::string>(false, inputFields, "inputFields");*/
  }
  
};
