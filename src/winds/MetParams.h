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

/** @file MetParams.h */

#pragma once

/*
 * This class is a container relating to sensors and metric
 * information read from the xml.
 */
#include <algorithm>

#include "util/ParseInterface.h"
#include "util/ParseException.h"
#include "Sensor.h"

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <string>

namespace pt = boost::property_tree;

/**
 * @class MetParams
 *
 * Container relating to sensors and metric information
 * read from the xml.
 *
 * @sa Sensor
 * @sa ParseInterface
 */
class MetParams : public ParseInterface
{
private:
public:
  int z0_domain_flag = 0; /**< :document this: */
  std::vector<Sensor *> sensors; /**< :document this: */

  std::vector<std::string> sensorName; /**< :document this: */


  /**
   * :document this:
   */
  virtual void parseValues()
  {
    parsePrimitive<int>(false, z0_domain_flag, "z0_domain_flag");
    parseMultiElements<Sensor>(false, sensors, "sensor");

    parseMultiPrimitives<std::string>(false, sensorName, "sensorName");
    for (auto i = 0u; i < sensorName.size(); i++) {
      sensorName[i] = QESfs::get_absolute_path(sensorName[i]);
    }
  }
};
