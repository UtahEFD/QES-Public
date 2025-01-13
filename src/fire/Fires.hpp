/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file Fires.hpp
 * @brief This class contains variables that define information necessary for running the fire code.
 */
#ifndef FIRES_HPP
#define FIRES_HPP

#include <string>
#include "util/ParseInterface.h"
#include "util/ParseVector.h"
#include "util/Vector3.h"
#include "util/Vector3Int.h"
#include "winds/DTEHeightField.h"
#include "ignition.h"


class Fires : public ParseInterface
{
private:
public:
  int numFires, fuelType, fieldFlag;
  float fireDur, fmc, courant, cure;
  std::vector<ignition *> IG;
  std::string fuelFile;
  virtual void parseValues()
  {
    parsePrimitive<float>(true, fireDur, "fireDur");
    parsePrimitive<int>(false, numFires, "numFires");
    parsePrimitive<int>(true, fuelType, "fuelType");
    parsePrimitive<float>(true, fmc, "fmc");
    parsePrimitive<float>(true, courant, "courant");
    parseMultiElements<ignition>(false, IG, "ignition");
    parsePrimitive<int>(true, fieldFlag, "fieldFlag");
    parsePrimitive<float>(true, cure, "fmc");
    fuelFile = "";
    parsePrimitive<std::string>(false, fuelFile, "fuelMap");
    if (fuelFile != ""){
      fuelFile = QESfs::get_absolute_path(fuelFile);
    }
  }


  void parseTree(pt::ptree t)
  {
    setTree(t);
    setParents("root");
    parseValues();

  }
};
#endif
