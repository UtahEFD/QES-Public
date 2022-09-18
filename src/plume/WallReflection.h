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

/** @file WallReflection.h
 * @brief
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <cstring>

#include "util/QEStime.h"
#include "util/calcTime.h"
#include "util/Vector3.h"
//#include "Matrix3.h"
#include "Random.h"

#include "util/QESNetCDFOutput.h"
#include "PlumeOutput.h"
#include "PlumeOutputParticleData.h"

#include "PlumeInputData.hpp"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

#include "Interp.h"
#include "InterpNearestCell.h"
#include "InterpPowerLaw.h"
#include "InterpTriLinear.h"

class Plume;

class WallReflection
{
public:
  WallReflection()
  {}
  ~WallReflection()
  {}
  virtual bool reflect(const WINDSGeneralData *,
                       const Plume *,
                       double &,
                       double &,
                       double &,
                       double &,
                       double &,
                       double &,
                       double &,
                       double &,
                       double &) = 0;
};

class WallReflection_DoNothing : public WallReflection
{
public:
  WallReflection_DoNothing()
  {}
  ~WallReflection_DoNothing()
  {}

  virtual bool reflect(const WINDSGeneralData *WGD,
                       const Plume *plume,
                       double &xPos,
                       double &yPos,
                       double &zPos,
                       double &disX,
                       double &disY,
                       double &disZ,
                       double &uFluct,
                       double &vFluct,
                       double &wFluct)
  {
    return true;
  }
};

class WallReflection_SetToInactive : public WallReflection
{
public:
  WallReflection_SetToInactive()
  {}
  ~WallReflection_SetToInactive()
  {}

  virtual bool reflect(const WINDSGeneralData *WGD,
                       const Plume *plume,
                       double &xPos,
                       double &yPos,
                       double &zPos,
                       double &disX,
                       double &disY,
                       double &disZ,
                       double &uFluct,
                       double &vFluct,
                       double &wFluct);
};
