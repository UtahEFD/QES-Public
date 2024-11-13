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
 * @file Smoke.h
 * @brief This class specifies smoke sources for QES-Fire and QES-Plume integration
 */

#ifndef SMOKE_H
#define SMOKE_H

#include "Fire.h"
#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "util/Vector3.h"
#include "util/Vector3Int.h"
#include "winds/DTEHeightField.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include "SourceFire.h"
#include "plume/Plume.hpp"

using namespace std;
class Fire;
class Plume;
class Smoke
{
public:
  Smoke();
  void genSmoke(WINDSGeneralData *, Fire *, Plume *);

  
  void source();

 private:
  int nx,ny,nz;
  float dx,dy,dz;
  float x_pos,y_pos,z_pos;
  float ppt;
};

#endif

