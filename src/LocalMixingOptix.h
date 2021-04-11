/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

/** @file LocalMixingOptix.h */

#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <limits>

#include "LocalMixing.h"
#include "NetCDFInput.h"

#include "Mesh.h"

#ifdef HAS_OPTIX
#include "OptixRayTrace.h"
#endif

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class WINDSInputData;
class WINDSGeneralData;

/**
 * @class LocalMixingOptix
 * @brief :document this:
 * @sa LocalMixing
 */
class LocalMixingOptix : public LocalMixing
{
private:

protected:

public:

    LocalMixingOptix()
    {}
    ~LocalMixingOptix()
    {}

    /**
     * Defines the mixing length as the height above the ground.
     */
    void defineMixingLength(const WINDSInputData*,WINDSGeneralData*);

};
