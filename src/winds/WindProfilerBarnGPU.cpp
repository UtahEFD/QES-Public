/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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

/**
 * @file Sensor.cpp
 * @brief Collection of variables containing information relevant to
 * sensors read from an xml.
 *
 * @sa ParseInterface
 * @sa TimeSeries
 */

#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <limits>

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "WindProfilerBarnGPU.h"


void WindProfilerBarnGPU::interpolateWindProfile(const WINDSInputData *WID, WINDSGeneralData *WGD)
{

  sensorsProfiles(WID, WGD);
  int num_sites = available_sensor_id.size();

  if (num_sites == 1) {
    singleSensorInterpolation(WGD);
  } else {
    // If number of sites are more than one
    // Apply 2D Barnes scheme to interpolate site velocity profiles to the whole domain
    auto startBarnesGPU = std::chrono::high_resolution_clock::now();
    BarnesInterpolationGPU(WID, WGD);
    auto finishBarnesGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsedBarnesGPU = finishBarnesGPU - startBarnesGPU;
    // std::cout << "Elapsed time for Barnes interpolation on GPU: " << elapsedBarnesGPU.count() << " s\n";
  }
  return;
}
