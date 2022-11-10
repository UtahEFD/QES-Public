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

/** @file LocalMixingSerial.h */

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

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class WINDSInputData;
class WINDSGeneralData;

/**
 * @class LocalMixingSerial
 * @brief :document this:
 * @sa LocalMixing
 */
class LocalMixingSerial : public LocalMixing
{
private:
  std::vector<int> wall_right_indices, wall_left_indices;
  std::vector<int> wall_back_indices, wall_front_indices;
  std::vector<int> wall_below_indices, wall_above_indices;

  // grid information
  std::vector<float> x_fc, x_cc;
  std::vector<float> y_fc, y_cc;
  std::vector<float> z_fc, z_cc;

  /**
   * This function propagate the distance in fuild cell form
   * the wall for the each solid element.
   *
   * @note This method is relatively inefficient and should be used only wiht small domains.
   * @warning This is a serial ONLY method.
   */
  void getMinDistWall(WINDSGeneralData *, int);

protected:
public:
  LocalMixingSerial()
  {}
  ~LocalMixingSerial()
  {}

  /**
   * Defines the mixing length with the serial
   * method (CANNOT be parallelized).
   */
  void defineMixingLength(const WINDSInputData *, WINDSGeneralData *);
};
