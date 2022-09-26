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

/** @file GIStool.h */

#pragma once

#include <math.h>
#include <algorithm>

class GIStool
{
public:
  /**
   * Converts UTM to lat/lon and vice versa of the sensor coordiantes.
   *
   * @param rlon :document this:
   * @param rlat :document this:
   * @param rx :document this:
   * @param ry :document this:
   * @param UTM_PROJECTION_ZONE :document this:
   * @param iway :document this:
   */
  static void UTMConverter(float &rlon, float &rlat, float &rx, float &ry, int &UTM_PROJECTION_ZONE, int iway);

  /**
   * Calculates the convergence value based on lat/lon input.
   *
   * @param lon :document this:
   * @param lat :document this:
   * @param site_UTM_zone :document this:
   * @param convergense :document this:
   */
  static void getConvergence(float &lon, float &lat, int &site_UTM_zone, float &convergence);

private:
  GIStool() {}
};
