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

/** @file WINDSGeneralData.h */

#pragma once

#include <vector>
#include <netcdf>
#include <cmath>

#include "WindProfilerType.h"

class WindProfilerSensorType : public WindProfilerType
{
protected:
  std::vector<float> u_prof;
  std::vector<float> v_prof;
  std::vector<int> available_sensor_id;
  std::vector<int> site_id;

  float asl_percent = 0.05;
  float abl_height = 200;

  void sensorsProfiles(const WINDSInputData *WID, WINDSGeneralData *WGD);

  void singleSensorInterpolation(WINDSGeneralData *WGD);

public:
  WindProfilerSensorType()
  {}
  virtual ~WindProfilerSensorType()
  {}

  virtual void interpolateWindProfile(const WINDSInputData *, WINDSGeneralData *) = 0;
};
