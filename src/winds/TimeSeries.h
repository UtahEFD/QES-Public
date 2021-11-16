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

/** @file TimeSeries.h */

#pragma once

#include "boost/date_time/posix_time/posix_time.hpp"
#include "util/ParseInterface.h"

class URBInputData;
class URBGeneralData;

namespace bt = boost::posix_time;

/**
 * @class TimeSeries
 * @brief :document this:
 *
 * @sa ParseInterface
 */
class TimeSeries : public ParseInterface
{
private:
public:
  int site_blayer_flag = 1; /**< :document this: */
  float site_z0; /**< :document this: */
  float site_p = 0.0; /**< :document this: */

  ///@{
  /** :document this: */
  std::vector<float> site_wind_dir, site_z_ref, site_U_ref;
  ///@}

  float site_one_overL; /**< :document this: */
  float site_canopy_H, site_atten_coeff; /**< :document this: */

  std::string timeStamp = ""; /**< :document this: */
  time_t timeEpoch = -1; /**< :document this: */
  bt::ptime timePosix; /**< :document this: */


  /**
   * :document this:
   */
  virtual void parseValues()
  {
    parsePrimitive<std::string>(false, timeStamp, "timeStamp");
    parsePrimitive<time_t>(false, timeEpoch, "timeEpoch");
    parsePrimitive<int>(false, site_blayer_flag, "boundaryLayerFlag");
    parsePrimitive<float>(true, site_z0, "siteZ0");
    parsePrimitive<float>(false, site_p, "siteP");
    parsePrimitive<float>(true, site_one_overL, "reciprocal");
    parseMultiPrimitives<float>(true, site_z_ref, "height");
    parseMultiPrimitives<float>(true, site_U_ref, "speed");
    parseMultiPrimitives<float>(true, site_wind_dir, "direction");
    parsePrimitive<float>(false, site_canopy_H, "canopyHeight");
    parsePrimitive<float>(false, site_atten_coeff, "attenuationCoefficient");


    if (timeStamp == "" && timeEpoch == -1) {
      std::cout << "[WARNING] no timestamp provided" << std::endl;
      timeStamp = "2020-01-01T00:00";
      timePosix = bt::from_iso_extended_string(timeStamp);
      timeEpoch = bt::to_time_t(timePosix);
    } else if (timeStamp != "" && timeEpoch == -1) {
      timePosix = bt::from_iso_extended_string(timeStamp);
      timeEpoch = bt::to_time_t(timePosix);
    } else if (timeEpoch != -1 && timeStamp == "") {
      timePosix = bt::from_time_t(timeEpoch);
      timeStamp = bt::to_iso_extended_string(timePosix);
    } else {
      timePosix = bt::from_iso_extended_string(timeStamp);
      bt::ptime testtime = bt::from_time_t(timeEpoch);
      if (testtime != timePosix) {
        std::cerr << "[ERROR] invalid timeStamp (timeEpoch != timeStamp)\n";
        exit(EXIT_FAILURE);
      }
    }
  }
};
