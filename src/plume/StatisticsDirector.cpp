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

/** @file StatisticsDirector.h
 * @brief
 */

#include "StatisticsDirector.h"

#include "PlumeInputData.hpp"
#include "PLUMEGeneralData.h"

StatisticsDirector::StatisticsDirector(const PlumeInputData *PID, PLUMEGeneralData *PGD)
{
  averagingStartTime = PGD->getSimTimeStart() + PID->colParams->averagingStartTime;
  averagingPeriod = PID->colParams->averagingPeriod;
  nextOutputTime = averagingStartTime + averagingPeriod;
}

void StatisticsDirector::attach(const std::string &key, StatisticsInterface *s)
{
  if (elements.find(key) == elements.end()) {
    elements.insert({ key, s });
  } else {
    exit(1);
  }
}

void StatisticsDirector::compute(QEStime &timeIn, const float &timeStep)
{
  // reset buffer if needed
  // Note: buffers need to be persistent for output.
  if (need_reset) {
    for (const auto &e : elements) {
      e.second->reset();
    }
    need_reset = false;
  }

  if (timeIn > averagingStartTime) {
    // incrementation of the averaging time
    ongoingAveragingTime += timeStep;

    for (const auto &e : elements) {
      e.second->collect(timeIn, timeStep);
    }

    if (timeIn >= nextOutputTime) {
      // compute the stats
      for (const auto &e : elements) {
        e.second->finalize(timeIn);
      }

      // reset variables
      need_reset = true;
      // for output (need to be refined)
      need_output = true;

      // set nest output time
      nextOutputTime = nextOutputTime + averagingPeriod;
    }
  }
}