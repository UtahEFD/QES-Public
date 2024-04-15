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

StatisticsDirector::StatisticsDirector(const PlumeInputData *PID, PLUMEGeneralData *PGD, QESFileOutput_v2 *outfile)
{
  m_startCollectionTime = PGD->getSimTimeStart() + PID->colParams->averagingStartTime;
  m_collectionPeriod = PID->colParams->averagingPeriod;
  m_nextOutputTime = m_startCollectionTime + m_collectionPeriod;
  m_statsFile = outfile;
  m_statsFile->setStartTime(PGD->getSimTimeStart());
}

void StatisticsDirector::attach(const std::string &key, DataSource *s)
{
  if (elements.find(key) == elements.end()) {
    elements.insert({ key, s });
    m_statsFile->attachDataSource(s);
  } else {
    std::cerr << "[!!!ERROR!!!]\tstat with key = " << key << " already exists" << std::endl;
    exit(1);
  }
}

void StatisticsDirector::compute(QEStime &timeIn, const float &timeStep)
{
  if (timeIn > m_startCollectionTime) {

    for (const auto &e : elements) {
      e.second->collect(timeIn, timeStep);
    }

    if (timeIn >= m_nextOutputTime) {
      // compute the stats

      m_statsFile->newTimeEntry(timeIn);
      for (const auto &e : elements) {
        e.second->prepareDataAndPushToFile(timeIn);
      }

      // set nest output time
      m_nextOutputTime = m_nextOutputTime + m_collectionPeriod;
    }
  }
}