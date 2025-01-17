/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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

StatisticsDirector::StatisticsDirector(const QEStime &startCollectionTime,
                                       const float &collectionPeriod,
                                       QESFileOutput_Interface *outfile)
  : m_startCollectionTime(startCollectionTime), m_collectionPeriod(collectionPeriod),
    m_statsFile(outfile)
{
  m_nextOutputTime = m_startCollectionTime + m_collectionPeriod;
}

StatisticsDirector::~StatisticsDirector()
{
  for (const auto &ds : dataSources) {
    delete ds.second;
  }
  delete m_statsFile;
}

void StatisticsDirector::attach(const std::string &key, DataSource *s)
{
  if (dataSources.find(key) == dataSources.end()) {
    dataSources.insert({ key, s });
    m_statsFile->attachDataSource(s);
  } else {
    std::cerr << "[!!!ERROR!!!]\tstat with key = " << key << " already exists" << std::endl;
    exit(1);
  }
}

void StatisticsDirector::compute(QEStime &timeIn, const float &timeStep)
{
  if (timeIn > m_startCollectionTime) {

    for (const auto &ds : dataSources) {
      ds.second->collect(timeIn, timeStep);
    }

    if (timeIn >= m_nextOutputTime) {
      // compute the stats

      m_statsFile->newTimeEntry(timeIn);
      for (const auto &ds : dataSources) {
        ds.second->prepareDataAndPushToFile(timeIn);
      }

      // set nest output time
      m_nextOutputTime = m_nextOutputTime + m_collectionPeriod;
    }
  }
}