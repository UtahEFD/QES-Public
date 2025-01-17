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

#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include <cstring>

#include "util/QEStime.h"
#include "util/QESOutputInterface.h"

#include "util/QESFileOutput_v2.h"
#include "util/DataSource.h"
#include "util/QESNetCDFOutput_v2.h"

#include "Statistics.h"

class PlumeInputData;
class PLUMEGeneralData;

class StatisticsDirector
{
public:
  /**
   * /brief Adds a key value pair to the container.
   *
   * Given a key (as a string), and a value of any type, this will store
   * the data in the container for retrieval later.
   *
   * @param startCollectionTime Time to start collecting.
   * @param collectionPeriod    collection period.
   * @param outfile             output file for statistics.
   */

  StatisticsDirector(const QEStime &, const float &, QESFileOutput_Interface *);

  ~StatisticsDirector();

  DataSource *get(const std::string &key) { return dataSources[key]; }
  bool is_set(const std::string &key) { return !(dataSources.find(key) == dataSources.end()); }
  typename std::map<std::string, DataSource *>::iterator begin() { return dataSources.begin(); }
  typename std::map<std::string, DataSource *>::iterator end() { return dataSources.end(); }

  void attach(const std::string &key, DataSource *s);

  void compute(QEStime &, const float &);

  QESFileOutput_Interface *getOutputFile() { return m_statsFile; }

protected:
  std::map<std::string, DataSource *> dataSources;

  QEStime m_startCollectionTime;
  QEStime m_nextOutputTime;
  float m_collectionPeriod = 0;

  QESFileOutput_Interface *m_statsFile{};

private:
  StatisticsDirector() = default;
};