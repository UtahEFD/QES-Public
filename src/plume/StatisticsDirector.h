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

#pragma once

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util/QEStime.h"
#include "util/QESOutputInterface.h"

#include "Statistics.h"

class PlumeInputData;
class PLUMEGeneralData;

class StatisticsDirector
{
public:
  StatisticsDirector(const PlumeInputData *, PLUMEGeneralData *);

  ~StatisticsDirector() = default;

  Statistics *get(const std::string &key) { return elements[key]; }
  typename std::unordered_map<std::string, Statistics *>::iterator begin() { return elements.begin(); }
  typename std::unordered_map<std::string, Statistics *>::iterator end() { return elements.end(); }

  void attach(const std::string &key, Statistics *s);

  void compute(QEStime &, const float &);

  bool getOutputStatus() { return need_output; }
  void resetOutputStatus() { need_output = false; }

protected:
  std::unordered_map<std::string, Statistics *> elements;

  QEStime averagingStartTime;
  QEStime nextOutputTime;
  float averagingPeriod = 0;
  float ongoingAveragingTime = 0;

private:
  StatisticsDirector() = default;

  bool need_reset = false;
  bool need_output = false;
};