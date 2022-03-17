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

/** @file QESTime.cpp */

#include "QEStime.h"


void QEStime::setTime(double t)
{
  m_time = t;
}
QEStime &QEStime::operator=(const double &t)
{
  m_time = t;
  return *this;
}

void QEStime::setTimestamp()
{
  m_timestamp_mode = true;
  std::cout << btime::to_iso_extended_string(m_ptime) << std::endl;
}

void QEStime::setTimestamp(time_t t)
{
  m_timestamp_mode = true;
  m_ptime = btime::from_time_t(t);
}

void QEStime::setTimestamp(std::string t)
{
  m_timestamp_mode = true;
  m_ptime = btime::from_iso_extended_string(t);
}

void QEStime::setTimestamp(time_t t1, std::string t2)
{
  m_timestamp_mode = true;
  m_ptime = btime::from_iso_extended_string(t2);
  btime::ptime testtime = btime::from_time_t(t1);
  if (testtime != m_ptime) {
    std::cerr << "[ERROR] invalid timeStamp (timeEpoch != timeStamp)\n";
    exit(EXIT_FAILURE);
  }
}

QEStime &QEStime::operator=(const std::string &t)
{
  m_timestamp_mode = true;
  m_ptime = btime::from_iso_extended_string(t);
  return *this;
}

void QEStime::increment(float dt)
{
  m_time += dt;
  if (m_timestamp_mode) {
    btime::time_duration tt = btime::milliseconds((int)(1000.0d * (double)dt));
    m_ptime += tt;
  }
}
