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

/** @file QESTime.cpp */

#include "QEStime.h"


void QEStime::setTimeToNow()
{
  btime::ptime UTCtime = btime::second_clock::universal_time();
  m_ptime = UTCtime;
}

void QEStime::setTimestamp(time_t t)
{
  m_ptime = btime::from_time_t(t);
}

void QEStime::setTimestamp(std::string t)
{
  m_ptime = btime::from_iso_extended_string(t);
}

QEStime &QEStime::operator=(const std::string &t)
{
  m_ptime = btime::from_iso_extended_string(t);
  return *this;
}

QEStime &QEStime::operator=(const btime::ptime &t)
{
  m_ptime = t;
  return *this;
}

std::string QEStime::getTimestamp()
{
  return btime::to_iso_extended_string(m_ptime);
}

void QEStime::getTimestamp(std::string &inout)
{
  inout = btime::to_iso_extended_string(m_ptime);
}

time_t QEStime::getEpochTime()
{
  btime::ptime epoch(bgreg::date(1970, 1, 1));
  time_t x = (m_ptime - epoch).total_seconds();
  return x;
}

void QEStime::increment(float dt)
{
  btime::time_duration tt = btime::milliseconds((int)round(1000.0d * (double)dt));
  m_ptime += tt;
}

QEStime &QEStime::operator+=(const float &dt)
{
  btime::time_duration tt = btime::milliseconds((int)round(1000.0d * (double)dt));
  m_ptime += tt;
  return *this;
}

QEStime QEStime::operator+(const float &dt)
{
  btime::time_duration tt = btime::milliseconds((int)round(1000.0d * (double)dt));
  QEStime t = this->m_ptime + tt;
  return t;
}

double QEStime::operator-(const QEStime &t2)
{
  double x = (m_ptime - t2.m_ptime).total_milliseconds() / 1000.0;
  return x;
}

bool QEStime::operator==(const std::string &t)
{
  btime::ptime testtime = btime::from_iso_extended_string(t);
  return m_ptime == testtime;
}

bool QEStime::operator==(const QEStime &t)
{
  return m_ptime == t.m_ptime;
}

bool QEStime::operator!=(const QEStime &t)
{
  return m_ptime != t.m_ptime;
}

bool QEStime::operator<=(const QEStime &t)
{
  return m_ptime <= t.m_ptime;
}

bool QEStime::operator<(const QEStime &t)
{
  return m_ptime < t.m_ptime;
}

bool QEStime::operator>=(const QEStime &t)
{
  return m_ptime >= t.m_ptime;
}

bool QEStime::operator>(const QEStime &t)
{
  return m_ptime > t.m_ptime;
}

double QEStime::operator%(const double &t)
{
  btime::ptime epoch(bgreg::date(1970, 1, 1));
  return ((m_ptime - epoch).total_milliseconds() % int(t * 1000.0)) / 1000.0;
}

std::ostream &operator<<(std::ostream &os, QEStime &t)
{
  os << t.getTimestamp();
  return os;
}
