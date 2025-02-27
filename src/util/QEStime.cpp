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

void QEStime::setTimestamp(const btime::ptime &t)
{
  m_ptime = t;
}

void QEStime::setTimestamp(const time_t &t)
{
  m_ptime = btime::from_time_t(t);
}

void QEStime::setTimestamp(const std::string &t)
{
  m_ptime = btime::from_iso_extended_string(t);
}

QEStime &QEStime::operator=(const btime::ptime &t)
{
  m_ptime = t;
  return *this;
}

QEStime &QEStime::operator=(const time_t &t)
{
  m_ptime = btime::from_time_t(t);
  return *this;
}

QEStime &QEStime::operator=(const std::string &t)
{
  m_ptime = btime::from_iso_extended_string(t);
  return *this;
}

std::string QEStime::getTimestamp() const
{
  return btime::to_iso_extended_string(m_ptime);
}

void QEStime::getTimestamp(std::string &inout) const
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
  btime::time_duration tt = btime::milliseconds((unsigned long)round(1000.0 * (double)dt));
  m_ptime += tt;
}

QEStime &QEStime::operator+=(const float &dt)
{
  btime::time_duration tt = btime::milliseconds((unsigned long)round(1000.0 * (double)dt));
  m_ptime += tt;
  return *this;
}

QEStime QEStime::operator+(const float &dt)
{
  btime::time_duration tt = btime::milliseconds((unsigned long)round(1000.0 * (double)dt));
  QEStime t(this->m_ptime + tt);
  return t;
}

double QEStime::operator-(const QEStime &t2)
{
  double x = (m_ptime - t2.m_ptime).total_milliseconds() / 1000.0;
  return x;
}

bool QEStime::operator==(const std::string &t) const
{
  btime::ptime testtime = btime::from_iso_extended_string(t);
  return m_ptime == testtime;
}

bool QEStime::operator==(const QEStime &t) const
{
  return m_ptime == t.m_ptime;
}

bool QEStime::operator!=(const QEStime &t) const
{
  return m_ptime != t.m_ptime;
}

bool QEStime::operator<=(const QEStime &t) const
{
  return m_ptime <= t.m_ptime;
}

bool QEStime::operator<(const QEStime &t) const
{
  return m_ptime < t.m_ptime;
}

bool QEStime::operator>=(const QEStime &t) const
{
  return m_ptime >= t.m_ptime;
}

bool QEStime::operator>(const QEStime &t) const
{
  return m_ptime > t.m_ptime;
}

double QEStime::operator%(const double &t)
{
  btime::ptime epoch(bgreg::date(1970, 1, 1));
  return ((m_ptime - epoch).total_milliseconds() % int(t * 1000.0)) / 1000.0;
}

std::ostream &operator<<(std::ostream &os, const QEStime &t)
{
  os << t.getTimestamp();
  return os;
}
