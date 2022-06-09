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

/** @file QEStime.h */

#pragma once

#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>

namespace bgreg = boost::gregorian;
namespace btime = boost::posix_time;

class QEStime
{
public:
  QEStime()
  {
    btime::ptime UTCtime = btime::second_clock::universal_time();
    m_ptime = UTCtime;
    //btime::ptime epoch(bgreg::date(1970, 1, 1));
    //m_ptime = epoch;
  }

  QEStime(time_t t)
  {
    m_ptime = btime::from_time_t(t);
  }
  QEStime(std::string t)
  {
    m_ptime = btime::from_iso_extended_string(t);
  }
  QEStime(btime::ptime t)
  {
    m_ptime = t;
  }

  ~QEStime() {}

  void setTime(double);
  QEStime &operator=(const double &);
  QEStime &operator=(const btime::ptime &);

  void setTimestamp(time_t);
  void setTimestamp(std::string);
  QEStime &operator=(const std::string &);

  std::string getTimestamp();
  double getTime();
  time_t getEpochTime();

  void increment(float);
  QEStime &operator+=(const float &);
  QEStime operator+(const float &);

  double operator-(const QEStime &);

  bool operator==(const std::string &);
  bool operator==(const QEStime &);
  bool operator!=(const QEStime &);

  bool operator<=(const QEStime &);
  bool operator<(const QEStime &);

  bool operator>=(const QEStime &);
  bool operator>(const QEStime &);

  double operator%(const double &);

  friend std::ostream &operator<<(std::ostream &, QEStime &);

private:
  btime::ptime m_ptime;
};
