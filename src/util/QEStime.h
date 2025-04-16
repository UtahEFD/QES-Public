/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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
    // btime::ptime epoch(bgreg::date(1970, 1, 1));
    // m_ptime = epoch;
  }

  explicit QEStime(time_t t)
  {
    m_ptime = btime::from_time_t(t);
  }
  explicit QEStime(const std::string &t)
  {
    m_ptime = btime::from_iso_extended_string(t);
  }
  explicit QEStime(btime::ptime t)
  {
    m_ptime = t;
  }

  ~QEStime() = default;

  void setTimeToNow();
  void setTimestamp(const btime::ptime &);
  void setTimestamp(const time_t &);
  void setTimestamp(const std::string &);

  QEStime &operator=(const btime::ptime &);
  QEStime &operator=(const time_t &);
  QEStime &operator=(const std::string &);


  std::string getTimestamp() const;
  void getTimestamp(std::string &) const;
  time_t getEpochTime();


  void increment(float);
  QEStime &operator+=(const float &);
  QEStime operator+(const float &);

  double operator-(const QEStime &);

  bool operator==(const std::string &) const;
  bool operator==(const QEStime &) const;
  bool operator!=(const QEStime &) const;

  bool operator<=(const QEStime &) const;
  bool operator<(const QEStime &) const;

  bool operator>=(const QEStime &) const;
  bool operator>(const QEStime &) const;

  double operator%(const double &);

  friend std::ostream &operator<<(std::ostream &, const QEStime &);

private:
  btime::ptime m_ptime;
};
