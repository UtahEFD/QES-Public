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

#include <boost/date_time/posix_time/posix_time.hpp>

namespace bgreg = boost::gregorian;
namespace btime = boost::posix_time;

class QEStime
{
public:
  QEStime() {}

  ~QEStime() {}

  void setTime(double);
  QEStime &operator=(const double &);

  void setTimestamp();
  void setTimestamp(time_t);
  void setTimestamp(std::string);
  void setTimestamp(time_t, std::string);
  QEStime &operator=(const std::string &);

  bool timestampMode()
  {
    return m_timestamp_mode;
  }

  std::string getTimestamp()
  {
    if (m_timestamp_mode)
      return btime::to_iso_extended_string(m_ptime);
    else
      return "0000-00-00T00:00:00";
  }

  double getTime()
  {
    return m_time;
  }

  time_t getEpochTime()
  {
    btime::ptime epoch(bgreg::date(1970, 1, 1));
    btime::time_duration::sec_type x = (m_ptime - epoch).total_seconds();
    return time_t(x);
  }

  void increment(float);

private:
  bool m_timestamp_mode = false;
  double m_time = 0.0;
  btime::ptime m_ptime;
};
