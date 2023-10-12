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

/** @file ReleaseType_continuous.hpp
 * @brief This class represents a specific release type.
 *
 * @note Child of ReleaseType
 * @sa ReleaseType
 */

#pragma once

#include "ReleaseType.hpp"

class ReleaseType_duration : public ReleaseType
{
private:
  // note that this also inherits data members ParticleReleaseType m_rType, int m_parPerTimestep, double m_releaseStartTime,
  //  double m_releaseEndTime, and int m_numPar from ReleaseType.
  // guidelines for how to set these variables within an inherited ReleaseType are given in ReleaseType.hpp.

protected:
public:
  // Default constructor
  ReleaseType_duration() : ReleaseType(ParticleReleaseType::duration)
  {
  }

  // destructor
  ~ReleaseType_duration() = default;

  void parseValues() override
  {
    parsePrimitive<double>(true, m_releaseStartTime, "releaseStartTime");
    parsePrimitive<double>(true, m_releaseEndTime, "releaseEndTime");
    parsePrimitive<int>(true, m_parPerTimestep, "parPerTimestep");
  }


  void calcReleaseInfo(const double &timestep, const double &simDur) override
  {
    // set the overall releaseType variables from the variables found in this class
    double releaseDur = m_releaseEndTime - m_releaseStartTime;
    if (releaseDur <= 0) {
      std::cerr << "[ERROR]" << std::endl;
      exit(1);
    }
    int nReleaseTimes = std::ceil(releaseDur / timestep);
    m_numPar = m_parPerTimestep * nReleaseTimes;
  }
};
