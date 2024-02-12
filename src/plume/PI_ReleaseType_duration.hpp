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

/** @file ReleaseType_continuous.hpp
 * @brief This class represents a specific release type.
 *
 * @note Child of ReleaseType
 * @sa ReleaseType
 */

#pragma once

#include "PI_ReleaseType.hpp"

class PI_ReleaseType_duration : public PI_ReleaseType
{
private:
  // note that this also inherits data members ParticleReleaseType m_rType, int m_particlePerTimestep, double m_releaseStartTime,
  //  double m_releaseEndTime, and int m_numPar from ReleaseType.
  // guidelines for how to set these variables within an inherited ReleaseType are given in ReleaseType.hpp.

protected:
public:
  // Default constructor
  PI_ReleaseType_duration() : PI_ReleaseType(ParticleReleaseType::duration)
  {
  }

  // destructor
  ~PI_ReleaseType_duration() = default;

  void parseValues() override
  {
    parsePrimitive<double>(true, m_releaseStartTime, "releaseStartTime");
    parsePrimitive<double>(true, m_releaseEndTime, "releaseEndTime");
    parsePrimitive<int>(true, m_particlePerTimestep, "particlePerTimestep");
    parsePrimitive<double>(false, m_totalMass, "totalMass");
    parsePrimitive<double>(false, m_massPerSec, "massPerSec");
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
    m_numPar = m_particlePerTimestep * nReleaseTimes;
    if (m_totalMass != 0.0 && m_massPerSec != 0.0) {
      std::cerr << "[ERROR]" << std::endl;
      exit(1);
    } else if (m_totalMass != 0.0) {
      m_massPerSec = m_totalMass / releaseDur;
      m_massPerParticle = m_totalMass / m_numPar;
    } else if (m_massPerSec != 0.0) {
      m_totalMass = m_massPerSec * releaseDur;
      m_massPerParticle = m_massPerSec / (m_particlePerTimestep / timestep);
    } else {
      m_massPerParticle = 0.0;
    }
  }
};
