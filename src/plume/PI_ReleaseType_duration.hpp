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
  float m_totalMass = 0;
  float m_massPerSec = 0;
  float m_duration = 0.0;

protected:
public:
  // Default constructor
  PI_ReleaseType_duration() : PI_ReleaseType()
  {
  }

  // destructor
  ~PI_ReleaseType_duration() override = default;

  void parseValues() override
  {
    parsePrimitive<float>(true, m_releaseStartTime, "releaseStartTime");
    parsePrimitive<float>(false, m_releaseEndTime, "releaseEndTime");
    parsePrimitive<float>(false, m_duration, "releaseDuration");
    parsePrimitive<int>(true, m_particlePerTimestep, "particlePerTimestep");
    parsePrimitive<float>(false, m_totalMass, "totalMass");
    parsePrimitive<float>(false, m_massPerSec, "massPerSec");

    if (m_duration != 0.0 && m_releaseEndTime != -1) {
      throw std::runtime_error("[ReleaseType_duration] at MOST ONE must be set: releaseEndTime or releaseDuration");
    } else if (m_duration == 0.0 && m_releaseEndTime == -1) {
      throw std::runtime_error("[ReleaseType_duration] at LEAST ONE must be set: releaseEndTime or releaseDuration");
    } else if (m_duration != 0.0 && m_releaseEndTime == -1) {
      m_releaseEndTime = m_releaseStartTime + m_duration;
    } else {
      m_duration = m_releaseEndTime - m_releaseStartTime;
    }

    if (m_releaseEndTime < m_releaseStartTime) {
      throw std::runtime_error("[ReleaseType_duration] invalid number of particles");
    }
  }

  void initialize(const float &timestep) override
  {
    // set the overall releaseType variables from the variables found in this class
    if (m_duration <= 0) {
      std::cerr << "[ERROR]" << std::endl;
      exit(1);
    }
    int nReleaseTimes = std::ceil(m_duration / timestep);
    int numPar = m_particlePerTimestep * nReleaseTimes;
    if (m_totalMass != 0.0 && m_massPerSec != 0.0) {
      std::cerr << "[ERROR]" << std::endl;
      exit(1);
    } else if (m_totalMass != 0.0) {
      m_massPerSec = m_totalMass / m_duration;
      m_massPerTimestep = m_totalMass / (float)m_particlePerTimestep;
    } else if (m_massPerSec != 0.0) {
      m_totalMass = m_massPerSec * m_duration;
      m_massPerTimestep = m_massPerSec * timestep;
    } else {
      m_massPerTimestep = 0.0;
    }
  }

  SourceReleaseController *create(QESDataTransport &data) override
  {
    QEStime start = QEStime("2020-01-01T00:00") + m_releaseStartTime;
    QEStime end = start + m_releaseEndTime;

    return new SourceReleaseController_base(start, end, m_particlePerTimestep, m_massPerTimestep);
  }
};
