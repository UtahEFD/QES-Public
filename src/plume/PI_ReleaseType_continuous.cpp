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

/** @file ReleaseType_continuous.cpp
 * @brief This class represents a specific release type.
 *
 * @note Child of ReleaseType
 * @sa ReleaseType
 */

#include "PI_ReleaseType_continuous.hpp"
#include "PLUMEGeneralData.h"

void PI_ReleaseType_continuous::parseValues()
{
  parsePrimitive<int>(true, m_particlePerTimestep, "particlePerTimestep");
  parsePrimitive<float>(false, m_releaseStartTime, "releaseStartTime");
  parsePrimitive<float>(false, m_massPerSec, "massPerSec");
  parsePrimitive<float>(false, m_massPerTimestep, "massPerTimestep");

  if (m_releaseStartTime == -1) {
    m_releaseStartTime = 0;
  }
  m_releaseEndTime = 1.0E10;

  if (m_massPerSec != 0.0 && m_massPerTimestep != 0.0) {
    throw std::runtime_error("[ReleaseType_continuous] at MOST ONE must be set: massPerSec or massPerTimestep");
  }
}

void PI_ReleaseType_continuous::initialize(const float &timestep)
{
  // set the overall releaseType variables from the variables found in this class
  if (m_massPerSec != 0.0 && m_massPerTimestep == 0.0) {
    m_massPerTimestep = m_massPerSec * timestep;
  }
}

SourceReleaseController *PI_ReleaseType_continuous::create(QESDataTransport &data)
{
  auto PGD = data.get_ref<PLUMEGeneralData *>("PGD");
  QEStime start = PGD->getSimTimeStart() + m_releaseStartTime;
  QEStime end = start + m_releaseEndTime;

  return new SourceReleaseController_base(start, end, m_particlePerTimestep, m_massPerTimestep);
}