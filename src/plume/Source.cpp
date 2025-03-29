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

/** @file Source.cpp
 * @brief  This class represents a generic source type
 *
 * @note
 */

#include "Source.h"

Source::Source(SourceReleaseController *r)
  : m_release(r)
{
  auto sourceID = SourceIDGen::getInstance();
  m_id = sourceID->get();

  m_components.emplace_back(new SetParticleID());
  m_components.emplace_back(new SetMass(m_release));
}


bool Source::isActive(const QEStime &currTime) const
{
  return (currTime >= m_release->startTime() && currTime <= m_release->endTime());
}

int Source::generate(const QEStime &currTime)
{
  // query how many particle need to be released
  if (isActive(currTime)) {
    // m_releaseType->m_particlePerTimestep;
    int n = m_release->particles(currTime);
    for (auto c : m_components)
      c->generate(currTime, n, m_data);

    // update source counter
    if (m_data.contains("mass")) {
      for (auto m : m_data.get_ref<std::vector<float>>("mass"))
        total_mass += m;

    }
    total_particle_released += n;

    return n;
  } else {
    m_data.clear();
    return 0;
  }
}
