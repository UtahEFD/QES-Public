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

/** @file SourceGeometrySphereShell.cpp
 * @brief
 */

#include "SourceGeometrySphereShell.h"
#include "PLUMEGeneralData.h"

SourceGeometrySphereShell::SourceGeometrySphereShell(const vec3 &min, const float &radius)
  : m_x(min), m_radius(radius)
{
  prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
  normal = std::normal_distribution<float>(0.0, 1.0);
}

void SourceGeometrySphereShell::generate(const QEStime &currTime, const int &n, QESDataTransport &data)
{
  std::vector<vec3> init(n);

  for (int k = 0; k < n; ++k) {
    // init[k] = { m_posX_0 + t * m_diffX, m_posY_0 + t * m_diffY, m_posZ_0 + t * m_diffZ };
    float nx = normal(prng);
    float ny = normal(prng);
    float nz = normal(prng);
    float overn = 1 / sqrt(nx * nx + ny * ny + nz * nz);
    init[k]._1 = m_x._1 + m_radius * nx * overn;
    init[k]._2 = m_x._2 + m_radius * ny * overn;
    init[k]._3 = m_x._3 + m_radius * nz * overn;
  }
  data.put("position", init);
}
