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

/** @file SourceGeometryFullDomain.cpp
 * @brief
 */

#include "SourceGeometryFullDomain.h"

#include "PI_SourceGeometry.hpp"
#include "PI_SourceGeometry_Cube.hpp"
#include "PI_SourceGeometry_FullDomain.hpp"
#include "PI_SourceGeometry_Line.hpp"
#include "PI_SourceGeometry_Point.hpp"
#include "PI_SourceGeometry_SphereShell.hpp"

#include "PLUMEGeneralData.h"

SourceGeometryFullDomain::SourceGeometryFullDomain()
{
  prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
  uniform = std::uniform_real_distribution<float>(0.0, 1.0);
}

SourceGeometryFullDomain::SourceGeometryFullDomain(const PI_SourceGeometry_FullDomain *param)
{
  prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
  uniform = std::uniform_real_distribution<float>(0.0, 1.0);
}

void SourceGeometryFullDomain::initialize(const WINDSGeneralData *WGD, const PLUMEGeneralData *PGD)
{
  PGD->interp->getDomainBounds(xStart, yStart, zStart, xEnd, yEnd, zEnd);
}

void SourceGeometryFullDomain::generate(const QEStime &currTime, const int &n, QESDataTransport &data)
{
  std::vector<vec3> init(n);

  for (int k = 0; k < n; ++k) {
    // init[k] = { m_posX_0 + t * m_diffX, m_posY_0 + t * m_diffY, m_posZ_0 + t * m_diffZ };
    init[k]._1 = uniform(prng) * (xEnd - xStart) + xStart;
    init[k]._2 = uniform(prng) * (yEnd - yStart) + yStart;
    init[k]._3 = uniform(prng) * (zEnd - zStart) + zStart;
  }
  data.put("position", init);
}