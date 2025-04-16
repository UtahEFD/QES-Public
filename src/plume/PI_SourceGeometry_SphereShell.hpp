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

/** @file SourceCircle.hpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once

#include "PI_SourceComponent.hpp"
#include "SourceGeometrySphereShell.h"

class PI_SourceGeometry_SphereShell : public PI_SourceComponent
{
private:
  float m_posX = -1.0;
  float m_posY = -1.0;
  float m_posZ = -1.0;
  float radius = -1.0;

protected:
public:
  // Default constructor
  PI_SourceGeometry_SphereShell() : PI_SourceComponent() {}

  // destructor
  ~PI_SourceGeometry_SphereShell() = default;

  void parseValues() override
  {
    parsePrimitive<float>(true, m_posX, "posX");
    parsePrimitive<float>(true, m_posY, "posY");
    parsePrimitive<float>(true, m_posZ, "posZ");
    parsePrimitive<float>(true, radius, "radius");
  }

  SourceComponent *create(QESDataTransport &data) override;
};
