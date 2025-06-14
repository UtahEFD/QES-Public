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

/** @file SourceCube.cpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once

#include "PI_SourceComponent.hpp"
#include "SourceGeometryCube.h"

class PI_SourceGeometry_Cube : public PI_SourceComponent
{
private:
  float m_minX = -1.0;
  float m_minY = -1.0;
  float m_minZ = -1.0;
  float m_maxX = -1.0;
  float m_maxY = -1.0;
  float m_maxZ = -1.0;

protected:
public:
  // Default constructor
  PI_SourceGeometry_Cube() : PI_SourceComponent() {}

  // destructor
  ~PI_SourceGeometry_Cube() = default;

  void parseValues() override
  {
    parsePrimitive<float>(true, m_minX, "minX");
    parsePrimitive<float>(true, m_minY, "minY");
    parsePrimitive<float>(true, m_minZ, "minZ");
    parsePrimitive<float>(true, m_maxX, "maxX");
    parsePrimitive<float>(true, m_maxY, "maxY");
    parsePrimitive<float>(true, m_maxZ, "maxZ");
  }

  SourceComponent *create(QESDataTransport &data) override;
};
