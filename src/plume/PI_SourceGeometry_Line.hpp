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

/** @file SourceLine.hpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once

#include "PI_SourceComponent.hpp"
#include "SourceGeometryLine.h"

class PI_SourceGeometry_Line : public PI_SourceComponent
{
private:
  float m_posX_0 = -1.0;
  float m_posY_0 = -1.0;
  float m_posZ_0 = -1.0;
  float m_posX_1 = -1.0;
  float m_posY_1 = -1.0;
  float m_posZ_1 = -1.0;

protected:
public:
  // Default constructor
  PI_SourceGeometry_Line() : PI_SourceComponent()
  {
  }

  // destructor
  ~PI_SourceGeometry_Line() = default;

  void parseValues() override
  {
    parsePrimitive<float>(true, m_posX_0, "m_posX_0");
    parsePrimitive<float>(true, m_posY_0, "m_posY_0");
    parsePrimitive<float>(true, m_posZ_0, "m_posZ_0");
    parsePrimitive<float>(true, m_posX_1, "m_posX_1");
    parsePrimitive<float>(true, m_posY_1, "m_posY_1");
    parsePrimitive<float>(true, m_posZ_1, "m_posZ_1");
  }

  SourceComponent *create(QESDataTransport &data) override;
};
