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

/** @file SourcePoint.hpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once


#include "PI_SourceGeometry.hpp"
#include "SourceGeometryPoint.h"

class PI_SourceGeometry_Point : public PI_SourceGeometry
{
private:
  // position of the source
  float posX = -1.0f;
  float posY = -1.0f;
  float posZ = -1.0f;

protected:
public:
  // Default constructor
  PI_SourceGeometry_Point() : PI_SourceGeometry(SourceShape::point)
  {
  }

  // destructor
  ~PI_SourceGeometry_Point() = default;


  void parseValues() override
  {
    parsePrimitive<float>(true, posX, "posX");
    parsePrimitive<float>(true, posY, "posY");
    parsePrimitive<float>(true, posZ, "posZ");
  }


  void checkPosInfo(const float &domainXstart,
                    const float &domainXend,
                    const float &domainYstart,
                    const float &domainYend,
                    const float &domainZstart,
                    const float &domainZend) override;

  void setInitialPosition(vec3 &) override;

  SourceComponent *create(QESDataTransport &data) override
  {
    // return new SourceGeometryPoint(this);
    return new SourceGeometryPoint({ posX, posY, posZ });
  }
};
