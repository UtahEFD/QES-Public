/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

#include "SourceGeometry.hpp"

class SourceGeometry_Cube : public SourceGeometry
{
private:
  // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
  // guidelines for how to set these variables within an inherited source are given in SourceType.

  double m_minX = -1.0;
  double m_minY = -1.0;
  double m_minZ = -1.0;
  double m_maxX = -1.0;
  double m_maxY = -1.0;
  double m_maxZ = -1.0;
  double sourceStrength = 0.0;// total mass released (g)

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> uniformDistribution;

protected:
public:
  // Default constructor
  SourceGeometry_Cube() : SourceGeometry(SourceShape::cube)
  {
    prng = std::mt19937(rd());// Standard mersenne_twister_engine seeded with rd()
    uniformDistribution = std::uniform_real_distribution<>(0.0, 1.0);
  }

  // destructor
  ~SourceGeometry_Cube() = default;

  void parseValues() override
  {
    parsePrimitive<double>(true, m_minX, "minX");
    parsePrimitive<double>(true, m_minY, "minY");
    parsePrimitive<double>(true, m_minZ, "minZ");
    parsePrimitive<double>(true, m_maxX, "maxX");
    parsePrimitive<double>(true, m_maxY, "maxY");
    parsePrimitive<double>(true, m_maxZ, "maxZ");

    parsePrimitive<double>(false, sourceStrength, "sourceStrength");
  }


  void checkPosInfo(const double &domainXstart,
                    const double &domainXend,
                    const double &domainYstart,
                    const double &domainYend,
                    const double &domainZstart,
                    const double &domainZend) override;

  void setInitialPosition(double &, double &, double &) override;
};
