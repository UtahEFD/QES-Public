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

/** @file SourceCircle.hpp
 * @brief This class represents a specific source type.
 *
 * @note Child of SourceType
 * @sa SourceType
 */

#pragma once


#include "SourceType.hpp"
#include "winds/WINDSGeneralData.h"
// #include "Particles.hpp"

class SourceCircle : public SourceType
{
private:
  // note that this also inherits public data members ReleaseType* m_rType and SourceShape m_sShape.
  // guidelines for how to set these variables within an inherited source are given in SourceType.

  double posX;
  double posY;
  double posZ;
  double radius;
  double sourceStrength = 0.0;// total mass released (g)
protected:
public:
  // Default constructor
  SourceCircle()
  {
  }

  // destructor
  ~SourceCircle()
  {
  }


  virtual void parseValues()
  {
    m_sShape = SourceShape::circle;

    setReleaseType();
    setParticleType();

    parsePrimitive<double>(true, posX, "posX");
    parsePrimitive<double>(true, posY, "posY");
    parsePrimitive<double>(true, posZ, "posZ");
    parsePrimitive<double>(true, radius, "radius");

    parsePrimitive<double>(false, sourceStrength, "sourceStrength");
  }


  void checkPosInfo(const double &domainXstart, const double &domainXend, const double &domainYstart, const double &domainYend, const double &domainZstart, const double &domainZend);


  int emitParticles(const float dt, const float currTime, std::list<Particle *> &emittedParticles);
};
